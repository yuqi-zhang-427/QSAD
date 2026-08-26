# --*-- coding:utf-8 --*--
# @time: 2026-03-30
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File: md_annealer.py

"""
OpenMM simulated annealing refiner.

Runs repeated heat → cool cycles to escape local energy minima that
are inaccessible to gradient-descent minimizers alone.  Tracks the
lowest-energy frame across all cycles and returns it as a heavy-atom PDB.

Protocol
--------
1. PDBFixer  : add hydrogens, fix missing heavy atoms.
2. Bonded-only minimization: relieve gross geometry violations safely.
3. CA-restrained full-force minimization: relax sidechains while keeping
   the backbone trace from the ca2all reconstruction.
4. Annealing cycles: heat to T_high → gradual cooling to T_low → minimize.
   The lowest-energy frame across all cycles is tracked.
5. Final unconstrained minimization on the best frame.
6. Strip hydrogens; return heavy-atom PDB string.

The CA restraints in steps 3–4 are set to a *looser* value than the
scorer uses (2000 vs 5000 kJ/mol/nm²), allowing the backbone to drift
slightly during annealing while still preserving the overall topology.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_LOG = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────

@dataclass
class MDAnnealerConfig:
    """
    Parameters
    ----------
    force_field / implicit_solvent :
        OpenMM XML files.  Implicit solvent avoids expensive water equilibration.
    bonded_min_steps :
        L-BFGS iterations for bonded-only pre-minimization.
    restrained_min_steps :
        L-BFGS iterations for CA-restrained full-force minimization.
    ca_restraint_k :
        CA harmonic restraint strength (kJ/mol/nm²).  Looser than the scorer
        (2000 vs 5000) to let the backbone sample nearby conformations.
    anneal_cycles :
        Number of heat → cool cycles.
    high_temp_K / low_temp_K :
        Temperature range.  500 K provides enough kinetic energy to cross
        ~2 kT barriers; 300 K is physiological.
    high_temp_steps / cool_steps :
        MD steps per heating and cooling phase.
    timestep_ps :
        Langevin integrator timestep (1 fs = 0.001 ps is conservative).
    final_min_steps :
        L-BFGS iterations for post-annealing minimization.
    platform :
        OpenMM platform ("CPU", "CUDA", "OpenCL", "Reference").
    """
    force_field: str = "amber14-all.xml"
    implicit_solvent: str = "implicit/gbn2.xml"

    bonded_min_steps: int = 500
    restrained_min_steps: int = 1000
    ca_restraint_k: float = 2000.0          # kJ/mol/nm²

    anneal_cycles: int = 3
    high_temp_K: float = 500.0
    low_temp_K: float = 300.0
    high_temp_steps: int = 2000
    cool_steps: int = 3000
    n_cool_segments: int = 5                # gradual cooling segments
    timestep_ps: float = 0.001

    final_min_steps: int = 1000
    final_tolerance: float = 10.0           # kJ/mol/nm

    platform: str = "CPU"


# ── Annealer ───────────────────────────────────────────────────────────────

class MDAnnealer:
    """
    OpenMM simulated annealing refiner.

    Usage
    -----
    >>> annealer = MDAnnealer()
    >>> refined_pdb = annealer.anneal(pdb_string)
    """

    def __init__(self, config: Optional[MDAnnealerConfig] = None) -> None:
        self.config = config or MDAnnealerConfig()

        try:
            import openmm
            import openmm.app as app
            import openmm.unit as unit
            from pdbfixer import PDBFixer
        except ImportError as exc:
            raise ImportError(
                "OpenMM ≥ 8 and PDBFixer required for MDAnnealer.\n"
                "Install: conda install -c conda-forge openmm pdbfixer\n"
                f"{exc}"
            ) from exc

        self._openmm = openmm
        self._app = app
        self._unit = unit
        self._PDBFixer = PDBFixer

        ff_files = [self.config.force_field, self.config.implicit_solvent]
        self._forcefield = app.ForceField(*ff_files)
        _LOG.info("MDAnnealer: force field loaded (%s + %s).", *ff_files)

    # ── Public API ─────────────────────────────────────────────────────

    def anneal(self, pdb_string: str) -> str:
        """
        Run the full annealing protocol on a PDB string.

        Returns a heavy-atom PDB string of the lowest-energy frame found.
        """
        openmm = self._openmm
        app = self._app
        unit = self._unit

        cfg = self.config

        # ── 1. PDBFixer ───────────────────────────────────────────────
        fixer = self._fix_structure(pdb_string)
        topology = fixer.topology

        try:
            platform = openmm.Platform.getPlatformByName(cfg.platform)
        except Exception:
            _LOG.warning("Platform '%s' not available; falling back to CPU.", cfg.platform)
            platform = openmm.Platform.getPlatformByName("CPU")

        orig_pos_nm = np.array([[p.x, p.y, p.z] for p in fixer.positions])
        ca_indices = [a.index for a in topology.atoms() if a.name == "CA"]

        # ── 2. Bonded-only minimization ───────────────────────────────
        bonded_sys = self._bonded_only_system(topology)
        bonded_integ = openmm.VerletIntegrator(0.001 * unit.picosecond)
        bonded_sim = app.Simulation(topology, bonded_sys, bonded_integ, platform)
        bonded_sim.context.setPositions(fixer.positions)
        try:
            bonded_sim.minimizeEnergy(maxIterations=cfg.bonded_min_steps)
        except Exception as exc:
            _LOG.warning("Bonded minimization stumbled (%s); expanding positions.", exc)
            self._expand(bonded_sim, fixer.positions, 2.5)
            bonded_sim.minimizeEnergy(maxIterations=cfg.bonded_min_steps * 2)

        state1 = bonded_sim.context.getState(getPositions=True)
        pos_after_bonded = state1.getPositions()
        del bonded_sim, bonded_integ, bonded_sys

        # ── 3. CA-restrained full minimization ────────────────────────
        full_sys = self._full_system(topology)
        self._add_ca_restraints(full_sys, ca_indices, orig_pos_nm, cfg.ca_restraint_k)

        main_integ = openmm.LangevinMiddleIntegrator(
            cfg.low_temp_K * unit.kelvin,
            1.0 / unit.picosecond,
            cfg.timestep_ps * unit.picosecond,
        )
        sim = app.Simulation(topology, full_sys, main_integ, platform)
        sim.context.setPositions(pos_after_bonded)

        try:
            sim.minimizeEnergy(maxIterations=cfg.restrained_min_steps)
        except Exception as exc:
            _LOG.warning("Restrained minimization issue: %s", exc)

        # ── 4. Annealing cycles ───────────────────────────────────────
        best_energy = float("inf")
        best_positions = None

        _scoring_groups = set(range(31))  # exclude group 31 (restraints)

        for cycle in range(cfg.anneal_cycles):
            # Heat
            main_integ.setTemperature(cfg.high_temp_K * unit.kelvin)
            sim.step(cfg.high_temp_steps)

            # Gradual cooling
            steps_seg = max(cfg.cool_steps // cfg.n_cool_segments, 1)
            for seg in range(cfg.n_cool_segments):
                frac = (seg + 1) / cfg.n_cool_segments
                t = cfg.high_temp_K + (cfg.low_temp_K - cfg.high_temp_K) * frac
                main_integ.setTemperature(t * unit.kelvin)
                sim.step(steps_seg)

            # Minimize after cooling
            sim.minimizeEnergy(maxIterations=cfg.final_min_steps // 2)

            # Score (excluding restraints)
            state_c = sim.context.getState(
                getEnergy=True, getPositions=True, groups=_scoring_groups,
            )
            e = state_c.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            _LOG.debug("Anneal cycle %d/%d: energy=%.1f kJ/mol", cycle + 1, cfg.anneal_cycles, e)

            if e < best_energy:
                best_energy = e
                best_positions = state_c.getPositions()

        if best_positions is None:
            best_positions = sim.context.getState(getPositions=True).getPositions()

        # ── 5. Final unconstrained minimization ───────────────────────
        sim.context.setPositions(best_positions)
        sim.context.setParameter("k_ca_restraint", 0.0)  # zero out restraints
        try:
            tol = cfg.final_tolerance * unit.kilojoules_per_mole / unit.nanometer
            sim.minimizeEnergy(maxIterations=cfg.final_min_steps, tolerance=tol)
        except Exception as exc:
            _LOG.warning("Final minimization issue: %s", exc)

        final_state = sim.context.getState(getPositions=True)

        # ── 6. Return heavy-atom PDB ──────────────────────────────────
        return self._heavy_atom_pdb(topology, final_state.getPositions())

    # ── Helpers ────────────────────────────────────────────────────────

    def _fix_structure(self, pdb_string: str):
        PDBFixer = self._PDBFixer
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
            tmp.write(pdb_string)
            tmp_path = tmp.name
        try:
            fixer = PDBFixer(filename=tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.positions = self._uniform_expand(fixer.positions, scale=2.5)
        fixer.addMissingHydrogens(7.0)
        return fixer

    def _uniform_expand(self, positions, scale: float = 2.5):
        openmm = self._openmm
        unit = self._unit
        pos = np.array([[p.x, p.y, p.z] for p in positions], dtype=np.float64)
        centroid = pos.mean(axis=0)
        n = len(pos)
        phi = (1 + np.sqrt(5)) / 2
        idx = np.arange(n, dtype=np.float64)
        jitter = np.column_stack([
            np.sin(idx * phi) * 0.002,
            np.cos(idx * phi * 2) * 0.002,
            np.sin(idx * phi * 3 + 1) * 0.002,
        ])
        expanded = centroid + (pos + jitter - centroid) * scale
        new_pos = [openmm.Vec3(float(p[0]), float(p[1]), float(p[2])) for p in expanded]
        return unit.Quantity(new_pos, unit.nanometer)

    def _expand(self, simulation, positions, scale: float):
        openmm = self._openmm
        unit = self._unit
        pos = np.array([[p.x, p.y, p.z] for p in positions])
        centroid = pos.mean(axis=0)
        expanded = centroid + (pos - centroid) * scale
        new_pos = [openmm.Vec3(float(p[0]), float(p[1]), float(p[2])) for p in expanded]
        simulation.context.setPositions(unit.Quantity(new_pos, unit.nanometer))

    def _bonded_only_system(self, topology):
        openmm = self._openmm
        app = self._app
        system = self._forcefield.createSystem(
            topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds,
        )
        keep = (
            openmm.HarmonicBondForce,
            openmm.HarmonicAngleForce,
            openmm.PeriodicTorsionForce,
            openmm.CMMotionRemover,
        )
        to_remove = [
            i for i in range(system.getNumForces())
            if not isinstance(system.getForce(i), keep)
        ]
        for i in reversed(to_remove):
            system.removeForce(i)
        return system

    def _full_system(self, topology):
        app = self._app
        return self._forcefield.createSystem(
            topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds,
        )

    def _add_ca_restraints(self, system, ca_indices, orig_pos_nm, k: float):
        openmm = self._openmm
        restraint = openmm.CustomExternalForce(
            "0.5*k_ca_restraint*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
        )
        restraint.addGlobalParameter("k_ca_restraint", k)
        restraint.addPerParticleParameter("x0")
        restraint.addPerParticleParameter("y0")
        restraint.addPerParticleParameter("z0")
        restraint.setForceGroup(31)
        for ca_idx in ca_indices:
            x0, y0, z0 = float(orig_pos_nm[ca_idx, 0]), float(orig_pos_nm[ca_idx, 1]), float(orig_pos_nm[ca_idx, 2])
            restraint.addParticle(ca_idx, [x0, y0, z0])
        system.addForce(restraint)

    def _heavy_atom_pdb(self, topology, positions) -> str:
        """Write heavy-atom PDB from OpenMM topology and positions.

        Produces column-exact PDB ATOM records compatible with PyRosetta:
          cols 1-6  : "ATOM  "
          cols 7-11 : serial
          col  12   : space
          cols 13-16: atom name
          col  17   : altLoc (space)      ← critical: separate from residue name
          cols 18-20: residue name
          col  21   : space
          col  22   : chain ID
          cols 23-26: residue seq number
          col  27   : insertion code (space)
          cols 28-30: spaces
          cols 31-54: X Y Z
          cols 55-60: occupancy
          cols 61-66: B-factor
          cols 77-78: element symbol
        """
        pos_nm = np.array([[p.x, p.y, p.z] for p in positions])
        pos_ang = pos_nm * 10.0  # nm → Å

        lines = ["REMARK  MDAnnealer output — heavy atoms only\n"]
        serial = 1
        res_seq_counter = 0
        prev_res_uid = None

        for atom in topology.atoms():
            if atom.element is None or atom.element.symbol == "H":
                continue

            res = atom.residue
            res_uid = (res.chain.id, res.id)   # unique residue key
            if res_uid != prev_res_uid:
                res_seq_counter += 1
                prev_res_uid = res_uid

            res_name = res.name[:3].upper()
            chain_id = (res.chain.id or "A")[:1]
            aname = atom.name
            # PDB atom-name convention: 4-char field, left-aligned for
            # multi-char elements (e.g. " CA "), right-pad otherwise
            aname_padded = f" {aname:<3}" if len(aname) < 4 else f"{aname:<4}"
            x, y, z = pos_ang[atom.index]
            elem = atom.element.symbol[:2].rjust(2)

            # Column-exact format: note the space for altLoc after aname_padded
            lines.append(
                f"ATOM  {serial:5d} {aname_padded} {res_name:>3} {chain_id}"
                f"{res_seq_counter:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
                f"          {elem}\n"
            )
            serial += 1

        lines.append("END\n")
        return "".join(lines)


__all__ = ["MDAnnealerConfig", "MDAnnealer"]
