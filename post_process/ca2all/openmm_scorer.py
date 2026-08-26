# --*-- conding:utf-8 --*--
# @time:3/28/26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:openmm_scorer.py

"""
Heavy-duty molecular-mechanics scorer using OpenMM.

Pipeline for each candidate structure:

1. **PDBFixer** -- add missing heavy atoms, add all hydrogens, cap termini.
2. **Force field** -- AMBER ff14SB protein force field.
3. **Implicit solvent** -- Generalised Born / Neck-2 (GBn2) model, which
   captures solvation free energy without explicit water.
4. **Energy minimisation** -- L-BFGS for a configurable number of steps to
   relieve local strain introduced by the lattice→all-atom reconstruction.
5. **Scoring** -- the post-minimisation potential energy is the final score.
   Lower energy = more physically reasonable structure.

The minimisation step is critical: it allows bond lengths / angles to relax
from the idealised NERF placement, so the *remaining* energy primarily
reflects non-local quality (contacts, packing, solvation).

Requires: ``openmm``, ``pdbfixer``.
"""

from __future__ import annotations

import io
import logging
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_LOG = logging.getLogger(__name__)

try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    from pdbfixer import PDBFixer

    _HAS_OPENMM = True
except ImportError:
    _HAS_OPENMM = False


# ── Configuration ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OpenMMScorerConfig:
    """
    Configuration for the OpenMM molecular-mechanics scorer.

    Parameters
    ----------
    force_field        : OpenMM force-field XML files.
    implicit_solvent   : implicit-solvent XML (GBn2 recommended).
    minimise_max_iter  : L-BFGS iterations for energy minimisation.
                         0 = score without minimisation (fast but noisy).
    minimise_tolerance : convergence tolerance (kJ/mol/nm).
    temperature        : simulation temperature for solvation model (K).
    platform_name      : OpenMM platform ("CPU", "CUDA", "OpenCL", or "Reference").
    constraints        : bond-length constraint method (None, HBonds, AllBonds).
    nonbonded_cutoff   : non-bonded cutoff (A); only used with explicit solvent.
    score_components   : if True, decompose energy by force group.
    """
    force_field: Tuple[str, ...] = ("amber14-all.xml",)
    implicit_solvent: str = "implicit/gbn2.xml"
    minimise_max_iter: int = 200
    minimise_tolerance: float = 10.0       # kJ/mol/nm
    temperature: float = 300.0             # K
    platform_name: str = "CPU"
    constraints: Optional[str] = "HBonds"  # speeds up minimisation
    nonbonded_cutoff: float = 0.0          # 0 = NoCutoff (implicit solvent)
    score_components: bool = True


# ── Scorer ─────────────────────────────────────────────────────────────────

class OpenMMScorer:
    """
    Score a full-atom structure using OpenMM molecular mechanics.

    Usage::

        scorer = OpenMMScorer()
        breakdown = scorer.score(build_result)
        # breakdown.total = potential energy in kJ/mol (lower is better)
    """

    def __init__(self, cfg: Optional[OpenMMScorerConfig] = None) -> None:
        if not _HAS_OPENMM:
            raise ImportError(
                "OpenMM and PDBFixer are required for OpenMMScorer. "
                "Install with: conda install -c conda-forge openmm pdbfixer"
            )
        self.cfg = cfg or OpenMMScorerConfig()
        # Pre-load force field (reusable across calls)
        ff_files = list(self.cfg.force_field) + [self.cfg.implicit_solvent]
        self._forcefield = app.ForceField(*ff_files)
        _LOG.info("OpenMM force field loaded: %s", ff_files)

    def score(self, build_result: Any) -> "ScoreBreakdown":
        """
        Score a BuildResult from ca2all.

        Returns ScoreBreakdown with total = potential energy (kJ/mol),
        lower_is_better = True.
        """
        from ..rerank.models import ScoreBreakdown

        pdb_text = build_result.write_pdb()
        try:
            energy, terms = self._evaluate(pdb_text)
        except Exception as exc:
            _LOG.warning("OpenMM scoring failed: %s", exc)
            return ScoreBreakdown(
                total=float("inf"),
                terms={"error": 1.0, "message": str(exc)[:100]},
                engine="openmm_amber14_gbn2",
                lower_is_better=True,
            )

        return ScoreBreakdown(
            total=energy,
            terms=terms,
            engine="openmm_amber14_gbn2",
            lower_is_better=True,
        )

    def score_pdb(self, pdb_text: str) -> Tuple[float, Dict[str, float]]:
        """Score a PDB string directly.  Returns (energy_kJ_mol, terms)."""
        return self._evaluate(pdb_text)

    # ── Core evaluation pipeline ───────────────────────────────────────

    def _evaluate(self, pdb_text: str) -> Tuple[float, Dict[str, float]]:
        """
        Full pipeline: fix → CA-restrained minimise → score (without restraints).

        Lattice-reconstructed structures have severe atom overlaps.  We use:

        Stage 1  Bonded-only minimisation (separate system, no non-bonded)
                 to relieve gross geometry violations safely.
        Stage 2  Full-force CA-restrained minimisation.  Harmonic restraints
                 on CA atoms keep the backbone trace intact while side chains,
                 hydrogens, and local geometry relax.
        Score    After minimisation, the energy is evaluated WITHOUT the
                 restraint force, so it purely reflects physical quality.

        CA restraints are critical: without them, all structures converge to
        similar local minima and discriminative power is lost.
        """
        terms: Dict[str, float] = {}

        # ── 1. PDBFixer: add hydrogens, fix missing atoms ─────────
        fixer = self._fix_structure(pdb_text)
        n_atoms = fixer.topology.getNumAtoms()
        n_residues = fixer.topology.getNumResidues()
        terms["n_atoms_fixed"] = float(n_atoms)
        terms["n_residues"] = float(n_residues)

        try:
            platform = openmm.Platform.getPlatformByName(self.cfg.platform_name)
        except Exception:
            platform = openmm.Platform.getPlatformByName("Reference")

        # Record original CA positions (in nm for OpenMM)
        orig_pos_nm = np.array([[p.x, p.y, p.z] for p in fixer.positions])
        ca_indices = [a.index for a in fixer.topology.atoms() if a.name == "CA"]

        max_iter = self.cfg.minimise_max_iter
        tol_unit = unit.kilojoules_per_mole / unit.nanometer

        if max_iter > 0:
            # ── Stage 1: Bonded-only minimisation (separate system) ──
            bonded_system = self._create_bonded_only_system(fixer.topology)
            bonded_integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
            bonded_sim = app.Simulation(
                fixer.topology, bonded_system, bonded_integrator, platform,
            )
            bonded_sim.context.setPositions(fixer.positions)

            try:
                bonded_sim.minimizeEnergy(
                    maxIterations=max_iter,
                    tolerance=self.cfg.minimise_tolerance * tol_unit,
                )
                terms["stage1_bonded"] = 1.0
            except Exception:
                self._set_expanded_positions(bonded_sim, fixer.positions, 2.0)
                try:
                    bonded_sim.minimizeEnergy(
                        maxIterations=max_iter, tolerance=100.0 * tol_unit,
                    )
                    terms["stage1_bonded"] = 2.0
                except Exception:
                    self._set_expanded_positions(bonded_sim, fixer.positions, 4.0)
                    bonded_sim.minimizeEnergy(
                        maxIterations=max_iter * 3, tolerance=1000.0 * tol_unit,
                    )
                    terms["stage1_bonded"] = 4.0

            state1 = bonded_sim.context.getState(getEnergy=True, getPositions=True)
            stage1_positions = state1.getPositions()
            terms["energy_after_stage1"] = state1.getPotentialEnergy().value_in_unit(
                unit.kilojoules_per_mole
            )
            del bonded_sim, bonded_integrator, bonded_system

            # ── Stage 2: Full forces + CA restraints ─────────────────
            full_system = self._create_system(fixer.topology)

            # Add CA harmonic restraints (strong: 1000 kJ/mol/nm²)
            restraint_force = openmm.CustomExternalForce(
                "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
            )
            restraint_force.addGlobalParameter("k", 5000.0)  # kJ/mol/nm²
            restraint_force.addPerParticleParameter("x0")
            restraint_force.addPerParticleParameter("y0")
            restraint_force.addPerParticleParameter("z0")
            # Use ORIGINAL (pre-expansion) CA positions as restraint targets.
            # This pulls the expanded structure back to the intended backbone.
            for ca_idx in ca_indices:
                x0 = orig_pos_nm[ca_idx, 0]
                y0 = orig_pos_nm[ca_idx, 1]
                z0 = orig_pos_nm[ca_idx, 2]
                restraint_force.addParticle(ca_idx, [x0, y0, z0])

            # Put restraint in a separate force group for exclusion from scoring
            restraint_group = 31  # highest group, excluded from scoring
            restraint_force.setForceGroup(restraint_group)
            full_system.addForce(restraint_force)

            full_integrator = openmm.LangevinMiddleIntegrator(
                self.cfg.temperature * unit.kelvin,
                1.0 / unit.picosecond,
                0.002 * unit.picosecond,
            )
            simulation = app.Simulation(
                fixer.topology, full_system, full_integrator, platform,
            )
            simulation.context.setPositions(stage1_positions)

            # Minimise with restraints -- extra iterations needed because the
            # structure must contract from the expanded state back to the
            # original CA positions under the restraint pull.
            try:
                simulation.minimizeEnergy(
                    maxIterations=max_iter * 5,
                    tolerance=self.cfg.minimise_tolerance * tol_unit,
                )
                terms["stage2_restrained"] = 1.0
            except Exception:
                cur_state = simulation.context.getState(getPositions=True)
                self._set_expanded_positions(
                    simulation, cur_state.getPositions(), 1.5,
                )
                try:
                    simulation.minimizeEnergy(
                        maxIterations=max_iter * 3, tolerance=100.0 * tol_unit,
                    )
                    terms["stage2_restrained"] = 1.5
                except Exception:
                    self._set_expanded_positions(
                        simulation, cur_state.getPositions(), 3.0,
                    )
                    simulation.minimizeEnergy(
                        maxIterations=max_iter * 5, tolerance=1000.0 * tol_unit,
                    )
                    terms["stage2_restrained"] = 3.0

        else:
            full_system = self._create_system(fixer.topology)
            full_integrator = openmm.LangevinMiddleIntegrator(
                self.cfg.temperature * unit.kelvin,
                1.0 / unit.picosecond,
                0.002 * unit.picosecond,
            )
            simulation = app.Simulation(
                fixer.topology, full_system, full_integrator, platform,
            )
            simulation.context.setPositions(fixer.positions)
            restraint_group = 31

        # ── 3. Score WITHOUT restraint contribution ───────────────
        # Get energy from all groups EXCEPT the restraint group (31)
        all_groups_except_restraint = set(range(31))  # groups 0-30
        state = simulation.context.getState(
            getEnergy=True, groups=all_groups_except_restraint,
        )
        total_energy = state.getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        terms["energy_post_minimisation"] = total_energy

        # Also record restraint energy (CA displacement penalty)
        r_state = simulation.context.getState(
            getEnergy=True, groups={restraint_group},
        )
        terms["restraint_energy"] = r_state.getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )

        # ── 4. Energy decomposition ──────────────────────────────
        if self.cfg.score_components:
            _, component_terms = self._get_energy_components(
                simulation, exclude_group=restraint_group,
            )
            terms.update(component_terms)

        # ── 5. CA drift ──────────────────────────────────────────
        min_state = simulation.context.getState(getPositions=True)
        min_pos_raw = min_state.getPositions(asNumpy=True)
        min_pos_ang = np.array(min_pos_raw) * 10.0  # nm → angstrom
        orig_pos_ang = orig_pos_nm * 10.0
        if len(ca_indices) >= 3:
            ca_orig = orig_pos_ang[ca_indices]
            ca_min = min_pos_ang[ca_indices]
            drift = float(np.sqrt(np.mean(np.sum((ca_min - ca_orig) ** 2, axis=1))))
            terms["ca_minimisation_drift"] = drift

        return total_energy, terms

    @staticmethod
    def _set_expanded_positions(simulation, positions, scale: float):
        """Expand positions outward from centroid by *scale* factor."""
        pos = np.array([[p.x, p.y, p.z] for p in positions])
        centroid = pos.mean(axis=0)
        expanded = centroid + (pos - centroid) * scale
        new_pos = [openmm.Vec3(float(p[0]), float(p[1]), float(p[2])) for p in expanded]
        simulation.context.setPositions(unit.Quantity(new_pos, unit.nanometer))

    # ── Internal helpers ───────────────────────────────────────────────

    def _fix_structure(self, pdb_text: str) -> "PDBFixer":
        """Use PDBFixer to prepare the structure for OpenMM.

        Lattice-reconstructed structures have severe atom overlaps that cause
        PDBFixer's ``addMissingHydrogens`` (which runs an internal minimiser)
        to fail with NaN.  We work around this with a multi-step approach:

        1. Fix missing heavy atoms with PDBFixer (no minimisation).
        2. Run a bonded-only (harmonic bonds + angles + torsions) minimisation
           on the heavy-atom structure to relieve geometry violations.
        3. Add hydrogens to the relaxed geometry (now safe for PDBFixer).
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
            tmp.write(pdb_text)
            tmp_path = tmp.name

        try:
            fixer = PDBFixer(filename=tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Step 1: Fix missing heavy atoms (no minimisation needed)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()

        # Step 2: Deterministic uniform expansion from centroid to resolve
        # overlaps before H addition.  Uses a fixed expansion factor (not
        # structure-dependent) so that all candidates are treated equally.
        fixer.positions = self._uniform_expand(fixer.positions, scale=2.5)

        # Step 3: Add hydrogens (safe now that atoms are spread apart)
        fixer.addMissingHydrogens(7.0)

        return fixer

    @staticmethod
    def _uniform_expand(positions, scale: float = 2.5):
        """Expand all positions uniformly outward from the centroid.

        Also adds a tiny deterministic per-atom jitter (based on atom index)
        to break exactly-overlapping positions without introducing
        structure-dependent randomness.

        This is deliberately structure-independent: every candidate gets the
        same expansion factor, avoiding the bias where more compact structures
        (which have more clashes) receive more perturbation.
        """
        pos = np.array([[p.x, p.y, p.z] for p in positions], dtype=np.float64)
        centroid = pos.mean(axis=0)

        # Small deterministic jitter based on atom index (golden ratio hash)
        n = len(pos)
        phi = (1 + np.sqrt(5)) / 2
        idx = np.arange(n, dtype=np.float64)
        jitter = np.column_stack([
            np.sin(idx * phi) * 0.002,        # ~0.02 A
            np.cos(idx * phi * 2) * 0.002,
            np.sin(idx * phi * 3 + 1) * 0.002,
        ])
        pos = pos + jitter

        # Uniform expansion from centroid
        expanded = centroid + (pos - centroid) * scale

        new_pos = [openmm.Vec3(float(p[0]), float(p[1]), float(p[2]))
                   for p in expanded]
        return unit.Quantity(new_pos, unit.nanometer)

    def _create_bonded_only_system(
        self, topology: "app.Topology",
    ) -> "openmm.System":
        """Create a System with only bonded forces (bonds, angles, torsions).

        Non-bonded and GB solvation forces are removed so that minimisation
        never encounters NaN from atom overlaps.
        """
        system = self._create_system(topology)
        bonded_types = (
            openmm.HarmonicBondForce,
            openmm.HarmonicAngleForce,
            openmm.PeriodicTorsionForce,
            openmm.CMMotionRemover,
        )
        to_remove = []
        for i in range(system.getNumForces()):
            if not isinstance(system.getForce(i), bonded_types):
                to_remove.append(i)
        for i in reversed(to_remove):
            system.removeForce(i)
        return system

    def _create_system(self, topology: "app.Topology") -> "openmm.System":
        """Create an OpenMM System with the configured force field."""
        kwargs = {
            "nonbondedMethod": app.NoCutoff,   # appropriate for implicit solvent
        }
        if self.cfg.constraints == "HBonds":
            kwargs["constraints"] = app.HBonds
        elif self.cfg.constraints == "AllBonds":
            kwargs["constraints"] = app.AllBonds

        return self._forcefield.createSystem(topology, **kwargs)

    def _get_energy_components(
        self, simulation: "app.Simulation",
        exclude_group: int = 31,
    ) -> Tuple[float, Dict[str, float]]:
        """Decompose energy by force, excluding the restraint group."""
        system = simulation.system
        terms: Dict[str, float] = {}
        total = 0.0

        # Assign each force to its own group (skip forces in exclude_group)
        group_idx = 0
        force_groups = {}
        for i in range(system.getNumForces()):
            force = system.getForce(i)
            if force.getForceGroup() == exclude_group:
                continue
            force.setForceGroup(group_idx)
            force_groups[group_idx] = force
            group_idx += 1

        simulation.context.reinitialize(preserveState=True)

        for gidx, force in force_groups.items():
            state = simulation.context.getState(getEnergy=True, groups={gidx})
            e = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            force_name = type(force).__name__
            terms[f"force_{force_name}"] = e
            total += e

        terms["energy_total_decomposed"] = total
        return total, terms


# ── Convenience batch scorer ───────────────────────────────────────────────

def score_builds_with_openmm(
    build_results: List[Any],
    cfg: Optional[OpenMMScorerConfig] = None,
) -> List[Tuple[float, Dict[str, float]]]:
    """
    Score a batch of BuildResult objects.  Returns list of (energy, terms).
    """
    scorer = OpenMMScorer(cfg)
    results = []
    for i, build in enumerate(build_results):
        try:
            breakdown = scorer.score(build)
            results.append((breakdown.total, breakdown.terms))
        except Exception as exc:
            _LOG.warning("Build %d failed: %s", i, exc)
            results.append((float("inf"), {"error": 1.0}))
    return results


__all__ = [
    "OpenMMScorer",
    "OpenMMScorerConfig",
    "score_builds_with_openmm",
]
