# --*-- coding:utf-8 --*--
# @time: 2026-03-30
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File: local_rebuilder.py

"""
Local structure rebuilding for high-energy regions using PyRosetta.

For each WeakRegion identified by EnergyAnalyzer, applies:
1. Backbone perturbation via SmallMover / ShearMover with Monte Carlo acceptance.
2. Expanded rotamer repacking for the region and its shell neighbors.
3. Local backbone + sidechain minimization.

This stage actively explores alternative conformations in problematic segments,
allowing the structure to escape shallow local minima before global relaxation.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from .energy_analyzer import WeakRegion, _ensure_rosetta

_LOG = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────

@dataclass
class LocalRebuilderConfig:
    """
    Parameters
    ----------
    n_backbone_trials :
        MC backbone perturbation attempts per weak region.
    bb_temperature :
        kT for Metropolis acceptance (higher → more aggressive moves).
    angle_max_loop :
        Max backbone angle perturbation (degrees) for loop residues.
    angle_max_helix :
        Max backbone angle perturbation for helix/strand residues.
    n_repack_cycles :
        Rotamer repacking cycles after backbone moves.
    use_extra_rotamers :
        If True, use expanded ex1/ex2 rotamer library.
    padding :
        Extra residues on each side of the weak region included in moves.
    """
    n_backbone_trials: int = 30
    bb_temperature: float = 2.0
    angle_max_loop: float = 8.0
    angle_max_helix: float = 5.0
    n_repack_cycles: int = 3
    use_extra_rotamers: bool = True
    padding: int = 2
    backbone_moves: bool = True


# ── Rebuilder ──────────────────────────────────────────────────────────────

class LocalRebuilder:
    """
    Local structure rebuilder for high-energy regions.

    Accepts a PDB string and a list of WeakRegions; returns a new PDB string
    with those regions rebuilt and locally minimized.
    """

    def __init__(self, config: Optional[LocalRebuilderConfig] = None) -> None:
        self.config = config or LocalRebuilderConfig()
        self._scorefxn: Any = None

    def _get_scorefxn(self):
        if self._scorefxn is None:
            _, ScoreFunctionFactory = _ensure_rosetta()
            self._scorefxn = ScoreFunctionFactory.create_score_function("ref2015")
        return self._scorefxn

    def rebuild(self, pdb_string: str, weak_regions: List[WeakRegion]) -> str:
        """
        Rebuild high-energy regions and return an improved PDB string.

        Parameters
        ----------
        pdb_string : str
            Full-atom PDB (heavy atoms; PyRosetta will handle H).
        weak_regions : List[WeakRegion]
            Sorted high-energy segments (worst first).

        Returns
        -------
        str
            PDB string of the best (lowest-energy) structure found.
        """
        pyrosetta, _ = _ensure_rosetta()

        import pyrosetta.rosetta.core.scoring as _cs
        from pyrosetta.rosetta.core.kinematics import MoveMap
        from pyrosetta.rosetta.protocols.minimization_packing import (
            MinMover, PackRotamersMover,
        )
        from pyrosetta.rosetta.core.pack.task import TaskFactory
        from pyrosetta.rosetta.core.pack.task.operation import (
            RestrictToRepacking, ExtraRotamers,
        )
        from pyrosetta.rosetta.protocols.simple_moves import SmallMover, ShearMover
        from pyrosetta.rosetta.protocols.moves import MonteCarlo

        sfxn = self._get_scorefxn()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
            tmp.write(pdb_string)
            tmp_path = tmp.name
        try:
            pose = pyrosetta.pose_from_pdb(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        n_res = pose.total_residue()
        best_pose = pose.clone()
        best_score = float(sfxn(best_pose))

        cfg = self.config

        for region in weak_regions:
            # Convert 0-based to 1-based Rosetta residue indices with padding
            r_start = max(1, region.start + 1 - cfg.padding)
            r_end = min(n_res, region.end + 1 + cfg.padding)
            region_res = list(range(r_start, r_end + 1))

            # ── MoveMap for this region ───────────────────────────────
            mm = MoveMap()
            mm.set_bb(False)
            mm.set_chi(False)
            for i in region_res:
                if cfg.backbone_moves:
                    mm.set_bb(i, True)
                mm.set_chi(i, True)

            # ── Backbone movers (skipped in sidechain-only mode) ──────
            if cfg.backbone_moves:
                small = SmallMover(mm, cfg.bb_temperature, 1)
                small.angle_max("H", cfg.angle_max_helix)
                small.angle_max("E", cfg.angle_max_helix)
                small.angle_max("L", cfg.angle_max_loop)

                shear = ShearMover(mm, cfg.bb_temperature, 1)
                shear.angle_max("H", cfg.angle_max_helix)
                shear.angle_max("E", cfg.angle_max_helix)
                shear.angle_max("L", cfg.angle_max_loop)

                mc = MonteCarlo(best_pose, sfxn, cfg.bb_temperature)

                for trial in range(cfg.n_backbone_trials):
                    trial_pose = best_pose.clone()
                    (small if trial % 2 == 0 else shear).apply(trial_pose)
                    mc.boltzmann(trial_pose)

                mc.recover_low(best_pose)

            # ── Rotamer repacking ─────────────────────────────────────
            tf = TaskFactory()
            tf.push_back(RestrictToRepacking())
            if cfg.use_extra_rotamers:
                tf.push_back(ExtraRotamers(0, 1, 1))  # ex1 ex2

            packer = PackRotamersMover(sfxn)
            packer.task_factory(tf)
            for _ in range(cfg.n_repack_cycles):
                packer.apply(best_pose)

            # ── Local minimization ────────────────────────────────────
            min_mover = MinMover(
                mm, sfxn, "lbfgs_armijo_nonmonotone", 0.01, True,
            )
            min_mover.apply(best_pose)

            new_score = float(sfxn(best_pose))
            _LOG.debug(
                "  Local rebuild region [%d-%d]: %.3f → %.3f",
                region.start, region.end, best_score, new_score,
            )
            best_score = new_score

        return _pose_to_pdb_string(best_pose)


# ── Helper ─────────────────────────────────────────────────────────────────

def _pose_to_pdb_string(pose: Any) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
        out_path = tmp.name
    try:
        pose.dump_pdb(out_path)
        with open(out_path) as f:
            return f.read()
    finally:
        Path(out_path).unlink(missing_ok=True)


__all__ = ["LocalRebuilderConfig", "LocalRebuilder"]
