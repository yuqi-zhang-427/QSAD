# --*-- coding:utf-8 --*--
# @time: 2026-03-30
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File: global_relaxer.py

"""
Aggressive multi-round Cartesian FastRelax using PyRosetta.

Unlike the constrained FastRelax used in the ca2all reranker (which preserves
the lattice-derived backbone trace), this relaxer:

* Does NOT constrain to starting coordinates — the backbone can move freely.
* Runs in Cartesian space — avoids torsion-space local minima.
* Repeats for multiple independent rounds, keeping the best result.
* Uses an expanded rotamer library (ex1/ex2) for better sidechain packing.

These settings give the relaxer a much larger conformational search radius,
enabling it to find lower-energy states that constrained relax would miss.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .energy_analyzer import _ensure_rosetta
from .local_rebuilder import _pose_to_pdb_string

_LOG = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────

@dataclass
class GlobalRelaxerConfig:
    """
    Parameters
    ----------
    n_rounds :
        Number of independent FastRelax runs.  Each run starts from the
        current best structure.  Best of all rounds is returned.
    cartesian :
        If True, use Cartesian-space FastRelax (ref2015_cart score function).
        More freedom than torsion-space, but ~2× slower.
    constrain_to_start :
        If False (default), no harmonic coordinate constraints are applied.
        The backbone is free to rearrange substantially.
    use_extra_rotamers :
        If True, use ex1/ex2 expanded rotamer library.
    max_iter :
        Minimizer iterations per FastRelax cycle.
    score_function :
        Base score function name.  Cartesian mode automatically switches to
        the ``_cart`` variant.
    """
    n_rounds: int = 3
    cartesian: bool = True
    constrain_to_start: bool = False
    use_extra_rotamers: bool = True
    max_iter: int = 500
    score_function: str = "ref2015"
    backbone_moves: bool = True


# ── Relaxer ────────────────────────────────────────────────────────────────

class GlobalRelaxer:
    """
    Multi-round aggressive Cartesian FastRelax (no start constraints).

    Usage
    -----
    >>> relaxer = GlobalRelaxer()
    >>> refined_pdb = relaxer.relax(pdb_string)
    """

    def __init__(self, config: Optional[GlobalRelaxerConfig] = None) -> None:
        self.config = config or GlobalRelaxerConfig()
        self._sfxn: Any = None

    def _get_sfxn(self):
        if self._sfxn is None:
            _, ScoreFunctionFactory = _ensure_rosetta()
            sfxn_name = (
                self.config.score_function + "_cart"
                if self.config.cartesian
                else self.config.score_function
            )
            try:
                self._sfxn = ScoreFunctionFactory.create_score_function(sfxn_name)
            except Exception:
                _LOG.warning(
                    "Score function '%s' not found; falling back to '%s'.",
                    sfxn_name, self.config.score_function,
                )
                self._sfxn = ScoreFunctionFactory.create_score_function(
                    self.config.score_function
                )
        return self._sfxn

    def relax(self, pdb_string: str) -> str:
        """
        Run multi-round aggressive FastRelax.

        Returns the PDB string of the lowest-score structure found.
        """
        pyrosetta, _ = _ensure_rosetta()

        from pyrosetta.rosetta.core.kinematics import MoveMap
        from pyrosetta.rosetta.protocols.relax import FastRelax
        from pyrosetta.rosetta.core.pack.task import TaskFactory
        from pyrosetta.rosetta.core.pack.task.operation import ExtraRotamers

        sfxn = self._get_sfxn()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
            tmp.write(pdb_string)
            tmp_path = tmp.name
        try:
            pose = pyrosetta.pose_from_pdb(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        initial_score = float(sfxn(pose))
        best_pose = pose.clone()
        best_score = initial_score

        mm = MoveMap()
        mm.set_bb(self.config.backbone_moves)
        mm.set_chi(True)
        mm.set_jump(self.config.backbone_moves)

        for rnd in range(self.config.n_rounds):
            fr = FastRelax()
            fr.set_scorefxn(sfxn.clone())
            fr.constrain_relax_to_start_coords(self.config.constrain_to_start)
            fr.max_iter(self.config.max_iter)

            if self.config.cartesian:
                fr.cartesian(True)

            if self.config.use_extra_rotamers:
                tf = TaskFactory()
                tf.push_back(ExtraRotamers(0, 1, 1))
                fr.set_task_factory(tf)

            fr.set_movemap(mm)

            working = best_pose.clone()
            fr.apply(working)

            round_score = float(sfxn(working))
            _LOG.debug(
                "  GlobalRelaxer round %d/%d: %.3f → %.3f",
                rnd + 1, self.config.n_rounds, best_score, round_score,
            )

            if round_score < best_score:
                best_score = round_score
                best_pose = working

        _LOG.info(
            "GlobalRelaxer: %.3f → %.3f (%+.3f)",
            initial_score, best_score, best_score - initial_score,
        )

        return _pose_to_pdb_string(best_pose)


__all__ = ["GlobalRelaxerConfig", "GlobalRelaxer"]
