# --*-- coding:utf-8 --*--
# @time: 2026-03-30
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File: ca_anchor_refine.py

"""
CA-anchored backbone refinement using PyRosetta.

Problem
-------
NERF backbone reconstruction places atom positions using ideal peptide
geometry propagated from residue 1.  Small fitting errors accumulate along
the chain, shifting the final CA positions by up to several Å relative to the
original lattice CA trace.  This raises the Cα RMSD even though the lattice
already provides a good coarse-grained conformation.

Solution
--------
After NERF builds the initial full-atom structure, apply strong harmonic
coordinate constraints that pull each CA back toward its original scaled
lattice position.  A Cartesian-space energy minimisation then adjusts ALL
backbone and sidechain atoms while satisfying both the constraints (CA → target)
and the energy function (good bond geometry, no clashes).  A final rotamer
repacking polishes the sidechains.

Result: CA positions ≈ input lattice positions (RMSD preserved from Stage 2).
"""

from __future__ import annotations

import logging
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

_LOG = logging.getLogger(__name__)

# ── PyRosetta singleton ────────────────────────────────────────────────────

_ROSETTA_READY: bool = False
_PR_NS: Any = None   # cached pyrosetta namespace


def _ensure_rosetta():
    """Initialise PyRosetta once; return (pyrosetta, ScoreFunctionFactory)."""
    global _ROSETTA_READY, _PR_NS
    if not _ROSETTA_READY:
        import pyrosetta
        from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
        pyrosetta.init(
            "-mute all -ignore_unrecognized_res -ignore_zero_occupancy false "
            "-run:constant_seed -run:jran 42",
            silent=True,
        )
        _PR_NS = pyrosetta
        _ROSETTA_READY = True
    from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
    return _PR_NS, ScoreFunctionFactory


# ── Configuration ──────────────────────────────────────────────────────────

@dataclass
class CAAnchorRefineConfig:
    """
    Parameters
    ----------
    ca_k : float
        Harmonic spring constant for CA coordinate constraints
        (kcal/mol/Å²).  Higher values keep CAs tighter to their
        target lattice positions; values < 10 allow significant drift.
        Default 50 gives ≈ 0.1-0.2 Å tolerance.
    coord_cst_weight : float
        Score-function weight for the coordinate_constraint term.
        Effective stiffness = ca_k * coord_cst_weight.
    cart_min_tol : float
        Gradient tolerance for Cartesian minimisation.
    n_min_cycles : int
        Number of Cartesian minimisation passes.
    pack_sidechains : bool
        If True, run PackRotamersMover + chi MinMover after minimisation
        to give well-packed sidechains for scoring.
    use_extra_rotamers : bool
        If True, enable the ex1/ex2 rotamer library.
    """
    ca_k: float = 50.0
    coord_cst_weight: float = 5.0
    cart_min_tol: float = 0.001
    n_min_cycles: int = 2
    pack_sidechains: bool = True
    use_extra_rotamers: bool = True


# ── Refiner ────────────────────────────────────────────────────────────────

class CAAnchorRefiner:
    """
    Refines a full-atom structure so its CA atoms match the target lattice
    positions while maintaining good backbone / sidechain geometry.

    Usage
    -----
    >>> refiner = CAAnchorRefiner()
    >>> refined_pdb = refiner.refine(pdb_string, scaled_ca_positions)
    """

    def __init__(self, config: Optional[CAAnchorRefineConfig] = None) -> None:
        self.config = config or CAAnchorRefineConfig()

    def refine(self, pdb_string: str, target_ca: np.ndarray) -> str:
        """
        Refine the full-atom structure toward target CA positions.

        Parameters
        ----------
        pdb_string : str
            Full-atom PDB (heavy atoms) from NERF reconstruction.
        target_ca : (N, 3) ndarray
            Desired CA positions in Å (scaled lattice coords from Stage 2).

        Returns
        -------
        str
            Refined PDB string with CA positions close to target_ca.
        """
        pyrosetta, ScoreFunctionFactory = _ensure_rosetta()

        from pyrosetta.rosetta.core.scoring import ScoreType
        from pyrosetta.rosetta.core.kinematics import MoveMap
        from pyrosetta.rosetta.protocols.minimization_packing import (
            MinMover, PackRotamersMover,
        )
        from pyrosetta.rosetta.core.pack.task import TaskFactory
        from pyrosetta.rosetta.core.pack.task.operation import (
            RestrictToRepacking, ExtraRotamers,
        )
        from pyrosetta.rosetta.core.scoring.constraints import CoordinateConstraint
        from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
        from pyrosetta.rosetta.core.id import AtomID
        from pyrosetta.rosetta.numeric import xyzVector_double_t

        cfg = self.config
        target_ca = np.asarray(target_ca, dtype=float)

        # ── Load pose ────────────────────────────────────────────────
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pdb", delete=False
        ) as tmp:
            tmp.write(pdb_string)
            tmp_path = tmp.name
        try:
            pose = pyrosetta.pose_from_pdb(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        n_res = pose.total_residue()
        pre_ca_rmsd = _ca_rmsd_to_target(pose, target_ca)
        _LOG.debug("CAAnchorRefiner: pre-refine CA RMSD to target = %.3f Å", pre_ca_rmsd)

        # ── Build constrained score function ─────────────────────────
        sfxn_cart = ScoreFunctionFactory.create_score_function("ref2015_cart")
        sfxn_cart.set_weight(ScoreType.coordinate_constraint, cfg.coord_cst_weight)

        # ── Add CA coordinate constraints → scaled_ca targets ────────
        sd = 1.0 / math.sqrt(max(cfg.ca_k, 1e-6))   # HarmonicFunc sd param
        anchor_id = AtomID(1, 1)                      # global reference atom

        n_cst = 0
        for i in range(min(n_res, len(target_ca))):
            seqpos = i + 1
            res = pose.residue(seqpos)
            if not res.has("CA"):
                continue
            ca_idx = res.atom_index("CA")
            ca_id = AtomID(ca_idx, seqpos)
            xyz = xyzVector_double_t(
                float(target_ca[i, 0]),
                float(target_ca[i, 1]),
                float(target_ca[i, 2]),
            )
            func = HarmonicFunc(0.0, sd)
            cst = CoordinateConstraint(ca_id, anchor_id, xyz, func)
            pose.add_constraint(cst)
            n_cst += 1

        _LOG.debug("Added %d CA coordinate constraints (k=%.1f, sd=%.4f Å).",
                   n_cst, cfg.ca_k, sd)

        # ── Cartesian minimisation (backbone + sidechain) ────────────
        mm_all = MoveMap()
        mm_all.set_bb(True)
        mm_all.set_chi(True)

        for cycle in range(cfg.n_min_cycles):
            min_mover = MinMover(
                mm_all, sfxn_cart,
                "lbfgs_armijo_nonmonotone",
                cfg.cart_min_tol,
                True,   # use nb list
            )
            min_mover.cartesian(True)
            min_mover.apply(pose)

        post_ca_rmsd = _ca_rmsd_to_target(pose, target_ca)
        _LOG.info(
            "CAAnchorRefiner: CA RMSD to lattice target  %.3f → %.3f Å",
            pre_ca_rmsd, post_ca_rmsd,
        )

        # ── Remove constraints before sidechain packing ─────────────
        pose.remove_constraints()

        # ── Sidechain repacking ──────────────────────────────────────
        if cfg.pack_sidechains:
            sfxn_std = ScoreFunctionFactory.create_score_function("ref2015")

            tf = TaskFactory()
            tf.push_back(RestrictToRepacking())
            if cfg.use_extra_rotamers:
                tf.push_back(ExtraRotamers(0, 1, 1))

            packer = PackRotamersMover(sfxn_std)
            packer.task_factory(tf)
            packer.apply(pose)

            mm_chi = MoveMap()
            mm_chi.set_bb(False)
            mm_chi.set_chi(True)
            min_chi = MinMover(
                mm_chi, sfxn_std,
                "lbfgs_armijo_nonmonotone", 0.01, True,
            )
            min_chi.apply(pose)

        return _pose_to_pdb_string(pose)


# ── Geometry helpers ────────────────────────────────────────────────────────

def _ca_rmsd_to_target(pose: Any, target_ca: np.ndarray) -> float:
    """Compute RMSD between pose CA atoms and target_ca positions."""
    n_res = pose.total_residue()
    n = min(n_res, len(target_ca))
    pred = []
    for i in range(n):
        seqpos = i + 1
        res = pose.residue(seqpos)
        if res.has("CA"):
            xyz = res.xyz("CA")
            pred.append([xyz.x, xyz.y, xyz.z])
        else:
            pred.append(target_ca[i].tolist())
    pred = np.array(pred, dtype=float)
    ref = target_ca[:n]
    diff = pred - ref
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def _pose_to_pdb_string(pose: Any) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
        out_path = tmp.name
    try:
        pose.dump_pdb(out_path)
        with open(out_path) as f:
            return f.read()
    finally:
        Path(out_path).unlink(missing_ok=True)


__all__ = ["CAAnchorRefineConfig", "CAAnchorRefiner"]
