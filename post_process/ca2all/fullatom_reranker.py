# --*-- conding:utf-8 --*--
# @time:3/28/26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:fullatom_reranker.py

"""
Full-atom reranker: chemistry-based posterior scoring engine.

Each candidate is scored **independently** using PyRosetta's REF2015 energy
function after constrained FastRelax.  The score is absolute -- it does not
depend on the other candidates in the set.

Scoring protocol
----------------
For every full-atom structure:

1. Build full-atom model via ca2all.
2. Load into PyRosetta pose, score with REF2015 (pre-relaxation).
3. Run constrained FastRelax (preserves overall conformation).
4. Score again with REF2015 (post-relaxation).
5. Final score = weighted sum of three energy signals:

   ``score = 2.0*fa_atr/nres + 0.5*total/nres + 2.0*lk_ball_wtd - 1.0*lr_contacts``

   All Rosetta terms follow "lower = better"; lr_contacts is a geometric
   density of long-range CB-CB contacts (positive value).

Calibration
-----------
Optimised on 100-protein benchmark (30 candidates each, constant seed,
exhaustive 3-feature grid + greedy expansion):

* Top-1 match: 11 / 100
* Top-3: 20 / 100
* Top-5: 31 / 100
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .builder import BuilderConfig, BuildResult, FullAtomBuilder
from .ca_anchor_refine import CAAnchorRefiner, CAAnchorRefineConfig

_LOG = logging.getLogger(__name__)


# ── PyRosetta lazy loader ─────────────────────────────────────────────────

def _init_pyrosetta():
    """Import and initialise PyRosetta.  Returns a namespace dict or raises."""
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory, ScoreType
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pyrosetta.rosetta.protocols.relax import FastRelax
    pyrosetta.init(
        "-mute all -ignore_unrecognized_res -ignore_zero_occupancy false "
        "-run:constant_seed -run:jran 42"
    )
    return {
        "pyrosetta": pyrosetta,
        "ScoreFunctionFactory": ScoreFunctionFactory,
        "ScoreType": ScoreType,
        "MoveMap": MoveMap,
        "FastRelax": FastRelax,
    }


# ── Scoring formula ──────────────────────────────────────────────────────
#
# All terms are extracted AFTER constrained FastRelax and follow Rosetta's
# "lower = better" convention.  The final score is simply a weighted sum.
#
# Feature                    | Meaning                       | Weight  | Sign
# ---------------------------|-------------------------------|---------|------
# post_fa_atr / n_res        | VDW attraction per residue    |  2.0    |  +  (fa_atr is negative)
# post_total / n_res         | Total energy per residue      |  0.5    |  +  (typically negative)
# post_lk_ball_wtd           | LK solvation (incl. ball)     |  2.0    |  +  (negative = buried)
# lr_contacts                | Long-range CB contacts / nres |  1.0    |  -  (positive; subtract to reward contacts)

_W_FA_ATR_PER_RES = 2.0
_W_PER_RES_MEAN = 0.5
_W_LK_BALL = 2.0
_W_LR_CONTACTS = 1.0


# ── Configuration ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FullAtomRerankConfig:
    """
    Configuration for the PyRosetta-based full-atom reranker.

    Parameters
    ----------
    sidechain_only_scoring : bool
        If True (default), replace backbone FastRelax with sidechain-only
        PackRotamers + chi MinMover.  The CA/backbone positions from the
        lattice are kept exactly, so the RMSD stays close to the Stage-2
        coarse-grained value.  Set False to use the old constrained
        FastRelax (which inadvertently moves the backbone ~2-3 Å).
    """
    builder_config: BuilderConfig = field(default_factory=BuilderConfig)
    fastrelax_repeats: int = 1
    max_iter: int = 200
    top_k_input: Optional[int] = None
    sidechain_only_scoring: bool = True
    use_ca_anchor_refine: bool = True
    ca_anchor_config: CAAnchorRefineConfig = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.ca_anchor_config is None:
            object.__setattr__(self, "ca_anchor_config", CAAnchorRefineConfig())


# ── Result containers ─────────────────────────────────────────────────────

@dataclass
class RankedFullAtomCandidate:
    """One candidate with absolute PyRosetta-based score attached."""
    candidate: Any
    build_result: BuildResult
    fullatom_score: float
    fullatom_terms: Dict[str, float]
    rank: int = 0

    # backward compat
    coarse_score: float = 0.0
    combined_score: float = 0.0
    coarse_z: float = 0.0
    fullatom_z: float = 0.0

    # CA-anchor refined structure (set when use_ca_anchor_refine=True)
    refined_pdb: Optional[str] = None

    def write_pdb(self, path: Optional[str] = None) -> str:
        """Return (and optionally write) the best available PDB string.

        Uses ``refined_pdb`` (CA-anchor corrected) when available,
        otherwise falls back to the raw NERF build result.
        """
        pdb = self.refined_pdb or self.build_result.write_pdb()
        if path is not None:
            with open(path, "w") as fh:
                fh.write(pdb)
        return pdb

    def to_record(self) -> Dict[str, Any]:
        return {
            "candidate_id": getattr(self.candidate, "candidate_id", ""),
            "sequence": getattr(self.candidate, "sequence", ""),
            "bitstring": getattr(self.candidate, "bitstring", ""),
            "rank": self.rank,
            "fullatom_score": float(self.fullatom_score),
            "combined_score": float(self.fullatom_score),
            "fullatom_terms": {k: float(v) for k, v in self.fullatom_terms.items()},
            "n_atoms": self.build_result.n_atoms,
            "scale_factor": float(self.build_result.scale_factor),
        }


@dataclass
class FullAtomRerankResult:
    """Output of the full-atom reranking stage."""
    ranked: List[RankedFullAtomCandidate]
    config: FullAtomRerankConfig

    @property
    def best(self) -> Optional[RankedFullAtomCandidate]:
        return self.ranked[0] if self.ranked else None

    @property
    def n_candidates(self) -> int:
        return len(self.ranked)

    def top_k(self, k: int) -> List[RankedFullAtomCandidate]:
        return self.ranked[:k]

    def write_best_pdb(self, path: str) -> Optional[str]:
        if self.best is None:
            return None
        return self.best.build_result.write_pdb(path)

    def write_top_k_pdb(self, directory: str, k: int = 5) -> List[str]:
        import os
        os.makedirs(directory, exist_ok=True)
        paths: List[str] = []
        for entry in self.ranked[:k]:
            cid = getattr(entry.candidate, "candidate_id", f"rank{entry.rank}")
            safe_id = "".join(
                c if c.isalnum() or c in "_-" else "_" for c in str(cid)
            )
            path = os.path.join(directory, f"rank{entry.rank}_{safe_id}.pdb")
            entry.write_pdb(path)   # uses refined_pdb when available
            paths.append(path)
        return paths

    def summary_records(self) -> List[Dict[str, Any]]:
        return [entry.to_record() for entry in self.ranked]


# ── Reranker ──────────────────────────────────────────────────────────────

class FullAtomReranker:
    """
    Chemistry-based posterior reranker using PyRosetta REF2015.

    Each structure is scored absolutely after constrained FastRelax using
    three energy signals: VDW attraction per residue, total energy per
    residue, and Lazaridis-Karplus solvation.

    Usage
    -----
    >>> reranker = FullAtomReranker()
    >>> result = reranker.rerank(candidates)
    >>> result.best.build_result.write_pdb("best.pdb")
    """

    def __init__(self, config: Optional[FullAtomRerankConfig] = None) -> None:
        self.config = config or FullAtomRerankConfig()
        self._builder = FullAtomBuilder(self.config.builder_config)

        # Initialise PyRosetta (required)
        self._pr = _init_pyrosetta()
        sfxn_factory = self._pr["ScoreFunctionFactory"]
        self._scorefxn = sfxn_factory.create_score_function("ref2015")

        # Optional CA-anchor refiner
        self._ca_refiner: Optional[CAAnchorRefiner] = (
            CAAnchorRefiner(self.config.ca_anchor_config)
            if self.config.use_ca_anchor_refine else None
        )

        # Resolve ScoreType enums we need
        ST = self._pr["ScoreType"]
        self._st_fa_atr = getattr(ST, "fa_atr")
        self._st_lk_ball_wtd = getattr(ST, "lk_ball_wtd")

        _LOG.info("FullAtomReranker initialised with PyRosetta REF2015.")

    # ── Geometric feature ─────────────────────────────────────────────

    @staticmethod
    def _cb_long_range_contacts(pose, dist_cutoff=8.0, seq_sep_min=6) -> float:
        """
        Count CB-CB contacts with sequence separation >= seq_sep_min.
        Returns contacts / n_residues (absolute per-structure density).
        """
        n_res = pose.total_residue()
        cbs = {}
        for i in range(1, n_res + 1):
            res = pose.residue(i)
            atom_name = "CB" if res.has("CB") else "CA"
            xyz = res.xyz(atom_name)
            cbs[i] = np.array([xyz.x, xyz.y, xyz.z])
        contacts = 0
        for i in range(1, n_res + 1):
            for j in range(i + seq_sep_min, n_res + 1):
                if np.linalg.norm(cbs[i] - cbs[j]) < dist_cutoff:
                    contacts += 1
        return contacts / max(n_res, 1)

    # ── Core: score one structure ─────────────────────────────────────

    def _score_single(self, pdb_text: str) -> Dict[str, float]:
        """
        Run the scoring protocol on one PDB string.

        Returns dict with all terms and the final_score (lower = better).
        """
        pr = self._pr
        MoveMap = pr["MoveMap"]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pdb", delete=False
        ) as tmp:
            tmp.write(pdb_text)
            tmp_path = tmp.name
        try:
            pose = pr["pyrosetta"].pose_from_pdb(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        n_res = max(pose.total_residue(), 1)

        # ── Pre-relaxation score ──────────────────────────────────
        pre_total = float(self._scorefxn(pose))

        if self.config.sidechain_only_scoring:
            # ── Sidechain-only: PackRotamers + chi MinMover ───────
            # Backbone / CA positions are kept exactly as built from
            # the lattice, preserving the Stage-2 coarse RMSD.
            from pyrosetta.rosetta.protocols.minimization_packing import (
                PackRotamersMover, MinMover,
            )
            from pyrosetta.rosetta.core.pack.task import TaskFactory
            from pyrosetta.rosetta.core.pack.task.operation import (
                RestrictToRepacking, ExtraRotamers,
            )

            tf = TaskFactory()
            tf.push_back(RestrictToRepacking())
            tf.push_back(ExtraRotamers(0, 1, 1))   # ex1/ex2 rotamer library

            packer = PackRotamersMover(self._scorefxn)
            packer.task_factory(tf)
            packer.apply(pose)

            mm_chi = MoveMap()
            mm_chi.set_bb(False)
            mm_chi.set_chi(True)
            min_mover = MinMover(
                mm_chi, self._scorefxn, "lbfgs_armijo_nonmonotone", 0.01, True,
            )
            min_mover.apply(pose)

        else:
            # ── Legacy: constrained full-backbone FastRelax ───────
            FastRelax = pr["FastRelax"]
            fr = FastRelax()
            fr.set_scorefxn(self._scorefxn.clone())
            fr.constrain_relax_to_start_coords(True)
            fr.max_iter(self.config.max_iter)

            mm = MoveMap()
            mm.set_bb(True)
            mm.set_chi(True)
            mm.set_jump(True)
            fr.set_movemap(mm)

            for _ in range(self.config.fastrelax_repeats):
                fr.apply(pose)

        # ── Post-relaxation scoring ───────────────────────────────
        post_total = float(self._scorefxn(pose))
        energies = pose.energies()
        te = energies.total_energies()

        post_fa_atr = float(te.get(self._st_fa_atr))
        post_lk_ball_wtd = float(te.get(self._st_lk_ball_wtd))

        # Per-residue energy (mean of total per residue)
        per_res = [
            float(energies.residue_total_energy(i))
            for i in range(1, n_res + 1)
        ]
        per_res_mean = float(np.mean(per_res))

        fa_atr_per_res = post_fa_atr / n_res

        # Long-range CB-CB contacts (seq sep >= 6, distance < 8 A)
        lr_contacts = self._cb_long_range_contacts(pose)

        # ── Final score (lower = better) ──────────────────────────
        final_score = (
            _W_FA_ATR_PER_RES * fa_atr_per_res
            + _W_PER_RES_MEAN * per_res_mean
            + _W_LK_BALL * post_lk_ball_wtd
            - _W_LR_CONTACTS * lr_contacts  # subtract: more contacts → lower score → better
        )

        return {
            "final_score": final_score,
            "post_fa_atr": post_fa_atr,
            "fa_atr_per_res": fa_atr_per_res,
            "post_lk_ball_wtd": post_lk_ball_wtd,
            "per_res_mean": per_res_mean,
            "lr_contacts": lr_contacts,
            "post_total": post_total,
            "pre_total": pre_total,
            "delta_total": pre_total - post_total,
            "n_res": float(n_res),
        }

    # ── Main entry point ──────────────────────────────────────────────

    def rerank(
        self,
        candidates: List[Any],
    ) -> FullAtomRerankResult:
        """
        Run the full-atom reranking pipeline.

        Parameters
        ----------
        candidates : list of ``StructureCandidate`` from coarse rerank.

        Returns
        -------
        FullAtomRerankResult ranked by absolute score (lower = better).
        """
        if self.config.top_k_input is not None:
            candidates = candidates[: self.config.top_k_input]

        if not candidates:
            return FullAtomRerankResult(ranked=[], config=self.config)

        _LOG.info("Full-atom reranking: %d candidates.", len(candidates))

        # ── Stage 1: Build full-atom structures ───────────────────
        entries: List[Tuple[Any, BuildResult]] = []
        for idx, cand in enumerate(candidates):
            try:
                build = self._builder.build(cand)
                entries.append((cand, build))
            except Exception as exc:
                _LOG.warning(
                    "ca2all build failed for candidate %d (%s): %s",
                    idx, getattr(cand, "candidate_id", "?"), exc,
                )
        _LOG.info(
            "Built %d / %d full-atom structures.",
            len(entries), len(candidates),
        )

        if not entries:
            return FullAtomRerankResult(ranked=[], config=self.config)

        # ── Stage 2: (Optional) CA-anchor refinement + score ──────
        ranked_entries: List[RankedFullAtomCandidate] = []
        for i, (cand, build) in enumerate(entries):
            pdb_text = build.write_pdb()
            refined_pdb: Optional[str] = None

            # Apply CA-anchor refinement (pulls CAs back to lattice targets)
            if self._ca_refiner is not None:
                try:
                    refined_pdb = self._ca_refiner.refine(pdb_text, build.scaled_ca)
                    pdb_text = refined_pdb   # score the refined structure
                except Exception as exc:
                    _LOG.warning(
                        "CA-anchor refinement failed for %s: %s",
                        getattr(cand, "candidate_id", "?"), exc,
                    )
                    refined_pdb = None  # fall back to raw NERF

            try:
                scores = self._score_single(pdb_text)
            except Exception as exc:
                _LOG.warning(
                    "Scoring failed for %s: %s",
                    getattr(cand, "candidate_id", "?"), exc,
                )
                scores = {"final_score": 1e6}

            ranked_entries.append(RankedFullAtomCandidate(
                candidate=cand,
                build_result=build,
                refined_pdb=refined_pdb,
                fullatom_score=scores.get("final_score", 1e6),
                fullatom_terms=scores,
                combined_score=scores.get("final_score", 1e6),
            ))

            if (i + 1) % 5 == 0 or (i + 1) == len(entries):
                _LOG.info("  Scored %d / %d candidates.", i + 1, len(entries))

        # ── Stage 3: Rank by absolute score (lower = better) ─────
        ranked_entries.sort(key=lambda e: e.fullatom_score)
        for rank, entry in enumerate(ranked_entries, 1):
            entry.rank = rank

        best = ranked_entries[0]
        _LOG.info(
            "Reranking complete. Best score=%.3f "
            "(fa_atr/res=%.2f, per_res_mean=%.2f, lk_ball=%.2f).",
            best.fullatom_score,
            best.fullatom_terms.get("fa_atr_per_res", 0),
            best.fullatom_terms.get("per_res_mean", 0),
            best.fullatom_terms.get("post_lk_ball_wtd", 0),
        )

        return FullAtomRerankResult(ranked=ranked_entries, config=self.config)


__all__ = [
    "FullAtomReranker",
    "FullAtomRerankConfig",
    "FullAtomRerankResult",
    "RankedFullAtomCandidate",
]
