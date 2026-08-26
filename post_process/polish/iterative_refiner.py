# --*-- coding:utf-8 --*--
# @time: 2026-03-30
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File: iterative_refiner.py

"""
Self-consistent iterative structure refiner.

Coordinates the three heavy-duty refinement stages in a loop until the
structure converges or the iteration budget is exhausted:

    1. EnergyAnalyzer  → identify high-energy weak regions
    2. LocalRebuilder  → backbone perturbation + rotamer repacking of weak regions
    3. MDAnnealer      → simulated annealing to escape local minima
    4. GlobalRelaxer   → unconstrained Cartesian FastRelax

Convergence criterion: Cα RMSD between consecutive iterations < tol.
Best (lowest-energy) structure across all iterations is returned.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from .energy_analyzer import EnergyAnalyzer, WeakRegion
from .local_rebuilder import LocalRebuilder, LocalRebuilderConfig
from .md_annealer import MDAnnealer, MDAnnealerConfig
from .global_relaxer import GlobalRelaxer, GlobalRelaxerConfig

_LOG = logging.getLogger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class PolishedCandidate:
    """
    Result of iterative refinement for one full-atom candidate.

    Attributes
    ----------
    candidate_id : str
    sequence : str
    source_rank : int
        Rank in the Stage-3 output (1 = best).
    source_score : float
        PyRosetta score from Stage 3 (lower = better).
    initial_pdb : str
        PDB string before refinement.
    refined_pdb : str
        PDB string of the best (lowest-energy) structure found.
    final_score : float
        PyRosetta energy of the refined structure.
    iterations_run : int
        Number of completed refinement iterations.
    energy_per_iter : List[float]
        Energy after each iteration (index 0 = before first iteration).
    rmsd_between_iters : List[float]
        Cα RMSD between consecutive iterations.
    weak_regions_per_iter : List[int]
        Number of weak regions found at the start of each iteration.
    """
    candidate_id: str
    sequence: str
    source_rank: int
    source_score: float
    initial_pdb: str
    refined_pdb: str
    final_score: float
    iterations_run: int
    energy_per_iter: List[float] = field(default_factory=list)
    rmsd_between_iters: List[float] = field(default_factory=list)
    weak_regions_per_iter: List[int] = field(default_factory=list)

    @property
    def score_improvement(self) -> float:
        """Positive = improvement (energy decreased)."""
        return self.source_score - self.final_score


# ── Configuration ──────────────────────────────────────────────────────────

@dataclass
class IterativeRefinerConfig:
    """
    Parameters
    ----------
    max_iter :
        Maximum number of refinement iterations.
    convergence_tol_angstrom :
        If Cα RMSD between iterations drops below this value, stop early.
    energy_percentile :
        Passed to EnergyAnalyzer: residues above this percentile are flagged.
    use_local_rebuild / use_md_anneal / use_global_relax :
        Toggle individual stages (set False to reduce compute cost).
    rebuilder_config / annealer_config / relaxer_config :
        Per-stage configs.
    """
    max_iter: int = 5
    convergence_tol_angstrom: float = 0.05

    energy_percentile: float = 0.75

    use_local_rebuild: bool = True
    use_md_anneal: bool = True
    use_global_relax: bool = True

    sidechain_only_mode: bool = False
    """
    If True, fix the backbone and optimise only side chains.

    Automatically sets:
      - rebuilder_config.backbone_moves = False  (no SmallMover/ShearMover)
      - use_md_anneal = False                    (MDAnnealer moves backbone)
      - relaxer_config.backbone_moves = False    (chi-only FastRelax)

    This prevents the RMSD regression caused by unconstrained backbone drift
    during aggressive energy minimisation.
    """

    rebuilder_config: LocalRebuilderConfig = field(default_factory=LocalRebuilderConfig)
    annealer_config: MDAnnealerConfig = field(default_factory=MDAnnealerConfig)
    relaxer_config: GlobalRelaxerConfig = field(default_factory=GlobalRelaxerConfig)


# ── Refiner ────────────────────────────────────────────────────────────────

class IterativeRefiner:
    """
    Self-consistent iterative structure refiner.

    Takes a ``RankedFullAtomCandidate`` from Stage 3 and returns a
    ``PolishedCandidate`` with the best refined structure.
    """

    def __init__(self, config: Optional[IterativeRefinerConfig] = None) -> None:
        self.config = config or IterativeRefinerConfig()

        # Apply sidechain-only overrides
        if self.config.sidechain_only_mode:
            self.config.rebuilder_config.backbone_moves = False
            self.config.use_md_anneal = False
            self.config.relaxer_config.backbone_moves = False
            _LOG.info("IterativeRefiner: sidechain_only_mode active — backbone frozen.")

        self._analyzer = EnergyAnalyzer(energy_percentile=self.config.energy_percentile)
        self._rebuilder: Optional[LocalRebuilder] = (
            LocalRebuilder(self.config.rebuilder_config)
            if self.config.use_local_rebuild else None
        )
        self._annealer: Optional[MDAnnealer] = (
            MDAnnealer(self.config.annealer_config)
            if self.config.use_md_anneal else None
        )
        self._relaxer: Optional[GlobalRelaxer] = (
            GlobalRelaxer(self.config.relaxer_config)
            if self.config.use_global_relax else None
        )

    def refine(self, ranked_candidate: Any) -> PolishedCandidate:
        """
        Iteratively refine one RankedFullAtomCandidate.

        Parameters
        ----------
        ranked_candidate : RankedFullAtomCandidate
            Direct output of Stage 3 (Ca2AllPipeline).

        Returns
        -------
        PolishedCandidate with the best structure found.
        """
        cand_id = getattr(ranked_candidate.candidate, "candidate_id", "unknown")
        sequence = getattr(ranked_candidate.candidate, "sequence", "")
        source_rank = int(getattr(ranked_candidate, "rank", 0))
        source_score = float(ranked_candidate.fullatom_score)

        # Prefer CA-anchor refined structure from Stage 3 (if available)
        refined = getattr(ranked_candidate, "refined_pdb", None)
        initial_pdb = refined if refined else ranked_candidate.build_result.write_pdb()
        current_pdb = initial_pdb
        best_pdb = initial_pdb

        # Initial energy from the analyzer (unified with refiner scoring)
        try:
            init_analysis = self._analyzer.analyze(current_pdb)
            current_energy = init_analysis.total_energy
        except Exception as exc:
            _LOG.warning("Initial energy analysis failed (%s); using stage-3 score.", exc)
            current_energy = source_score

        best_energy = current_energy
        energy_history = [current_energy]
        rmsd_history: List[float] = []
        weak_counts: List[int] = []

        _LOG.info(
            "IterativeRefiner: starting %s (rank=%d, energy=%.3f)",
            cand_id, source_rank, current_energy,
        )

        for iteration in range(self.config.max_iter):
            _LOG.info("  Iteration %d/%d", iteration + 1, self.config.max_iter)
            candidate_pdb = current_pdb

            try:
                # ── Step 1: energy analysis ───────────────────────────
                analysis = self._analyzer.analyze(candidate_pdb)
                weak_regions = analysis.weak_regions
                weak_counts.append(len(weak_regions))
                _LOG.info("    Weak regions: %d", len(weak_regions))

                # ── Step 2: local rebuild ─────────────────────────────
                if self._rebuilder and weak_regions:
                    try:
                        candidate_pdb = self._rebuilder.rebuild(candidate_pdb, weak_regions)
                    except Exception as exc:
                        _LOG.warning("    LocalRebuilder failed: %s", exc)

                # ── Step 3: MD annealing ──────────────────────────────
                if self._annealer:
                    try:
                        candidate_pdb = self._annealer.anneal(candidate_pdb)
                    except Exception as exc:
                        _LOG.warning("    MDAnnealer failed: %s", exc)

                # ── Step 4: global relaxation ─────────────────────────
                if self._relaxer:
                    try:
                        candidate_pdb = self._relaxer.relax(candidate_pdb)
                    except Exception as exc:
                        _LOG.warning("    GlobalRelaxer failed: %s", exc)

                # ── Step 5: score and track ───────────────────────────
                new_analysis = self._analyzer.analyze(candidate_pdb)
                new_energy = new_analysis.total_energy

                rmsd = _ca_rmsd(current_pdb, candidate_pdb)
                rmsd_history.append(rmsd)
                energy_history.append(new_energy)

                _LOG.info(
                    "    Energy: %.3f → %.3f (%+.3f) | Cα RMSD: %.3f Å",
                    current_energy, new_energy, new_energy - current_energy,
                    rmsd,
                )

                # Track global best
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pdb = candidate_pdb

                # Always continue from the new structure (accept unconditionally)
                current_pdb = candidate_pdb
                current_energy = new_energy

                # ── Step 6: convergence check ─────────────────────────
                # In sidechain-only mode backbone never moves, so Cα RMSD is
                # always 0 — skip the RMSD check and rely only on max_iter.
                if not self.config.sidechain_only_mode and rmsd < self.config.convergence_tol_angstrom:
                    _LOG.info(
                        "    Converged: Cα RMSD=%.4f < tol=%.4f",
                        rmsd, self.config.convergence_tol_angstrom,
                    )
                    break

            except Exception as exc:
                _LOG.error("  Iteration %d failed: %s", iteration + 1, exc)
                break

        _LOG.info(
            "IterativeRefiner done: %s  energy %.3f → %.3f  (%d iters)",
            cand_id, source_score, best_energy, len(rmsd_history),
        )

        return PolishedCandidate(
            candidate_id=cand_id,
            sequence=sequence,
            source_rank=source_rank,
            source_score=source_score,
            initial_pdb=initial_pdb,
            refined_pdb=best_pdb,
            final_score=best_energy,
            iterations_run=len(rmsd_history),
            energy_per_iter=energy_history,
            rmsd_between_iters=rmsd_history,
            weak_regions_per_iter=weak_counts,
        )


# ── Geometry helpers ───────────────────────────────────────────────────────

def _extract_ca(pdb_string: str) -> np.ndarray:
    """Extract Cα coordinates from a PDB string."""
    positions = []
    for line in pdb_string.splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
            try:
                positions.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                pass
    return np.array(positions, dtype=float) if positions else np.zeros((0, 3))


def _ca_rmsd(pdb1: str, pdb2: str) -> float:
    """Kabsch-aligned Cα RMSD between two PDB strings."""
    ca1 = _extract_ca(pdb1)
    ca2 = _extract_ca(pdb2)
    n = min(len(ca1), len(ca2))
    if n < 3:
        return 0.0
    ca1, ca2 = ca1[:n].copy(), ca2[:n].copy()
    ca1 -= ca1.mean(axis=0)
    ca2 -= ca2.mean(axis=0)
    H = ca1.T @ ca2
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    ca1_rot = ca1 @ R.T
    return float(np.sqrt(np.mean(np.sum((ca1_rot - ca2) ** 2, axis=1))))


__all__ = [
    "PolishedCandidate",
    "IterativeRefinerConfig",
    "IterativeRefiner",
    "_ca_rmsd",
    "_extract_ca",
]
