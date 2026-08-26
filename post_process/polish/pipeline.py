# --*-- coding:utf-8 --*--
# @time: 2026-03-30
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File: pipeline.py

"""
Stage-4 pipeline: full-atom ranked candidates → self-consistently refined structures.

Takes the ``FullAtomRerankResult`` produced by ``Ca2AllPipeline`` (Stage 3) and
applies heavy iterative refinement to the top-k candidates via ``IterativeRefiner``,
maximising structural quality and minimising potential energy.

Usage
-----
>>> from post_process.polish import PolishPipeline, PolishPipelineConfig
>>> pipeline = PolishPipeline(PolishPipelineConfig(top_k=5, output_dir="out/polished"))
>>> polished = pipeline.run(ca2all_result)   # FullAtomRerankResult from Stage 3
>>> for p in polished:
...     print(p.candidate_id, p.final_score, f"improvement={p.score_improvement:+.3f}")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .iterative_refiner import IterativeRefiner, IterativeRefinerConfig, PolishedCandidate

_LOG = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────

@dataclass
class PolishPipelineConfig:
    """
    Parameters
    ----------
    top_k :
        Number of top-ranked Stage-3 candidates to refine.
    refiner_config :
        Configuration forwarded to IterativeRefiner.
    output_dir :
        If set, write final PDB files here.
    save_initial :
        If True, also write the pre-refinement (Stage-3) PDB for comparison.
    """
    top_k: int = 5
    refiner_config: IterativeRefinerConfig = field(
        default_factory=IterativeRefinerConfig
    )
    output_dir: Optional[str] = None
    save_initial: bool = False


# ── Pipeline ───────────────────────────────────────────────────────────────

class PolishPipeline:
    """
    Stage-4 pipeline: iterative heavy refinement of full-atom candidates.

    Each candidate is processed serially (PyRosetta is not thread-safe).
    The final list is sorted by ``final_score`` (lower = better).
    """

    def __init__(self, config: Optional[PolishPipelineConfig] = None) -> None:
        self.config = config or PolishPipelineConfig()

    def run(self, ranked_result: Any) -> List[PolishedCandidate]:
        """
        Refine top-k candidates from Stage 3.

        Parameters
        ----------
        ranked_result : FullAtomRerankResult
            Direct output of ``Ca2AllPipeline.run()``.

        Returns
        -------
        List[PolishedCandidate]
            Sorted by ``final_score`` (best first).
        """
        top_k_candidates = ranked_result.top_k(self.config.top_k)
        if not top_k_candidates:
            _LOG.warning("PolishPipeline: no candidates to refine.")
            return []

        _LOG.info(
            "PolishPipeline: refining %d candidate(s) with up to %d iterations each.",
            len(top_k_candidates),
            self.config.refiner_config.max_iter,
        )

        refiner = IterativeRefiner(self.config.refiner_config)
        results: List[PolishedCandidate] = []

        for cand in top_k_candidates:
            cid = getattr(cand.candidate, "candidate_id", f"rank{cand.rank}")
            _LOG.info("--- Polishing candidate: %s (rank=%d) ---", cid, cand.rank)
            polished = refiner.refine(cand)
            results.append(polished)

            _LOG.info(
                "  %s: score %.3f → %.3f (%+.3f) in %d iter(s)",
                polished.candidate_id,
                polished.source_score,
                polished.final_score,
                polished.score_improvement,
                polished.iterations_run,
            )

        # Sort by final score
        results.sort(key=lambda r: r.final_score)

        # Write PDB files
        if self.config.output_dir:
            os.makedirs(self.config.output_dir, exist_ok=True)
            for rank, polished in enumerate(results, 1):
                safe_id = "".join(
                    c if c.isalnum() or c in "_-" else "_"
                    for c in str(polished.candidate_id)
                )[:60]
                pdb_path = os.path.join(
                    self.config.output_dir, f"polished_rank{rank}_{safe_id}.pdb"
                )
                with open(pdb_path, "w") as fh:
                    fh.write(polished.refined_pdb)
                _LOG.info("Wrote %s", pdb_path)

                if self.config.save_initial:
                    init_path = os.path.join(
                        self.config.output_dir,
                        f"initial_rank{rank}_{safe_id}.pdb",
                    )
                    with open(init_path, "w") as fh:
                        fh.write(polished.initial_pdb)

        return results


__all__ = ["PolishPipelineConfig", "PolishPipeline"]
