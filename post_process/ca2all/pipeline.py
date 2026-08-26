# --*-- coding:utf-8 --*--
# @time: 2026-03-30
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File: pipeline.py

"""
Stage-3 pipeline: coarse-grained candidates → full-atom ranked structures.

Thin orchestration layer that wires FullAtomBuilder + FullAtomReranker into a
single callable entry point, compatible with the output of the coarse-rerank
stage (Stage 2) and the input of the polish stage (Stage 4).

Usage
-----
>>> from post_process.ca2all import Ca2AllPipeline, Ca2AllPipelineConfig
>>> pipeline = Ca2AllPipeline(Ca2AllPipelineConfig(top_k=5, output_dir="out/ca2all"))
>>> result = pipeline.run(candidates)   # List[StructureCandidate] from Stage 2
>>> result.best.build_result.write_pdb("best.pdb")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .fullatom_reranker import (
    FullAtomReranker,
    FullAtomRerankConfig,
    FullAtomRerankResult,
)

_LOG = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────

@dataclass
class Ca2AllPipelineConfig:
    """
    Configuration for the Stage-3 Ca2All pipeline.

    Parameters
    ----------
    top_k :
        Number of top-ranked candidates to output (and optionally write to disk).
    reranker_config :
        Config forwarded to FullAtomReranker (PyRosetta scoring / FastRelax).
    output_dir :
        If set, write PDB files for the top-k candidates here.
    """
    top_k: int = 5
    reranker_config: FullAtomRerankConfig = field(default_factory=FullAtomRerankConfig)
    output_dir: Optional[str] = None


# ── Pipeline ───────────────────────────────────────────────────────────────

class Ca2AllPipeline:
    """
    Stage-3 pipeline: coarse lattice candidates → full-atom ranked structures.

    1. Builds full-atom structures (backbone + sidechains) for all candidates.
    2. Scores each with PyRosetta REF2015 + constrained FastRelax.
    3. Ranks by absolute score (lower = better).
    4. Returns the top-k candidates in a ``FullAtomRerankResult``.
    5. Optionally writes PDB files to ``output_dir``.

    The input ``List[StructureCandidate]`` is built via
    ``StructureCandidate.from_decoded()`` from decoded lattice structures, or is
    any list produced by the coarse-rerank stage.
    """

    def __init__(self, config: Optional[Ca2AllPipelineConfig] = None) -> None:
        self.config = config or Ca2AllPipelineConfig()
        self._reranker = FullAtomReranker(self.config.reranker_config)

    def run(self, candidates: List[Any]) -> FullAtomRerankResult:
        """
        Execute the full-atom reranking pipeline.

        Parameters
        ----------
        candidates : List[StructureCandidate]
            Coarse-grained candidates from Stage 2.

        Returns
        -------
        FullAtomRerankResult
            Ranked full-atom structures, best first.
            The ``.best`` attribute holds the top candidate.
        """
        _LOG.info("Ca2AllPipeline: processing %d candidates.", len(candidates))

        result = self._reranker.rerank(candidates)

        if result.best is not None:
            _LOG.info(
                "Ca2AllPipeline: ranked %d candidates. "
                "Best score=%.3f (rank 1).",
                result.n_candidates,
                result.best.fullatom_score,
            )
        else:
            _LOG.warning("Ca2AllPipeline: no candidates could be ranked.")

        if self.config.output_dir:
            paths = result.write_top_k_pdb(self.config.output_dir, k=self.config.top_k)
            _LOG.info(
                "Wrote %d PDB files to %s.", len(paths), self.config.output_dir
            )

        return result


__all__ = ["Ca2AllPipeline", "Ca2AllPipelineConfig"]
