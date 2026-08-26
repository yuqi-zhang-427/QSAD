# --*-- coding:utf-8 --*--
# @time: 3/31/26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File: scorer.py

"""
Thin wrapper that exposes a single ``score_backbone`` function
used by LandscapeBuilder to score decoded lattice conformations.

Supported scorer types
----------------------
"lattice_contact"
    LatticeContactScorer — physics-based MJ contact energy + steric +
    compactness + burial + contact-order.  This is the primary scorer
    and the one validated on the QSAD benchmark.

"mm"
    MolecularMechanicsStructureScorer — force-field-like backbone energy
    (bond + angle + LJ repulsion/attraction + compactness).

Both scorers operate on scaled CA coordinates and require no PyRosetta.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


def score_backbone(
    main_positions: np.ndarray,
    sequence: str,
    count: float = 1.0,
    prob: float = 0.0,
    scorer_type: str = "lattice_contact",
) -> Tuple[float, Dict[str, float]]:
    """
    Score a single backbone conformation.

    Parameters
    ----------
    main_positions : (L, 3) array of CA coordinates in Ångströms.
    sequence       : amino-acid sequence string.
    count          : sampling count (used as weight, not posterior evidence).
    prob           : empirical sampling probability.
    scorer_type    : "lattice_contact" | "mm".

    Returns
    -------
    (total_energy, terms_dict)
        total_energy — scalar float (lower = more stable).
        terms_dict   — per-term breakdown dictionary.
    """
    from post_process.rerank.models import StructureCandidate

    candidate = StructureCandidate(
        candidate_id="__landscape_tmp__",
        sequence=sequence,
        bitstring="",
        main_positions=main_positions,
        side_positions=[],
        weight=count,
        metadata={"count": count, "prob": prob},
    )

    try:
        if scorer_type == "mm":
            from post_process.rerank.scoring import MolecularMechanicsStructureScorer
            scorer = MolecularMechanicsStructureScorer()
        else:
            from post_process.rerank.scoring import LatticeContactScorer
            scorer = LatticeContactScorer()

        result = scorer.score(candidate)
        return float(result.total), dict(result.terms)

    except Exception as exc:
        # Gracefully return inf so this conformation is simply excluded
        return math.inf, {"error": str(exc)}


class BatchScorer:
    """
    Reusable scorer instance for scoring many conformations of the same
    protein without repeatedly constructing the scorer object.
    """

    def __init__(self, scorer_type: str = "lattice_contact") -> None:
        self._scorer_type = scorer_type
        self._scorer = self._build_scorer(scorer_type)

    @staticmethod
    def _build_scorer(scorer_type: str):
        if scorer_type == "mm":
            from post_process.rerank.scoring import MolecularMechanicsStructureScorer
            return MolecularMechanicsStructureScorer()
        from post_process.rerank.scoring import LatticeContactScorer
        return LatticeContactScorer()

    def score(
        self,
        main_positions: np.ndarray,
        sequence: str,
        count: float = 1.0,
        prob: float = 0.0,
    ) -> Tuple[float, Dict[str, float]]:
        """Score one conformation; returns (total, terms)."""
        from post_process.rerank.models import StructureCandidate

        candidate = StructureCandidate(
            candidate_id="__batch__",
            sequence=sequence,
            bitstring="",
            main_positions=main_positions,
            side_positions=[],
            weight=count,
            metadata={"count": count, "prob": prob},
        )
        try:
            result = self._scorer.score(candidate)
            return float(result.total), dict(result.terms)
        except Exception as exc:
            return math.inf, {"error": str(exc)}
