# --*-- conding:utf-8 --*--
# @time:3/26/26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:models.py

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import numpy as np


class DecodedStructureLike(Protocol):
    """
    Structural protocol for decoded lattice conformations.

    Any object that carries these four attributes can be passed directly to
    ``StructureCandidate.from_decoded()``.  This keeps ``post_process.rerank``
    free of any runtime dependency on ``post_process.decom``.
    """

    bitstring: str
    sequence: str
    main_positions: np.ndarray
    side_positions: List[Optional[np.ndarray]]


def _as_positions(array_like: Any, field_name: str) -> np.ndarray:
    positions = np.asarray(array_like, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"{field_name} must have shape (N, 3), got {positions.shape}")
    return positions


def _as_optional_positions(items: Any) -> List[Optional[np.ndarray]]:
    normalized: List[Optional[np.ndarray]] = []
    for item in items or []:
        if item is None:
            normalized.append(None)
        else:
            normalized.append(_as_positions([item], "side_positions")[0])
    return normalized


@dataclass
class ScoreBreakdown:
    total: float
    terms: Dict[str, float] = field(default_factory=dict)
    engine: str = ""
    lower_is_better: bool = True

    def to_record(self, prefix: str = "score_") -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            f"{prefix}total": float(self.total),
            f"{prefix}engine": self.engine,
            f"{prefix}lower_is_better": self.lower_is_better,
        }
        for key, value in self.terms.items():
            payload[f"{prefix}{key}"] = float(value)
        return payload


@dataclass
class StructureCandidate:
    candidate_id: str
    sequence: str
    bitstring: str
    main_positions: np.ndarray
    side_positions: List[Optional[np.ndarray]] = field(default_factory=list)
    full_atom_positions: Optional[np.ndarray] = None
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    coarse_score: Optional[ScoreBreakdown] = None
    score: Optional[ScoreBreakdown] = None
    reference_rmsd: Optional[float] = None

    def __post_init__(self) -> None:
        self.main_positions = _as_positions(self.main_positions, "main_positions")
        self.side_positions = _as_optional_positions(self.side_positions)
        if self.full_atom_positions is not None:
            self.full_atom_positions = _as_positions(self.full_atom_positions, "full_atom_positions")

    @classmethod
    def from_decoded(
        cls,
        decoded: DecodedStructureLike,
        candidate_id: Optional[str] = None,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "StructureCandidate":
        return cls(
            candidate_id=candidate_id or decoded.bitstring,
            sequence=decoded.sequence,
            bitstring=decoded.bitstring,
            main_positions=decoded.main_positions,
            side_positions=decoded.side_positions,
            weight=weight,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_record(
        cls,
        record: Dict[str, Any],
        candidate_id_key: str = "candidate_id",
    ) -> "StructureCandidate":
        return cls(
            candidate_id=str(record.get(candidate_id_key) or record.get("bitstring") or ""),
            sequence=str(record.get("sequence") or ""),
            bitstring=str(record.get("bitstring") or ""),
            main_positions=record.get("main_positions") or [],
            side_positions=record.get("side_positions") or [],
            full_atom_positions=record.get("full_atom_positions"),
            weight=float(record.get("weight", 1.0) or 1.0),
            metadata=dict(record),
        )

    def primary_positions(self, prefer_full_atom: bool = False) -> np.ndarray:
        if prefer_full_atom and self.full_atom_positions is not None:
            return self.full_atom_positions
        return self.main_positions

    def sampling_count(self) -> float:
        try:
            value = float(self.metadata.get("count", self.weight) or self.weight or 0.0)
        except Exception:
            value = float(self.weight or 0.0)
        return max(value, 0.0)

    def sampling_probability(self) -> float:
        try:
            value = float(self.metadata.get("prob", 0.0) or 0.0)
        except Exception:
            value = 0.0
        return max(value, 0.0)

    def support_value(self) -> float:
        return math.log1p(self.sampling_count()) + self.sampling_probability()

    def clone(self, **changes: Any) -> "StructureCandidate":
        payload: Dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "sequence": self.sequence,
            "bitstring": self.bitstring,
            "main_positions": self.main_positions.copy(),
            "side_positions": [
                None if coords is None else np.asarray(coords, dtype=float).copy()
                for coords in self.side_positions
            ],
            "full_atom_positions": None if self.full_atom_positions is None else self.full_atom_positions.copy(),
            "weight": self.weight,
            "metadata": dict(self.metadata),
            "coarse_score": self.coarse_score,
            "score": self.score,
            "reference_rmsd": self.reference_rmsd,
        }
        payload.update(changes)
        return StructureCandidate(**payload)

    def to_record(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "sequence": self.sequence,
            "bitstring": self.bitstring,
            "weight": float(self.weight),
            "main_positions": self.main_positions.tolist(),
            "side_positions": [
                coords.tolist() if coords is not None else None for coords in self.side_positions
            ],
            "reference_rmsd": None if self.reference_rmsd is None else float(self.reference_rmsd),
        }
        if self.full_atom_positions is not None:
            payload["full_atom_positions"] = self.full_atom_positions.tolist()
        if self.coarse_score is not None:
            payload.update(self.coarse_score.to_record(prefix="coarse_score_"))
        if self.score is not None:
            payload.update(self.score.to_record())
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload
