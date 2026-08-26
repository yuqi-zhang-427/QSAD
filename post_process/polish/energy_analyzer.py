# --*-- coding:utf-8 --*--
# @time: 2026-03-30
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File: energy_analyzer.py

"""
Per-residue energy analysis using PyRosetta.

Identifies contiguous high-energy segments (WeakRegion) that are candidates
for local rebuilding in the polish self-consistent iteration loop.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

_LOG = logging.getLogger(__name__)

# Module-level PyRosetta init guard
_ROSETTA_READY = False


def _ensure_rosetta() -> Any:
    global _ROSETTA_READY
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory

    if not _ROSETTA_READY:
        pyrosetta.init(
            "-mute all -ignore_unrecognized_res -ignore_zero_occupancy false "
            "-run:constant_seed -run:jran 42",
            silent=True,
        )
        _ROSETTA_READY = True

    return pyrosetta, ScoreFunctionFactory


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class WeakRegion:
    """Contiguous segment of high-energy residues (0-based indices)."""
    residue_indices: List[int]
    mean_energy: float
    max_energy: float
    region_type: str = "high_energy"

    @property
    def start(self) -> int:
        return self.residue_indices[0]

    @property
    def end(self) -> int:
        return self.residue_indices[-1]

    @property
    def length(self) -> int:
        return len(self.residue_indices)


@dataclass
class EnergyAnalysis:
    per_residue_energies: List[float]
    weak_regions: List[WeakRegion]
    total_energy: float
    mean_energy: float
    n_residues: int


# ── Analyzer ──────────────────────────────────────────────────────────────

class EnergyAnalyzer:
    """
    Identifies high-energy residue segments using PyRosetta per-residue scoring.

    Parameters
    ----------
    energy_percentile :
        Residues above this percentile (e.g. 0.75 → top 25%) are flagged.
    max_region_size :
        Regions larger than this are split into sub-segments.
    merge_gap :
        Adjacent flagged residues within this distance are merged.
    """

    def __init__(
        self,
        energy_percentile: float = 0.75,
        max_region_size: int = 12,
        merge_gap: int = 2,
    ) -> None:
        self.energy_percentile = energy_percentile
        self.max_region_size = max_region_size
        self.merge_gap = merge_gap
        self._scorefxn: Any = None

    def _get_scorefxn(self):
        if self._scorefxn is None:
            _, ScoreFunctionFactory = _ensure_rosetta()
            self._scorefxn = ScoreFunctionFactory.create_score_function("ref2015")
        return self._scorefxn

    def analyze(self, pdb_string: str) -> EnergyAnalysis:
        """Score a PDB string and return per-residue energy analysis."""
        pyrosetta, _ = _ensure_rosetta()
        sfxn = self._get_scorefxn()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
            tmp.write(pdb_string)
            tmp_path = tmp.name
        try:
            pose = pyrosetta.pose_from_pdb(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        n_res = pose.total_residue()
        total_energy = float(sfxn(pose))
        energies_obj = pose.energies()

        per_res = [
            float(energies_obj.residue_total_energy(i)) for i in range(1, n_res + 1)
        ]

        threshold = float(np.percentile(per_res, self.energy_percentile * 100))
        flagged = [i for i, e in enumerate(per_res) if e > threshold]
        weak_regions = self._group_into_regions(flagged, per_res)

        return EnergyAnalysis(
            per_residue_energies=per_res,
            weak_regions=weak_regions,
            total_energy=total_energy,
            mean_energy=float(np.mean(per_res)),
            n_residues=n_res,
        )

    def _group_into_regions(
        self, flagged: List[int], per_res: List[float]
    ) -> List[WeakRegion]:
        if not flagged:
            return []

        # Merge nearby flagged residues
        groups: List[List[int]] = []
        current = [flagged[0]]
        for idx in flagged[1:]:
            if idx - current[-1] <= self.merge_gap:
                current.append(idx)
            else:
                groups.append(current)
                current = [idx]
        groups.append(current)

        regions: List[WeakRegion] = []
        for grp in groups:
            full = list(range(grp[0], grp[-1] + 1))
            # Split oversized groups
            for i in range(0, len(full), self.max_region_size):
                seg = full[i : i + self.max_region_size]
                energies_seg = [per_res[j] for j in seg if j < len(per_res)]
                if not energies_seg:
                    continue
                regions.append(WeakRegion(
                    residue_indices=seg,
                    mean_energy=float(np.mean(energies_seg)),
                    max_energy=float(np.max(energies_seg)),
                ))

        regions.sort(key=lambda r: r.mean_energy, reverse=True)
        return regions


__all__ = ["WeakRegion", "EnergyAnalysis", "EnergyAnalyzer"]
