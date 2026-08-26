# --*-- coding:utf-8 --*--
# @time: 3/31/26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File: data_types.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SampledConformation:
    """
    One unique backbone conformation recovered from quantum sampling.

    The conformation is identified by the *canonical bitstring* — the
    rightmost ``expected_bitstring_length`` bits which encode the
    main-chain turn sequence.  Multiple full measurement bitstrings
    (which differ in side-chain bits we ignore here) can map to the
    same canonical form; their counts are aggregated.

    Attributes
    ----------
    bitstring_key   : canonical (main-chain-only) bitstring used as unique ID.
    full_bitstring  : one representative full-length measurement bitstring.
    sequence        : amino-acid sequence string.
    main_positions  : shape (L, 3) backbone CA coordinates (Å, scaled).
    energy          : LatticeContactScorer total energy (lower = better).
    energy_terms    : per-term energy breakdown dictionary.
    count           : aggregated measurement count across all groups.
    prob            : aggregated empirical sampling probability.
    saw_valid       : True if the conformation passes SAW overlap check.
    """

    bitstring_key: str
    full_bitstring: str
    sequence: str
    main_positions: np.ndarray          # (L, 3)
    energy: float
    energy_terms: Dict[str, float] = field(default_factory=dict)
    count: int = 1
    prob: float = 0.0
    saw_valid: bool = True


@dataclass
class LandscapeData:
    """
    Complete energy landscape dataset for one protein sequence.

    Contains:
    * the filtered, scored list of SampledConformation objects;
    * 2-D PCA projection coordinates;
    * interpolated energy surface on a regular grid;
    * summary statistics (funnel score, basin count, etc.).
    """

    protein_id: str
    sequence: str
    conformations: List[SampledConformation]

    # PCA results
    pca_coords: np.ndarray              # (N, 2)
    pca_explained_variance: np.ndarray  # (2,) fraction of variance

    # Interpolated surface (grid)
    grid_x: np.ndarray                  # (G, G) meshgrid X
    grid_y: np.ndarray                  # (G, G) meshgrid Y
    grid_z: np.ndarray                  # (G, G) interpolated energy

    # Key indices
    min_energy_idx: int                 # index in conformations[]

    # Summary statistics
    median_energy: float
    funnel_score: float                 # Pearson corr(dist_to_min, energy)
    n_valid: int                        # valid SAW conformations processed
    n_total_unique: int                 # total unique canonical bitstrings
    n_basins: int                       # estimated number of energy basins

    # ------------------------------------------------------------------ #
    #  Convenience properties                                              #
    # ------------------------------------------------------------------ #

    @property
    def energies(self) -> np.ndarray:
        return np.array([c.energy for c in self.conformations], dtype=float)

    @property
    def probs(self) -> np.ndarray:
        return np.array([c.prob for c in self.conformations], dtype=float)

    @property
    def counts(self) -> np.ndarray:
        return np.array([c.count for c in self.conformations], dtype=float)

    @property
    def min_energy_conformation(self) -> SampledConformation:
        return self.conformations[self.min_energy_idx]

    # ------------------------------------------------------------------ #
    #  Serialisation                                                       #
    # ------------------------------------------------------------------ #

    def to_summary_record(self) -> Dict[str, Any]:
        """Compact JSON-serialisable summary (no large arrays)."""
        e = self.energies
        return {
            "protein_id": self.protein_id,
            "sequence": self.sequence,
            "L": len(self.sequence),
            "n_valid": self.n_valid,
            "n_total_unique": self.n_total_unique,
            "min_energy": float(e.min()),
            "max_energy": float(e.max()),
            "mean_energy": float(e.mean()),
            "median_energy": float(self.median_energy),
            "std_energy": float(e.std()),
            "funnel_score": float(self.funnel_score),
            "n_basins": self.n_basins,
            "pca_var_pc1": float(self.pca_explained_variance[0]),
            "pca_var_pc2": float(self.pca_explained_variance[1]),
            "min_energy_bitstring": self.min_energy_conformation.bitstring_key,
        }

    def to_full_record(self) -> Dict[str, Any]:
        """Full record including per-conformation arrays for JSON export."""
        record = self.to_summary_record()
        record.update(
            {
                "energies": [float(x) for x in self.energies],
                "probs": [float(x) for x in self.probs],
                "counts": [int(x) for x in self.counts],
                "pca_pc1": [float(x) for x in self.pca_coords[:, 0]],
                "pca_pc2": [float(x) for x in self.pca_coords[:, 1]],
                "bitstring_keys": [c.bitstring_key for c in self.conformations],
                "energy_terms_list": [c.energy_terms for c in self.conformations],
            }
        )
        return record
