# --*-- coding:utf-8 --*--
# @time: 3/31/26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File: builder.py

"""
LandscapeBuilder
================
Converts a quantum-sampling CSV file into a LandscapeData object by:

  1. Loading the merged sampling CSV.
  2. Aggregating counts/probs by full bitstring.
  3. Extracting canonical (main-chain-only) bitstrings and deduplicating.
  4. Decoding each canonical bitstring to backbone CA coordinates.
  5. Filtering self-avoiding-walk (SAW) validity.
  6. Scoring each valid conformation with LatticeContactScorer.
  7. Running PCA on the flattened, scaled backbone coordinate vectors.
  8. Interpolating an energy surface on a 2-D grid via RBF.
  9. Computing summary statistics (funnel score, basin count).
 10. Returning a LandscapeData instance.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import LandscapeConfig
from .data_types import LandscapeData, SampledConformation
from .scorer import BatchScorer

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _check_saw_valid(positions: np.ndarray, eps: float = 0.15) -> bool:
    """
    O(N²) SAW validity check on raw lattice positions.

    Returns False if any two non-adjacent residues overlap (distance < eps).
    Positions are on the integer-ish tetrahedral lattice (step ≈ 1/√3 ≈ 0.577).
    eps=0.15 is well below the nearest-neighbour distance and above round-off.
    """
    n = len(positions)
    if n < 3:
        return True
    # Vectorised for speed
    for i in range(n - 2):
        diff = positions[i + 2 :] - positions[i]  # (n-i-2, 3)
        dists = np.linalg.norm(diff, axis=1)
        if np.any(dists < eps):
            return False
    return True


def _estimate_basins(
    pca_coords: np.ndarray,
    energies: np.ndarray,
    n_grid: int = 20,
) -> int:
    """
    Rough estimate of the number of distinct energy basins.

    Algorithm:
      1. Bin the scattered (PC1, PC2) points onto a coarse grid, recording
         the minimum energy in each cell.
      2. Mark cells at or below the 35th-percentile energy as "low-energy".
      3. Count connected components of that binary mask.

    Returns at least 1.
    """
    from scipy.ndimage import label as sp_label

    x, y = pca_coords[:, 0], pca_coords[:, 1]
    x_range = float(x.max() - x.min())
    y_range = float(y.max() - y.min())

    if x_range < 1e-6 or y_range < 1e-6:
        return 1

    grid = np.full((n_grid, n_grid), np.inf)
    xi = np.clip(
        ((x - x.min()) / x_range * (n_grid - 1)).astype(int), 0, n_grid - 1
    )
    yi = np.clip(
        ((y - y.min()) / y_range * (n_grid - 1)).astype(int), 0, n_grid - 1
    )
    for k in range(len(energies)):
        if energies[k] < grid[xi[k], yi[k]]:
            grid[xi[k], yi[k]] = energies[k]

    filled_mask = grid < np.inf
    if not filled_mask.any():
        return 1

    threshold = float(np.percentile(grid[filled_mask], 35))
    low_mask = (grid <= threshold) & filled_mask
    _, n_features = sp_label(low_mask)
    return max(1, int(n_features))


# ---------------------------------------------------------------------------
#  Main builder
# ---------------------------------------------------------------------------

class LandscapeBuilder:
    """
    Build a LandscapeData from a QSAD sampling CSV file.

    Parameters
    ----------
    cfg : LandscapeConfig
        Analysis configuration.  Defaults to LandscapeConfig() if None.
    """

    def __init__(self, cfg: Optional[LandscapeConfig] = None) -> None:
        self.cfg = cfg or LandscapeConfig()

    # ------------------------------------------------------------------ #
    #  Public entry-points                                                 #
    # ------------------------------------------------------------------ #

    def build_from_csv(
        self,
        csv_path: str,
        protein_id: Optional[str] = None,
        sequence: Optional[str] = None,
    ) -> Optional[LandscapeData]:
        """Load a sampling CSV and build the landscape.  Returns None on failure."""
        if not os.path.isfile(csv_path):
            _LOG.warning(f"CSV not found: {csv_path}")
            return None

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            _LOG.error(f"Failed to read {csv_path}: {exc}")
            return None

        if df.empty or "bitstring" not in df.columns:
            return None

        if sequence is None and "sequence" in df.columns:
            sequence = str(df["sequence"].dropna().iloc[0])
        if protein_id is None and "protein" in df.columns:
            protein_id = str(df["protein"].dropna().iloc[0])

        return self.build_from_dataframe(
            df,
            protein_id=protein_id or "unknown",
            sequence=sequence or "",
        )

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        protein_id: str,
        sequence: str,
    ) -> Optional[LandscapeData]:
        """Build the landscape from an already-loaded sampling DataFrame."""
        from post_process.decom.decoder import BinaryStructureDecoder, StructureDecoderConfig
        from post_process.rerank.geometry import scale_backbone_positions

        L = len(sequence)
        if L < 4:
            _LOG.warning(f"[{protein_id}] Sequence too short (L={L}), skipping.")
            return None

        if "count" not in df.columns:
            df = df.copy()
            df["count"] = 1
        if "prob" not in df.columns:
            df = df.copy()
            df["prob"] = df["count"] / df["count"].sum()

        # ---- 1. Build decoder (main-chain only, all-False hot-vec) ----------
        hot_vec = [False] * L
        dec_cfg = StructureDecoderConfig(
            side_chain_hot_vector=hot_vec,
            fifth_bit=self.cfg.fifth_bit,
            strict=self.cfg.strict_decode,
        )
        decoder = BinaryStructureDecoder(dec_cfg)
        expected_len = dec_cfg.expected_bitstring_length

        # ---- 2. Extract canonical (main-chain) bits, THEN aggregate --------
        # Critical: extract rightmost expected_len bits first so that multiple
        # full-length bitstrings with identical main-chain encoding merge.
        df_work = df.copy()
        df_work["canonical"] = df_work["bitstring"].apply(
            lambda bs: str(bs).strip()[-expected_len:]
            if len(str(bs).strip()) >= expected_len else None
        )
        df_work = df_work.dropna(subset=["canonical"])
        df_work = df_work[df_work["canonical"].str.len() == expected_len]

        # Aggregate count/prob by canonical; also keep one representative full-bs
        canonical_agg = (
            df_work.groupby("canonical", sort=False)
            .agg(count=("count", "sum"), prob=("prob", "sum"))
            .reset_index()
        )
        first_full = (
            df_work.groupby("canonical", sort=False)["bitstring"]
            .first()
            .rename("full_bs")
        )
        canonical_agg = canonical_agg.join(first_full, on="canonical")

        n_total_canonical = len(canonical_agg)

        # ---- 3. Sampling strategy ------------------------------------------
        # For short proteins the majority of canonical bitstrings are valid SAWs.
        # For long proteins (L ≥ 13) valid SAWs are rare (~0.5–2%) and are
        # distributed throughout the frequency distribution — NOT concentrated
        # at the most frequent bitstrings (which are quantum noise patterns).
        # Strategy: sample broadly (random) rather than sort-by-frequency.
        _MAX_SAW_SCREEN = 40_000   # decode+SAW check at most this many
        _MAX_VALID_SCORE = self.cfg.max_unique_conformations  # score at most this many valid SAWs

        if n_total_canonical <= _MAX_SAW_SCREEN:
            # Small enough: process everything
            sample_df = canonical_agg
        else:
            # Large pool: stratified random sample to cover frequency range
            sample_df = canonical_agg.sample(
                n=_MAX_SAW_SCREEN, random_state=42, replace=False
            )

        # ---- 4. Build canonical_map ----------------------------------------
        canonical_map: Dict[str, Dict] = {}
        for _, row in sample_df.iterrows():
            canonical = str(row["canonical"])
            canonical_map[canonical] = {
                "full_bs": str(row.get("full_bs", canonical)),
                "count": int(row["count"]),
                "prob": float(row["prob"]),
            }

        _LOG.info(
            f"[{protein_id}] L={L}  total_unique_canonical={n_total_canonical}  "
            f"screened={len(canonical_map)}"
        )

        # ---- 4-6. Decode → filter SAW → score  ------------------------------
        scorer = BatchScorer(self.cfg.scorer)
        conformations: List[SampledConformation] = []

        for canonical_bs, info in canonical_map.items():
            # Decode
            try:
                ds = decoder.decode(canonical_bs, sequence)
            except Exception:
                continue

            raw_positions = ds.main_positions  # unscaled lattice coords

            # SAW check on unscaled positions (integer-ish grid)
            if not _check_saw_valid(raw_positions, eps=self.cfg.overlap_eps):
                continue

            # Scale backbone
            if self.cfg.scale_backbone:
                positions = scale_backbone_positions(raw_positions)
            else:
                positions = raw_positions.copy()

            # Score
            energy, terms = scorer.score(
                main_positions=positions,
                sequence=sequence,
                count=float(info["count"]),
                prob=float(info["prob"]),
            )
            if not math.isfinite(energy):
                continue

            conformations.append(
                SampledConformation(
                    bitstring_key=canonical_bs,
                    full_bitstring=info["full_bs"],
                    sequence=sequence,
                    main_positions=positions,
                    energy=energy,
                    energy_terms=terms,
                    count=int(info["count"]),
                    prob=float(info["prob"]),
                    saw_valid=True,
                )
            )

        # Cap valid conformations for PCA/plotting performance
        if len(conformations) > _MAX_VALID_SCORE:
            # Sort by sampling count (most-represented first)
            conformations.sort(key=lambda c: c.count, reverse=True)
            conformations = conformations[:_MAX_VALID_SCORE]

        n_valid = len(conformations)
        _LOG.info(f"[{protein_id}] valid conformations after SAW + score filter: {n_valid}")

        if n_valid < self.cfg.min_unique_conformations:
            _LOG.warning(
                f"[{protein_id}] Only {n_valid} valid conformations "
                f"(min required: {self.cfg.min_unique_conformations}) — skipping."
            )
            return None

        # ---- 7. PCA on flattened backbone coordinates ----------------------
        coord_matrix = np.array(
            [c.main_positions.flatten() for c in conformations], dtype=float
        )  # (N, L*3)
        energies = np.array([c.energy for c in conformations], dtype=float)

        from sklearn.decomposition import PCA

        n_comp = min(self.cfg.n_pca_components, coord_matrix.shape[0] - 1, coord_matrix.shape[1])
        pca = PCA(n_components=max(1, n_comp))
        pca_coords_raw = pca.fit_transform(coord_matrix)
        explained = pca.explained_variance_ratio_

        # Always return (N, 2) — pad if needed
        if pca_coords_raw.shape[1] < 2:
            pad = np.zeros((pca_coords_raw.shape[0], 2 - pca_coords_raw.shape[1]))
            pca_coords = np.hstack([pca_coords_raw, pad])
            explained = np.pad(explained, (0, 2 - len(explained)))
        else:
            pca_coords = pca_coords_raw[:, :2]
            explained = np.array(explained[:2])

        # ---- 8. RBF surface interpolation ----------------------------------
        grid_x, grid_y, grid_z = self._interpolate_surface(pca_coords, energies)

        # ---- 9. Summary statistics -----------------------------------------
        min_idx = int(np.argmin(energies))
        dists_to_min = np.linalg.norm(pca_coords - pca_coords[min_idx], axis=1)

        if len(dists_to_min) > 5 and dists_to_min.std() > 1e-9:
            corr_matrix = np.corrcoef(dists_to_min, energies)
            funnel_score = float(corr_matrix[0, 1])
            if not math.isfinite(funnel_score):
                funnel_score = 0.0
        else:
            funnel_score = 0.0

        n_basins = _estimate_basins(pca_coords, energies)

        return LandscapeData(
            protein_id=protein_id,
            sequence=sequence,
            conformations=conformations,
            pca_coords=pca_coords,
            pca_explained_variance=explained,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_z=grid_z,
            min_energy_idx=min_idx,
            median_energy=float(np.median(energies)),
            funnel_score=funnel_score,
            n_valid=n_valid,
            n_total_unique=n_total_canonical,
            n_basins=n_basins,
        )

    # ------------------------------------------------------------------ #
    #  Private: RBF surface interpolation                                  #
    # ------------------------------------------------------------------ #

    def _interpolate_surface(
        self,
        pca_coords: np.ndarray,
        energies: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build a smooth energy surface over the 2-D PCA space.

        1. Clip outlier energies at ``energy_percentile_clip`` percentile.
        2. Build a regular (grid_resolution × grid_resolution) mesh.
        3. Interpolate with RBFInterpolator (thin-plate-spline); fall back
           to scipy.interpolate.griddata on failure.

        Returns (GX, GY, GZ) meshgrid arrays of shape (G, G).
        """
        e_clip_val = float(np.percentile(energies, self.cfg.energy_percentile_clip))
        e_plot = np.clip(energies, None, e_clip_val)

        x = pca_coords[:, 0]
        y = pca_coords[:, 1]

        x_range = float(x.max() - x.min())
        y_range = float(y.max() - y.min())
        margin_x = max(0.08 * x_range, 1e-3)
        margin_y = max(0.08 * y_range, 1e-3)

        G = self.cfg.grid_resolution
        xi = np.linspace(x.min() - margin_x, x.max() + margin_x, G)
        yi = np.linspace(y.min() - margin_y, y.max() + margin_y, G)
        GX, GY = np.meshgrid(xi, yi)
        grid_pts = np.column_stack([GX.ravel(), GY.ravel()])

        try:
            from scipy.interpolate import RBFInterpolator
            smoothing = self.cfg.rbf_smoothing_factor * len(pca_coords)
            rbf = RBFInterpolator(
                pca_coords,
                e_plot,
                kernel=self.cfg.rbf_kernel,
                smoothing=smoothing,
            )
            GZ = rbf(grid_pts).reshape(GX.shape)
        except Exception as exc:
            _LOG.debug(f"RBFInterpolator failed ({exc}), falling back to griddata.")
            from scipy.interpolate import griddata
            GZ = griddata(
                pca_coords, e_plot, (GX, GY),
                method="linear",
                fill_value=float(e_clip_val),
            )

        return GX, GY, GZ
