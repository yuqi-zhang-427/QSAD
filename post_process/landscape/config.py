# --*-- coding:utf-8 --*--
# @time: 3/31/26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File: config.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class LandscapeConfig:
    """
    Configuration for protein energy landscape analysis and visualization.

    Attributes
    ----------
    Decoder
        fifth_bit, strict_decode — passed to BinaryStructureDecoder.

    Sampling / filtering
        max_unique_conformations : cap on unique decoded conformations (performance).
        min_unique_conformations : minimum needed for a meaningful landscape.
        overlap_eps              : overlap tolerance for SAW validity check.

    Scoring
        scorer : "lattice_contact" (physics-based MJ + steric) or "mm" (force-field).

    PCA
        n_pca_components : always 2 for 2-D landscape projection.
        scale_backbone   : rescale CA–CA spacing to 3.8 Å before PCA.

    RBF surface interpolation
        rbf_kernel           : kernel for RBFInterpolator (thin_plate_spline recommended).
        rbf_smoothing_factor : smoothing multiplier (× N_samples) for RBF.
        grid_resolution      : grid points per axis for surface mesh.
        energy_percentile_clip : clip energy outliers above this percentile for display.

    Visualization
        figsize, dpi, colormap, surface_alpha, scatter_size, scatter_alpha,
        top_k_annotate : number of best-energy structures annotated on 2-D plot.

    Output
        output_dir  : root output directory.
        save_data   : persist per-protein JSON data.
        save_plots  : persist per-protein overview PNG.
    """

    # ---- Decoder ----
    fifth_bit: bool = False
    strict_decode: bool = False

    # ---- Sampling / filtering ----
    max_unique_conformations: int = 3000
    min_unique_conformations: int = 40
    overlap_eps: float = 0.15          # lattice units; positions are on integer grid

    # ---- Scoring ----
    scorer: str = "lattice_contact"    # "lattice_contact" | "mm"

    # ---- PCA ----
    n_pca_components: int = 2
    scale_backbone: bool = True        # rescale to 3.8 Å CA–CA spacing

    # ---- RBF surface interpolation ----
    rbf_kernel: str = "thin_plate_spline"
    rbf_smoothing_factor: float = 0.3  # smoothing = factor × N
    grid_resolution: int = 60          # grid points per axis
    energy_percentile_clip: float = 93.0

    # ---- Visualization ----
    figsize: Tuple[float, float] = (24, 6.6)
    dpi: int = 600
    font_scale: float = 1.0
    colormap: str = "RdYlBu_r"        # red = high energy, blue = low energy
    surface_alpha: float = 0.62
    scatter_size: float = 18.0
    scatter_alpha: float = 0.72
    top_k_annotate: int = 3

    # ---- Output ----
    output_dir: str = "landscape_result"
    save_data: bool = True
    save_plots: bool = True
