# --*-- coding:utf-8 --*--
# @time: 3/31/26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File: __init__.py

"""
post_process.landscape
======================
Energy landscape analysis and visualisation for QSAD quantum-sampling results.

Public API
----------
LandscapeConfig
    Analysis configuration (decoder, scoring, PCA, RBF, visualisation settings).

LandscapeData
    Fully-processed landscape: PCA coordinates, interpolated surface,
    energy/probability arrays, summary statistics.

SampledConformation
    One valid, scored backbone conformation from quantum sampling.

LandscapeBuilder
    CSV → decode → filter → score → PCA → interpolate → LandscapeData.

LandscapePlotter
    LandscapeData → four-panel matplotlib figure (3-D surface, 2-D contour,
    energy histogram, sampling-probability vs energy).

LandscapeAnalysisPipeline
    High-level orchestrator: single-protein and batch modes,
    automatic JSON/PNG/PDF export.

Quick start
-----------
    from post_process.landscape import LandscapeConfig, LandscapeAnalysisPipeline

    cfg = LandscapeConfig(output_dir="landscape_result")
    pipeline = LandscapeAnalysisPipeline(cfg)

    # Single protein
    data = pipeline.run_single(
        "1a0q",
        "Sampling_result/sampling_result/1a0q/samples_1a0q_all_ibm.csv",
    )

    # Batch
    results = pipeline.run_batch(
        summary_csv="Sampling_result/sampling_result/quantum_data_summary.csv",
        sampling_dir="Sampling_result/sampling_result",
    )

    # Detailed PDF
    pipeline.generate_detail_pdf(data, "img/1a0q_landscape_detail.pdf")
"""

from .config import LandscapeConfig
from .data_types import LandscapeData, SampledConformation
from .builder import LandscapeBuilder
from .visualizer import LandscapePlotter
from .pipeline import LandscapeAnalysisPipeline

__all__ = [
    "LandscapeConfig",
    "LandscapeData",
    "SampledConformation",
    "LandscapeBuilder",
    "LandscapePlotter",
    "LandscapeAnalysisPipeline",
]
