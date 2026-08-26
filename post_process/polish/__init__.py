# --*-- coding:utf-8 --*--
# @time: 2026-03-30
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File: __init__.py

"""
Stage-4 self-consistent iterative structure refinement.

Takes the top-k full-atom candidates from Stage 3 (Ca2All) and applies
heavy-duty multi-stage refinement to minimise energy and improve structural
quality, targeting the lowest achievable RMSD from native.

Pipeline
--------
1. EnergyAnalyzer  -- per-residue PyRosetta scoring to find weak segments.
2. LocalRebuilder  -- backbone MC perturbation + rotamer repacking of weak regions.
3. MDAnnealer      -- OpenMM simulated annealing to escape local minima.
4. GlobalRelaxer   -- unconstrained Cartesian FastRelax (multi-round).
5. IterativeRefiner -- self-consistent loop over steps 1-4 until convergence.
6. PolishPipeline  -- top-level entry point (Stage 3 → Stage 4).

Quick start
-----------
>>> from post_process.ca2all import Ca2AllPipeline, Ca2AllPipelineConfig
>>> from post_process.polish import PolishPipeline, PolishPipelineConfig
>>>
>>> ca2all_result = Ca2AllPipeline(Ca2AllPipelineConfig(top_k=5)).run(candidates)
>>> polished = PolishPipeline(PolishPipelineConfig(output_dir="out/polished")).run(ca2all_result)
>>> for p in polished:
...     print(p.candidate_id, p.final_score, f"improvement={p.score_improvement:+.3f}")
"""

from .energy_analyzer import EnergyAnalyzer, EnergyAnalysis, WeakRegion
from .local_rebuilder import LocalRebuilder, LocalRebuilderConfig
from .md_annealer import MDAnnealer, MDAnnealerConfig
from .global_relaxer import GlobalRelaxer, GlobalRelaxerConfig
from .iterative_refiner import (
    IterativeRefiner,
    IterativeRefinerConfig,
    PolishedCandidate,
)
from .pipeline import PolishPipeline, PolishPipelineConfig

__all__ = [
    # top-level entry
    "PolishPipeline",
    "PolishPipelineConfig",
    # refiner
    "IterativeRefiner",
    "IterativeRefinerConfig",
    "PolishedCandidate",
    # stages
    "EnergyAnalyzer",
    "EnergyAnalysis",
    "WeakRegion",
    "LocalRebuilder",
    "LocalRebuilderConfig",
    "MDAnnealer",
    "MDAnnealerConfig",
    "GlobalRelaxer",
    "GlobalRelaxerConfig",
]
