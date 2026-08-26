#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File: run_landscape.py
"""
run_landscape.py
================
Energy-landscape analysis over recorded utility-scale QPU samples.

Reads the QPU runs shipped in ``qpu_samples/``, decodes every sampled
bitstring, keeps the self-avoiding-walk (SAW) conformations, scores them with
the lattice contact potential, projects them onto two principal components,
and writes one JSON per protein plus an aggregate summary:

    landscape_result/
    ├── data/<pdbid>_landscape.json   energies, probabilities, PCA coords,
    │                                 funnel score, basin count
    ├── plots/<pdbid>_landscape.png   per-protein overview (quick look)
    └── landscape_summary.csv         one row per protein

``plot_landscapes.py`` turns the JSON files into the paper figures.

Usage
-----
    python run_landscape.py                     # both shipped proteins
    python run_landscape.py --proteins 4y79     # just one
    python run_landscape.py --max-samples 5000  # deeper (slower)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
_LOG = logging.getLogger("run_landscape")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Recorded QPU samples shipped with the artifact.
SAMPLING_DIR = os.path.join(SCRIPT_DIR, "qpu_samples")
SUMMARY_CSV = os.path.join(SAMPLING_DIR, "quantum_data_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Energy-landscape analysis over recorded QPU samples"
    )
    parser.add_argument(
        "--output", default="landscape_result",
        help="Root output directory (default: landscape_result)",
    )
    parser.add_argument(
        "--proteins", nargs="+",
        help="Subset of PDB IDs to analyse (default: every protein in "
             "qpu_samples/quantum_data_summary.csv)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=2500,
        help="Cap on unique decoded conformations per protein (default: 2500)",
    )
    parser.add_argument(
        "--scorer", default="lattice_contact",
        choices=["lattice_contact", "mm"],
        help="Scoring function (default: lattice_contact)",
    )
    args = parser.parse_args()

    if not os.path.isfile(SUMMARY_CSV):
        _LOG.error("Sample catalogue not found: %s", SUMMARY_CSV)
        sys.exit(1)

    from post_process.landscape import LandscapeConfig, LandscapeAnalysisPipeline
    import pandas as pd

    cfg = LandscapeConfig(
        output_dir=args.output,
        max_unique_conformations=args.max_samples,
        scorer=args.scorer,
        save_data=True,
        save_plots=True,
    )
    pipeline = LandscapeAnalysisPipeline(cfg)

    summary_df = pd.read_csv(SUMMARY_CSV)
    protein_ids: Optional[List[str]] = args.proteins or summary_df["pdb_id"].tolist()
    _LOG.info("Analysing %d protein(s): %s", len(protein_ids), ", ".join(protein_ids))

    results = pipeline.run_batch(
        summary_csv=SUMMARY_CSV,
        sampling_dir=SAMPLING_DIR,
        protein_ids=protein_ids,
    )
    _LOG.info("Completed: %d/%d landscapes built", len(results), len(protein_ids))

    if not results:
        _LOG.error("No landscapes built. Check that qpu_samples/ is intact.")
        sys.exit(1)

    for pid, data in sorted(results.items()):
        rec = data.to_summary_record()
        _LOG.info(
            "  %s: L=%s  n_valid=%s  unique=%s  funnel=%.3f  basins=%s  E_min=%.3f",
            pid, rec.get("L"), rec.get("n_valid"), rec.get("n_total_unique"),
            float(rec.get("funnel_score", float("nan"))),
            rec.get("n_basins"),
            float(rec.get("min_energy", float("nan"))),
        )

    _LOG.info("JSON  -> %s", os.path.join(args.output, "data"))
    _LOG.info("Summary -> %s", os.path.join(args.output, "landscape_summary.csv"))
    _LOG.info("Next: python plot_landscapes.py")


if __name__ == "__main__":
    main()
