#!/usr/bin/env python3
"""
plot_landscapes.py
==================
Render the paper's energy-landscape figures from the recorded QPU samples in
``qpu_samples/``:

    1a0q -> Fig. 9   (L=16, 120 qubits)
    4y79 -> Fig. 10  (L=9,  49 qubits)

Output: ``landscape_result/figures/<pdbid>_landscape.{pdf,png}``

This is the exact rendering path used for the figures in the paper: the
landscape is rebuilt from the samples and drawn by the analysis module's own
plotter, with figure-level titles stripped (the caption carries them).

Usage
-----
    python plot_landscapes.py            # both paper figures
    python plot_landscapes.py 4y79       # just one
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figs.paper_figure_style import figure_size, style_value

SUMMARY_CSV = ROOT / "qpu_samples" / "quantum_data_summary.csv"
SAMPLING_DIR = ROOT / "qpu_samples"
OUT_DIR = ROOT / "landscape_result" / "figures"

# Proteins shown in the paper: 1a0q -> Fig. 9, 4y79 -> Fig. 10.
SELECTED = ["1a0q", "4y79"]


def _landscape_config(pdbid: str):
    from post_process.landscape import LandscapeConfig

    figure_id = f"{pdbid}_landscape"
    figsize = figure_size(figure_id, (25.5, 6.9), group="landscape")
    font_scale = float(style_value("font_scale", 1.0, figure_id=figure_id, group="landscape"))
    return LandscapeConfig(
        output_dir=str(ROOT / "landscape_result"),
        save_data=False,
        save_plots=False,
        figsize=figsize,
        font_scale=font_scale,
    )


def _strip_figure_level_titles(fig) -> None:
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        suptitle.remove()
    for text in list(fig.texts):
        text.remove()


def main() -> None:
    import matplotlib.pyplot as plt

    from post_process.landscape import LandscapeAnalysisPipeline

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(SUMMARY_CSV)
    seq_map = {
        str(row["pdb_id"]).strip(): str(row["sequence"]).strip()
        for _, row in summary_df.iterrows()
    }

    targets = sys.argv[1:] or SELECTED
    for pdbid in targets:
        cfg = _landscape_config(pdbid)
        pipeline = LandscapeAnalysisPipeline(cfg)
        csv_path = SAMPLING_DIR / pdbid / f"samples_{pdbid}_all_ibm.csv"
        sequence = seq_map.get(pdbid)
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing sampling CSV for {pdbid}: {csv_path}")

        print(f"[render] {pdbid}")
        data = pipeline.run_single(pdbid, str(csv_path), sequence)
        if data is None:
            raise RuntimeError(f"Landscape build returned None for {pdbid}")

        for ext in ("pdf", "png"):
            out_path = OUT_DIR / f"{pdbid}_landscape.{ext}"
            fig = pipeline.plotter.plot(data, save_path=None, fmt=ext)
            _strip_figure_level_titles(fig)
            fig.savefig(out_path, format=ext, dpi=cfg.dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
