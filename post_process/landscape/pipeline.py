# --*-- coding:utf-8 --*--
# @time: 3/31/26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File: pipeline.py

"""
LandscapeAnalysisPipeline
=========================
High-level orchestrator that wires together LandscapeBuilder and
LandscapePlotter.

Typical use
-----------
Single protein::

    from post_process.landscape import LandscapeConfig, LandscapeAnalysisPipeline
    cfg = LandscapeConfig(output_dir="landscape_result")
    pipeline = LandscapeAnalysisPipeline(cfg)
    data = pipeline.run_single("1a0q", "path/to/samples_1a0q_all_ibm.csv")

Batch::

    results = pipeline.run_batch(
        summary_csv="Sampling_result/sampling_result/quantum_data_summary.csv",
        sampling_dir="Sampling_result/sampling_result",
    )

Detailed PDF::

    pipeline.generate_detail_pdf(data, "img/1a0q_landscape_detail.pdf")
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .builder import LandscapeBuilder
from .config import LandscapeConfig
from .data_types import LandscapeData
from .visualizer import LandscapePlotter

_LOG = logging.getLogger(__name__)


class LandscapeAnalysisPipeline:
    """
    Full pipeline from sampling CSV → landscape data → figures.

    Parameters
    ----------
    cfg : LandscapeConfig  (optional)
    """

    def __init__(self, cfg: Optional[LandscapeConfig] = None) -> None:
        self.cfg = cfg or LandscapeConfig()
        self.builder = LandscapeBuilder(self.cfg)
        self.plotter = LandscapePlotter(self.cfg)
        os.makedirs(self.cfg.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Single protein                                                       #
    # ------------------------------------------------------------------ #

    def run_single(
        self,
        protein_id: str,
        csv_path: str,
        sequence: Optional[str] = None,
    ) -> Optional[LandscapeData]:
        """
        Run landscape analysis for one protein.

        Parameters
        ----------
        protein_id : PDB-style identifier (used for filenames).
        csv_path   : path to merged sampling CSV.
        sequence   : amino-acid sequence; inferred from CSV if None.

        Returns
        -------
        LandscapeData or None if the build failed.
        """
        _LOG.info(f"[{protein_id}] Building landscape …")
        data = self.builder.build_from_csv(csv_path, protein_id=protein_id,
                                           sequence=sequence)
        if data is None:
            _LOG.warning(f"[{protein_id}] Landscape build returned None.")
            return None

        _LOG.info(
            f"[{protein_id}] done — "
            f"n_valid={data.n_valid}  funnel={data.funnel_score:.3f}  "
            f"basins={data.n_basins}"
        )

        if self.cfg.save_data:
            self._save_json(data)

        if self.cfg.save_plots:
            plot_dir = os.path.join(self.cfg.output_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig = self.plotter.plot(data, save_path=None, fmt="pdf")
            for fmt in ("pdf", "png"):
                plot_path = os.path.join(plot_dir, f"{protein_id}_landscape.{fmt}")
                fig.savefig(
                    plot_path,
                    format=fmt,
                    dpi=self.cfg.dpi if fmt == "pdf" else min(300, self.cfg.dpi),
                    bbox_inches="tight",
                    facecolor="white",
                )
                _LOG.info(f"[{protein_id}] Figure saved → {plot_path}")
            plt.close(fig)

        return data

    # ------------------------------------------------------------------ #
    #  Batch                                                                #
    # ------------------------------------------------------------------ #

    def run_batch(
        self,
        summary_csv: str,
        sampling_dir: str,
        protein_ids: Optional[List[str]] = None,
    ) -> Dict[str, LandscapeData]:
        """
        Run landscape analysis for multiple proteins.

        Parameters
        ----------
        summary_csv  : path to quantum_data_summary.csv
                       (columns: pdb_id, sequence, L, …).
        sampling_dir : directory containing per-protein sub-directories.
        protein_ids  : subset of IDs to process; None = all in summary.

        Returns
        -------
        Dict mapping protein_id → LandscapeData (only successful builds).
        """
        summary_df = pd.read_csv(summary_csv)
        if protein_ids is not None:
            summary_df = summary_df[summary_df["pdb_id"].isin(protein_ids)]

        results: Dict[str, LandscapeData] = {}
        summary_records = []
        total = len(summary_df)

        for idx, (_, row) in enumerate(summary_df.iterrows(), 1):
            pid = str(row["pdb_id"])
            seq = str(row["sequence"])
            csv_path = os.path.join(
                sampling_dir, pid, f"samples_{pid}_all_ibm.csv"
            )
            _LOG.info(f"[{idx}/{total}] {pid}")

            if not os.path.isfile(csv_path):
                _LOG.warning(f"  CSV not found: {csv_path}")
                continue

            data = self.run_single(pid, csv_path, seq)
            if data is not None:
                results[pid] = data
                summary_records.append(data.to_summary_record())

        # ---- Aggregate summary CSV ----------------------------------------
        if summary_records:
            out_path = os.path.join(self.cfg.output_dir, "landscape_summary.csv")
            pd.DataFrame(summary_records).sort_values(
                "funnel_score", ascending=False
            ).to_csv(out_path, index=False)
            _LOG.info(f"Summary CSV → {out_path}")

        _LOG.info(
            f"Batch complete: {len(results)}/{total} proteins built successfully."
        )
        return results

    # ------------------------------------------------------------------ #
    #  Detailed PDF export                                                  #
    # ------------------------------------------------------------------ #

    def generate_detail_pdf(
        self,
        data: LandscapeData,
        out_path: str,
    ) -> str:
        """
        Render a high-quality PDF of the four-panel landscape figure.

        Parameters
        ----------
        data     : LandscapeData from run_single / run_batch.
        out_path : destination path (must end in .pdf).

        Returns
        -------
        Absolute path of the saved PDF.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig = self.plotter.plot(data, save_path=out_path, fmt="pdf")
        plt.close(fig)
        _LOG.info(f"PDF → {out_path}")
        return os.path.abspath(out_path)

    # ------------------------------------------------------------------ #
    #  Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _save_json(self, data: LandscapeData) -> None:
        """Persist per-protein landscape data as JSON."""
        data_dir = os.path.join(self.cfg.output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        record = data.to_full_record()
        path = os.path.join(data_dir, f"{data.protein_id}_landscape.json")
        with open(path, "w") as fh:
            json.dump(record, fh, indent=2, default=_json_default)
        _LOG.debug(f"JSON data → {path}")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
