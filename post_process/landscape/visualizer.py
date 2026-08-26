# --*-- coding:utf-8 --*--
# @time: 3/31/26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File: visualizer.py

"""
LandscapePlotter
================
Four-panel figure for a single protein energy landscape.

Panel layout
------------
  ┌──────────────┬──────────────┬──────────────┬──────────────┐
  │ (A) 3-D      │ (B) 2-D      │ (C) Energy   │ (D) Sampling │
  │ surface      │ contour map  │ distribution │ prob/energy  │
  └──────────────┴──────────────┴──────────────┴──────────────┘

White background, publication-ready typography.
All outputs are saved as PDF (vector) at 600 ppi.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Colour palette — white background, high-contrast
# ---------------------------------------------------------------------------
_BG          = "white"
_PANEL_BG    = "white"
_GRID_COLOR  = "#cccccc"
_SPINE_COLOR = "#555555"
_TEXT        = "#1a1a1a"

_ACCENT_MIN  = "#d62728"   # red star  — global minimum
_ACCENT_PROB = "#1f77b4"   # blue diamond — max probability
_HIST_COLOR  = "#4878d0"   # histogram bars
_LINE_KDE    = "#ee854a"   # KDE curve (orange)
_LINE_BOLTZ  = "#6acc65"   # Boltzmann fit (green)
_LINE_VMIN   = "#d62728"   # vertical line: minimum
_LINE_VMED   = "#956cb4"   # vertical line: median
_LINE_VTOP5  = "#8c613c"   # vertical line: top-5%


class LandscapePlotter:
    """
    Create the four-panel energy landscape figure for one protein.

    Parameters
    ----------
    cfg : LandscapeConfig  (optional, defaults to LandscapeConfig())
    """

    def __init__(self, cfg=None) -> None:
        from .config import LandscapeConfig
        self.cfg = cfg or LandscapeConfig()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def plot(
        self,
        data,
        save_path: Optional[str] = None,
        fmt: str = "pdf",
    ):
        """
        Build and optionally save the four-panel figure.

        Parameters
        ----------
        data       : LandscapeData instance.
        save_path  : full output path (extension determines format).
                     If None the figure is only returned, not saved.
        fmt        : "pdf" (default, vector) | "png".

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.colors as mcolors
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fs = float(getattr(self.cfg, "font_scale", 1.0) or 1.0)

        # ---- Global rcParams for this figure -----------------------------
        with plt.rc_context({
            "font.family":       "sans-serif",
            "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size":         12 * fs,
            "axes.titlesize":    14 * fs,
            "axes.labelsize":    12 * fs,
            "xtick.labelsize":   11 * fs,
            "ytick.labelsize":   11 * fs,
            "legend.fontsize":   10 * fs,
            "axes.titleweight":  "normal",
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "figure.facecolor":  _BG,
            "axes.facecolor":    _PANEL_BG,
            "axes.edgecolor":    _SPINE_COLOR,
            "axes.grid":         True,
            "grid.color":        _GRID_COLOR,
            "grid.linewidth":    0.5,
            "text.color":        _TEXT,
            "xtick.color":       _TEXT,
            "ytick.color":       _TEXT,
            "pdf.fonttype":      42,
            "ps.fonttype":       42,
        }):
            fig = self._build_figure(data, fmt, fs)

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(
                save_path,
                format=fmt,
                dpi=self.cfg.dpi,
                bbox_inches="tight",
                facecolor="white",
            )
            _LOG.info(f"Figure saved → {save_path}")

        return fig

    # ------------------------------------------------------------------ #
    #  Internal figure construction                                         #
    # ------------------------------------------------------------------ #

    def _build_figure(self, data, fmt, fs: float = 1.0):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.colors as mcolors
        from matplotlib import cm

        energies = data.energies
        e_min = float(energies.min())
        e_max_display = float(np.percentile(energies, self.cfg.energy_percentile_clip))
        norm = mcolors.Normalize(vmin=e_min, vmax=e_max_display)
        cmap = cm.get_cmap(self.cfg.colormap)

        base_w, base_h = self.cfg.figsize
        fig_w = base_w * (1.0 + 0.10 * max(fs - 1.0, 0.0))
        fig_h = base_h * (1.0 + 0.18 * max(fs - 1.0, 0.0))
        title_y = min(0.985, 0.965 + 0.02 * max(fs - 1.0, 0.0))
        subtitle_y = min(title_y - 0.045, 0.92 + 0.01 * max(fs - 1.0, 0.0))
        grid_top = 0.80 - 0.035 * max(fs - 1.0, 0.0)
        panel_wspace = 0.20 + 0.10 * max(fs - 1.0, 0.0)
        grid_right = 0.905
        cbar_left = 0.915
        cbar_width = 0.014

        # ---- Figure skeleton -------------------------------------------
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=self.cfg.dpi)
        fig.patch.set_facecolor(_BG)

        # 1×4 horizontal layout with a shared colorbar on the right
        gs = gridspec.GridSpec(
            1, 4,
            figure=fig,
            width_ratios=[1.16, 1.12, 1.01, 1.06],
            wspace=panel_wspace,
            left=0.035, right=grid_right,
            top=grid_top, bottom=0.14,
        )

        ax3d    = fig.add_subplot(gs[0, 0], projection="3d")
        ax2d    = fig.add_subplot(gs[0, 1])
        ax_hist = fig.add_subplot(gs[0, 2])
        ax_prob = fig.add_subplot(gs[0, 3])

        # ---- Panels ----------------------------------------------------
        self._panel_3d_surface(ax3d,    data, norm, cmap, e_max_display, fs)
        self._panel_2d_contour(ax2d,    data, norm, cmap, e_max_display, fs)
        self._panel_energy_hist(ax_hist, data, e_max_display, fs)
        self._panel_prob_vs_energy(ax_prob, data, e_max_display, fs)

        # ---- Shared colorbar (right edge, spans full height) ------------
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([cbar_left, 0.16, cbar_width, 0.64])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Energy (a.u.)", fontsize=12 * fs, labelpad=10)
        cbar.ax.tick_params(labelsize=11 * fs)
        for spine in cbar.ax.spines.values():
            spine.set_edgecolor(_SPINE_COLOR)

        # ---- Overall title ---------------------------------------------
        ev1 = float(data.pca_explained_variance[0])
        ev2 = float(data.pca_explained_variance[1])
        fig.suptitle(
            f"Protein Energy Landscape · {data.protein_id.upper()} ({data.sequence})",
            fontsize=15 * fs,
            color=_TEXT,
            y=title_y,
            va="top",
        )
        fig.text(
            0.5, 0.92,
            f"L = {len(data.sequence)}   |   "
            f"$N_{{\\mathrm{{valid}}}}$ = {data.n_valid}   |   "
            f"Funnel score = {data.funnel_score:.3f}   |   "
            f"Basins $\\approx$ {data.n_basins}   |   "
            f"PC1 + PC2 = {(ev1 + ev2)*100:.1f}% variance",
            ha="center",
            va="top",
            fontsize=11.5 * fs,
            color=_TEXT,
        )
        fig.texts[-1].set_y(subtitle_y)

        return fig

    # ------------------------------------------------------------------ #
    #  Panel A: 3-D surface                                                #
    # ------------------------------------------------------------------ #

    def _panel_3d_surface(self, ax, data, norm, cmap, e_max_display, fs: float = 1.0):
        GX, GY, GZ = data.grid_x, data.grid_y, data.grid_z
        GZ_clip     = np.clip(GZ, None, e_max_display)
        energies    = data.energies
        e_clip      = np.clip(energies, None, e_max_display)

        # Surface
        ax.plot_surface(
            GX, GY, GZ_clip,
            cmap=cmap, norm=norm,
            alpha=self.cfg.surface_alpha,
            linewidth=0, antialiased=True,
            rcount=50, ccount=50,
        )

        # Sample scatter (semi-transparent)
        sc_colors = cmap(norm(e_clip))
        ax.scatter(
            data.pca_coords[:, 0],
            data.pca_coords[:, 1],
            e_clip,
            c=sc_colors,
            s=self.cfg.scatter_size * 0.7,
            alpha=0.55,
            edgecolors="none",
            depthshade=True,
            zorder=4,
        )

        # Global minimum
        mi = data.min_energy_idx
        ax.scatter(
            [data.pca_coords[mi, 0]],
            [data.pca_coords[mi, 1]],
            [float(energies[mi])],
            c=_ACCENT_MIN, s=160, marker="*", zorder=10,
            label=f"Min  E = {energies[mi]:.3f}",
        )

        ev1, ev2 = data.pca_explained_variance
        _style_3d_ax(ax, ev1, ev2, fs)
        ax.set_title("(A) 3-D Energy Surface", pad=10)
        ax.legend(fontsize=10 * fs, loc="upper left",
                  facecolor=_BG, edgecolor=_SPINE_COLOR)

    # ------------------------------------------------------------------ #
    #  Panel B: 2-D contour                                                #
    # ------------------------------------------------------------------ #

    def _panel_2d_contour(self, ax, data, norm, cmap, e_max_display, fs: float = 1.0):
        GX, GY, GZ = data.grid_x, data.grid_y, data.grid_z
        GZ_clip     = np.clip(GZ, None, e_max_display)
        energies    = data.energies
        e_clip      = np.clip(energies, None, e_max_display)

        ax.set_facecolor(_PANEL_BG)

        # Filled contour
        levels = np.linspace(float(energies.min()), e_max_display, 28)
        ax.contourf(GX, GY, GZ_clip, levels=levels,
                    cmap=cmap, norm=norm, alpha=0.85)

        # Topographic iso-lines
        ax.contour(
            GX, GY, GZ_clip, levels=levels[::4],
            colors="#555555", alpha=0.35, linewidths=0.6,
        )

        # Scatter overlay
        ax.scatter(
            data.pca_coords[:, 0], data.pca_coords[:, 1],
            c=e_clip, cmap=cmap, norm=norm,
            s=self.cfg.scatter_size,
            alpha=self.cfg.scatter_alpha,
            edgecolors="white", linewidths=0.2,
            zorder=5,
        )

        # Global minimum (red star)
        mi = data.min_energy_idx
        ax.scatter(
            data.pca_coords[mi, 0], data.pca_coords[mi, 1],
            c=_ACCENT_MIN, s=220, marker="*", zorder=10,
            edgecolors="none",
            label=f"Min  E = {energies[mi]:.3f}",
        )

        # Highest-probability conformation (blue diamond)
        probs = data.probs
        max_prob_idx = int(np.argmax(probs))
        ax.scatter(
            data.pca_coords[max_prob_idx, 0],
            data.pca_coords[max_prob_idx, 1],
            c=_ACCENT_PROB, s=140, marker="D", zorder=9,
            edgecolors="none",
            label=f"Max P  p = {probs[max_prob_idx]:.4f}",
        )

        # Annotate top-k lowest-energy structures
        for rank, idx in enumerate(np.argsort(energies)[: self.cfg.top_k_annotate]):
            ax.annotate(
                f"#{rank+1}",
                xy=(data.pca_coords[idx, 0], data.pca_coords[idx, 1]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=10 * fs,
                color=_ACCENT_MIN, zorder=11,
            )

        ev1, ev2 = data.pca_explained_variance
        _style_2d_ax(ax, ev1, ev2, fs)
        ax.set_title("(B) 2-D Energy Landscape", pad=10)
        ax.legend(fontsize=10 * fs, loc="upper right",
                  facecolor=_BG, edgecolor=_SPINE_COLOR,
                  framealpha=0.9)

    # ------------------------------------------------------------------ #
    #  Panel C: Energy histogram                                           #
    # ------------------------------------------------------------------ #

    def _panel_energy_hist(self, ax, data, e_max_display, fs: float = 1.0):
        energies = data.energies
        e_show   = energies[energies <= e_max_display]
        if len(e_show) == 0:
            e_show = energies

        n_bins = max(20, min(55, len(e_show) // 8))
        ax.hist(
            e_show, bins=n_bins, density=True,
            color=_HIST_COLOR, alpha=0.50,
            edgecolor="white", linewidth=0.4,
            label=f"N = {len(energies)}",
        )

        # KDE overlay
        if len(e_show) > 15:
            try:
                from scipy.stats import gaussian_kde
                kde     = gaussian_kde(e_show, bw_method=0.25)
                x_range = np.linspace(e_show.min(), e_show.max(), 500)
                ax.plot(x_range, kde(x_range),
                        color=_LINE_KDE, linewidth=2.2, label="KDE")
            except Exception:
                pass

        # Reference verticals
        e_min    = float(energies.min())
        e_median = float(np.median(energies))
        e_top5   = float(np.percentile(energies, 5))
        legend_fs = min(10 * fs, 11.0)

        ax.axvline(e_min,    color=_LINE_VMIN,  lw=1.8, ls="--",
                   label=f"Min = {e_min:.2f}")
        ax.axvline(e_median, color=_LINE_VMED,  lw=1.6, ls=":",
                   label=f"Median = {e_median:.2f}")
        ax.axvline(e_top5,   color=_LINE_VTOP5, lw=1.6, ls="-.",
                   label=f"Top-5% = {e_top5:.2f}")

        ax.set_xlabel("Energy (a.u.)")
        ax.set_ylabel("Density")
        ax.set_title("(C) Energy Distribution", pad=10)
        ax.legend(fontsize=legend_fs, facecolor=_BG, edgecolor=_SPINE_COLOR,
                  framealpha=0.9)

    # ------------------------------------------------------------------ #
    #  Panel D: Prob vs Energy                                             #
    # ------------------------------------------------------------------ #

    def _panel_prob_vs_energy(self, ax, data, e_max_display, fs: float = 1.0):
        from matplotlib.ticker import MaxNLocator

        energies = data.energies
        probs    = data.probs
        mask     = energies <= e_max_display
        legend_fs = min(10 * fs, 11.0)

        ax.scatter(
            energies[mask], probs[mask],
            c=_HIST_COLOR, s=self.cfg.scatter_size,
            alpha=0.55, edgecolors="none",
            label="Sampled",
        )

        # Boltzmann fit: log(p) ≈ −β·E + const
        try:
            e_fit = energies[mask]
            p_fit = probs[mask]
            if len(e_fit) > 20 and p_fit.sum() > 0:
                log_p  = np.log(np.clip(p_fit / p_fit.sum(), 1e-15, None))
                coeffs = np.polyfit(e_fit, log_p, 1)
                beta_fit = -float(coeffs[0])
                e_range  = np.linspace(e_fit.min(), e_fit.max(), 400)
                p_pred   = np.exp(np.polyval(coeffs, e_range))
                p_pred  *= p_fit.mean() / (p_pred.mean() + 1e-15)
                ax.plot(
                    e_range, p_pred,
                    color=_LINE_BOLTZ, linewidth=2.2,
                    label=f"Boltzmann fit  β = {beta_fit:.2f}",
                    zorder=5,
                )
        except Exception:
            pass

        ax.set_xlabel("Energy (a.u.)")
        # Keep larger font while avoiding cross-panel overlap in dense layouts.
        ax.set_ylabel("")
        ax.text(
            0.02, 0.50, "Sampling Probability",
            transform=ax.transAxes,
            rotation=90,
            va="center",
            ha="left",
            fontsize=11 * fs,
            color=_TEXT,
        )
        ax.set_title("(D) Sampling Frequency vs Energy", pad=10)
        ax.legend(
            fontsize=legend_fs,
            loc="upper left",
            facecolor=_BG,
            edgecolor=_SPINE_COLOR,
            framealpha=0.9,
        )
        ax.yaxis.set_major_locator(MaxNLocator(5))


# ---------------------------------------------------------------------------
#  Axes styling helpers
# ---------------------------------------------------------------------------

def _style_3d_ax(ax, ev1: float, ev2: float, fs: float = 1.0) -> None:
    """White-background styling for a 3-D axes."""
    ax.set_xlabel(f"PC1 ({ev1*100:.1f}%)", fontsize=11 * fs, labelpad=6)
    ax.set_ylabel(f"PC2 ({ev2*100:.1f}%)", fontsize=11 * fs, labelpad=6)
    ax.set_zlabel("Energy",                fontsize=11 * fs, labelpad=6)
    ax.tick_params(labelsize=9 * fs)

    # Transparent pane faces, light pane edges
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(_GRID_COLOR)

    ax.grid(True, color=_GRID_COLOR, linewidth=0.4)
    ax.set_facecolor(_PANEL_BG)


def _style_2d_ax(ax, ev1: float, ev2: float, fs: float = 1.0) -> None:
    """White-background styling for a 2-D axes."""
    ax.set_xlabel(f"PC1 ({ev1*100:.1f}%)", fontsize=12 * fs)
    ax.set_ylabel(f"PC2 ({ev2*100:.1f}%)", fontsize=12 * fs)
    ax.tick_params(labelsize=11 * fs)
    ax.set_facecolor(_PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE_COLOR)
        spine.set_linewidth(0.8)
