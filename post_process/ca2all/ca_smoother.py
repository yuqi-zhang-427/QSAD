# --*-- coding:utf-8 --*--
# @time: 2026-03-30
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File: ca_smoother.py

"""
Cα trace smoother for lattice-derived coarse-grained structures.

Lattice coordinates produce angular, jagged Cα traces.  NERF reconstruction
from such traces accumulates frame errors, leading to ~2-3 Å higher RMSD
than the underlying coarse-grained coordinates suggest.

This module applies iterative Gaussian smoothing + bidirectional distance
projection to produce a physically realistic Cα trace while preserving the
3.8 Å inter-residue spacing required by the backbone reconstructor.

Algorithm
---------
1. Gaussian smoothing (sigma controls neighbourhood radius).
2. Iterative bidirectional distance projection:
   - Forward pass: each CA_i is projected onto the sphere of radius d_target
     centred at CA_{i-1}.
   - Backward pass: reverse direction.
   Repeat until bond-length RMS error < tol or max_iter reached.
3. Optionally centre the smoothed trace at the same centroid as the input.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def smooth_ca_trace(
    ca: np.ndarray,
    sigma: float = 1.5,
    target_distance: float = 3.8,
    max_iter: int = 100,
    tol: float = 1e-4,
    preserve_centroid: bool = True,
) -> np.ndarray:
    """
    Smooth a Cα trace while preserving inter-residue bond lengths.

    Parameters
    ----------
    ca : (N, 3) array
        Scaled Cα positions (must already be in Å with ~3.8 Å spacing).
    sigma : float
        Gaussian smoothing half-width in residue units.  Larger values
        produce smoother traces; values < 0.5 are nearly no-ops.
    target_distance : float
        Desired Cα–Cα bond length (Å).  Should match BuilderConfig.target_ca_distance.
    max_iter : int
        Maximum distance-projection iterations.
    tol : float
        Convergence: stop when RMS bond-length error (Å) drops below this.
    preserve_centroid : bool
        If True, translate the smoothed trace so its centroid matches the input.

    Returns
    -------
    (N, 3) array of smoothed Cα positions with ~target_distance spacing.
    """
    ca = np.asarray(ca, dtype=float)
    n = len(ca)
    if n <= 2:
        return ca.copy()

    centroid_in = ca.mean(axis=0)

    # ── 1. Gaussian smoothing ────────────────────────────────────────────
    smoothed = _gaussian_smooth(ca, sigma)

    # ── 2. Iterative bidirectional distance projection ───────────────────
    smoothed = _project_distances(smoothed, target_distance, max_iter, tol)

    # ── 3. Restore centroid ──────────────────────────────────────────────
    if preserve_centroid:
        smoothed += centroid_in - smoothed.mean(axis=0)

    return smoothed


# ── Internal helpers ────────────────────────────────────────────────────────

def _gaussian_smooth(ca: np.ndarray, sigma: float) -> np.ndarray:
    """Apply 1-D Gaussian smoothing along the sequence axis."""
    n = len(ca)
    if sigma <= 0.0:
        return ca.copy()

    # Kernel half-width: 3σ, clamped to sequence length
    half = min(int(np.ceil(3 * sigma)), n - 1)
    r = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-0.5 * (r / sigma) ** 2)
    kernel /= kernel.sum()

    smoothed = np.empty_like(ca)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n - 1, i + half)
        k_lo = lo - (i - half)
        k_hi = k_lo + (hi - lo)
        w = kernel[k_lo: k_hi + 1]
        w /= w.sum()
        smoothed[i] = (ca[lo: hi + 1] * w[:, None]).sum(axis=0)

    return smoothed


def _project_distances(
    ca: np.ndarray,
    d: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """
    Iterative bidirectional projection so all consecutive CA pairs are
    exactly distance d apart, minimising overall displacement from input.
    """
    pos = ca.copy()
    n = len(pos)

    for _ in range(max_iter):
        # Forward pass
        for i in range(1, n):
            v = pos[i] - pos[i - 1]
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                pos[i] = pos[i - 1] + v * (d / norm)

        # Backward pass
        for i in range(n - 2, -1, -1):
            v = pos[i] - pos[i + 1]
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                pos[i] = pos[i + 1] + v * (d / norm)

        # Convergence check
        dists = np.linalg.norm(pos[1:] - pos[:-1], axis=1)
        rms_err = float(np.sqrt(np.mean((dists - d) ** 2)))
        if rms_err < tol:
            break

    return pos


__all__ = ["smooth_ca_trace"]
