# --*-- conding:utf-8 --*--
# @time:3/26/26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:geometry.py

from __future__ import annotations

import math

import numpy as np


def scale_backbone_positions(main_positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(main_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected positions with shape (N, 3), got {positions.shape}")
    if len(positions) <= 1:
        return positions.astype(float, copy=True)
    step = float(np.median(np.linalg.norm(positions[1:] - positions[:-1], axis=1)))
    scale = 3.8 / max(step, 1e-8)
    return positions * scale


def kabsch_rmsd(predicted: np.ndarray, reference: np.ndarray) -> float:
    p = np.asarray(predicted, dtype=float)
    r = np.asarray(reference, dtype=float)
    if p.shape != r.shape:
        raise ValueError(f"Shape mismatch for RMSD: {p.shape} vs {r.shape}")
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"Expected positions with shape (N, 3), got {p.shape}")

    p_centered = p - p.mean(axis=0)
    r_centered = r - r.mean(axis=0)
    covariance = p_centered.T @ r_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    aligned = p_centered @ rotation
    return math.sqrt(float(np.mean(np.sum((aligned - r_centered) ** 2, axis=1))))
