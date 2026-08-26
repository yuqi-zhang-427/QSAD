# --*-- conding:utf-8 --*--
# @time:10/30/25 18:04
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:feature_calculator.py


from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd  # only needed when output_format == "parquet"
    _PANDAS_OK = True
except Exception:
    _PANDAS_OK = False


# =========================
#          CONFIG
# =========================

@dataclass
class FeatureConfig:
    # IO
    output_path: str = "features.jsonl"
    output_format: str = "jsonl"  # "jsonl" | "parquet"
    carry_keys: Tuple[str, ...] = ("pdb_id", "case_id", "residue_range", "id", "bitstring")

    # Global/density (Å; will be auto-scaled to lattice units)
    contact_cutoff: float = 8.0
    clash_min_dist: float = 3.4
    nn_exclude: int = 1  # will be promoted to >=2 for lattice duplicates

    # Secondary-structure proxies (Å; will be auto-scaled)
    alpha_i3_max: float = 5.4
    alpha_i4_max: float = 6.2
    beta_pair_max: float = 7.0  # |i-j|>=3 and within this => beta-like

    # Burial / packing (Å; will be auto-scaled)
    burial_r: float = 6.0  # Gaussian kernel radius
    pack_rep_max: float = 4.5
    pack_att_min: float = 5.5
    pack_att_max: float = 7.5

    # Pseudo-atom reconstruction (Å; will be auto-scaled)
    use_pseudo_atoms: bool = True
    ca_n_dist: float = 1.46
    ca_cb_dist: float = 1.52
    ca_o_dist: float = 1.24

    # H-bond (pseudo N-O) window (Å; will be auto-scaled)
    hb_min: float = 2.6
    hb_max: float = 3.2

    # Sequence classes
    hydrophobic: str = "AVLIMFWYV"
    charged: str = "KRDEH"
    polar: str = "STNQYC"

    # Fail-soft behavior
    ignore_bad_rows: bool = True


# =========================
#     FEATURE CALCULATOR
# =========================

class StructuralFeatureCalculator:
    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg

        self._hydrophobic = set(cfg.hydrophobic.upper())
        self._charged = set(cfg.charged.upper())
        self._polar = set(cfg.polar.upper())

        if self.cfg.output_format not in ("jsonl", "parquet"):
            raise ValueError("output_format must be 'jsonl' or 'parquet'.")

    # ---------- public API ----------

    def evaluate_jsonl(self, decoded_jsonl_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        out_path = output_path or self.cfg.output_path
        if self.cfg.output_format == "jsonl":
            written = self._run_streaming_jsonl(decoded_jsonl_path, out_path)
        else:
            if not _PANDAS_OK:
                raise RuntimeError("Pandas not available but output_format='parquet' requested.")
            written = self._run_to_parquet(decoded_jsonl_path, out_path)
        return {"written": written, "output_path": out_path, "format": self.cfg.output_format}

    # ---------- writers ----------

    def _run_streaming_jsonl(self, in_path: str, out_path: str) -> int:
        cnt = 0
        with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                try:
                    features = self._compute_features(rec)
                except Exception as e:
                    if not self.cfg.ignore_bad_rows:
                        raise
                    features = {"_feature_error": str(e)}
                    for k in self.cfg.carry_keys:
                        if k in rec:
                            features[k] = rec[k]
                fout.write(json.dumps(features) + "\n")
                cnt += 1
        return cnt

    def _run_to_parquet(self, in_path: str, out_path: str) -> int:
        rows: List[Dict[str, Any]] = []
        with open(in_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                try:
                    rows.append(self._compute_features(rec))
                except Exception as e:
                    if not self.cfg.ignore_bad_rows:
                        raise
                    row = {"_feature_error": str(e)}
                    for k in self.cfg.carry_keys:
                        if k in rec:
                            row[k] = rec[k]
                    rows.append(row)
        if not rows:
            with open(out_path, "wb") as f:
                pass
            return 0
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_parquet(out_path, index=False)
        return len(rows)

    # ---------- per-record ----------

    def _compute_features(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        seq = str(rec["sequence"])
        X = self._as_np(rec.get("main_positions", []))  # (N,3)
        self._validate_lengths(seq, X)

        N = X.shape[0]

        # ----- lattice scale estimation and Å->lattice conversion -----
        AA_CA = 3.8  # Å, canonical Cα–Cα distance
        adj = np.linalg.norm(X[1:] - X[:-1], axis=1) if N > 1 else np.array([])
        step = float(np.median(adj)) if adj.size else 1.0  # lattice step length
        scale = AA_CA / max(step, 1e-8)                   # Å per lattice unit

        # local thresholds (all in lattice units after scaling)
        contact_cutoff = self.cfg.contact_cutoff / scale
        clash_min_dist = self.cfg.clash_min_dist / scale
        alpha_i3_max = self.cfg.alpha_i3_max / scale
        alpha_i4_max = self.cfg.alpha_i4_max / scale
        beta_pair_max = self.cfg.beta_pair_max / scale
        burial_r = self.cfg.burial_r / scale
        pack_rep_max = self.cfg.pack_rep_max / scale
        pack_att_min = self.cfg.pack_att_min / scale
        pack_att_max = self.cfg.pack_att_max / scale
        ca_n_dist = self.cfg.ca_n_dist / scale
        ca_cb_dist = self.cfg.ca_cb_dist / scale
        ca_o_dist = self.cfg.ca_o_dist / scale
        hb_min = self.cfg.hb_min / scale
        hb_max = self.cfg.hb_max / scale
        nn_exclude = max(2, self.cfg.nn_exclude)  # robust to lattice duplicates

        # pairwise distances
        D = self._pdist(X) if N else np.zeros((0, 0), dtype=float)

        out: Dict[str, Any] = {}
        for k in self.cfg.carry_keys:
            if k in rec:
                out[k] = rec[k]

        # ---------- A: global/density ----------
        out["length"] = float(N)
        out["Rg"] = self._radius_of_gyration(X) if N else 0.0
        out["end_to_end"] = float(np.linalg.norm(X[-1] - X[0])) if N > 1 else 0.0
        out["contact_density"] = self._contact_density(D, contact_cutoff) if N else 0.0

        # ---------- B: clashes / nearest-neighbor ----------
        out["clash_count"] = float(self._clash_count(D, clash_min_dist))
        p10, p50, p90 = self._nn_dists(D, nn_exclude)
        out["nn_dist_p10"] = p10
        out["nn_dist_p50"] = p50
        out["nn_dist_p90"] = p90

        # ---------- C: secondary-structure-ish proxies ----------
        out["alpha_hits"] = float(self._alpha_hits(X, alpha_i3_max, alpha_i4_max))
        out["beta_hits"] = float(self._beta_hits(D, beta_pair_max))
        denom = max(1.0, float(N))
        out["rama_allowed_ratio"] = (out["alpha_hits"] + out["beta_hits"]) / denom

        # ---------- D: burial / packing ----------
        b_mean, b_max = self._burial_stats(D, burial_r)
        out["burial_mean"] = b_mean
        out["burial_max"] = b_max
        rep, att = self._packing_counts(D, pack_rep_max, pack_att_min, pack_att_max)
        out["packing_rep_count"] = float(rep)
        out["packing_att_count"] = float(att)

        # ---------- E: sequence stats ----------
        out.update(self._seq_stats(seq))

        # ---------- F: pseudo atoms ----------
        if self.cfg.use_pseudo_atoms and N >= 2:
            Npos, Opos, CBpos = self._pseudo_atoms_from_ca(X, ca_n_dist, ca_o_dist, ca_cb_dist)
            if len(Npos) and len(Opos):
                D_no = self._pdist_cross(Npos, Opos)
                mask = (D_no >= hb_min) & (D_no <= hb_max)
                hb_cnt = int(mask.sum())
                out["hb_count_pseudo"] = float(hb_cnt)
                out["hb_density_pseudo"] = float(hb_cnt) / max(1.0, float(N))
            else:
                out["hb_count_pseudo"] = 0.0
                out["hb_density_pseudo"] = 0.0

            if len(CBpos) >= 2:
                D_cb = self._pdist(CBpos)
                m = np.triu(np.ones_like(D_cb, dtype=bool), k=2)
                out["cbeta_rep_count"] = float((D_cb[m] < pack_rep_max).sum())
                m2 = (D_cb >= pack_att_min) & (D_cb <= pack_att_max)
                out["cbeta_att_count"] = float((m2 & m).sum())
            else:
                out["cbeta_rep_count"] = 0.0
                out["cbeta_att_count"] = 0.0
        else:
            out["hb_count_pseudo"] = 0.0
            out["hb_density_pseudo"] = 0.0
            out["cbeta_rep_count"] = 0.0
            out["cbeta_att_count"] = 0.0

        # ---------- G: copy energy components if present (EXCLUDING E_total) ----------
        for k, v in rec.items():
            if isinstance(v, (int, float)) and k.startswith("E_") and k != "E_total":
                out[k] = float(v)

        return out

    # =========================
    #        UTILITIES
    # =========================

    @staticmethod
    def _as_np(positions: List[List[float]]) -> np.ndarray:
        X = np.asarray(positions, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError(f"main_positions must be (N,3), got {X.shape}")
        return X

    @staticmethod
    def _validate_lengths(seq: str, X: np.ndarray) -> None:
        if len(seq) != len(X):
            raise ValueError(f"len(sequence)={len(seq)} != len(main_positions)={len(X)}")

    @staticmethod
    def _pdist(X: np.ndarray) -> np.ndarray:
        diff = X[:, None, :] - X[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))

    @staticmethod
    def _pdist_cross(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        diff = A[:, None, :] - B[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))

    @staticmethod
    def _radius_of_gyration(X: np.ndarray) -> float:
        c = X.mean(axis=0, keepdims=True)
        return float(np.sqrt(((X - c) ** 2).sum(axis=1).mean()))

    @staticmethod
    def _contact_density(D: np.ndarray, cutoff: float) -> float:
        N = D.shape[0]
        if N < 3:
            return 0.0
        mask = np.triu(np.ones_like(D, dtype=bool), k=2)
        return float((D[mask] < cutoff).sum()) / float(N)

    @staticmethod
    def _clash_count(D: np.ndarray, thr: float) -> int:
        if D.size == 0:
            return 0
        mask = np.triu(np.ones_like(D, dtype=bool), k=2)
        return int((D[mask] < thr).sum())

    @staticmethod
    def _nn_dists(D: np.ndarray, nn_exclude: int) -> Tuple[float, float, float]:
        N = D.shape[0]
        if N == 0:
            return (0.0, 0.0, 0.0)
        nn: List[float] = []
        for i in range(N):
            m = np.ones(N, dtype=bool)
            m[i] = False
            for k in range(1, nn_exclude + 1):
                if i - k >= 0: m[i - k] = False
                if i + k < N: m[i + k] = False
            cand = D[i][m]
            if cand.size:
                nn.append(float(np.min(cand)))
        if not nn:
            return (0.0, 0.0, 0.0)
        arr = np.array(nn, dtype=float)
        return (float(np.percentile(arr, 10)),
                float(np.percentile(arr, 50)),
                float(np.percentile(arr, 90)))

    @staticmethod
    def _alpha_hits(X: np.ndarray, i3_max: float, i4_max: float) -> int:
        N = len(X)
        cnt = 0
        for i in range(N):
            j = i + 3
            if j < N and np.linalg.norm(X[j] - X[i]) <= i3_max:
                cnt += 1
            j = i + 4
            if j < N and np.linalg.norm(X[j] - X[i]) <= i4_max:
                cnt += 1
        return cnt

    @staticmethod
    def _beta_hits(D: np.ndarray, beta_pair_max: float) -> int:
        N = D.shape[0]
        if N < 3:
            return 0
        mask = np.triu(np.ones_like(D, dtype=bool), k=3)  # |i-j| >= 3
        return int((D[mask] <= beta_pair_max).sum())

    @staticmethod
    def _burial_stats(D: np.ndarray, burial_r: float) -> Tuple[float, float]:
        N = D.shape[0]
        if N == 0:
            return (0.0, 0.0)
        B = np.exp(- (D / burial_r) ** 2)
        np.fill_diagonal(B, 0.0)
        vec = B.sum(axis=1)
        return float(vec.mean()), float(vec.max(initial=0.0))

    @staticmethod
    def _packing_counts(D: np.ndarray, rep_max: float, att_min: float, att_max: float) -> Tuple[int, int]:
        if D.size == 0:
            return (0, 0)
        mask = np.triu(np.ones_like(D, dtype=bool), k=2)
        rep = int(((D < rep_max) & mask).sum())
        att = int((((D >= att_min) & (D <= att_max)) & mask).sum())
        return rep, att

    def _seq_stats(self, seq: str) -> Dict[str, float]:
        N = len(seq)
        if N == 0:
            return dict(frac_hydrophobic=0.0, frac_charged=0.0, frac_polar=0.0,
                        n_proline=0.0, n_glycine=0.0)
        s = seq.upper()
        n_h = sum(aa in self._hydrophobic for aa in s)
        n_c = sum(aa in self._charged for aa in s)
        n_p = sum(aa in self._polar for aa in s)
        return dict(
            frac_hydrophobic=n_h / N,
            frac_charged=n_c / N,
            frac_polar=n_p / N,
            n_proline=float(s.count("P")),
            n_glycine=float(s.count("G")),
        )

    @staticmethod
    def _pseudo_atoms_from_ca(X: np.ndarray, ca_n: float, ca_o: float, ca_cb: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Very light Cα-only pseudo-atom construction:
        - Tangent t_i ~ normalize((CA_{i+1}-CA_i) + (CA_i-CA_{i-1}))
        - Normal n_i ~ normalize(cross(t_i, z) or cross(t_i, y) as fallback)
        - Place N_i at CA_i - t_i * ca_n
        - Place O_i at CA_i + t_i * ca_o
        - Place Cβ_i at CA_i + n_i * ca_cb
        """
        N = len(X)
        if N == 1:
            return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))

        fwd = np.zeros_like(X)
        bwd = np.zeros_like(X)
        fwd[:-1] = X[1:] - X[:-1]
        bwd[1:] = X[1:] - X[:-1]
        bwd = -bwd
        t = fwd + bwd
        t[0] = X[1] - X[0]
        t[-1] = X[-1] - X[-2]
        t = StructuralFeatureCalculator._safe_normalize_rows(t)

        n = np.cross(t, np.array([0.0, 0.0, 1.0]))
        zero_mask = (np.linalg.norm(n, axis=1) < 1e-8)
        n[zero_mask] = np.cross(t[zero_mask], np.array([0.0, 1.0, 0.0]))
        n = StructuralFeatureCalculator._safe_normalize_rows(n)

        Npos = X - t * ca_n
        Opos = X + t * ca_o
        CBpos = X + n * ca_cb

        return Npos, Opos, CBpos

    @staticmethod
    def _safe_normalize_rows(V: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        L = np.linalg.norm(V, axis=1, keepdims=True)
        L[L < eps] = 1.0
        return V / L
