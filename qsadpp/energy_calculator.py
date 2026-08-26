# --*-- conding:utf-8 --*--
# @time:10/23/25 17:08
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:energy_calculator.py

from __future__ import annotations
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

_LOG = logging.getLogger(__name__)
if not _LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

_MJ_MATRIX_PATH = os.path.join(os.path.dirname(__file__), "mj_matrix.txt")


@dataclass
class EnergyConfig:
    # geometric baselines
    r_min: float = 1.0                      # soft steric wall radius for nonbonded CA-CA
    r_contact: float = 1.4                  # legacy hard contact cutoff (only if soft_contact=False)
    d0: float = 1.0                         # target bond length for CA-CA

    # penalties
    lambda_overlap: float = 1000.0          # penalty per true overlap pair

    # MJ soft-contact kernel
    soft_contact: bool = True
    contact_r0: float = 1.8                 # length scale in exp(-(r/r0)^p)
    contact_p: float = 6.0                  # exponent
    use_sigmoid_contact: bool = False
    sigmoid_alpha: float = 5.0

    # NEW: hydrophobic burial
    burial_use_cb: bool = True              # compute burial on pseudo-CB if available
    burial_radius: float = 3.5              # Gaussian length scale for burial counting
    burial_ref: float = 6.0                 # reference burial level B_ref
    # simple hydrophobicity scale (positive = hydrophobic preference)
    hydrophobicity: Dict[str, float] = field(default_factory=lambda: {
        "A": 1.8, "V": 4.2, "I": 4.5, "L": 3.8, "M": 1.9, "F": 2.8, "W": -0.9, "Y": -1.3,
        "C": 2.5, "P": -1.6, "G": -0.4, "S": -0.8, "T": -0.7, "N": -3.5, "Q": -3.5,
        "D": -3.5, "E": -3.5, "K": -3.9, "R": -4.5, "H": -3.2
    })

    # NEW: virtual Cβ packing (pseudo-CB positions from CA chain)
    cbeta_enable: bool = True
    cbeta_rep_r: float = 3.5                # repulsive radius
    cbeta_att_r: float = 5.0                # mild attractive distance
    cbeta_eps_rep: float = 1.0              # repulsion strength
    cbeta_eps_att: float = 0.2              # mild attraction strength

    # NEW: main-chain hydrogen bond (CA-based proxy, soft directional potential)
    hb_enable: bool = True
    hb_r0: float = 3.6                      # preferred CA-CA (proxy) distance for H-bond
    hb_sigma_r: float = 0.4                 # radial width
    hb_cos0: float = 0.8                    # preferred alignment between local tangents
    hb_kappa: float = 10.0                  # angular sharpness
    hb_min_sep: int = 3                     # |i-j| >= hb_min_sep

    # NEW: Ramachandran-like potential (CA-based pseudo-phi/psi)
    rama_enable: bool = True
    rama_mode: str = "generic"              # "generic" mixture prior or "table"
    rama_table_path: Optional[str] = None   # optional external table: lines "phi psi logP"
    rama_sigma_deg: float = 30.0            # Gaussian width for generic wells
    rama_alpha_center: Tuple[float, float] = (-60.0, -45.0)  # (phi, psi) in degrees
    rama_beta_center: Tuple[float, float] = (-120.0, 120.0)
    rama_mix_alpha: float = 0.55            # mixture weights for generic prior
    rama_mix_beta: float = 0.45

    # per-term weights
    weights: Dict[str, float] = field(default_factory=lambda: {
        "steric": 1.0,
        "geom": 0.5,
        "bond": 0.2,
        "mj": 2.0,
        "dihedral": 0.3,
        "hydroph": 0.5,      # NEW
        "cbeta": 0.4,        # NEW
        "hb": 0.6,           # NEW
        "rama": 0.5,         # NEW
    })

    # normalization strategy
    normalize_mode: str = "per_term"        # "per_term" or "legacy"
    normalize: bool = True                  # used only if normalize_mode == "legacy"

    output_path: str = "decoded_with_energy.jsonl"


class LatticeEnergyCalculator:
    """
    Energy for main-chain-only lattice proteins with added secondary-structure-sensitive terms.

    Implemented components (each per-term normalized):
      - Bond length     (E_bond)
      - Bending angle   (E_geom)
      - Dihedral torsion (E_dihedral)
      - Steric soft-wall + overlap count penalty (E_steric)
      - MJ soft-contact (E_mj)
      - Hydrophobic burial (E_hydroph)          [NEW]
      - Virtual Cβ packing (E_cbeta)            [NEW]
      - Main-chain H-bond proxy (E_hb)          [NEW]
      - Ramachandran-like CA-based prior (E_rama) [NEW]
    """

    def __init__(self, cfg: EnergyConfig = EnergyConfig()):
        self.cfg = cfg
        self._mj_matrix, self._aa_index = self._load_mj_matrix()
        self._rama_table = None
        if self.cfg.rama_enable and self.cfg.rama_mode == "table" and self.cfg.rama_table_path:
            self._rama_table = self._load_rama_table(self.cfg.rama_table_path)

    # ---------------- I/O helpers ----------------
    def _load_mj_matrix(self) -> Tuple[np.ndarray, Dict[str, int]]:
        path = _MJ_MATRIX_PATH
        if not os.path.exists(path):
            _LOG.warning("MJ matrix not found in package: %s; all E_MJ=0", path)
            return np.zeros((20, 20)), {}
        with open(path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        headers = lines[0].split()
        mat = np.zeros((len(headers), len(headers)))
        for i, line in enumerate(lines[1:]):
            vals = [float(x) for x in line.split()]
            for j in range(len(vals)):
                mat[i, j] = vals[j]
                mat[j, i] = vals[j]
        aa_index = {aa: i for i, aa in enumerate(headers)}
        _LOG.info("Loaded MJ matrix from package: %s (%d residues)", path, len(aa_index))
        return mat, aa_index

    def _load_rama_table(self, path: str) -> np.ndarray:
        """
        Load a simple table: each line 'phi_deg psi_deg logP'. Will be interpolated via RBF-like kernel.
        """
        try:
            arr = []
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    sp = line.split()
                    if len(sp) < 3:
                        continue
                    arr.append([float(sp[0]), float(sp[1]), float(sp[2])])
            if not arr:
                _LOG.warning("Empty Ramachandran table: %s", path)
                return None
            return np.array(arr, dtype=float)  # (M,3) [phi, psi, logP]
        except Exception as e:
            _LOG.warning("Failed to load Ramachandran table %s: %s", path, e)
            return None

    # ---------------- math helpers ----------------
    @staticmethod
    def _pairwise_dist(pts: np.ndarray) -> np.ndarray:
        diff = pts[:, None, :] - pts[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    @staticmethod
    def _safe_mean(x_sum_or_vec, denom: int) -> float:
        # x_sum_or_vec can be a scalar sum or a vector to be summed
        val = float(np.sum(x_sum_or_vec))
        return val / max(1, denom)

    @staticmethod
    def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.clip(n, eps, None)

    @staticmethod
    def _dihedral_angles(P: np.ndarray) -> np.ndarray:
        N = P.shape[0]
        if N < 4:
            return np.array([], dtype=float)
        b0 = P[1:-2] - P[0:-3]
        b1 = P[2:-1] - P[1:-2]
        b2 = P[3:   ] - P[2:-1]
        b1n = LatticeEnergyCalculator._normalize(b1)
        n0 = np.cross(b0, b1)
        n1 = np.cross(b1, b2)
        n0n = LatticeEnergyCalculator._normalize(n0)
        n1n = LatticeEnergyCalculator._normalize(n1)
        m1 = np.cross(n0n, b1n)
        x = np.sum(n0n * n1n, axis=1)
        y = np.sum(m1 * n1n, axis=1)
        return np.arctan2(y, x)  # radians

    # ---------------- virtual Cβ positions ----------------
    def _pseudo_cb_positions(self, P: np.ndarray) -> np.ndarray:
        """
        Build pseudo-CB from CA using a simple geometric construction.
        Uses local frame from CA[i-1], CA[i], CA[i+1]. Endpoints fall back to tangential approximation.
        """
        N = len(P)
        if N < 3:
            return P.copy()
        CB = np.zeros_like(P)
        # internal residues
        v_prev = P[1:-1] - P[0:-2]
        v_next = P[2:] - P[1:-1]
        t_prev = self._normalize(v_prev)
        t_next = self._normalize(v_next)
        t = self._normalize(t_prev + t_next)  # smoothed tangent
        n = self._normalize(np.cross(t_prev, t_next))  # binormal-like
        # choose a perpendicular direction approximately mimicking tetrahedral offset
        u = self._normalize(np.cross(n, t))
        offset = 0.5  # lattice-scale CB offset; can be tuned
        CB[1:-1] = P[1:-1] + offset * u
        # endpoints: extrapolate
        if N >= 2:
            t0 = self._normalize(P[1] - P[0])
            CB[0] = P[0] + offset * t0.squeeze()
            tL = self._normalize(P[-1] - P[-2])
            CB[-1] = P[-1] + offset * tL.squeeze()
        return CB

    # ---------------- hydrophobic burial ----------------
    def _hydrophobic_burial(self, P: np.ndarray, seq: str) -> float:
        """
        Burial score: B_i = sum_j exp(-(r_ij / r_b)^2) for chosen atoms (CA or pseudo-CB).
        Energy: E_hydroph = mean_i [ - h(aa_i) * (B_i - B_ref) ] so hydrophobics prefer higher burial.
        """
        if self.cfg.burial_use_cb:
            sites = self._pseudo_cb_positions(P)
        else:
            sites = P
        N = len(sites)
        if N == 0:
            return 0.0
        dist = self._pairwise_dist(sites)
        r_b = max(1e-8, self.cfg.burial_radius)
        K = np.exp(-(dist / r_b) ** 2)
        np.fill_diagonal(K, 0.0)
        B = np.sum(K, axis=1)  # burial count per residue
        hmap = self.cfg.hydrophobicity
        h = np.array([hmap.get(a, 0.0) for a in seq[:N]], dtype=float)
        E_vec = -h * (B - float(self.cfg.burial_ref))
        return self._safe_mean(E_vec, N)

    # ---------------- virtual Cβ packing ----------------
    def _cbeta_packing(self, P: np.ndarray) -> float:
        """
        Pair potential on pseudo-CB:
          - soft repulsion for r < cbeta_rep_r
          - mild attraction around cbeta_att_r (Gaussian well)
        Per-pair mean.
        """
        if not self.cfg.cbeta_enable or len(P) < 3:
            return 0.0
        CB = self._pseudo_cb_positions(P)
        dist = self._pairwise_dist(CB)
        N = len(CB)
        mask = np.triu(np.ones((N, N), dtype=bool), k=2)
        D = dist[mask]
        num_pairs = int(np.count_nonzero(mask))
        if num_pairs == 0:
            return 0.0
        # repulsive quadratic wall
        rep = np.clip(self.cfg.cbeta_rep_r - D, 0.0, None)
        E_rep = float(self.cfg.cbeta_eps_rep) * np.sum(rep * rep)
        # mild attractive Gaussian around target
        att = np.exp(-((D - float(self.cfg.cbeta_att_r)) ** 2) / (2.0 * 0.5 ** 2))
        E_att = -float(self.cfg.cbeta_eps_att) * np.sum(att)
        return (E_rep + E_att) / num_pairs

    # ---------------- hydrogen bond (CA-proxy) ----------------
    def _hbond_energy(self, P: np.ndarray) -> float:
        """
        CA-based proxy: favor pairs with |i-j|>=hb_min_sep whose CA-CA distance is near hb_r0
        and local tangents aligned (cos ~ hb_cos0). Per-pair mean.
        """
        if not self.cfg.hb_enable or len(P) < 4:
            return 0.0
        N = len(P)
        dist = self._pairwise_dist(P)
        tang = np.zeros_like(P)
        tang[1:-1] = self._normalize(P[2:] - P[:-2]).squeeze()
        tang[0] = self._normalize(P[1] - P[0]).squeeze()
        tang[-1] = self._normalize(P[-1] - P[-2]).squeeze()

        pairs_sum = 0.0
        pairs_cnt = 0
        r0 = float(self.cfg.hb_r0)
        sig_r = max(1e-6, float(self.cfg.hb_sigma_r))
        cos0 = float(self.cfg.hb_cos0)
        kappa = float(self.cfg.hb_kappa)
        min_sep = int(self.cfg.hb_min_sep)

        for i in range(N):
            for j in range(i + min_sep, N):
                rij = dist[i, j]
                # radial weight
                w_r = np.exp(-((rij - r0) ** 2) / (2.0 * sig_r ** 2))
                if w_r < 1e-8:
                    continue
                # directional weight (favor aligned/antialigned tangents)
                cos_t = float(np.dot(tang[i], tang[j]))
                # von Mises-like bump around cos0 and -cos0
                w_th = np.exp(kappa * (cos_t - cos0)) + np.exp(kappa * (-cos_t - cos0))
                w = w_r * w_th
                pairs_sum += -w  # H-bond lowers energy
                pairs_cnt += 1

        if pairs_cnt == 0:
            return 0.0
        return pairs_sum / pairs_cnt

    # ---------------- Ramachandran-like (CA-based) ----------------
    def _rama_energy(self, P: np.ndarray) -> float:
        """
        CA-based pseudo Ramachandran:
          For residue i, define pseudo-phi as dihedral(CA[i-2],CA[i-1],CA[i],CA[i+1]),
          and pseudo-psi as dihedral(CA[i-1],CA[i],CA[i+1],CA[i+2]).
        Energy per-residue mean:
          - If mode="table" and table provided: E = -log P(phi,psi) via kernel interpolation.
          - Else generic: mixture of two Gaussian wells centered at (alpha, beta) in degrees.
        """
        N = len(P)
        if not self.cfg.rama_enable or N < 5:
            return 0.0

        # compute pseudo-phi/psi arrays (in degrees)
        phi = []
        psi = []
        # pseudo-phi at i uses (i-2,i-1,i,i+1), valid for i=2..N-2
        # pseudo-psi at i uses (i-1,i,i+1,i+2), valid for i=2..N-3
        dih = self._dihedral_angles  # reuse dihedral on subsequences
        # precompute once on whole chain for speed? We'll compute locally for clarity.
        for i in range(2, N - 2):
            # pseudo-phi
            phi_val = float(self._dihedral_angles(P[i - 2:i + 2])[-1])  # last angle from 4 points
            # pseudo-psi
            psi_val = float(self._dihedral_angles(P[i - 1:i + 3])[-1])
            phi.append(np.degrees(phi_val))
            psi.append(np.degrees(psi_val))

        phi = np.array(phi, dtype=float)
        psi = np.array(psi, dtype=float)
        if phi.size == 0:
            return 0.0

        if self.cfg.rama_mode == "table" and self._rama_table is not None:
            tab = self._rama_table  # (M,3): phi, psi, logP
            # kernel interpolation in (phi,psi) with periodic wrap handled approximately by min wrap
            # For robustness, use an RBF with width sigma
            sig = max(1.0, float(self.cfg.rama_sigma_deg))
            E = 0.0
            for ph, ps in zip(phi, psi):
                dphi = np.minimum(np.abs(ph - tab[:, 0]), 360.0 - np.abs(ph - tab[:, 0]))
                dpsi = np.minimum(np.abs(ps - tab[:, 1]), 360.0 - np.abs(ps - tab[:, 1]))
                w = np.exp(-0.5 * ((dphi / sig) ** 2 + (dpsi / sig) ** 2))
                if np.sum(w) < 1e-8:
                    # fallback to mean logP
                    logP = float(np.mean(tab[:, 2]))
                else:
                    # weighted average of logP
                    logP = float(np.sum(w * tab[:, 2]) / np.sum(w))
                E += -logP
            return E / phi.size

        # generic two-well prior (alpha & beta basins), smooth and robust
        sig = max(1.0, float(self.cfg.rama_sigma_deg))
        a_phi, a_psi = self.cfg.rama_alpha_center
        b_phi, b_psi = self.cfg.rama_beta_center
        mix_a = float(self.cfg.rama_mix_alpha)
        mix_b = float(self.cfg.rama_mix_beta)
        eps = 1e-8

        def wrap_delta(a, b):
            d = np.abs(a - b)
            return np.minimum(d, 360.0 - d)

        E = 0.0
        for ph, ps in zip(phi, psi):
            dpa = (wrap_delta(ph, a_phi) / sig) ** 2 + (wrap_delta(ps, a_psi) / sig) ** 2
            dpb = (wrap_delta(ph, b_phi) / sig) ** 2 + (wrap_delta(ps, b_psi) / sig) ** 2
            Pa = np.exp(-0.5 * dpa)
            Pb = np.exp(-0.5 * dpb)
            P = mix_a * Pa + mix_b * Pb + eps
            E += -np.log(P)
        return E / phi.size

    # ---------------- soft-contact kernel ----------------
    def _soft_contact_weight(self, r: np.ndarray) -> np.ndarray:
        if self.cfg.use_sigmoid_contact:
            return 1.0 / (1.0 + np.exp(-self.cfg.sigmoid_alpha * (self.cfg.r_contact - r)))
        r0 = max(1e-8, self.cfg.contact_r0)
        p = float(self.cfg.contact_p)
        return np.exp(- (r / r0) ** p)

    # ---------------- main entry ----------------
    def compute_energy(self, main_positions: np.ndarray, sequence: str) -> Dict[str, float]:
        N = len(main_positions)
        if N < 3:
            return {
                "E_total": 0.0, "E_steric": 0.0, "E_geom": 0.0, "E_bond": 0.0, "E_mj": 0.0,
                "E_dihedral": 0.0, "E_hb": 0.0, "E_hydroph": 0.0, "E_cbeta": 0.0, "E_rama": 0.0
            }

        # 0) backbone vectors
        P = np.asarray(main_positions, dtype=float)
        v = np.diff(P, axis=0)
        bond_lengths = np.linalg.norm(v, axis=1)

        # 1) bond (per-bond mean)
        E_bond_raw = np.sum((bond_lengths - self.cfg.d0) ** 2)
        E_bond = self._safe_mean(E_bond_raw, N - 1)

        # 2) bending (per-angle mean)
        if N >= 3:
            v1, v2 = v[:-1], v[1:]
            denom = np.clip(np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1), 1e-12, None)
            cos_t = np.clip(np.sum(v1 * v2, axis=1) / denom, -1.0, 1.0)
            E_geom = self._safe_mean(1.0 - cos_t, N - 2)
        else:
            E_geom = 0.0

        # 3) dihedral (per-dihedral mean)
        diheds = self._dihedral_angles(P)
        if diheds.size > 0:
            E_dihedral = self._safe_mean(1.0 - np.cos(diheds), N - 3)
        else:
            E_dihedral = 0.0

        # 4) steric soft-wall + overlap (per-nonbonded pair mean)
        dist = self._pairwise_dist(P)
        nb_mask = np.triu(np.ones((N, N), dtype=bool), k=2)
        D = dist[nb_mask]
        num_pairs = int(np.count_nonzero(nb_mask)) or 1
        rmin = float(self.cfg.r_min)
        wall = (rmin - D[D < rmin]) ** 2
        E_steric_wall = float(np.sum(wall))
        overlap_eps = 1e-8
        overlap_count = int(np.count_nonzero(D < overlap_eps))
        E_overlap = overlap_count * float(self.cfg.lambda_overlap)
        E_steric = (E_steric_wall + E_overlap) / num_pairs

        # 5) MJ soft-contact (per-nonbonded pair mean)
        E_mj_raw = 0.0
        if len(self._aa_index) > 0:
            if self.cfg.soft_contact:
                w_r = self._soft_contact_weight(D)
                ii, jj = np.triu_indices(N, k=2)
                for k, (i, j) in enumerate(zip(ii, jj)):
                    ai = self._aa_index.get(sequence[i])
                    aj = self._aa_index.get(sequence[j])
                    if ai is None or aj is None:
                        continue
                    wij = float(w_r[k])
                    if wij <= 0.0:
                        continue
                    E_mj_raw += wij * float(self._mj_matrix[ai, aj])
            else:
                for i in range(N - 1):
                    ai = self._aa_index.get(sequence[i])
                    if ai is None:
                        continue
                    for j in range(i + 2, N):
                        aj = self._aa_index.get(sequence[j])
                        if aj is None:
                            continue
                        if dist[i, j] <= self.cfg.r_contact:
                            E_mj_raw += float(self._mj_matrix[ai, aj])
        E_mj = E_mj_raw / num_pairs

        # 6) NEW: hydrophobic burial (per-residue mean)
        E_hydroph = self._hydrophobic_burial(P, sequence)

        # 7) NEW: virtual Cβ packing (per-pair mean)
        E_cbeta = self._cbeta_packing(P)

        # 8) NEW: main-chain hydrogen bond proxy (per-pair mean)
        E_hb = self._hbond_energy(P)

        # 9) NEW: Ramachandran-like prior (per-residue mean)
        E_rama = self._rama_energy(P)

        # 10) combine with weights
        w = self.cfg.weights
        E_total = (
            w.get("steric", 1.0)   * E_steric +
            w.get("geom", 0.5)     * E_geom +
            w.get("bond", 0.2)     * E_bond +
            w.get("mj", 2.0)       * E_mj +
            w.get("dihedral", 0.3) * E_dihedral +
            w.get("hydroph", 0.5)  * E_hydroph +
            w.get("cbeta", 0.4)    * E_cbeta +
            w.get("hb", 0.6)       * E_hb +
            w.get("rama", 0.5)     * E_rama
        )

        # 11) optional legacy normalization
        if self.cfg.normalize_mode.lower() == "legacy":
            if self.cfg.normalize:
                E_total = E_total / max(1, N)

        out = {
            "E_total": float(E_total),
            "E_steric": float(E_steric),
            "E_geom": float(E_geom),
            "E_bond": float(E_bond),
            "E_mj": float(E_mj),
            # diagnostics
            "E_dihedral": float(E_dihedral),
            "E_hb": float(E_hb),
            "E_hydroph": float(E_hydroph),
            "E_cbeta": float(E_cbeta),
            "E_rama": float(E_rama),
        }
        return out

    # ---------------- batch API ----------------
    def evaluate_jsonl(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        out_path = output_path or self.cfg.output_path
        written = 0
        with open(input_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for idx, line in enumerate(fin):
                if limit and idx >= limit:
                    break
                rec = json.loads(line)
                if "main_positions" not in rec or "sequence" not in rec:
                    _LOG.warning("Skipping line %d: missing data", idx)
                    continue
                P = np.array(rec["main_positions"], dtype=float)
                seq = rec["sequence"]
                energies = self.compute_energy(P, seq)
                rec.update(energies)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
        _LOG.info("Wrote %d records with energy to %s", written, out_path)
        return {"written": written, "path": out_path}
