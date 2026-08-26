# --*-- conding:utf-8 --*--
# @time:3/28/26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:fullatom_scorer.py

"""
Full-atom chemical quality scorer for reconstructed structures.

Evaluates six dimensions of structural quality that are **only possible**
after Calpha-to-all-atom reconstruction:

1. Backbone geometry  -- bond-length / bond-angle / omega deviations
2. Ramachandran       -- phi/psi dihedral quality
3. Steric clashes     -- all heavy-atom van der Waals overlap
4. Hydrogen bonds     -- real N...O distance + angle criteria
5. Side-chain packing -- inter-residue contact density
6. Compactness        -- full-atom radius of gyration

The scorer implements the ``StructureScorer`` protocol from
``post_process.rerank`` so it plugs directly into the existing pipeline.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .backbone import _normalize
from .constants import (
    N_CA_BOND,
    CA_C_BOND,
    C_N_BOND,
    C_O_BOND,
    CA_CB_BOND,
    N_CA_C_ANGLE,
    CA_C_N_ANGLE,
    C_N_CA_ANGLE,
    CA_C_O_ANGLE,
)

_LOG = logging.getLogger(__name__)


# ── Van der Waals radii (A) for heavy-atom clash detection ─────────────────
_VDW_RADII: Dict[str, float] = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
}

# Ideal bond lengths for scoring reference (atom_pair -> length in A)
_IDEAL_BONDS: Dict[Tuple[str, str], float] = {
    ("N", "CA"):  N_CA_BOND,
    ("CA", "C"):  CA_C_BOND,
    ("C", "N"):   C_N_BOND,
    ("C", "O"):   C_O_BOND,
    ("CA", "CB"): CA_CB_BOND,
}

# Ideal bond angles for scoring reference (atom_triple -> angle in degrees)
_IDEAL_ANGLES: Dict[Tuple[str, str, str], float] = {
    ("N", "CA", "C"):   N_CA_C_ANGLE,
    ("CA", "C", "N"):   CA_C_N_ANGLE,
    ("C", "N", "CA"):   C_N_CA_ANGLE,
    ("CA", "C", "O"):   CA_C_O_ANGLE,
}


# ── Ramachandran allowed regions (simplified) ──────────────────────────────
# Each region is (phi_center, psi_center, phi_halfwidth, psi_halfwidth)
# in degrees.  A residue is "allowed" if it falls in any region.
_RAMA_REGIONS: List[Tuple[float, float, float, float]] = [
    # beta-sheet
    (-120.0, 130.0, 60.0, 60.0),
    # alpha-helix
    (-60.0, -45.0, 40.0, 40.0),
    # left-handed helix (rare but allowed)
    (60.0, 45.0, 40.0, 40.0),
    # polyproline II
    (-75.0, 150.0, 35.0, 35.0),
]


# ── Geometry helpers ───────────────────────────────────────────────────────

def _bond_length(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(b - a))


def _bond_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle A-B-C in degrees."""
    ba = _normalize(a - b)
    bc = _normalize(c - b)
    cos_val = float(np.clip(np.dot(ba, bc), -1.0, 1.0))
    return math.degrees(math.acos(cos_val))


def _dihedral_deg(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray,
) -> float:
    """Dihedral angle A-B-C-D in degrees, range (-180, 180]."""
    b1 = b - a
    b2 = c - b
    b3 = d - c
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-10 or n2_norm < 1e-10:
        return 0.0
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    m = np.cross(n1, _normalize(b2))
    x = float(np.dot(n1, n2))
    y = float(np.dot(m, n2))
    return math.degrees(math.atan2(-y, x))


def _angle_distance(angle1: float, angle2: float) -> float:
    """Shortest angular distance in degrees (0-180)."""
    d = abs(angle1 - angle2) % 360.0
    return min(d, 360.0 - d)


def _guess_element(name: str) -> str:
    name = name.strip()
    if not name:
        return "C"
    for prefix in ("N", "O", "S", "C"):
        if name.startswith(prefix):
            return prefix
    return "C"


# ── Scorer configuration ──────────────────────────────────────────────────

@dataclass(frozen=True)
class FullAtomScoringConfig:
    """
    Weights and thresholds for the full-atom chemical quality scorer.

    Weight calibration notes (from grid search over 100 lattice-derived proteins,
    2880 weight combinations, optimizing top-1 match count):

    - Backbone geometry (bond, angle, omega, Ramachandran) shows LOW variance
      across candidates because all share the same NERF reconstruction.
      These terms are DISABLED.
    - Hydrogen bonds are the strongest discriminator (Spearman rho = -0.10
      with RMSD, 72% consistency across proteins).  Weight UP from 3 -> 5.
    - Compactness / radius of gyration (rho = +0.13): compact structures have
      lower RMSD.  Weight UP from 3 -> 5.
    - Steric clashes are weakly useful (rho = -0.08) but act as regularization
      against over-rewarding compactness.  Weight DOWN from 1.5 -> 0.5.
    - Side-chain packing has weak signal; weight DOWN from 2 -> 1.

    Grid search result: 12/100 top-1 match (up from 8/100 with previous weights).
    """
    # ── Backbone geometry (low discriminating power for lattice models) ──
    w_bond_length: float = 0.0
    w_bond_angle: float = 0.0
    w_omega: float = 0.0            # anti-correlated: disabled
    omega_tolerance_deg: float = 20.0

    # ── Ramachandran (anti-correlated for lattice models) ──
    w_ramachandran: float = 0.0      # disabled: anti-correlated with RMSD
    rama_outlier_penalty: float = 2.0

    # ── Steric clashes (regularization against over-rewarding compactness) ──
    w_clash: float = 0.5
    clash_overlap_threshold: float = 0.4
    bonded_exclusion_seq_dist: int = 3

    # ── Hydrogen bonds (best discriminator) ──
    w_hbond: float = 5.0
    hbond_dist_max: float = 3.5
    hbond_angle_min: float = 100.0

    # ── Side-chain packing (weak signal) ──
    w_packing: float = 1.0
    packing_contact_dist: float = 4.5
    packing_seq_sep: int = 3

    # ── Compactness (strong discriminator) ──
    w_compactness: float = 5.0
    target_rg_per_residue: float = 2.0


# ── Main scorer ───────────────────────────────────────────────────────────

class FullAtomScorer:
    """
    Scores a reconstructed full-atom structure on chemical quality.

    Implements the ``StructureScorer`` protocol (``score()`` returns
    ``ScoreBreakdown``).

    Input: a ``BuildResult`` from ca2all, or any object carrying:
      - ``backbone.n_positions``, ``backbone.ca_positions``,
        ``backbone.c_positions``, ``backbone.o_positions``
      - ``full_atoms`` list with ``.name``, ``.residue_index``, ``.position``
      - ``sequence``
    """

    def __init__(self, cfg: Optional[FullAtomScoringConfig] = None) -> None:
        self.cfg = cfg or FullAtomScoringConfig()

    def score(self, build_result) -> "ScoreBreakdown":
        """
        Score a BuildResult and return a ScoreBreakdown.

        Lower total = better structure.
        """
        from ..rerank.models import ScoreBreakdown

        terms: Dict[str, float] = {}

        # Extract arrays from BuildResult
        bb = build_result.backbone
        n_pos = bb.n_positions
        ca_pos = bb.ca_positions
        c_pos = bb.c_positions
        o_pos = bb.o_positions
        seq = build_result.sequence
        n_res = len(seq)
        atoms = build_result.full_atoms

        # ── 1. Backbone geometry ───────────────────────────────────
        bond_penalty, angle_penalty, omega_penalty = self._score_backbone_geometry(
            n_pos, ca_pos, c_pos, o_pos, n_res,
        )
        terms["bond_length_penalty"] = bond_penalty
        terms["bond_angle_penalty"] = angle_penalty
        terms["omega_penalty"] = omega_penalty

        # ── 2. Ramachandran ────────────────────────────────────────
        rama_penalty, rama_outlier_frac = self._score_ramachandran(
            n_pos, ca_pos, c_pos, n_res,
        )
        terms["ramachandran_penalty"] = rama_penalty
        terms["ramachandran_outlier_fraction"] = rama_outlier_frac

        # ── 3. Steric clashes ─────────────────────────────────────
        clash_count, clash_penalty = self._score_clashes(atoms)
        terms["clash_count"] = float(clash_count)
        terms["clash_penalty"] = clash_penalty

        # ── 4. Hydrogen bonds ──────────────────────────────────────
        hbond_count, hbond_score = self._score_hbonds(
            n_pos, ca_pos, c_pos, o_pos, n_res,
        )
        terms["hbond_count"] = float(hbond_count)
        terms["hbond_reward"] = hbond_score

        # ── 5. Side-chain packing ─────────────────────────────────
        packing_density, packing_score = self._score_packing(atoms, n_res)
        terms["packing_density"] = packing_density
        terms["packing_reward"] = packing_score

        # ── 6. Compactness ─────────────────────────────────────────
        rg, compactness_penalty = self._score_compactness(atoms, n_res)
        terms["radius_of_gyration"] = rg
        terms["compactness_penalty"] = compactness_penalty

        # ── Combine ────────────────────────────────────────────────
        total = (
            self.cfg.w_bond_length * bond_penalty
            + self.cfg.w_bond_angle * angle_penalty
            + self.cfg.w_omega * omega_penalty
            + self.cfg.w_ramachandran * rama_penalty
            + self.cfg.w_clash * clash_penalty
            - self.cfg.w_hbond * hbond_score       # reward (subtract)
            - self.cfg.w_packing * packing_score    # reward (subtract)
            + self.cfg.w_compactness * compactness_penalty
        )

        terms["total_penalties"] = (
            bond_penalty + angle_penalty + omega_penalty
            + rama_penalty + clash_penalty + compactness_penalty
        )
        terms["total_rewards"] = hbond_score + packing_score

        return ScoreBreakdown(
            total=float(total),
            terms=terms,
            engine="fullatom_chemical",
            lower_is_better=True,
        )

    # ── Dimension 1: Backbone geometry ─────────────────────────────────

    def _score_backbone_geometry(
        self,
        n_pos: np.ndarray,
        ca_pos: np.ndarray,
        c_pos: np.ndarray,
        o_pos: np.ndarray,
        n_res: int,
    ) -> Tuple[float, float, float]:
        """Returns (bond_penalty, angle_penalty, omega_penalty)."""
        bond_devs: List[float] = []
        angle_devs: List[float] = []
        omega_devs: List[float] = []

        for i in range(n_res):
            # Bond lengths within residue
            bl_n_ca = _bond_length(n_pos[i], ca_pos[i])
            bl_ca_c = _bond_length(ca_pos[i], c_pos[i])
            bl_c_o = _bond_length(c_pos[i], o_pos[i])
            bond_devs.append(abs(bl_n_ca - N_CA_BOND))
            bond_devs.append(abs(bl_ca_c - CA_C_BOND))
            bond_devs.append(abs(bl_c_o - C_O_BOND))

            # Peptide bond to next residue
            if i < n_res - 1:
                bl_c_n = _bond_length(c_pos[i], n_pos[i + 1])
                bond_devs.append(abs(bl_c_n - C_N_BOND))

            # Bond angles
            ang_n_ca_c = _bond_angle_deg(n_pos[i], ca_pos[i], c_pos[i])
            angle_devs.append(abs(ang_n_ca_c - N_CA_C_ANGLE))

            if i < n_res - 1:
                ang_ca_c_n = _bond_angle_deg(ca_pos[i], c_pos[i], n_pos[i + 1])
                angle_devs.append(abs(ang_ca_c_n - CA_C_N_ANGLE))

            if i > 0:
                ang_c_n_ca = _bond_angle_deg(c_pos[i - 1], n_pos[i], ca_pos[i])
                angle_devs.append(abs(ang_c_n_ca - C_N_CA_ANGLE))

            # Omega dihedral (CA[i-1]-C[i-1]-N[i]-CA[i])
            if i > 0:
                omega = _dihedral_deg(ca_pos[i - 1], c_pos[i - 1], n_pos[i], ca_pos[i])
                dev = _angle_distance(abs(omega), 180.0)
                if dev > self.cfg.omega_tolerance_deg:
                    omega_devs.append(dev - self.cfg.omega_tolerance_deg)

        bond_penalty = float(np.mean(bond_devs)) if bond_devs else 0.0
        angle_penalty = float(np.mean(angle_devs)) / 10.0 if angle_devs else 0.0
        omega_penalty = float(np.sum(omega_devs)) / max(n_res - 1, 1) if omega_devs else 0.0

        return bond_penalty, angle_penalty, omega_penalty

    # ── Dimension 2: Ramachandran ──────────────────────────────────────

    def _score_ramachandran(
        self,
        n_pos: np.ndarray,
        ca_pos: np.ndarray,
        c_pos: np.ndarray,
        n_res: int,
    ) -> Tuple[float, float]:
        """Returns (rama_penalty, outlier_fraction)."""
        outliers = 0
        scorable = 0

        for i in range(1, n_res - 1):
            # phi = C[i-1]-N[i]-CA[i]-C[i]
            phi = _dihedral_deg(c_pos[i - 1], n_pos[i], ca_pos[i], c_pos[i])
            # psi = N[i]-CA[i]-C[i]-N[i+1]
            psi = _dihedral_deg(n_pos[i], ca_pos[i], c_pos[i], n_pos[i + 1])

            scorable += 1
            if not self._in_allowed_region(phi, psi):
                outliers += 1

        if scorable == 0:
            return 0.0, 0.0

        outlier_frac = outliers / scorable
        penalty = outlier_frac * self.cfg.rama_outlier_penalty * scorable

        return penalty, outlier_frac

    def _in_allowed_region(self, phi: float, psi: float) -> bool:
        for phi_c, psi_c, phi_hw, psi_hw in _RAMA_REGIONS:
            if (_angle_distance(phi, phi_c) <= phi_hw and
                    _angle_distance(psi, psi_c) <= psi_hw):
                return True
        return False

    # ── Dimension 3: Steric clashes ────────────────────────────────────

    def _score_clashes(self, atoms: list) -> Tuple[int, float]:
        """Returns (clash_count, clash_penalty)."""
        n = len(atoms)
        if n < 2:
            return 0, 0.0

        positions = np.array([a.position for a in atoms])
        res_indices = np.array([a.residue_index for a in atoms])
        elements = [_guess_element(a.name) for a in atoms]
        radii = np.array([_VDW_RADII.get(e, 1.7) for e in elements])

        clash_count = 0
        total_overlap = 0.0
        threshold = self.cfg.clash_overlap_threshold
        seq_excl = self.cfg.bonded_exclusion_seq_dist

        # Pairwise distance matrix (vectorized for efficiency)
        for i in range(n):
            for j in range(i + 1, n):
                # Skip bonded or near-bonded atoms
                if abs(int(res_indices[i]) - int(res_indices[j])) < seq_excl:
                    continue
                dist = float(np.linalg.norm(positions[j] - positions[i]))
                min_dist = radii[i] + radii[j] - threshold
                if dist < min_dist:
                    overlap = min_dist - dist
                    clash_count += 1
                    total_overlap += overlap

        # Normalize by number of residues (count-based, not overlap-based)
        n_res = max(len(set(int(r) for r in res_indices)), 1)
        penalty = clash_count / max(n_res, 1)
        return clash_count, penalty

    # ── Dimension 4: Hydrogen bonds ────────────────────────────────────

    def _score_hbonds(
        self,
        n_pos: np.ndarray,
        ca_pos: np.ndarray,
        c_pos: np.ndarray,
        o_pos: np.ndarray,
        n_res: int,
    ) -> Tuple[int, float]:
        """
        Detect backbone hydrogen bonds: N[i]...O[j] with proper geometry.

        Returns (hbond_count, reward_score).
        """
        hbond_count = 0
        dist_max = self.cfg.hbond_dist_max
        angle_min = self.cfg.hbond_angle_min

        for i in range(n_res):
            for j in range(n_res):
                if abs(i - j) < 2:
                    continue

                # Donor: N[i], Acceptor: O[j]
                d_no = float(np.linalg.norm(n_pos[i] - o_pos[j]))
                if d_no > dist_max:
                    continue

                # Check angle: CA[i]-N[i]...O[j] should be > angle_min
                angle = _bond_angle_deg(ca_pos[i], n_pos[i], o_pos[j])
                if angle < angle_min:
                    continue

                hbond_count += 1

        # Reward: normalize by chain length (uncapped for better discrimination)
        reward = hbond_count / max(n_res, 1)

        return hbond_count, reward

    # ── Dimension 5: Side-chain packing ────────────────────────────────

    def _score_packing(
        self, atoms: list, n_res: int,
    ) -> Tuple[float, float]:
        """
        Count inter-residue side-chain contacts.

        Returns (contact_density, reward_score).
        """
        # Collect side-chain atoms (not backbone)
        backbone_names = {"N", "CA", "C", "O"}
        sc_atoms = [a for a in atoms if a.name not in backbone_names]
        if len(sc_atoms) < 2:
            return 0.0, 0.0

        contacts = 0
        total_pairs = 0
        cutoff = self.cfg.packing_contact_dist
        seq_sep = self.cfg.packing_seq_sep

        for i in range(len(sc_atoms)):
            for j in range(i + 1, len(sc_atoms)):
                if abs(sc_atoms[i].residue_index - sc_atoms[j].residue_index) < seq_sep:
                    continue
                total_pairs += 1
                dist = float(np.linalg.norm(sc_atoms[j].position - sc_atoms[i].position))
                if dist < cutoff:
                    contacts += 1

        density = contacts / max(len(sc_atoms), 1)
        # Reward: contact density per side-chain atom
        reward = density

        return density, reward

    # ── Dimension 6: Compactness ───────────────────────────────────────

    def _score_compactness(
        self, atoms: list, n_res: int,
    ) -> Tuple[float, float]:
        """
        Compute radius of gyration and penalize deviation from target.

        Returns (rg, compactness_penalty).
        """
        if not atoms:
            return 0.0, 0.0

        positions = np.array([a.position for a in atoms])
        centroid = positions.mean(axis=0)
        rg = float(np.sqrt(np.mean(np.sum((positions - centroid) ** 2, axis=1))))

        # Target Rg scales as ~c * sqrt(N)
        target_rg = self.cfg.target_rg_per_residue * math.sqrt(n_res)
        deviation = abs(rg - target_rg) / max(target_rg, 1e-6)

        # Linear penalty for deviation (better discrimination than quadratic)
        penalty = deviation

        return rg, penalty


__all__ = [
    "FullAtomScorer",
    "FullAtomScoringConfig",
]
