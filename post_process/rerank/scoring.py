# --*-- conding:utf-8 --*--
# @time:3/26/26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:scoring.py

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

from .geometry import scale_backbone_positions
from .models import ScoreBreakdown, StructureCandidate


def _find_mj_matrix_path() -> Optional[str]:
    """Locate the Miyazawa-Jernigan matrix file bundled with this package."""
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "resources", "mj_matrix.txt"),
        os.path.join(base, "..", "..", "Protein_Folding", "interactions", "resources", "mj_matrix.txt"),
    ]
    for path in candidates:
        normalized = os.path.normpath(path)
        if os.path.isfile(normalized):
            return normalized
    return None


def _load_mj_matrix(path: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, int]]:
    """Load the MJ interaction matrix. Returns (matrix, aa_index)."""
    if path is None:
        path = _find_mj_matrix_path()
    if path is None or not os.path.isfile(path):
        return np.zeros((20, 20)), {}
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    headers = lines[0].split()
    n = len(headers)
    mat = np.zeros((n, n))
    for i, line in enumerate(lines[1:]):
        vals = [float(x) for x in line.split()]
        for j in range(len(vals)):
            mat[i, j] = vals[j]
            mat[j, i] = vals[j]
    aa_index = {aa: idx for idx, aa in enumerate(headers)}
    return mat, aa_index


class StructureProposalGenerator(Protocol):
    def generate(self, candidate: StructureCandidate) -> List[StructureCandidate]: ...


class StructureScorer(Protocol):
    def score(self, candidate: StructureCandidate) -> ScoreBreakdown: ...


@dataclass
class IdentityStructureProposalGenerator:
    """Placeholder before a real CA-to-all / rotamer engine is connected."""

    def generate(self, candidate: StructureCandidate) -> List[StructureCandidate]:
        return [candidate]


@dataclass(frozen=True)
class HeuristicScoringConfig:
    """
    Coarse geometry score used only for prefiltering and cluster-source selection.

    This score is intentionally lightweight and should not be treated as the
    final structure-ranking criterion.
    """

    scale_backbone: bool = True
    clash_distance: float = 3.2
    contact_distance: float = 8.0
    min_sequence_separation: int = 2
    weight_overlap: float = 100.0
    weight_clash: float = 20.0
    weight_rg: float = 0.5
    weight_contact: float = 2.0
    weight_support: float = 1.0


@dataclass
class HeuristicStructureScorer:
    cfg: HeuristicScoringConfig = HeuristicScoringConfig()

    def score(self, candidate: StructureCandidate) -> ScoreBreakdown:
        positions = candidate.main_positions
        if self.cfg.scale_backbone:
            positions = scale_backbone_positions(positions)

        if len(positions) <= 2:
            return ScoreBreakdown(
                total=math.inf,
                terms={
                    "overlap_count": 0.0,
                    "clash_penalty": 0.0,
                    "rg": 0.0,
                    "contact_fraction": 0.0,
                    "support_bonus": 0.0,
                },
                engine="coarse_geometry_prefilter",
                lower_is_better=True,
            )

        distances = self._pairwise_distances(positions)
        mask = self._nonbonded_mask(len(positions), self.cfg.min_sequence_separation)
        nonbonded = distances[mask]

        overlap_count = float(np.count_nonzero(nonbonded < 1e-8))
        close = nonbonded[nonbonded < self.cfg.clash_distance]
        clash_penalty = float(np.mean((self.cfg.clash_distance - close) ** 2)) if close.size else 0.0
        contact_fraction = (
            float(np.count_nonzero(nonbonded <= self.cfg.contact_distance)) / float(nonbonded.size)
            if nonbonded.size
            else 0.0
        )
        rg = self._radius_of_gyration(positions)
        support_bonus = candidate.support_value()

        total = (
            self.cfg.weight_overlap * overlap_count
            + self.cfg.weight_clash * clash_penalty
            + self.cfg.weight_rg * rg
            - self.cfg.weight_contact * contact_fraction
            - self.cfg.weight_support * support_bonus
        )

        return ScoreBreakdown(
            total=float(total),
            terms={
                "overlap_count": overlap_count,
                "clash_penalty": clash_penalty,
                "rg": float(rg),
                "contact_fraction": contact_fraction,
                "support_bonus": support_bonus,
            },
            engine="coarse_geometry_prefilter",
            lower_is_better=True,
        )

    @staticmethod
    def _pairwise_distances(positions: np.ndarray) -> np.ndarray:
        diff = positions[:, None, :] - positions[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    @staticmethod
    def _nonbonded_mask(length: int, min_sequence_separation: int) -> np.ndarray:
        rows, cols = np.indices((length, length))
        return (cols > rows) & ((cols - rows) >= max(2, int(min_sequence_separation)))

    @staticmethod
    def _radius_of_gyration(positions: np.ndarray) -> float:
        center = positions.mean(axis=0)
        return float(np.sqrt(np.mean(np.sum((positions - center) ** 2, axis=1))))


@dataclass(frozen=True)
class IndependentEnergyConfig:
    """
    Coarse-grained backbone energy used by rerank.

    This deliberately avoids reusing any external Hamiltonian-style implementation.
    The scorer is backbone-only and treats the
    candidate as a coarse chain with local geometry, excluded volume, solvation-
    like hydrophobic packing, screened charge interactions, and a compactness
    envelope.
    """

    scale_backbone: bool = True
    bond_target: float = 3.8
    bond_sigma: float = 0.25
    angle_target_deg: float = 110.0
    angle_sigma_deg: float = 20.0
    torsion_weight_cos1: float = 0.35
    torsion_weight_cos3: float = 0.65
    repulsion_distance: float = 3.4
    min_distance_floor: float = 1.5
    hydrophobic_contact_center: float = 7.0
    hydrophobic_contact_sigma: float = 1.4
    packing_shell_center: float = 6.8
    packing_shell_sigma: float = 1.3
    charge_screen_length: float = 8.0
    compactness_prefactor: float = 2.2
    compactness_exponent: float = 0.38
    min_sequence_separation: int = 3
    length_transition_center: float = 8.0
    length_transition_span: float = 4.0
    short_weight_angle: float = 1.5
    short_weight_chirality: float = 2.0
    short_weight_trans: float = 0.5
    long_weight_angle: float = 1.0
    long_weight_chirality: float = 2.0
    long_weight_trans: float = 0.0
    long_weight_propensity: float = 6.0
    long_weight_hydrophobic_contacts: float = 6.0
    long_hydrophobic_cutoff: float = 6.8
    long_hydrophobic_min_gap: int = 4
    # Defaults are calibrated to keep rerank energy independent from the
    # sampling distribution while remaining stable on current backbone-only
    # smoke tests. Additional coarse terms stay available for later tuning.
    weight_bond: float = 0.0
    weight_angle: float = 0.0
    weight_torsion: float = 0.0
    weight_steric: float = 0.0
    weight_hydrophobic: float = 0.0
    weight_packing: float = 0.0
    weight_charge: float = 0.0
    weight_compactness: float = 0.0
    hydropathy: Dict[str, float] = field(
        default_factory=lambda: {
            "A": 1.8,
            "V": 4.2,
            "I": 4.5,
            "L": 3.8,
            "M": 1.9,
            "F": 2.8,
            "W": -0.9,
            "Y": -1.3,
            "C": 2.5,
            "P": -1.6,
            "G": -0.4,
            "S": -0.8,
            "T": -0.7,
            "N": -3.5,
            "Q": -3.5,
            "D": -3.5,
            "E": -3.5,
            "K": -3.9,
            "R": -4.5,
            "H": -3.2,
        },
        repr=False,
    )
    charge: Dict[str, float] = field(
        default_factory=lambda: {
            "D": -1.0,
            "E": -1.0,
            "K": 1.0,
            "R": 1.0,
            "H": 0.3,
        },
        repr=False,
    )
    beta_propensity: Dict[str, float] = field(
        default_factory=lambda: {
            "V": 1.7,
            "I": 1.6,
            "Y": 1.5,
            "F": 1.4,
            "W": 1.4,
            "T": 1.2,
            "C": 1.2,
            "L": 1.2,
            "M": 1.1,
            "Q": 1.1,
        },
        repr=False,
    )
    turn_propensity: Dict[str, float] = field(
        default_factory=lambda: {
            "G": 1.7,
            "P": 1.6,
            "N": 1.4,
            "D": 1.4,
            "S": 1.3,
            "C": 1.2,
            "K": 1.1,
            "R": 1.1,
            "H": 1.1,
        },
        repr=False,
    )
    left_handed_penalty: float = 2.0
    left_handed_tolerant: Dict[str, float] = field(
        default_factory=lambda: {
            "G": 0.5,
        },
        repr=False,
    )


@dataclass
class IndependentBackboneEnergyScorer:
    cfg: IndependentEnergyConfig = IndependentEnergyConfig()

    def score(self, candidate: StructureCandidate) -> ScoreBreakdown:
        positions = self._scaled_positions(candidate.main_positions)
        sequence = candidate.sequence or ""
        torsion_angles = self._torsion_angles(positions)

        bond = self._bond_energy(positions)
        angle = self._angle_energy(positions)
        torsion = self._torsion_energy(torsion_angles)
        steric = self._steric_energy(positions)
        hydrophobic = self._hydrophobic_energy(positions, sequence)
        packing = self._packing_energy(positions)
        charge = self._charge_energy(positions, sequence)
        compactness = self._compactness_energy(positions)
        chirality = self._chirality_energy(torsion_angles)
        trans = self._trans_energy(torsion_angles)
        sequence_propensity = self._sequence_propensity_energy(sequence, torsion_angles)
        long_hydrophobic = self._long_range_hydrophobic_contact_energy(positions, sequence)
        length_gate = self._length_gate(len(sequence) if sequence else len(positions))

        adaptive_local = (
            (1.0 - length_gate)
            * (
                self.cfg.short_weight_angle * angle
                + self.cfg.short_weight_chirality * chirality
                + self.cfg.short_weight_trans * trans
            )
            + length_gate
            * (
                self.cfg.long_weight_angle * angle
                + self.cfg.long_weight_chirality * chirality
                + self.cfg.long_weight_trans * trans
                + self.cfg.long_weight_propensity * sequence_propensity
                + self.cfg.long_weight_hydrophobic_contacts * long_hydrophobic
            )
        )

        total = (
            adaptive_local
            + self.cfg.weight_bond * bond
            + self.cfg.weight_angle * angle
            + self.cfg.weight_torsion * torsion
            + self.cfg.weight_steric * steric
            + self.cfg.weight_hydrophobic * hydrophobic
            + self.cfg.weight_packing * packing
            + self.cfg.weight_charge * charge
            + self.cfg.weight_compactness * compactness
        )

        return ScoreBreakdown(
            total=float(total),
            terms={
                "adaptive_local": float(adaptive_local),
                "length_gate": float(length_gate),
                "bond": float(bond),
                "angle": float(angle),
                "torsion": float(torsion),
                "chirality": float(chirality),
                "trans": float(trans),
                "sequence_propensity": float(sequence_propensity),
                "long_hydrophobic_contacts": float(long_hydrophobic),
                "steric": float(steric),
                "hydrophobic": float(hydrophobic),
                "packing": float(packing),
                "charge": float(charge),
                "compactness": float(compactness),
            },
            engine="independent_backbone_energy",
            lower_is_better=True,
        )

    def _scaled_positions(self, main_positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(main_positions, dtype=float)
        if not self.cfg.scale_backbone:
            return positions
        return scale_backbone_positions(positions)

    def _bond_energy(self, positions: np.ndarray) -> float:
        if len(positions) <= 1:
            return 0.0
        distances = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
        sigma = max(self.cfg.bond_sigma, 1e-6)
        return float(np.mean(((distances - self.cfg.bond_target) / sigma) ** 2))

    def _angle_energy(self, positions: np.ndarray) -> float:
        if len(positions) <= 2:
            return 0.0
        left = positions[1:-1] - positions[:-2]
        right = positions[2:] - positions[1:-1]
        norm_left = np.linalg.norm(left, axis=1)
        norm_right = np.linalg.norm(right, axis=1)
        valid = (norm_left > 1e-8) & (norm_right > 1e-8)
        if not np.any(valid):
            return 0.0
        cosine = np.sum(left[valid] * right[valid], axis=1) / (norm_left[valid] * norm_right[valid])
        cosine = np.clip(cosine, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cosine))
        sigma = max(self.cfg.angle_sigma_deg, 1e-6)
        return float(np.mean(((angle_deg - self.cfg.angle_target_deg) / sigma) ** 2))

    def _torsion_angles(self, positions: np.ndarray) -> np.ndarray:
        if len(positions) <= 3:
            return np.array([], dtype=float)
        b0 = positions[1:-2] - positions[:-3]
        b1 = positions[2:-1] - positions[1:-2]
        b2 = positions[3:] - positions[2:-1]
        n0 = np.cross(b0, b1)
        n1 = np.cross(b1, b2)
        n0_norm = np.linalg.norm(n0, axis=1)
        n1_norm = np.linalg.norm(n1, axis=1)
        b1_norm = np.linalg.norm(b1, axis=1)
        valid = (n0_norm > 1e-8) & (n1_norm > 1e-8) & (b1_norm > 1e-8)
        if not np.any(valid):
            return np.array([], dtype=float)
        n0u = n0[valid] / n0_norm[valid][:, None]
        n1u = n1[valid] / n1_norm[valid][:, None]
        b1u = b1[valid] / b1_norm[valid][:, None]
        m1 = np.cross(n0u, b1u)
        x = np.sum(n0u * n1u, axis=1)
        y = np.sum(m1 * n1u, axis=1)
        return np.arctan2(y, x)

    def _torsion_energy(self, torsion_angles: np.ndarray) -> float:
        if torsion_angles.size == 0:
            return 0.0
        periodic = (
            self.cfg.torsion_weight_cos1 * (1.0 - np.cos(torsion_angles))
            + self.cfg.torsion_weight_cos3 * (1.0 - np.cos(3.0 * torsion_angles))
        )
        return float(np.mean(periodic))

    @staticmethod
    def _chirality_energy(torsion_angles: np.ndarray) -> float:
        if torsion_angles.size == 0:
            return 0.0
        positive = np.clip(np.sin(torsion_angles), 0.0, None)
        return float(np.mean(positive ** 2))

    @staticmethod
    def _trans_energy(torsion_angles: np.ndarray) -> float:
        if torsion_angles.size == 0:
            return 0.0
        return float(np.mean(1.0 + np.cos(torsion_angles)))

    def _sequence_propensity_energy(self, sequence: str, torsion_angles: np.ndarray) -> float:
        if torsion_angles.size == 0 or not sequence:
            return 0.0
        torsion_deg = np.degrees(torsion_angles)
        values: List[float] = []
        for index, angle in enumerate(torsion_deg, start=1):
            residue = sequence[min(index, len(sequence) - 1)]
            if abs(float(angle)) >= 120.0:
                values.append(-float(self.cfg.beta_propensity.get(residue, 1.0)))
            elif float(angle) < 0.0:
                values.append(-float(self.cfg.turn_propensity.get(residue, 1.0)))
            else:
                values.append(float(self.cfg.left_handed_tolerant.get(residue, self.cfg.left_handed_penalty)))
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _long_range_hydrophobic_contact_energy(self, positions: np.ndarray, sequence: str) -> float:
        if not sequence:
            return 0.0
        values: List[float] = []
        for left in range(len(sequence)):
            if max(self.cfg.hydropathy.get(sequence[left], 0.0), 0.0) <= 0.0:
                continue
            for right in range(left + max(2, self.cfg.long_hydrophobic_min_gap), len(sequence)):
                if max(self.cfg.hydropathy.get(sequence[right], 0.0), 0.0) <= 0.0:
                    continue
                distance = float(np.linalg.norm(positions[left] - positions[right]))
                values.append(1.0 if distance <= self.cfg.long_hydrophobic_cutoff else 0.0)
        if not values:
            return 0.0
        return float(-sum(values) / len(values))

    def _length_gate(self, sequence_length: int) -> float:
        if self.cfg.length_transition_span <= 1e-8:
            return 1.0 if sequence_length >= self.cfg.length_transition_center else 0.0
        scaled = (float(sequence_length) - self.cfg.length_transition_center) / self.cfg.length_transition_span
        return float(np.clip(scaled, 0.0, 1.0))

    def _steric_energy(self, positions: np.ndarray) -> float:
        distances, _, _ = self._nonbonded_pairs(positions)
        if distances.size == 0:
            return 0.0
        close = distances[distances < self.cfg.repulsion_distance]
        if close.size == 0:
            return 0.0
        clipped = np.clip(close, self.cfg.min_distance_floor, None)
        penetration = (self.cfg.repulsion_distance - clipped) / self.cfg.repulsion_distance
        return float(np.mean(np.clip(penetration, 0.0, None) ** 4))

    def _hydrophobic_energy(self, positions: np.ndarray, sequence: str) -> float:
        distances, left_idx, right_idx = self._nonbonded_pairs(positions)
        if distances.size == 0:
            return 0.0
        values = []
        sigma = max(self.cfg.hydrophobic_contact_sigma, 1e-6)
        for distance, left, right in zip(distances, left_idx, right_idx):
            h_left = max(self.cfg.hydropathy.get(sequence[left], 0.0), 0.0) if left < len(sequence) else 0.0
            h_right = max(self.cfg.hydropathy.get(sequence[right], 0.0), 0.0) if right < len(sequence) else 0.0
            if h_left <= 0.0 or h_right <= 0.0:
                continue
            shell = math.exp(-((float(distance) - self.cfg.hydrophobic_contact_center) ** 2) / (2.0 * sigma ** 2))
            values.append(-(h_left * h_right) * shell)
        if not values:
            return 0.0
        return float(np.mean(values))

    def _packing_energy(self, positions: np.ndarray) -> float:
        distances, _, _ = self._nonbonded_pairs(positions)
        if distances.size == 0:
            return 0.0
        sigma = max(self.cfg.packing_shell_sigma, 1e-6)
        shell = np.exp(-((distances - self.cfg.packing_shell_center) ** 2) / (2.0 * sigma ** 2))
        return float(-np.mean(shell))

    def _charge_energy(self, positions: np.ndarray, sequence: str) -> float:
        distances, left_idx, right_idx = self._nonbonded_pairs(positions)
        if distances.size == 0:
            return 0.0
        values = []
        screen = max(self.cfg.charge_screen_length, 1e-6)
        for distance, left, right in zip(distances, left_idx, right_idx):
            q_left = self.cfg.charge.get(sequence[left], 0.0) if left < len(sequence) else 0.0
            q_right = self.cfg.charge.get(sequence[right], 0.0) if right < len(sequence) else 0.0
            if q_left == 0.0 or q_right == 0.0:
                continue
            dist_value = max(float(distance), self.cfg.min_distance_floor)
            values.append((q_left * q_right) * math.exp(-dist_value / screen) / dist_value)
        if not values:
            return 0.0
        return float(np.mean(values))

    def _compactness_energy(self, positions: np.ndarray) -> float:
        if len(positions) == 0:
            return 0.0
        rg = self._radius_of_gyration(positions)
        target = self.cfg.compactness_prefactor * (len(positions) ** self.cfg.compactness_exponent)
        if target <= 1e-8:
            return 0.0
        return float(((rg - target) / target) ** 2)

    def _nonbonded_pairs(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(positions) <= self.cfg.min_sequence_separation:
            empty = np.array([], dtype=float)
            empty_idx = np.array([], dtype=int)
            return empty, empty_idx, empty_idx
        diff = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(diff, axis=-1)
        rows, cols = np.indices((len(positions), len(positions)))
        mask = (cols > rows) & ((cols - rows) >= max(2, self.cfg.min_sequence_separation))
        return distances[mask], rows[mask], cols[mask]

    @staticmethod
    def _radius_of_gyration(positions: np.ndarray) -> float:
        center = positions.mean(axis=0)
        return float(np.sqrt(np.mean(np.sum((positions - center) ** 2, axis=1))))


@dataclass(frozen=True)
class MolecularMechanicsScoringConfig:
    """
    A force-field-inspired ranking score for refined candidates.

    This is not a full chemistry engine, but its terms are organized like a
    simple molecular mechanics objective so a future engine can replace it
    without changing the rerank pipeline.
    """

    scale_backbone: bool = True
    target_ca_distance: float = 3.8
    target_ca_angle_deg: float = 110.0
    attractive_distance: float = 7.5
    repulsive_distance: float = 3.5
    min_distance_floor: float = 1.0
    min_sequence_separation: int = 2
    weight_bond: float = 10.0
    weight_angle: float = 2.0
    weight_repulsion: float = 8.0
    weight_attraction: float = 1.5
    weight_compactness: float = 0.35
    weight_sidechain_clash: float = 6.0
    weight_support: float = 0.1


@dataclass
class MolecularMechanicsStructureScorer:
    cfg: MolecularMechanicsScoringConfig = MolecularMechanicsScoringConfig()

    def score(self, candidate: StructureCandidate) -> ScoreBreakdown:
        main_positions, side_positions = self._scaled_positions(candidate)
        atom_cloud = self._atom_cloud(main_positions, side_positions, candidate.full_atom_positions)

        bond_penalty = self._bond_penalty(main_positions)
        angle_penalty = self._angle_penalty(main_positions)
        repulsion = self._backbone_repulsion(main_positions)
        attraction = self._backbone_attraction(main_positions)
        compactness = self._radius_of_gyration(atom_cloud)
        sidechain_clash = self._sidechain_clash(main_positions, side_positions)
        support_bonus = candidate.support_value()

        total = (
            self.cfg.weight_bond * bond_penalty
            + self.cfg.weight_angle * angle_penalty
            + self.cfg.weight_repulsion * repulsion
            - self.cfg.weight_attraction * attraction
            + self.cfg.weight_compactness * compactness
            + self.cfg.weight_sidechain_clash * sidechain_clash
            - self.cfg.weight_support * support_bonus
        )

        return ScoreBreakdown(
            total=float(total),
            terms={
                "bond_penalty": float(bond_penalty),
                "angle_penalty": float(angle_penalty),
                "repulsion": float(repulsion),
                "attraction": float(attraction),
                "compactness": float(compactness),
                "sidechain_clash": float(sidechain_clash),
                "support_bonus": float(support_bonus),
            },
            engine="molecular_mechanics_like",
            lower_is_better=True,
        )

    def _scaled_positions(
        self,
        candidate: StructureCandidate,
    ) -> tuple[np.ndarray, List[np.ndarray]]:
        main_positions = np.asarray(candidate.main_positions, dtype=float)
        side_positions = [
            np.asarray(side, dtype=float)
            for side in candidate.side_positions
            if side is not None
        ]
        if not self.cfg.scale_backbone:
            return main_positions, side_positions
        if len(main_positions) <= 1:
            return main_positions.copy(), [side.copy() for side in side_positions]
        step = float(np.median(np.linalg.norm(main_positions[1:] - main_positions[:-1], axis=1)))
        scale = self.cfg.target_ca_distance / max(step, 1e-8)
        return main_positions * scale, [side * scale for side in side_positions]

    @staticmethod
    def _atom_cloud(
        main_positions: np.ndarray,
        side_positions: List[np.ndarray],
        full_atom_positions: np.ndarray | None,
    ) -> np.ndarray:
        if full_atom_positions is not None:
            return np.asarray(full_atom_positions, dtype=float)
        atoms: List[np.ndarray] = [np.asarray(main_positions, dtype=float)]
        if side_positions:
            atoms.append(np.vstack(side_positions))
        return np.vstack(atoms)

    def _bond_penalty(self, main_positions: np.ndarray) -> float:
        if len(main_positions) <= 1:
            return 0.0
        distances = np.linalg.norm(main_positions[1:] - main_positions[:-1], axis=1)
        return float(np.mean((distances - self.cfg.target_ca_distance) ** 2))

    def _angle_penalty(self, main_positions: np.ndarray) -> float:
        if len(main_positions) <= 2:
            return 0.0
        vectors_left = main_positions[1:-1] - main_positions[:-2]
        vectors_right = main_positions[2:] - main_positions[1:-1]
        norms_left = np.linalg.norm(vectors_left, axis=1)
        norms_right = np.linalg.norm(vectors_right, axis=1)
        valid = (norms_left > 1e-8) & (norms_right > 1e-8)
        if not np.any(valid):
            return 0.0
        cosines = np.sum(vectors_left[valid] * vectors_right[valid], axis=1) / (norms_left[valid] * norms_right[valid])
        cosines = np.clip(cosines, -1.0, 1.0)
        angles = np.degrees(np.arccos(cosines))
        return float(np.mean((angles - self.cfg.target_ca_angle_deg) ** 2) / 100.0)

    def _backbone_repulsion(self, main_positions: np.ndarray) -> float:
        distances = self._backbone_nonbonded(main_positions)
        if distances.size == 0:
            return 0.0
        clipped = np.clip(distances, self.cfg.min_distance_floor, None)
        repulsive = clipped[clipped < self.cfg.repulsive_distance]
        if repulsive.size == 0:
            return 0.0
        ratio = self.cfg.repulsive_distance / repulsive
        return float(np.mean(ratio ** 12 - 2.0 * ratio ** 6 + 1.0))

    def _backbone_attraction(self, main_positions: np.ndarray) -> float:
        distances = self._backbone_nonbonded(main_positions)
        if distances.size == 0:
            return 0.0
        contact = distances[distances <= self.cfg.attractive_distance]
        if contact.size == 0:
            return 0.0
        ratio = self.cfg.attractive_distance / np.clip(contact, self.cfg.min_distance_floor, None)
        return float(np.mean(ratio ** 6))

    def _sidechain_clash(self, main_positions: np.ndarray, side_positions: List[np.ndarray]) -> float:
        if not side_positions:
            return 0.0
        side_array = np.vstack(side_positions)
        backbone_diff = side_array[:, None, :] - main_positions[None, :, :]
        backbone_dist = np.linalg.norm(backbone_diff, axis=-1)
        side_diff = side_array[:, None, :] - side_array[None, :, :]
        side_dist = np.linalg.norm(side_diff, axis=-1)
        backbone_clashes = backbone_dist[backbone_dist < self.cfg.repulsive_distance]
        side_mask = np.triu(np.ones_like(side_dist, dtype=bool), k=1)
        side_clashes = side_dist[side_mask & (side_dist < self.cfg.repulsive_distance)]
        if backbone_clashes.size == 0 and side_clashes.size == 0:
            return 0.0
        penalty_terms: List[np.ndarray] = []
        if backbone_clashes.size:
            penalty_terms.append((self.cfg.repulsive_distance - backbone_clashes) ** 2)
        if side_clashes.size:
            penalty_terms.append((self.cfg.repulsive_distance - side_clashes) ** 2)
        return float(np.mean(np.concatenate(penalty_terms)))

    def _backbone_nonbonded(self, main_positions: np.ndarray) -> np.ndarray:
        if len(main_positions) <= 2:
            return np.array([], dtype=float)
        diff = main_positions[:, None, :] - main_positions[None, :, :]
        distances = np.linalg.norm(diff, axis=-1)
        rows, cols = np.indices((len(main_positions), len(main_positions)))
        mask = (cols > rows) & ((cols - rows) >= max(2, self.cfg.min_sequence_separation))
        return distances[mask]

    @staticmethod
    def _radius_of_gyration(positions: np.ndarray) -> float:
        if len(positions) == 0:
            return 0.0
        center = positions.mean(axis=0)
        return float(np.sqrt(np.mean(np.sum((positions - center) ** 2, axis=1))))


@dataclass
class CallableStructureScorer:
    """Adapter for future chemistry engines or custom score functions."""

    score_fn: Callable[[StructureCandidate], ScoreBreakdown | Dict[str, Any] | float]
    engine_name: str = "callable_scorer"
    lower_is_better: bool = True

    def score(self, candidate: StructureCandidate) -> ScoreBreakdown:
        result = self.score_fn(candidate)
        if isinstance(result, ScoreBreakdown):
            return result
        if isinstance(result, dict):
            terms = {
                str(key): float(value)
                for key, value in result.items()
                if isinstance(value, (int, float))
            }
            total = terms.get("total", terms.get("score", float("inf")))
            return ScoreBreakdown(
                total=float(total),
                terms=terms,
                engine=self.engine_name,
                lower_is_better=self.lower_is_better,
            )
        return ScoreBreakdown(
            total=float(result),
            terms={"score": float(result)},
            engine=self.engine_name,
            lower_is_better=self.lower_is_better,
        )


@dataclass(frozen=True)
class LatticeContactScoringConfig:
    """
    Lattice-aware scoring for tetrahedral lattice protein models.

    Uses Miyazawa-Jernigan contact potential (with virtual Cb side-chain
    contacts), cross-group consensus, contact order, residue-environment
    compatibility, and turn propensity to rank decoded structures.

    Design principle: quantum sampling provides the candidate pool but
    carries NO information about RMSD quality.  All ranking power comes
    from physics-based posterior terms.  ``weight_support`` is therefore
    set to **0.0** by default.
    """

    # --- overlap / steric --------------------------------------------------
    overlap_penalty: float = 1000.0
    overlap_eps: float = 1e-6
    steric_r_min: float = 0.8
    weight_steric: float = 10.0

    # --- MJ soft-contact kernel --------------------------------------------
    contact_r0: float = 2.5
    contact_p: float = 4.0
    min_sequence_separation: int = 2
    weight_mj: float = 3.0

    # --- Virtual Cb side-chain MJ contacts (NEW) --------------------------
    #   Cb positions are inferred from backbone geometry.
    #   Three sub-terms: main-main (existing), main-side, side-side.
    cb_offset: float = 0.7          # lattice-unit offset for virtual Cb
    weight_mj_main_side: float = 2.0
    weight_mj_side_side: float = 3.5

    # --- Contact density ---------------------------------------------------
    contact_cutoff: float = 2.5
    weight_contact_density: float = 2.0

    # --- Compactness (radius of gyration) ----------------------------------
    weight_compactness: float = 0.5
    compactness_prefactor: float = 0.8
    compactness_exponent: float = 0.5

    # --- Residue environment (bidirectional burial) ------------------------
    weight_burial: float = 0.5
    burial_radius: float = 3.0
    burial_ref: float = 3.0

    # --- Contact order (NEW) -----------------------------------------------
    weight_contact_order: float = 1.5
    co_contact_cutoff: float = 2.0
    co_target: float = 0.15

    # --- Turn propensity (NEW) ---------------------------------------------
    weight_turn_propensity: float = 1.0

    # --- Cross-group consensus (NEW) ---------------------------------------
    #   n_groups: number of independent quantum groups that sampled this
    #   bitstring.  Higher → more reproducible → more reliable.
    weight_consensus: float = 3.0
    max_groups: int = 10

    # --- Quantum sampling support (DEPRECATED as ranking signal) -----------
    #   Set to 0.0: quantum sampling is prior only, not posterior evidence.
    weight_support: float = 0.0

    hydrophobicity: Dict[str, float] = field(
        default_factory=lambda: {
            "A": 1.8, "V": 4.2, "I": 4.5, "L": 3.8, "M": 1.9, "F": 2.8,
            "W": -0.9, "Y": -1.3, "C": 2.5, "P": -1.6, "G": -0.4,
            "S": -0.8, "T": -0.7, "N": -3.5, "Q": -3.5, "D": -3.5,
            "E": -3.5, "K": -3.9, "R": -4.5, "H": -3.2,
        },
        repr=False,
    )

    # Turn-type propensity table (NEW)
    # Lattice turns 0,2 map to extended/beta-like; turns 1,3 map to
    # turn/bend-like conformations.
    turn_propensity_table: Dict[str, Dict[int, float]] = field(
        default_factory=lambda: {
            # strong beta formers → prefer extended turns (0, 2)
            "V": {0: 1.7, 1: 0.5, 2: 1.7, 3: 0.5},
            "I": {0: 1.6, 1: 0.5, 2: 1.6, 3: 0.5},
            "Y": {0: 1.5, 1: 0.6, 2: 1.5, 3: 0.6},
            "F": {0: 1.4, 1: 0.6, 2: 1.4, 3: 0.6},
            "W": {0: 1.4, 1: 0.6, 2: 1.4, 3: 0.6},
            "L": {0: 1.2, 1: 0.8, 2: 1.2, 3: 0.8},
            "C": {0: 1.2, 1: 0.8, 2: 1.2, 3: 0.8},
            "M": {0: 1.1, 1: 0.9, 2: 1.1, 3: 0.9},
            "A": {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
            "T": {0: 1.2, 1: 0.9, 2: 1.2, 3: 0.9},
            # strong turn formers → prefer bend turns (1, 3)
            "G": {0: 0.5, 1: 1.7, 2: 0.5, 3: 1.7},
            "P": {0: 0.5, 1: 1.6, 2: 0.5, 3: 1.6},
            "N": {0: 0.7, 1: 1.4, 2: 0.7, 3: 1.4},
            "D": {0: 0.7, 1: 1.4, 2: 0.7, 3: 1.4},
            "S": {0: 0.8, 1: 1.3, 2: 0.8, 3: 1.3},
            "K": {0: 0.9, 1: 1.1, 2: 0.9, 3: 1.1},
            "R": {0: 0.9, 1: 1.1, 2: 0.9, 3: 1.1},
            "H": {0: 0.9, 1: 1.1, 2: 0.9, 3: 1.1},
            "Q": {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
            "E": {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
        },
        repr=False,
    )

    mj_matrix_path: Optional[str] = None


@dataclass
class LatticeContactScorer:
    """
    Enhanced scorer for tetrahedral lattice protein models.

    Ranking is driven entirely by **posterior** physics-based terms.
    Quantum sampling provides the candidate pool but carries no weight
    in the final score (``weight_support=0``).

    Scoring terms (13 total):
      1. overlap / steric         – geometric validity
      2. MJ main-main             – backbone contact energy
      3. MJ main-side (virtual Cb)– backbone ↔ side-chain contacts  (NEW)
      4. MJ side-side (virtual Cb)– side-chain ↔ side-chain          (NEW)
      5. contact density           – fraction of pairs in contact
      6. compactness               – radius of gyration vs target
      7. residue environment       – bidirectional burial/exposure    (ENHANCED)
      8. contact order             – topology regularizer             (NEW)
      9. turn propensity           – residue-turn compatibility       (NEW)
     10. cross-group consensus     – independent reproducibility      (NEW)
     11. sampling support          – (disabled, weight=0)
    """

    cfg: LatticeContactScoringConfig = field(default_factory=LatticeContactScoringConfig)
    _mj_matrix: np.ndarray = field(init=False, repr=False)
    _aa_index: Dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._mj_matrix, self._aa_index = _load_mj_matrix(self.cfg.mj_matrix_path)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def score(self, candidate: StructureCandidate) -> ScoreBreakdown:
        positions = np.asarray(candidate.main_positions, dtype=float)
        sequence = candidate.sequence or ""
        n = len(positions)

        if n < 3:
            return ScoreBreakdown(
                total=math.inf,
                terms={},
                engine="lattice_contact",
                lower_is_better=True,
            )

        dist = self._pairwise_dist(positions)

        # 1. Overlap detection (any pair)
        all_nb = np.triu(np.ones((n, n), dtype=bool), k=1)
        overlap_count = float(np.count_nonzero(dist[all_nb] < self.cfg.overlap_eps))
        e_overlap = self.cfg.overlap_penalty * overlap_count

        # 2. Steric soft wall (non-bonded pairs, |i-j| >= min_sep)
        sep = max(2, self.cfg.min_sequence_separation)
        nb_mask = np.triu(np.ones((n, n), dtype=bool), k=sep)
        D = dist[nb_mask]
        num_pairs = max(1, D.size)
        wall = np.clip(self.cfg.steric_r_min - D, 0.0, None)
        e_steric = self.cfg.weight_steric * float(np.sum(wall ** 2)) / num_pairs

        # 3. MJ main-main soft-contact energy
        e_mj_mm = self._mj_energy_pairwise(D, n, sep, sequence, sequence)

        # 4-5. Virtual Cb side-chain MJ contacts (NEW)
        cb_positions = self._virtual_cb_positions(positions, sequence)
        e_mj_ms, e_mj_ss = self._sidechain_mj_energy(
            positions, cb_positions, sequence, n, sep,
        )

        # 6. Contact density (fraction of non-bonded pairs within cutoff)
        contact_fraction = (
            float(np.count_nonzero(D <= self.cfg.contact_cutoff)) / num_pairs
            if num_pairs > 0 else 0.0
        )

        # 7. Compactness (radius of gyration)
        center = positions.mean(axis=0)
        rg = float(np.sqrt(np.mean(np.sum((positions - center) ** 2, axis=1))))
        rg_target = self.cfg.compactness_prefactor * (n ** self.cfg.compactness_exponent)
        e_compact = ((rg - rg_target) / max(rg_target, 1e-6)) ** 2

        # 8. Bidirectional residue environment (ENHANCED)
        e_burial = self._burial_energy(dist, sequence)

        # 9. Contact order (NEW)
        e_co = self._contact_order_energy(dist, n)

        # 10. Turn propensity (NEW)
        e_turn = self._turn_propensity_energy(candidate, sequence)

        # 11. Cross-group consensus (NEW)
        e_consensus = self._consensus_energy(candidate)

        # 12. Sampling support (disabled by default)
        support = candidate.support_value()

        total = (
            e_overlap
            + e_steric
            + self.cfg.weight_mj * e_mj_mm
            + self.cfg.weight_mj_main_side * e_mj_ms
            + self.cfg.weight_mj_side_side * e_mj_ss
            - self.cfg.weight_contact_density * contact_fraction
            + self.cfg.weight_compactness * e_compact
            + self.cfg.weight_burial * e_burial
            + self.cfg.weight_contact_order * e_co
            - self.cfg.weight_turn_propensity * e_turn
            - self.cfg.weight_consensus * e_consensus
            - self.cfg.weight_support * support
        )

        return ScoreBreakdown(
            total=float(total),
            terms={
                "overlap_count": overlap_count,
                "e_overlap": float(e_overlap),
                "e_steric": float(e_steric),
                "e_mj_mm": float(e_mj_mm),
                "e_mj_ms": float(e_mj_ms),
                "e_mj_ss": float(e_mj_ss),
                "contact_fraction": contact_fraction,
                "rg": rg,
                "e_compactness": float(e_compact),
                "e_burial": float(e_burial),
                "e_contact_order": float(e_co),
                "e_turn_propensity": float(e_turn),
                "e_consensus": float(e_consensus),
                "support": float(support),
            },
            engine="lattice_contact",
            lower_is_better=True,
        )

    # ------------------------------------------------------------------
    #  MJ energy helpers
    # ------------------------------------------------------------------
    def _mj_energy_pairwise(
        self,
        D: np.ndarray,
        n: int,
        sep: int,
        seq_row: str,
        seq_col: str,
    ) -> float:
        """Weighted MJ energy for a pre-computed distance vector *D*."""
        if not self._aa_index or D.size == 0:
            return 0.0
        r0 = max(1e-8, self.cfg.contact_r0)
        w_contact = np.exp(-(D / r0) ** self.cfg.contact_p)
        ii, jj = np.triu_indices(n, k=sep)
        e_mj = 0.0
        for k_idx, (i, j) in enumerate(zip(ii, jj)):
            if i >= len(seq_row) or j >= len(seq_col):
                continue
            ai = self._aa_index.get(seq_row[i])
            aj = self._aa_index.get(seq_col[j])
            if ai is None or aj is None:
                continue
            w = float(w_contact[k_idx])
            if w <= 0.0:
                continue
            e_mj += w * float(self._mj_matrix[ai, aj])
        return e_mj / max(1, D.size)

    # ------------------------------------------------------------------
    #  Virtual Cb side-chain positions (NEW)
    # ------------------------------------------------------------------
    def _virtual_cb_positions(
        self,
        main_positions: np.ndarray,
        sequence: str,
    ) -> List[Optional[np.ndarray]]:
        """
        Estimate virtual Cb positions from backbone geometry.

        For each non-Glycine residue *i*, Cb is placed along the outward
        direction from the local centroid at distance ``cb_offset``.
        Glycine has no side chain → None.
        """
        n = len(main_positions)
        centroid = main_positions.mean(axis=0)
        cb_list: List[Optional[np.ndarray]] = []
        for i in range(n):
            if i < len(sequence) and sequence[i] == "G":
                cb_list.append(None)
                continue
            direction = main_positions[i] - centroid
            norm = float(np.linalg.norm(direction))
            if norm < 1e-8:
                cb_list.append(None)
            else:
                cb_list.append(main_positions[i] + (direction / norm) * self.cfg.cb_offset)
        return cb_list

    def _sidechain_mj_energy(
        self,
        main_pos: np.ndarray,
        cb_list: List[Optional[np.ndarray]],
        sequence: str,
        n: int,
        sep: int,
    ) -> Tuple[float, float]:
        """Return (e_mj_main_side, e_mj_side_side) from virtual Cb contacts."""
        if not self._aa_index:
            return 0.0, 0.0

        r0 = max(1e-8, self.cfg.contact_r0)
        p = self.cfg.contact_p
        e_ms = 0.0
        e_ss = 0.0
        count_ms = 0
        count_ss = 0

        for i in range(n):
            ai = self._aa_index.get(sequence[i]) if i < len(sequence) else None
            if ai is None:
                continue
            for j in range(i + sep, n):
                aj = self._aa_index.get(sequence[j]) if j < len(sequence) else None
                if aj is None:
                    continue

                cb_j = cb_list[j]
                cb_i = cb_list[i]

                # main_i ↔ side_j
                if cb_j is not None:
                    d_ms = float(np.linalg.norm(main_pos[i] - cb_j))
                    w = math.exp(-((d_ms / r0) ** p))
                    if w > 0.0:
                        e_ms += w * float(self._mj_matrix[ai, aj])
                        count_ms += 1

                # side_i ↔ main_j
                if cb_i is not None:
                    d_sm = float(np.linalg.norm(cb_i - main_pos[j]))
                    w = math.exp(-((d_sm / r0) ** p))
                    if w > 0.0:
                        e_ms += w * float(self._mj_matrix[ai, aj])
                        count_ms += 1

                # side_i ↔ side_j
                if cb_i is not None and cb_j is not None:
                    d_ss = float(np.linalg.norm(cb_i - cb_j))
                    w = math.exp(-((d_ss / r0) ** p))
                    if w > 0.0:
                        e_ss += w * float(self._mj_matrix[ai, aj])
                        count_ss += 1

        e_ms = e_ms / max(1, count_ms) if count_ms > 0 else 0.0
        e_ss = e_ss / max(1, count_ss) if count_ss > 0 else 0.0
        return e_ms, e_ss

    # ------------------------------------------------------------------
    #  Bidirectional burial/environment (ENHANCED)
    # ------------------------------------------------------------------
    def _burial_energy(self, dist: np.ndarray, sequence: str) -> float:
        """
        Bidirectional residue-environment energy.

        Penalizes:
          - hydrophobic residues that are EXPOSED  (few neighbours)
          - polar/charged residues that are BURIED  (too many neighbours)
        """
        n = dist.shape[0]
        if n < 3 or not sequence:
            return 0.0
        r_b = max(1e-8, self.cfg.burial_radius)
        kernel = np.exp(-(dist / r_b) ** 2)
        np.fill_diagonal(kernel, 0.0)
        burial = np.sum(kernel, axis=1)  # per-residue burial score
        hmap = self.cfg.hydrophobicity
        ref = self.cfg.burial_ref

        e_total = 0.0
        for i in range(min(n, len(sequence))):
            h = hmap.get(sequence[i], 0.0)
            b = float(burial[i])
            if h > 0:
                # Hydrophobic: penalize exposure (low burial)
                e_total += h * max(0.0, ref - b)
            else:
                # Polar/charged: penalize burial (high burial)
                e_total += (-h) * max(0.0, b - ref)
        return e_total / max(1, min(n, len(sequence)))

    # ------------------------------------------------------------------
    #  Contact order (NEW)
    # ------------------------------------------------------------------
    def _contact_order_energy(self, dist: np.ndarray, n: int) -> float:
        """
        Relative contact order penalty.

        CO = (1 / L * N_c) * sum(|i-j|) for contacts.
        Penalizes deviation from natural CO_target.
        """
        if n < 4:
            return 0.0
        threshold = self.cfg.co_contact_cutoff
        ii, jj = np.where(np.triu(dist < threshold, k=2))
        n_contacts = len(ii)
        if n_contacts == 0:
            return 1.0  # penalty for no contacts at all
        co = float(np.mean(np.abs(jj - ii))) / float(n)
        target = self.cfg.co_target
        return ((co - target) / max(target, 1e-6)) ** 2

    # ------------------------------------------------------------------
    #  Turn propensity (NEW)
    # ------------------------------------------------------------------
    def _turn_propensity_energy(
        self,
        candidate: StructureCandidate,
        sequence: str,
    ) -> float:
        """
        Score compatibility between residue types and lattice turn types.

        Returns a positive value (higher → better compatibility), used
        as a *bonus* in the total (subtracted with positive weight).
        """
        main_turns = candidate.metadata.get("main_turns")
        if not main_turns or not sequence:
            return 0.0
        table = self.cfg.turn_propensity_table
        total = 0.0
        count = 0
        # main_turns[k] encodes the turn from bead k to bead k+1.
        # The turn is associated with residue k+1 (the destination).
        for k, turn in enumerate(main_turns):
            residue_idx = k + 1
            if residue_idx >= len(sequence):
                break
            aa = sequence[residue_idx]
            aa_table = table.get(aa)
            if aa_table is None:
                total += 1.0  # neutral
            else:
                total += aa_table.get(int(turn), 1.0)
            count += 1
        if count == 0:
            return 0.0
        return total / count

    # ------------------------------------------------------------------
    #  Cross-group consensus (NEW)
    # ------------------------------------------------------------------
    def _consensus_energy(self, candidate: StructureCandidate) -> float:
        """
        Consensus bonus from cross-group reproducibility.

        Returns a normalized value in [0, 1] (higher → more groups agree),
        used as a *bonus* in the total.
        """
        n_groups = candidate.metadata.get("n_groups")
        if n_groups is None:
            return 0.0
        return float(n_groups) / max(1, self.cfg.max_groups)

    # ------------------------------------------------------------------
    #  Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _pairwise_dist(pts: np.ndarray) -> np.ndarray:
        diff = pts[:, None, :] - pts[None, :, :]
        return np.linalg.norm(diff, axis=-1)


# Backward-compatible aliases while the downstream package still speaks in
# terms of "completion".
StructureCompleter = StructureProposalGenerator
IdentityStructureCompleter = IdentityStructureProposalGenerator
