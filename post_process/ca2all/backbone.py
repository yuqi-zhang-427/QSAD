# --*-- conding:utf-8 --*--
# @time:3/28/26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:backbone.py

"""
Reconstruct full backbone atoms (N, CA, C, O) and CB from a C-alpha trace.

Algorithm overview
------------------
1.  Build a local orthonormal frame at each CA[i] from its neighbours.
2.  Place N[i] and C[i] using ideal bond lengths and the N-CA-C angle,
    oriented so that the peptide planes alternate (trans omega).
3.  Place O[i] in the CA[i]-C[i]-N[i+1] plane at the ideal C=O angle.
4.  Place CB[i] from the N-CA-C frame (except GLY).

The method is analogous to the "nerf" (Natural Extension Reference Frame)
placement used in PULCHRA / PD2 / OpenMM backbone builders, adapted for
the coarse-grained lattice input of this project.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import (
    N_CA_BOND,
    CA_C_BOND,
    C_O_BOND,
    C_N_BOND,
    CA_CB_BOND,
    N_CA_C_ANGLE,
    CA_C_O_ANGLE,
    CA_C_N_ANGLE,
    C_N_CA_ANGLE,
    N_CA_CB_ANGLE,
    C_CA_CB_ANGLE,
)

_LOG = logging.getLogger(__name__)


# ── NERF placement utility ─────────────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(n, 1e-12)


def nerf(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bond_length: float,
    bond_angle_deg: float,
    dihedral_deg: float,
) -> np.ndarray:
    """
    Place atom D given three reference atoms A-B-C using Natural Extension
    Reference Frame (NERF).

    Parameters
    ----------
    a, b, c : reference positions (grandparent, parent, bonded-atom)
    bond_length : |C-D|
    bond_angle_deg : angle(B-C-D) in degrees
    dihedral_deg : torsion(A-B-C-D) in degrees

    Returns
    -------
    Position of D as ndarray(3,).
    """
    theta = np.radians(bond_angle_deg)
    phi = np.radians(dihedral_deg)

    bc = _normalize(c - b)
    n = _normalize(np.cross(b - a, bc))
    m = np.cross(n, bc)

    d = np.array([
        -bond_length * np.cos(theta),
         bond_length * np.sin(theta) * np.cos(phi),
         bond_length * np.sin(theta) * np.sin(phi),
    ])

    # Transform to lab frame
    rot = np.column_stack([bc, m, n])
    return c + rot @ d


# ── Backbone reconstruction ───────────────────────────────────────────────

@dataclass
class BackboneAtom:
    """One backbone (or CB) atom with its identity."""
    name: str               # "N", "CA", "C", "O", "CB"
    residue_index: int      # 0-based
    position: np.ndarray    # shape (3,)


@dataclass
class BackboneResult:
    """Full backbone reconstruction output."""
    atoms: List[BackboneAtom] = field(default_factory=list)
    n_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    ca_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    c_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    o_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    cb_positions: Dict[int, np.ndarray] = field(default_factory=dict)

    def get_frame(self, residue_index: int) -> Dict[str, np.ndarray]:
        """Return {name: position} for all backbone atoms at residue_index."""
        return {a.name: a.position for a in self.atoms if a.residue_index == residue_index}


def _initial_peptide_frame(
    ca0: np.ndarray, ca1: np.ndarray, ca2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an initial reference frame for the first residue.

    Returns a synthetic triplet (ghost_A, ghost_B, CA[0]) so that the first
    N atom can be placed with NERF.
    """
    d01 = _normalize(ca1 - ca0)
    d02 = _normalize(ca2 - ca0)
    perp = np.cross(d01, d02)
    norm = np.linalg.norm(perp)
    if norm < 1e-6:
        # Collinear -- pick an arbitrary perpendicular
        arbitrary = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(d01, arbitrary)) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        perp = np.cross(d01, arbitrary)
    perp = _normalize(perp)

    # ghost_B is 1 A "behind" CA[0] along -d01
    ghost_b = ca0 - d01
    # ghost_A gives a well-defined dihedral
    ghost_a = ghost_b + perp
    return ghost_a, ghost_b, ca0


def reconstruct_backbone(
    ca_trace: np.ndarray,
    sequence: str = "",
    omega: float = 180.0,
) -> BackboneResult:
    """
    Build N, CA, C, O, CB for every residue from a scaled CA trace.

    Parameters
    ----------
    ca_trace : (N, 3) array of Calpha positions (already scaled to ~3.8 A).
    sequence : amino-acid sequence (1-letter). Used only to skip CB for GLY.
               If empty, CB is placed for every residue.
    omega    : peptide-bond dihedral in degrees (default 180 = trans).

    Returns
    -------
    BackboneResult with per-atom positions and convenience arrays.
    """
    ca = np.asarray(ca_trace, dtype=float)
    n_res = len(ca)
    if n_res < 2:
        raise ValueError("Need at least 2 CA positions for backbone reconstruction.")

    seq = sequence.upper() if sequence else "X" * n_res
    if len(seq) < n_res:
        seq = seq + "X" * (n_res - len(seq))

    atoms: List[BackboneAtom] = []
    n_pos = np.zeros((n_res, 3))
    c_pos = np.zeros((n_res, 3))
    o_pos = np.zeros((n_res, 3))
    cb_map: Dict[int, np.ndarray] = {}

    # ── Step 1: Place N and C for every residue ────────────────────────
    # We iterate from N-terminus to C-terminus.  For each residue we need
    # three reference points for NERF.  At the very start we create a
    # synthetic "ghost" frame.

    # Build initial ghost frame from first three (or two) CA atoms.
    if n_res >= 3:
        ghost_a, ghost_b, _ = _initial_peptide_frame(ca[0], ca[1], ca[2])
    else:
        d = _normalize(ca[1] - ca[0])
        arb = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(d, arb)) > 0.9:
            arb = np.array([0.0, 1.0, 0.0])
        perp = _normalize(np.cross(d, arb))
        ghost_b = ca[0] - d
        ghost_a = ghost_b + perp

    # Track the "previous C" position for placing the next N via peptide bond.
    prev_c: Optional[np.ndarray] = None
    prev_ca: Optional[np.ndarray] = None
    prev_n: Optional[np.ndarray] = None

    for i in range(n_res):
        # ── Place N[i] ─────────────────────────────────────────────
        if i == 0:
            # First residue: place N using the ghost frame
            n_i = nerf(ghost_a, ghost_b, ca[0], N_CA_BOND,
                        C_N_CA_ANGLE, 180.0)
        else:
            # N[i] is across the peptide bond from C[i-1]
            # Dihedral = omega (trans = 180)
            n_i = nerf(prev_ca, prev_c, ca[i] if prev_c is not None else ca[i],
                        C_N_BOND, CA_C_N_ANGLE, omega)
            # However, N must also be ~1.458 A from CA[i].
            # We adjust by placing N on the C[i-1]->CA[i] segment with
            # proper geometry.  Use NERF from (CA[i-1], C[i-1], CA[i]).
            n_i = nerf(prev_ca, prev_c, ca[i], N_CA_BOND,
                        C_N_CA_ANGLE, omega)

        n_pos[i] = n_i
        atoms.append(BackboneAtom("N", i, n_i.copy()))

        # ── CA[i] is given ─────────────────────────────────────────
        atoms.append(BackboneAtom("CA", i, ca[i].copy()))

        # ── Place C[i] ────────────────────────────────────────────
        if i < n_res - 1:
            # C[i] points roughly toward CA[i+1]
            # Use NERF from (N[i], CA[i], ...) with the N-CA-C angle.
            # The dihedral (psi-like) is chosen so that C[i] is in the
            # plane toward CA[i+1].
            #
            # Compute the ideal psi by projecting CA[i+1] direction.
            d_next = _normalize(ca[i + 1] - ca[i])
            d_n = _normalize(n_i - ca[i])
            # Desired C direction (from CA toward next CA, but at ideal angle)
            # We compute the dihedral that places C[i] closest to the
            # CA[i]->CA[i+1] direction.
            # Use a simple heuristic: psi that minimizes |C[i] - midpoint|
            # between CA[i] and CA[i+1].  In practice, alternating +/-120
            # works well for extended chains from lattice models.
            #
            # More robust: compute dihedral from the existing N placement.
            if prev_c is not None:
                ref_a = prev_c
            else:
                ref_a = ghost_b
            # Try a range of psi and pick the one that puts C closest to
            # the CA[i]->CA[i+1] midpoint (simple but effective).
            best_c = None
            best_dist = float("inf")
            target = ca[i] + d_next * CA_C_BOND  # ideal direction
            for psi_deg in range(-180, 180, 15):
                trial = nerf(ref_a, n_i, ca[i], CA_C_BOND, N_CA_C_ANGLE, float(psi_deg))
                dist = float(np.linalg.norm(trial - target))
                if dist < best_dist:
                    best_dist = dist
                    best_c = trial
            c_i = best_c
        else:
            # Last residue: place C using default psi
            if prev_c is not None:
                ref_a = prev_c
            else:
                ref_a = ghost_b
            c_i = nerf(ref_a, n_i, ca[i], CA_C_BOND, N_CA_C_ANGLE, 120.0)

        c_pos[i] = c_i
        atoms.append(BackboneAtom("C", i, c_i.copy()))

        # ── Place O[i] ────────────────────────────────────────────
        # O is in the peptide plane, opposite to N[i+1] direction.
        if i < n_res - 1:
            o_i = nerf(n_i, ca[i], c_i, C_O_BOND, CA_C_O_ANGLE, 180.0)
        else:
            o_i = nerf(n_i, ca[i], c_i, C_O_BOND, CA_C_O_ANGLE, 0.0)
        o_pos[i] = o_i
        atoms.append(BackboneAtom("O", i, o_i.copy()))

        # ── Place CB[i] ───────────────────────────────────────────
        if seq[i] != "G":
            cb_i = _place_cb(n_i, ca[i], c_i)
            cb_map[i] = cb_i
            atoms.append(BackboneAtom("CB", i, cb_i.copy()))

        prev_c = c_i
        prev_ca = ca[i]
        prev_n = n_i

    return BackboneResult(
        atoms=atoms,
        n_positions=n_pos,
        ca_positions=ca.copy(),
        c_positions=c_pos,
        o_positions=o_pos,
        cb_positions=cb_map,
    )


def _place_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Place CB using the standard N-CA-C frame.

    CB is positioned on the opposite side of the N-CA-C plane from the
    backbone carbonyl, at ideal tetrahedral geometry.
    """
    # Vectors from CA
    n_ca = _normalize(n - ca)
    c_ca = _normalize(c - ca)

    # Bisector of N-CA-C angle (points roughly "inward")
    bisector = _normalize(n_ca + c_ca)

    # Normal to N-CA-C plane
    perp = np.cross(n_ca, c_ca)
    norm = np.linalg.norm(perp)
    if norm < 1e-8:
        # Degenerate -- pick arbitrary perpendicular
        arb = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(n_ca, arb)) > 0.9:
            arb = np.array([0.0, 1.0, 0.0])
        perp = np.cross(n_ca, arb)
    perp = _normalize(perp)

    # CB direction: opposite bisector, tilted out of plane
    # The angle between the bisector and CB direction is about 35.5 degrees
    # (from ideal tetrahedral geometry: cos(35.5) ~ 0.814, sin(35.5) ~ 0.581)
    cb_dir = _normalize(-bisector * 0.814 + perp * 0.581)

    return ca + CA_CB_BOND * cb_dir


__all__ = [
    "BackboneAtom",
    "BackboneResult",
    "nerf",
    "reconstruct_backbone",
]
