# --*-- conding:utf-8 --*--
# @time:3/28/26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:sidechain.py

"""
Side-chain reconstruction from backbone + coarse-grained lattice bead positions.

Strategy
--------
1.  For each residue, look up its ``ResidueTopology`` (from constants.py).
2.  Place atoms in Z-matrix order using the NERF builder from backbone.py.
3.  **Lattice-guided chi1 override** -- when a coarse-grained side-chain
    bead position is available (from the quantum-sampled lattice model),
    we compute chi1 so that the CB->CG direction points toward that bead.
    This leverages the quantum sampling information instead of relying
    solely on library rotamers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .backbone import BackboneResult, nerf, _normalize
from .constants import (
    AtomPlacement,
    ResidueTopology,
    get_topology,
    ONE_TO_THREE,
)

_LOG = logging.getLogger(__name__)


@dataclass
class SidechainAtom:
    """One side-chain heavy atom."""
    name: str
    residue_index: int
    position: np.ndarray


@dataclass
class SidechainResult:
    """Collection of side-chain atoms for the whole chain."""
    atoms: List[SidechainAtom] = field(default_factory=list)

    def atoms_for_residue(self, residue_index: int) -> List[SidechainAtom]:
        return [a for a in self.atoms if a.residue_index == residue_index]


def _compute_dihedral(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray,
) -> float:
    """Compute dihedral angle A-B-C-D in degrees, range (-180, 180]."""
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
    return float(np.degrees(np.arctan2(-y, x)))


def _chi1_from_lattice_bead(
    n_pos: np.ndarray,
    ca_pos: np.ndarray,
    cb_pos: np.ndarray,
    bead_pos: np.ndarray,
) -> float:
    """
    Compute the chi1 dihedral that orients CB->CG toward the lattice bead.

    The coarse-grained bead position (from quantum sampling) represents the
    approximate center of mass of the side chain.  We project the CA->bead
    direction onto the CG placement plane and find the chi1 dihedral that
    best aligns with it.
    """
    # The desired CG direction (from CB toward bead)
    desired_dir = _normalize(bead_pos - cb_pos)

    # Scan chi1 values and pick the one whose CG direction best matches
    best_chi1 = -65.0  # fallback
    best_dot = -2.0
    for chi1_deg in range(-180, 180, 5):
        # Place a trial CG at standard bond length/angle
        trial_cg = nerf(n_pos, ca_pos, cb_pos, 1.52, 113.0, float(chi1_deg))
        trial_dir = _normalize(trial_cg - cb_pos)
        dot = float(np.dot(trial_dir, desired_dir))
        if dot > best_dot:
            best_dot = dot
            best_chi1 = float(chi1_deg)

    return best_chi1


def build_sidechains(
    backbone: BackboneResult,
    sequence: str,
    side_positions: Optional[List[Optional[np.ndarray]]] = None,
    use_lattice_chi1: bool = True,
) -> SidechainResult:
    """
    Build all side-chain heavy atoms from backbone geometry and topology.

    Parameters
    ----------
    backbone       : BackboneResult from ``reconstruct_backbone()``.
    sequence       : 1-letter amino-acid sequence.
    side_positions : coarse-grained side-chain bead positions from the lattice
                     decoder (one per residue, None for residues without a
                     side chain bead).  Used to guide chi1.
    use_lattice_chi1 : if True and a side-chain bead is available, override
                       chi1 with the lattice-derived angle.

    Returns
    -------
    SidechainResult containing all placed heavy atoms.
    """
    seq = sequence.upper()
    n_res = len(seq)
    result = SidechainResult()

    side_pos = side_positions or [None] * n_res
    if len(side_pos) < n_res:
        side_pos = list(side_pos) + [None] * (n_res - len(side_pos))

    for i in range(n_res):
        res1 = seq[i]

        # Skip GLY and ALA (no extra atoms beyond CB)
        try:
            topo = get_topology(res1)
        except KeyError:
            _LOG.warning("Unknown residue '%s' at position %d, skipping sidechain.", res1, i)
            continue

        if not topo.has_sidechain:
            continue

        # Gather backbone frame for this residue
        frame = backbone.get_frame(i)
        if "N" not in frame or "CA" not in frame or "C" not in frame:
            _LOG.warning("Incomplete backbone at residue %d, skipping sidechain.", i)
            continue

        n_pos = frame["N"]
        ca_pos = frame["CA"]
        c_pos = frame["C"]
        cb_pos = frame.get("CB")
        if cb_pos is None:
            continue  # GLY

        # Determine chi1 override from lattice bead
        chi1_override: Optional[float] = None
        if use_lattice_chi1 and side_pos[i] is not None:
            bead = np.asarray(side_pos[i], dtype=float)
            if bead.shape == (3,) and np.linalg.norm(bead - ca_pos) > 0.1:
                chi1_override = _chi1_from_lattice_bead(n_pos, ca_pos, cb_pos, bead)

        # Place each atom in Z-matrix order
        # Maintain a local name->position map for references
        placed: Dict[str, np.ndarray] = {
            "N": n_pos,
            "CA": ca_pos,
            "C": c_pos,
            "CB": cb_pos,
        }

        for ap in topo.heavy_atoms:
            ref_names = ap.reference_atoms  # (grandparent, parent, bonded)
            # Check all references are available
            if not all(rn in placed for rn in ref_names):
                _LOG.debug(
                    "Missing reference atom for %s at residue %d (need %s), skipping.",
                    ap.name, i, ref_names,
                )
                continue

            a_pos = placed[ref_names[0]]
            b_pos = placed[ref_names[1]]
            c_ref = placed[ref_names[2]]

            # Determine dihedral
            dihedral = ap.dihedral
            if ap.is_chi and ap.chi_index == 1 and chi1_override is not None:
                dihedral = chi1_override

            atom_pos = nerf(a_pos, b_pos, c_ref, ap.bond_length, ap.bond_angle, dihedral)
            placed[ap.name] = atom_pos
            result.atoms.append(SidechainAtom(ap.name, i, atom_pos.copy()))

    return result


__all__ = [
    "SidechainAtom",
    "SidechainResult",
    "build_sidechains",
]
