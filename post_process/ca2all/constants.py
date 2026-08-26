# --*-- conding:utf-8 --*--
# @time:3/28/26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:constants.py

"""
Amino-acid topology, ideal backbone geometry, and a compact rotamer library
used by the Calpha-to-all-atom reconstruction pipeline.

Coordinate system convention
----------------------------
All bond lengths are in angstroms (A).  All angles are in **degrees** and
converted to radians at point of use.

Rotamer library
---------------
Each residue stores one representative rotamer (the most-populated cluster
from the Dunbrack 2010 backbone-dependent library, backbone-independent
mode-1 values).  This is intentionally minimal -- the builder can optionally
override chi angles with the lattice side-chain bead direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Ideal backbone bond lengths (A) ────────────────────────────────────────
N_CA_BOND = 1.458
CA_C_BOND = 1.525
C_O_BOND = 1.229
C_N_BOND = 1.329          # peptide bond
CA_CB_BOND = 1.521

# ── Ideal backbone bond angles (degrees) ───────────────────────────────────
N_CA_C_ANGLE = 111.2      # tau angle
CA_C_N_ANGLE = 116.2
C_N_CA_ANGLE = 121.7
CA_C_O_ANGLE = 120.8
N_CA_CB_ANGLE = 110.5
C_CA_CB_ANGLE = 110.1

# ── Ideal peptide dihedral (degrees) ───────────────────────────────────────
OMEGA_TRANS = 180.0        # trans peptide bond (default)


# ── Amino acid metadata ────────────────────────────────────────────────────
ONE_TO_THREE: Dict[str, str] = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}

THREE_TO_ONE: Dict[str, str] = {v: k for k, v in ONE_TO_THREE.items()}


# ── Side-chain atom placement specification ────────────────────────────────

@dataclass(frozen=True)
class AtomPlacement:
    """
    Internal-coordinate instruction for placing one heavy atom.

    The atom is built from three already-placed reference atoms using
    (bond_length, bond_angle, dihedral) -- classic Z-matrix style.

    reference_atoms : (grandparent, parent, self-parent) -- the three atoms
                      that define the local frame.  The *last* name is the
                      atom this new atom is directly bonded to.
    bond_length     : distance to the bonded atom (A)
    bond_angle      : angle grandparent-parent-self (degrees)
    dihedral        : torsion grandparent-parent-bonded-self (degrees)
    is_chi          : if True the dihedral is a chi rotamer angle that can
                      be overridden by the lattice bead direction.
    chi_index       : 1-based chi number (1 = chi1, 2 = chi2 ...) when
                      ``is_chi`` is True.
    """
    name: str
    reference_atoms: Tuple[str, str, str]
    bond_length: float
    bond_angle: float
    dihedral: float
    is_chi: bool = False
    chi_index: int = 0


@dataclass(frozen=True)
class ResidueTopology:
    """Full heavy-atom topology for one amino-acid type."""
    name3: str
    name1: str
    heavy_atoms: List[AtomPlacement] = field(default_factory=list)

    @property
    def has_sidechain(self) -> bool:
        return len(self.heavy_atoms) > 0


# ── Helper shortcuts ───────────────────────────────────────────────────────
def _ap(name: str, refs: Tuple[str, str, str], bl: float, ba: float, dh: float,
        chi: bool = False, chi_idx: int = 0) -> AtomPlacement:
    return AtomPlacement(name, refs, bl, ba, dh, chi, chi_idx)


# ── Per-residue topologies ─────────────────────────────────────────────────
# Backbone atoms (N, CA, C, O) are placed by backbone.py.
# Everything below starts from CB which is also placed by backbone.py.
# Reference atoms always use backbone names: "N", "CA", "C", "CB".

_TOPOLOGIES: Dict[str, ResidueTopology] = {}


def _register(name1: str, atoms: List[AtomPlacement]) -> None:
    name3 = ONE_TO_THREE[name1]
    _TOPOLOGIES[name1] = ResidueTopology(name3=name3, name1=name1, heavy_atoms=atoms)
    _TOPOLOGIES[name3] = _TOPOLOGIES[name1]


# GLY -- no sidechain (not even CB)
_register("G", [])

# ALA -- only CB (placed by backbone.py, nothing extra)
_register("A", [])

# VAL -- CB -> CG1, CG2
_register("V", [
    _ap("CG1", ("N", "CA", "CB"), 1.527, 110.7, 177.0, chi=True, chi_idx=1),
    _ap("CG2", ("N", "CA", "CB"), 1.527, 110.4, -63.0),
])

# LEU -- CB -> CG -> CD1, CD2
_register("L", [
    _ap("CG",  ("N",  "CA", "CB"), 1.530, 116.3, -65.0, chi=True, chi_idx=1),
    _ap("CD1", ("CA", "CB", "CG"), 1.524, 110.7, 175.0, chi=True, chi_idx=2),
    _ap("CD2", ("CA", "CB", "CG"), 1.524, 110.7, -65.0),
])

# ILE -- CB -> CG1 -> CD1, CG2
_register("I", [
    _ap("CG1", ("N",  "CA", "CB"), 1.530, 110.3, -65.0, chi=True, chi_idx=1),
    _ap("CG2", ("N",  "CA", "CB"), 1.527, 110.7,  175.0),
    _ap("CD1", ("CA", "CB", "CG1"), 1.513, 113.8, 170.0, chi=True, chi_idx=2),
])

# PRO -- CB -> CG -> CD (ring closure to N handled by builder)
_register("P", [
    _ap("CG", ("N",  "CA", "CB"), 1.492, 104.5,  29.6, chi=True, chi_idx=1),
    _ap("CD", ("CA", "CB", "CG"), 1.503, 105.5, -34.8, chi=True, chi_idx=2),
])

# PHE -- CB -> CG -> CD1, CD2 -> CE1, CE2 -> CZ (aromatic ring)
_register("F", [
    _ap("CG",  ("N",  "CA", "CB"), 1.510, 113.9, -65.0, chi=True, chi_idx=1),
    _ap("CD1", ("CA", "CB", "CG"), 1.391, 120.7,  90.0, chi=True, chi_idx=2),
    _ap("CD2", ("CA", "CB", "CG"), 1.391, 120.7, -90.0),
    _ap("CE1", ("CB", "CG", "CD1"), 1.393, 120.0, 180.0),
    _ap("CE2", ("CB", "CG", "CD2"), 1.393, 120.0, 180.0),
    _ap("CZ",  ("CG", "CD1", "CE1"), 1.389, 120.0,   0.0),
])

# TYR -- like PHE + OH
_register("Y", [
    _ap("CG",  ("N",  "CA", "CB"), 1.510, 113.9, -65.0, chi=True, chi_idx=1),
    _ap("CD1", ("CA", "CB", "CG"), 1.391, 120.7,  90.0, chi=True, chi_idx=2),
    _ap("CD2", ("CA", "CB", "CG"), 1.391, 120.7, -90.0),
    _ap("CE1", ("CB", "CG", "CD1"), 1.393, 120.0, 180.0),
    _ap("CE2", ("CB", "CG", "CD2"), 1.393, 120.0, 180.0),
    _ap("CZ",  ("CG", "CD1", "CE1"), 1.389, 120.0,   0.0),
    _ap("OH",  ("CD1", "CE1", "CZ"), 1.376, 120.0, 180.0),
])

# TRP -- indole ring system
_register("W", [
    _ap("CG",  ("N",  "CA", "CB"), 1.498, 113.6, -65.0, chi=True, chi_idx=1),
    _ap("CD1", ("CA", "CB", "CG"), 1.365, 126.9, 90.0, chi=True, chi_idx=2),
    _ap("CD2", ("CA", "CB", "CG"), 1.433, 126.8, -90.0),
    _ap("NE1", ("CB", "CG", "CD1"), 1.374, 110.2, 180.0),
    _ap("CE2", ("CB", "CG", "CD2"), 1.412, 107.2, 180.0),
    _ap("CE3", ("CG", "CD2", "CE2"), 1.400, 118.6, 180.0),
    _ap("CZ2", ("CD2", "CE2", "NE1"), 1.394, 122.4, 180.0),
    _ap("CZ3", ("CD2", "CE3", "CZ2"), 1.382, 121.5,   0.0),
    _ap("CH2", ("CE2", "CZ2", "CZ3"), 1.368, 117.5,   0.0),
])

# SER -- CB -> OG
_register("S", [
    _ap("OG", ("N", "CA", "CB"), 1.417, 111.1, 64.0, chi=True, chi_idx=1),
])

# THR -- CB -> OG1, CG2
_register("T", [
    _ap("OG1", ("N", "CA", "CB"), 1.430, 109.2, 64.0, chi=True, chi_idx=1),
    _ap("CG2", ("N", "CA", "CB"), 1.521, 111.5, -60.0),
])

# CYS -- CB -> SG
_register("C", [
    _ap("SG", ("N", "CA", "CB"), 1.808, 113.8, -65.0, chi=True, chi_idx=1),
])

# MET -- CB -> CG -> SD -> CE
_register("M", [
    _ap("CG", ("N",  "CA", "CB"), 1.520, 113.7, -65.0, chi=True, chi_idx=1),
    _ap("SD", ("CA", "CB", "CG"), 1.791, 112.7, -65.0, chi=True, chi_idx=2),
    _ap("CE", ("CB", "CG", "SD"), 1.791, 100.9, 180.0, chi=True, chi_idx=3),
])

# ASP -- CB -> CG -> OD1, OD2
_register("D", [
    _ap("CG",  ("N",  "CA", "CB"), 1.516, 113.1, -65.0, chi=True, chi_idx=1),
    _ap("OD1", ("CA", "CB", "CG"), 1.249, 119.2,   0.0, chi=True, chi_idx=2),
    _ap("OD2", ("CA", "CB", "CG"), 1.249, 118.4, 180.0),
])

# ASN -- CB -> CG -> OD1, ND2
_register("N", [
    _ap("CG",  ("N",  "CA", "CB"), 1.516, 113.4, -65.0, chi=True, chi_idx=1),
    _ap("OD1", ("CA", "CB", "CG"), 1.231, 120.9,   0.0, chi=True, chi_idx=2),
    _ap("ND2", ("CA", "CB", "CG"), 1.328, 116.5, 180.0),
])

# GLU -- CB -> CG -> CD -> OE1, OE2
_register("E", [
    _ap("CG",  ("N",  "CA", "CB"), 1.520, 113.8, -65.0, chi=True, chi_idx=1),
    _ap("CD",  ("CA", "CB", "CG"), 1.516, 113.1, 180.0, chi=True, chi_idx=2),
    _ap("OE1", ("CB", "CG", "CD"), 1.249, 119.0,   0.0, chi=True, chi_idx=3),
    _ap("OE2", ("CB", "CG", "CD"), 1.249, 118.4, 180.0),
])

# GLN -- CB -> CG -> CD -> OE1, NE2
_register("Q", [
    _ap("CG",  ("N",  "CA", "CB"), 1.520, 113.8, -65.0, chi=True, chi_idx=1),
    _ap("CD",  ("CA", "CB", "CG"), 1.516, 113.1, 180.0, chi=True, chi_idx=2),
    _ap("OE1", ("CB", "CG", "CD"), 1.231, 120.9,   0.0, chi=True, chi_idx=3),
    _ap("NE2", ("CB", "CG", "CD"), 1.328, 116.5, 180.0),
])

# LYS -- CB -> CG -> CD -> CE -> NZ
_register("K", [
    _ap("CG", ("N",  "CA", "CB"), 1.520, 113.8, -65.0, chi=True, chi_idx=1),
    _ap("CD", ("CA", "CB", "CG"), 1.520, 111.8, 180.0, chi=True, chi_idx=2),
    _ap("CE", ("CB", "CG", "CD"), 1.520, 111.8, 180.0, chi=True, chi_idx=3),
    _ap("NZ", ("CG", "CD", "CE"), 1.489, 112.0, 180.0, chi=True, chi_idx=4),
])

# ARG -- CB -> CG -> CD -> NE -> CZ -> NH1, NH2
_register("R", [
    _ap("CG",  ("N",  "CA", "CB"), 1.520, 113.8, -65.0, chi=True, chi_idx=1),
    _ap("CD",  ("CA", "CB", "CG"), 1.520, 111.8, 180.0, chi=True, chi_idx=2),
    _ap("NE",  ("CB", "CG", "CD"), 1.460, 111.8, 180.0, chi=True, chi_idx=3),
    _ap("CZ",  ("CG", "CD", "NE"), 1.329, 124.2, 180.0, chi=True, chi_idx=4),
    _ap("NH1", ("CD", "NE", "CZ"), 1.326, 120.0,   0.0),
    _ap("NH2", ("CD", "NE", "CZ"), 1.326, 120.0, 180.0),
])

# HIS -- CB -> CG -> ND1, CD2 -> CE1, NE2 (imidazole ring)
_register("H", [
    _ap("CG",  ("N",  "CA", "CB"), 1.497, 113.8, -65.0, chi=True, chi_idx=1),
    _ap("ND1", ("CA", "CB", "CG"), 1.371, 122.7, 90.0, chi=True, chi_idx=2),
    _ap("CD2", ("CA", "CB", "CG"), 1.354, 130.6, -90.0),
    _ap("CE1", ("CB", "CG", "ND1"), 1.322, 108.2, 180.0),
    _ap("NE2", ("CB", "CG", "CD2"), 1.374, 108.2, 180.0),
])


def get_topology(residue: str) -> ResidueTopology:
    """Look up topology by 1-letter or 3-letter code (case-insensitive)."""
    key = residue.strip().upper()
    if key in _TOPOLOGIES:
        return _TOPOLOGIES[key]
    raise KeyError(f"Unknown residue type: {residue!r}")


def get_all_topologies() -> Dict[str, ResidueTopology]:
    """Return mapping of 1-letter code -> ResidueTopology for all 20 standard aa."""
    return {code: _TOPOLOGIES[code] for code in ONE_TO_THREE}


__all__ = [
    "AtomPlacement",
    "ResidueTopology",
    "get_topology",
    "get_all_topologies",
    "ONE_TO_THREE",
    "THREE_TO_ONE",
    "N_CA_BOND",
    "CA_C_BOND",
    "C_O_BOND",
    "C_N_BOND",
    "CA_CB_BOND",
    "N_CA_C_ANGLE",
    "CA_C_N_ANGLE",
    "C_N_CA_ANGLE",
    "CA_C_O_ANGLE",
    "N_CA_CB_ANGLE",
    "C_CA_CB_ANGLE",
    "OMEGA_TRANS",
]
