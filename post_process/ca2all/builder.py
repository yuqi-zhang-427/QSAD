# --*-- conding:utf-8 --*--
# @time:3/28/26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:builder.py

"""
Orchestrator: lattice coarse-grained model --> full-atom structure.

Pipeline stages
---------------
1. **Scale**     -- rescale lattice CA positions to real CA-CA distance (~3.8 A).
2. **Backbone**  -- reconstruct N, CA, C, O, CB from the scaled CA trace.
3. **Sidechain** -- build full side-chain heavy atoms using topology + rotamers,
                    optionally guided by lattice side-chain bead positions.
4. **Output**    -- merge into ``FullAtom`` list, optionally write PDB.

Usage
-----
>>> from post_process.ca2all import FullAtomBuilder, BuilderConfig
>>> builder = FullAtomBuilder()
>>> result = builder.build(candidate)        # StructureCandidate
>>> result.write_pdb("output.pdb")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .backbone import BackboneResult, reconstruct_backbone
from .sidechain import SidechainResult, build_sidechains
from .pdb_writer import FullAtom, atoms_from_reconstruction, write_pdb
from .ca_smoother import smooth_ca_trace

_LOG = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────

@dataclass
class BuilderConfig:
    """
    Configuration for the CA-to-all-atom builder.

    Parameters
    ----------
    target_ca_distance : ideal CA-CA distance for scaling (default 3.8 A).
    use_lattice_chi1   : if True, override chi1 with lattice bead direction.
    chain_id           : PDB chain identifier.
    omega              : peptide bond dihedral (180 = trans, 0 = cis).
    """
    target_ca_distance: float = 3.8
    use_lattice_chi1: bool = True
    chain_id: str = "A"
    omega: float = 180.0
    smooth_ca: bool = False
    smooth_ca_sigma: float = 0.8


# ── Build result ───────────────────────────────────────────────────────────

@dataclass
class BuildResult:
    """
    Result of full-atom reconstruction for one structure.

    Holds all intermediate products (scaled coordinates, backbone, sidechains)
    and provides convenience methods for PDB output and downstream use.
    """
    sequence: str
    scaled_ca: np.ndarray
    backbone: BackboneResult
    sidechain: SidechainResult
    full_atoms: List[FullAtom]
    scale_factor: float
    config: BuilderConfig = field(default_factory=BuilderConfig)

    @property
    def n_atoms(self) -> int:
        return len(self.full_atoms)

    @property
    def n_residues(self) -> int:
        return len(self.sequence)

    def all_positions(self) -> np.ndarray:
        """Return (N_atoms, 3) array of all atom positions."""
        return np.array([a.position for a in self.full_atoms], dtype=float)

    def ca_positions(self) -> np.ndarray:
        """Return (N_res, 3) array of CA positions."""
        return self.scaled_ca.copy()

    def backbone_positions(self) -> np.ndarray:
        """Return (N_res, 4, 3) array of N, CA, C, O positions."""
        n_res = self.n_residues
        result = np.zeros((n_res, 4, 3))
        result[:, 0, :] = self.backbone.n_positions
        result[:, 1, :] = self.backbone.ca_positions
        result[:, 2, :] = self.backbone.c_positions
        result[:, 3, :] = self.backbone.o_positions
        return result

    def write_pdb(
        self,
        output: Union[str, Path, None] = None,
        title: Optional[str] = None,
    ) -> str:
        """Write the full-atom structure to PDB format."""
        if title is None:
            title = f"QSAD CA2ALL  {self.sequence[:20]}  {self.n_atoms} atoms"
        return write_pdb(
            self.full_atoms,
            output=output,
            title=title,
            chain_id=self.config.chain_id,
        )

    def to_record(self) -> Dict[str, Any]:
        """Serialize key information to a JSON-friendly dict."""
        return {
            "sequence": self.sequence,
            "n_residues": self.n_residues,
            "n_atoms": self.n_atoms,
            "scale_factor": float(self.scale_factor),
            "scaled_ca": self.scaled_ca.tolist(),
            "all_positions": self.all_positions().tolist(),
        }


# ── Builder ────────────────────────────────────────────────────────────────

class FullAtomBuilder:
    """
    Converts lattice coarse-grained structures to full-atom representations.

    Accepts either raw arrays (``build_from_arrays``) or ``StructureCandidate``
    objects (``build``).  The latter integrates directly with the rerank pipeline.
    """

    def __init__(self, config: Optional[BuilderConfig] = None) -> None:
        self.config = config or BuilderConfig()

    # ── Core build from arrays ─────────────────────────────────────────

    def build_from_arrays(
        self,
        main_positions: np.ndarray,
        sequence: str,
        side_positions: Optional[List[Optional[np.ndarray]]] = None,
    ) -> BuildResult:
        """
        Full pipeline: scale -> backbone -> sidechain -> merge.

        Parameters
        ----------
        main_positions : (N, 3) lattice CA positions (un-scaled).
        sequence       : 1-letter amino-acid sequence.
        side_positions : per-residue coarse-grained side-chain bead positions.
        """
        ca = np.asarray(main_positions, dtype=float)
        seq = sequence.upper()

        # ── Stage 1: Scale ──────────────────────────────────────────
        scaled_ca, scale_factor = self._scale(ca)
        _LOG.info(
            "Scaled %d CA positions (factor=%.3f, median step=%.2f A).",
            len(ca), scale_factor, self.config.target_ca_distance,
        )

        # Scale side-chain positions with the same factor
        scaled_sides: Optional[List[Optional[np.ndarray]]] = None
        if side_positions is not None:
            scaled_sides = []
            for sp in side_positions:
                if sp is None:
                    scaled_sides.append(None)
                else:
                    scaled_sides.append(np.asarray(sp, dtype=float) * scale_factor)

        # ── Stage 1b: CA smoothing (optional) ───────────────────────
        if self.config.smooth_ca and len(scaled_ca) > 2:
            smoothed_ca = smooth_ca_trace(
                scaled_ca,
                sigma=self.config.smooth_ca_sigma,
                target_distance=self.config.target_ca_distance,
            )
            delta = float(np.sqrt(np.mean(np.sum((smoothed_ca - scaled_ca) ** 2, axis=1))))
            _LOG.info("CA smoothing: mean displacement=%.3f A (sigma=%.1f).", delta, self.config.smooth_ca_sigma)
            scaled_ca = smoothed_ca

        # ── Stage 2: Backbone ───────────────────────────────────────
        backbone = reconstruct_backbone(
            scaled_ca, sequence=seq, omega=self.config.omega,
        )
        _LOG.info("Backbone: %d atoms placed.", len(backbone.atoms))

        # ── Stage 3: Side chains ────────────────────────────────────
        sidechain = build_sidechains(
            backbone,
            sequence=seq,
            side_positions=scaled_sides,
            use_lattice_chi1=self.config.use_lattice_chi1,
        )
        _LOG.info("Sidechains: %d atoms placed.", len(sidechain.atoms))

        # ── Stage 4: Merge ──────────────────────────────────────────
        full_atoms = atoms_from_reconstruction(
            backbone.atoms,
            sidechain.atoms,
            sequence=seq,
            chain_id=self.config.chain_id,
        )
        _LOG.info("Full-atom model: %d total heavy atoms.", len(full_atoms))

        return BuildResult(
            sequence=seq,
            scaled_ca=scaled_ca,
            backbone=backbone,
            sidechain=sidechain,
            full_atoms=full_atoms,
            scale_factor=scale_factor,
            config=self.config,
        )

    # ── Build from StructureCandidate ──────────────────────────────────

    def build(self, candidate: Any) -> BuildResult:
        """
        Build full-atom structure from a ``StructureCandidate``.

        Reads ``main_positions``, ``sequence``, and ``side_positions`` from
        the candidate object (duck-typed for loose coupling).
        """
        main_positions = np.asarray(candidate.main_positions, dtype=float)
        sequence = str(getattr(candidate, "sequence", ""))
        side_positions = getattr(candidate, "side_positions", None)

        if not sequence:
            raise ValueError("Candidate has no sequence -- cannot build full-atom model.")

        return self.build_from_arrays(
            main_positions=main_positions,
            sequence=sequence,
            side_positions=side_positions,
        )

    # ── Batch build ────────────────────────────────────────────────────

    def build_many(self, candidates: List[Any]) -> List[BuildResult]:
        """Build full-atom structures for a list of candidates."""
        results: List[BuildResult] = []
        for idx, cand in enumerate(candidates):
            try:
                results.append(self.build(cand))
            except Exception as exc:
                _LOG.warning("Failed to build candidate %d: %s", idx, exc)
        return results

    # ── Internal helpers ───────────────────────────────────────────────

    def _scale(self, ca: np.ndarray) -> tuple:
        """Scale CA positions so median step equals target_ca_distance."""
        if len(ca) <= 1:
            return ca.copy(), 1.0
        steps = np.linalg.norm(ca[1:] - ca[:-1], axis=1)
        median_step = float(np.median(steps))
        if median_step < 1e-8:
            _LOG.warning("Degenerate CA trace (zero step size), skipping scaling.")
            return ca.copy(), 1.0
        factor = self.config.target_ca_distance / median_step
        return ca * factor, factor


__all__ = [
    "BuilderConfig",
    "BuildResult",
    "FullAtomBuilder",
]
