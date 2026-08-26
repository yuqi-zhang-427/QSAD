# --*-- conding:utf-8 --*--
# @time:3/28/26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:__init__.py

"""
Calpha-to-all-atom reconstruction for coarse-grained lattice models.

This subpackage converts the tetrahedral lattice conformations produced by
QSAD quantum sampling into full heavy-atom protein structures.  The pipeline
runs four stages:

1. **Scale** -- rescale lattice CA coordinates to physical CA-CA bond
   distances (~3.8 A).
2. **Backbone** -- reconstruct backbone heavy atoms (N, CA, C, O) and CB
   using ideal peptide geometry and the NERF placement algorithm.
3. **Sidechain** -- build side-chain heavy atoms from a per-residue
   topology / rotamer library, optionally guided by the coarse-grained
   side-chain bead positions from the lattice model.
4. **Output** -- merge all atoms and write standard PDB format.

Quick start
-----------
>>> from post_process.ca2all import FullAtomBuilder, FullAtomReranker
>>> builder = FullAtomBuilder()
>>> result = builder.build(candidate)     # StructureCandidate from rerank
>>> result.write_pdb("model.pdb")
>>>
>>> # Full-atom reranking (after coarse rerank)
>>> reranker = FullAtomReranker()
>>> rerank_result = reranker.rerank(coarse_ranked_candidates)
>>> rerank_result.best.build_result.write_pdb("best.pdb")
"""

from .backbone import BackboneAtom, BackboneResult, nerf, reconstruct_backbone
from .builder import BuilderConfig, BuildResult, FullAtomBuilder
from .ca_smoother import smooth_ca_trace
from .ca_anchor_refine import CAAnchorRefineConfig, CAAnchorRefiner
from .constants import (
    AtomPlacement,
    ResidueTopology,
    get_topology,
    get_all_topologies,
    ONE_TO_THREE,
    THREE_TO_ONE,
)
from .fullatom_scorer import FullAtomScorer, FullAtomScoringConfig
from .openmm_scorer import OpenMMScorer, OpenMMScorerConfig
from .fullatom_reranker import (
    FullAtomReranker,
    FullAtomRerankConfig,
    FullAtomRerankResult,
    RankedFullAtomCandidate,
)
from .pdb_writer import FullAtom, atoms_from_reconstruction, write_pdb
from .sidechain import SidechainAtom, SidechainResult, build_sidechains
from .pipeline import Ca2AllPipeline, Ca2AllPipelineConfig

__all__ = [
    # Stage-3 pipeline entry point
    "Ca2AllPipeline",
    "Ca2AllPipelineConfig",
    # builder (main entry point)
    "BuilderConfig",
    "BuildResult",
    "FullAtomBuilder",
    # ca smoother
    "smooth_ca_trace",
    # ca anchor refinement
    "CAAnchorRefineConfig",
    "CAAnchorRefiner",
    # full-atom scoring & reranking
    "FullAtomScorer",
    "FullAtomScoringConfig",
    "OpenMMScorer",
    "OpenMMScorerConfig",
    "FullAtomReranker",
    "FullAtomRerankConfig",
    "FullAtomRerankResult",
    "RankedFullAtomCandidate",
    # backbone
    "BackboneAtom",
    "BackboneResult",
    "nerf",
    "reconstruct_backbone",
    # sidechain
    "SidechainAtom",
    "SidechainResult",
    "build_sidechains",
    # pdb output
    "FullAtom",
    "atoms_from_reconstruction",
    "write_pdb",
    # constants
    "AtomPlacement",
    "ResidueTopology",
    "get_topology",
    "get_all_topologies",
    "ONE_TO_THREE",
    "THREE_TO_ONE",
]
