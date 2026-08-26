# --*-- conding:utf-8 --*--
# @time:10/23/25 17:20
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:orchestrator.py

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import pandas as pd

from qsadpp.io_reader import SampleReader, ReaderOptions
from qsadpp.coordinate_decoder import CoordinateBatchDecoder, CoordinateDecoderConfig
from qsadpp.energy_calculator import LatticeEnergyCalculator, EnergyConfig

# NEW: structural features (geometry + pseudo atoms)
try:
    from qsadpp.feature_calculator import FeatureConfig, StructuralFeatureCalculator
    _FEATURES_AVAILABLE = True
except Exception:
    _FEATURES_AVAILABLE = False

_LOG = logging.getLogger(__name__)
if not _LOG.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class OrchestratorConfig:
    """
    End-to-end pipeline:
      read -> per-group decode -> per-group energy -> aggregate per PDB -> (optional) per-group features -> aggregate.

    Notes:
      - We assume all CSVs under `pdb_dir` belong to the same_squences protein (pdbid).
      - For main-chain only, side_chain_hot_builder defaults to all-False of length N.
    """
    # Reader
    pdb_dir: str
    reader_options: ReaderOptions = field(default_factory=ReaderOptions)

    # Encoding options (main-chain only default)
    fifth_bit: bool = False
    side_chain_hot_builder: Optional[Callable[[str], list[bool]]] = None  # input: sequence -> list[bool]

    # Output directories and templates
    out_dir: str = "results"
    # Per-group intermediate file templates
    decoded_tpl: str = "{protein}_{label}_g{gid}.decoded.jsonl"
    energy_tpl: str = "{protein}_{label}_g{gid}.energy.jsonl"

    # Aggregation options
    aggregate_only: bool = True  # if True, keep only combined files and remove group files
    combined_names: Tuple[str, str] = ("decoded.jsonl", "energies.jsonl")  # (decoded, energy)

    # Energy defaults (tetrahedral lattice)
    energy_config: EnergyConfig = field(
        default_factory=lambda: EnergyConfig(
            r_min=0.5,
            r_contact=1.0,
            d0=0.57735,
            lambda_overlap=1000.0,
            weights={"steric": 1.0, "geom": 0.5, "bond": 0.2, "mj": 1.0},
            normalize=True,
            output_path="__unused__.jsonl",
        )
    )

    # Decode defaults (bitstring->turns->coords)
    decoder_output_format: str = "jsonl"  # "jsonl" | "parquet"
    strict_decode: bool = False

    # ---------- NEW: feature calculation controls ----------
    compute_features: bool = False                 # turn on to compute extra geometry/statistics features
    feature_from: str = "decoded"                  # "decoded" or "energies" as input to feature calculator
    feature_tpl: str = "{protein}_{label}_g{gid}.features.jsonl"  # per-group features file
    combined_feature_name: str = "features.jsonl"  # aggregated features filename under <out_dir>/<pdbid>

    # pass-through config to StructuralFeatureCalculator (safe defaults if you don't touch)
    feature_config: Optional["FeatureConfig"] = None


class PipelineOrchestrator:
    """
    Orchestrates:
      - stream groups from SampleReader
      - decode each group's bitstrings to coordinates
      - evaluate energies on decoded coordinates
      - optionally compute extra features on decoded/energies
      - append all groups to <out_dir>/<pdbid>/{decoded.jsonl, energies.jsonl, features.jsonl}
      - optionally remove per-group intermediate files
    """

    def __init__(self, cfg: OrchestratorConfig):
        self.cfg = cfg
        _ensure_dir(self.cfg.out_dir)

        # Reader
        self.reader = SampleReader(self.cfg.pdb_dir, self.cfg.reader_options)

        # Will be initialized after seeing the first group (we need pdbid)
        self.decoded_all_path: Optional[str] = None
        self.energy_all_path: Optional[str] = None
        self.features_all_path: Optional[str] = None
        self._pdb_root: Optional[str] = None

        # Reusable energy calculator (avoid reloading MJ per group)
        self.energy_calc = LatticeEnergyCalculator(self.cfg.energy_config)

        # Optional decoder cache keyed by sequence
        self._decoder_cache: Dict[str, CoordinateBatchDecoder] = {}

        # NEW: features calculator (lazy set after we know output_path)
        self.feat_calc: Optional["StructuralFeatureCalculator"] = None

        if self.cfg.compute_features and not _FEATURES_AVAILABLE:
            _LOG.warning("Feature calculator not available: qsadpp.feature_calculator import failed. "
                         "Set compute_features=False or ensure the module exists.")

    # ---------- helpers ----------

    def _infer_side_chain_hot(self, sequence: str) -> list[bool]:
        """Default: main-chain only => all False, length=len(sequence)."""
        if self.cfg.side_chain_hot_builder is not None:
            return list(self.cfg.side_chain_hot_builder(sequence))
        return [False] * len(sequence)

    def _build_decoder(self, sequence: str) -> CoordinateBatchDecoder:
        """Get or build a decoder for this sequence length."""
        if sequence not in self._decoder_cache:
            hot_vec = self._infer_side_chain_hot(sequence)
            dec_cfg = CoordinateDecoderConfig(
                side_chain_hot_vector=hot_vec,
                fifth_bit=self.cfg.fifth_bit,
                output_format=self.cfg.decoder_output_format,
                output_path="__set_per_group__",  # will be set per group
                bitstring_col="bitstring",
                sequence_col="sequence",
                strict=self.cfg.strict_decode,
            )
            self._decoder_cache[sequence] = CoordinateBatchDecoder(dec_cfg)
        return self._decoder_cache[sequence]

    def _group_paths(self, protein: str, label: str, gid: int) -> Tuple[str, str, str]:
        """Per-group intermediate files (kept or removed depending on `aggregate_only`)."""
        decoded_name = self.cfg.decoded_tpl.format(protein=protein, label=label, gid=gid)
        energy_name  = self.cfg.energy_tpl.format(protein=protein, label=label, gid=gid)
        feature_name = self.cfg.feature_tpl.format(protein=protein, label=label, gid=gid)
        return (
            os.path.join(self.cfg.out_dir, decoded_name),
            os.path.join(self.cfg.out_dir, energy_name),
            os.path.join(self.cfg.out_dir, feature_name),
        )

    def _init_pdb_aggregate_paths(self, pdbid: str) -> None:
        """Create <out_dir>/<pdbid>/ and prepare combined files."""
        self._pdb_root = os.path.join(self.cfg.out_dir, pdbid)
        _ensure_dir(self._pdb_root)
        decoded_name, energy_name = self.cfg.combined_names
        self.decoded_all_path = os.path.join(self._pdb_root, decoded_name)
        self.energy_all_path  = os.path.join(self._pdb_root, energy_name)
        # truncate (start fresh)
        open(self.decoded_all_path, "w", encoding="utf-8").close()
        open(self.energy_all_path, "w", encoding="utf-8").close()

        # NEW: aggregated features file (only if needed)
        if self.cfg.compute_features:
            self.features_all_path = os.path.join(self._pdb_root, self.cfg.combined_feature_name)
            open(self.features_all_path, "w", encoding="utf-8").close()

            # init feature calculator with final aggregated output path context
            if _FEATURES_AVAILABLE:
                fcfg = self.cfg.feature_config or FeatureConfig()
                # do not set output_path here; we will set per-group file below
                self.feat_calc = StructuralFeatureCalculator(fcfg)

    @staticmethod
    def _append_file(from_path: str, to_path: Optional[str]) -> None:
        """Append a JSONL file into a combined JSONL (line by line)."""
        if not to_path or not os.path.exists(from_path):
            return
        with open(from_path, "r", encoding="utf-8") as fin, open(to_path, "a", encoding="utf-8") as fout:
            for line in fin:
                fout.write(line)

    # ---------- main API ----------

    def run(self) -> Dict[str, int | str]:
        """
        Execute the full pipeline over all groups yielded by SampleReader.
        Returns summary with counts and combined file paths.
        """
        groups = 0
        decoded_rows_total = 0
        energy_rows_total = 0
        feature_rows_total = 0
        pdb_initialized = False

        for (protein, label, gid), df, meta in self.reader.iter_groups():
            groups += 1
            if len(df) == 0:
                _LOG.warning("Group (%s, %s, %s) is empty; skipping.", protein, label, gid)
                continue

            # Initialize combined files under <out_dir>/<pdbid> on first group
            if not pdb_initialized:
                self._init_pdb_aggregate_paths(protein)
                pdb_initialized = True

            if "sequence" not in df.columns:
                _LOG.warning("Group (%s, %s, %s) has no 'sequence' column; skipping.", protein, label, gid)
                continue

            # Prepare decoder for this sequence
            seq0 = str(df["sequence"].iloc[0])
            decoder = self._build_decoder(seq0)

            # Per-group intermediate paths
            decoded_path, energy_path, feature_path = self._group_paths(protein, label, gid)
            _ensure_dir(os.path.dirname(decoded_path))
            _ensure_dir(os.path.dirname(energy_path))
            _ensure_dir(os.path.dirname(feature_path))

            # ---- Decode ----
            in_df = df[[decoder.cfg.bitstring_col, decoder.cfg.sequence_col]].copy()
            decoder.cfg.output_path = decoded_path
            decoder.cfg.output_format = self.cfg.decoder_output_format

            _LOG.info("Decoding group (%s, %s, %s) -> %s", protein, label, gid, decoded_path)
            dec_summary = decoder.decode_and_save(in_df)
            decoded_rows_total += int(dec_summary.get("written", 0))

            # Append to combined decoded file
            self._append_file(decoded_path, self.decoded_all_path)

            # ---- Energy ----
            self.energy_calc.cfg.output_path = energy_path
            _LOG.info("Computing energies for group (%s, %s, %s) -> %s", protein, label, gid, energy_path)
            en_summary = self.energy_calc.evaluate_jsonl(decoded_path, energy_path)
            energy_rows_total += int(en_summary.get("written", 0))

            # Append to combined energies file
            self._append_file(energy_path, self.energy_all_path)

            # ---- NEW: Features (optional) ----
            if self.cfg.compute_features and self.feat_calc is not None:
                # choose source (decoded or energies)
                src_path = decoded_path if self.cfg.feature_from == "decoded" else energy_path
                # set per-group output
                self.feat_calc.cfg.output_path = feature_path
                _LOG.info(
                    "Computing features for group (%s, %s, %s) from %s -> %s",
                    protein, label, gid, self.cfg.feature_from, feature_path
                )
                fsum = self.feat_calc.evaluate_jsonl(src_path, output_path=feature_path)
                feature_rows_total += int(fsum.get("written", 0))

                # append to combined features
                self._append_file(feature_path, self.features_all_path)

            # Optionally remove intermediate files
            if self.cfg.aggregate_only:
                # Remove decoded/energy group files
                for p in (decoded_path, energy_path):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
                # Remove feature group file if created
                if self.cfg.compute_features:
                    try:
                        os.remove(feature_path)
                    except OSError:
                        pass

        _LOG.info(
            "Pipeline finished: %d groups, %d decoded rows, %d energy rows%s. Combined: %s, %s%s",
            groups, decoded_rows_total, energy_rows_total,
            (f", {feature_rows_total} feature rows" if self.cfg.compute_features else ""),
            self.decoded_all_path, self.energy_all_path,
            (f", {self.features_all_path}" if self.cfg.compute_features else "")
        )
        summary: Dict[str, int | str] = {
            "groups": groups,
            "decoded_rows": decoded_rows_total,
            "energy_rows": energy_rows_total,
            "decoded_all": self.decoded_all_path or "",
            "energies_all": self.energy_all_path or "",
        }
        if self.cfg.compute_features:
            summary["feature_rows"] = feature_rows_total
            summary["features_all"] = self.features_all_path or ""
        return summary
