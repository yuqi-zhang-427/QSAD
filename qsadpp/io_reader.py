# --*-- conding:utf-8 --*--
# @time:10/23/25 16:26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:io_reader.py

from __future__ import annotations

import os
import glob
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple, Any

import pandas as pd
import numpy as np


# Required schema inferred from your sample
REQUIRED_COLUMNS = [
    "L", "n_qubits", "shots", "beta", "seed", "label", "backend",
    "ibm_backend", "circuit_hash", "protein", "sequence", "group_id",
    "bitstring", "count", "prob",
]


def _default_dtype_map(as_float32: bool = True) -> Dict[str, Any]:
    """Return a dtype map optimized for memory while remaining safe."""
    flt = np.float32 if as_float32 else np.float64
    return {
        "L": np.int32,
        "n_qubits": np.int32,
        "shots": np.int32,
        "beta": flt,
        "seed": np.int32,
        "label": "string",
        "backend": "string",
        "ibm_backend": "string",
        "circuit_hash": "string",
        "protein": "string",
        "sequence": "string",
        "group_id": np.int32,
        "bitstring": "string",
        "count": np.int32,
        "prob": flt,
    }


def _categorize_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Convert repetitive string columns to Categorical to save memory."""
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def _validate_columns(df: pd.DataFrame, strict: bool) -> List[str]:
    """Ensure required columns exist; return missing list or raise if strict."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing and strict:
        raise ValueError(f"Missing required columns: {missing}")
    return missing


def _validate_and_fix_chunk(
    df: pd.DataFrame,
    strict: bool,
    bitlen_check: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Basic integrity checks and light normalization for a chunk.
    - Drop fully NA rows.
    - Ensure positive counts/prob (non-negative).
    - Optionally check bitstring length == n_qubits.
    - Coalesce duplicates within the chunk.
    """
    meta: Dict[str, Any] = {
        "dropped_na_rows": 0,
        "neg_count_rows": 0,
        "neg_prob_rows": 0,
        "bitlen_mismatch_rows": 0,
        "dedup_pairs": 0,
    }

    # Drop fully NA rows
    before = len(df)
    df = df.dropna(how="all")
    meta["dropped_na_rows"] = before - len(df)

    # Enforce non-negative count/prob (drop bad rows if not strict)
    if "count" in df.columns:
        mask_neg = df["count"] < 0
        if mask_neg.any():
            meta["neg_count_rows"] = int(mask_neg.sum())
            if strict:
                raise ValueError("Negative 'count' encountered.")
            df = df.loc[~mask_neg]
    if "prob" in df.columns:
        mask_negp = df["prob"] < 0
        if mask_negp.any():
            meta["neg_prob_rows"] = int(mask_negp.sum())
            if strict:
                raise ValueError("Negative 'prob' encountered.")
            df = df.loc[~mask_negp]

    # Optional: bitstring length check
    if bitlen_check and {"bitstring", "n_qubits"}.issubset(df.columns):
        # Vectorized length check using .str.len()
        bl = df["bitstring"].str.len()
        mismatch = bl != df["n_qubits"]
        if mismatch.any():
            meta["bitlen_mismatch_rows"] = int(mismatch.sum())
            if strict:
                raise ValueError("Bitstring length does not match n_qubits.")
            df = df.loc[~mismatch]

    # Coalesce duplicates by (protein, label, group_id, bitstring)
    key_cols = ["protein", "label", "group_id", "bitstring"]
    have_cols = [c for c in key_cols if c in df.columns]
    if len(have_cols) == len(key_cols):
        # Count duplicates
        dup_count = int(df.duplicated(subset=key_cols).sum())
        meta["dedup_pairs"] = dup_count

        # Aggregate counts/prob conservatively:
        # - sum counts
        # - re-derive prob if shots available (count/shots)
        # - keep first metadata row for the rest columns
        agg_spec = {}
        for c in df.columns:
            if c in key_cols:
                continue
            if c == "count":
                agg_spec[c] = "sum"
            elif c == "prob":
                # temporary placeholder; recompute after groupby if shots present
                agg_spec[c] = "first"
            else:
                agg_spec[c] = "first"
        if dup_count > 0:
            df = df.groupby(key_cols, as_index=False, sort=False).agg(agg_spec)
            if "shots" in df.columns and "count" in df.columns:
                with np.errstate(divide="ignore", invalid="ignore"):
                    df["prob"] = np.where(
                        (df["shots"] > 0) & (df["count"] >= 0),
                        df["count"] / df["shots"],
                        df.get("prob", 0.0),
                    )

    return df, meta


@dataclass
class ReaderOptions:
    chunksize: int = 100_000
    as_float32: bool = True
    strict: bool = False
    categorize_strings: bool = True
    include_all_csv: bool = False  # whether to include *_all_*.csv alongside group files
    bitlen_check: bool = True
    # columns that we will keep for downstream "string-based reconstruction"
    keep_columns: Tuple[str, ...] = (
        "protein", "label", "group_id", "sequence",
        "n_qubits", "bitstring", "count", "shots", "beta",
        "L", "seed", "backend", "ibm_backend", "circuit_hash", "prob",
    )


@dataclass
class SampleReader:
    """
    Stream CSVs under a pdb-specific directory and yield per-group DataFrames.
    It assumes that all CSVs under the directory share the same_squences amino-acid sequence.
    """
    pdb_dir: str
    options: ReaderOptions = field(default_factory=ReaderOptions)

    def _list_csv_files(self) -> List[str]:
        """
        List CSV files to be processed under pdb_dir.
        Excludes *_all_*.csv by default to avoid double-counting when group files exist.
        """
        pattern = os.path.join(self.pdb_dir, "*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            return []
        if self.options.include_all_csv:
            return files
        # Exclude *_all_*.csv if group files exist
        group_like = [f for f in files if "_group" in os.path.basename(f)]
        if len(group_like) > 0:
            files = [f for f in files if "_all_" not in os.path.basename(f)]
        return files

    def _read_csv_stream(self, path: str) -> Iterator[pd.DataFrame]:
        """
        Read a CSV file in chunks. Apply dtype map, keep only columns necessary,
        and perform light validation and normalization per chunk.
        """
        dtype_map = _default_dtype_map(self.options.as_float32)

        # Read in chunks to control memory
        reader = pd.read_csv(
            path,
            chunksize=self.options.chunksize,
            dtype=dtype_map,
            keep_default_na=True,
        )

        for chunk in reader:
            # Ensure required columns exist (soft check unless strict)
            missing = _validate_columns(chunk, self.options.strict)
            if missing:
                # Keep going with what's available if not strict
                pass

            # Optional: down-select columns we care about for downstream
            keep_cols = [c for c in self.options.keep_columns if c in chunk.columns]
            if len(keep_cols) < len(self.options.keep_columns) and self.options.strict:
                raise ValueError(f"Some required keep_columns are missing in {path}.")
            chunk = chunk[keep_cols].copy()

            # Light normalization and validation
            chunk, meta = _validate_and_fix_chunk(
                chunk,
                strict=self.options.strict,
                bitlen_check=self.options.bitlen_check,
            )

            # Optionally categorize repetitive strings for memory saving
            if self.options.categorize_strings:
                chunk = _categorize_columns(
                    chunk,
                    cols=["label", "backend", "ibm_backend", "protein", "sequence", "circuit_hash"],
                )

            # Attach source for traceability
            chunk["source_id"] = os.path.basename(path)

            yield chunk

    def iter_groups(
        self,
    ) -> Iterator[Tuple[Tuple[str, str, int], pd.DataFrame, Dict[str, Any]]]:
        """
        Iterate over (protein, label, group_id) groups across all CSVs in the pdb_dir.
        Yields (group_key, group_df, group_meta) where:
          - group_key = (protein, label, group_id)
          - group_df contains only rows for that group
          - group_meta provides light integrity information
        """
        files = self._list_csv_files()
        if not files:
            return  # nothing to yield

        # Buffer per (protein, label, group_id)
        buffers: Dict[Tuple[str, str, int], List[pd.DataFrame]] = {}
        counts: Dict[Tuple[str, str, int], Dict[str, int]] = {}

        def flush_group(key: Tuple[str, str, int]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
            parts = buffers.pop(key)
            df = pd.concat(parts, axis=0, ignore_index=True)
            # Compute group-level meta
            meta = {
                "rows": int(len(df)),
                "unique_bitstrings": int(df["bitstring"].nunique()) if "bitstring" in df.columns else int(len(df)),
                "sum_count": int(df["count"].sum()) if "count" in df.columns else 0,
                "shots": int(df["shots"].iloc[0]) if "shots" in df.columns and len(df) > 0 else -1,
                "source_files": sorted(df["source_id"].astype(str).unique().tolist()) if "source_id" in df.columns else [],
            }
            return df, meta

        # Stream all files
        for f in files:
            for chunk in self._read_csv_stream(f):
                if not {"protein", "label", "group_id"}.issubset(chunk.columns):
                    if self.options.strict:
                        raise ValueError("Missing one of required grouping columns: protein/label/group_id.")
                    # Skip if we cannot group
                    continue

                # Group within the chunk by (protein, label, group_id)
                gb = chunk.groupby(["protein", "label", "group_id"], sort=False, observed=False)
                for key, part in gb:
                    # Initialize buffer and counts
                    if key not in buffers:
                        buffers[key] = []
                        counts[key] = {"rows": 0}
                    buffers[key].append(part)
                    counts[key]["rows"] += len(part)

                    # Heuristic flush: if a group has reached ~24k rows, emit it
                    if counts[key]["rows"] >= 24000:
                        df, meta = flush_group(key)
                        counts.pop(key, None)
                        yield key, df, meta

        # Flush remaining groups
        for key in list(buffers.keys()):
            df, meta = flush_group(key)
            yield key, df, meta

    def read_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Read all CSVs and return:
          - a concatenated DataFrame across all groups,
          - a summary DataFrame with one row per (protein, label, group_id).
        """
        frames: List[pd.DataFrame] = []
        summary_rows: List[Dict[str, Any]] = []

        for (protein, label, gid), df, meta in self.iter_groups():
            frames.append(df)
            summary_rows.append({
                "protein": str(protein),
                "label": str(label),
                "group_id": int(gid),
                **meta,
            })

        if not frames:
            return pd.DataFrame(columns=list(self.options.keep_columns) + ["source_id"]), pd.DataFrame(
                columns=["protein", "label", "group_id", "rows", "unique_bitstrings", "sum_count", "shots", "source_files"]
            )

        all_df = pd.concat(frames, axis=0, ignore_index=True)
        summary_df = pd.DataFrame(summary_rows)
        return all_df, summary_df

    def schema(self) -> Dict[str, Any]:
        """Return the dtype map used for CSV reading."""
        return _default_dtype_map(self.options.as_float32)
