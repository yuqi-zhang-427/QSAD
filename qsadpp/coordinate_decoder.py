# --*-- conding:utf-8 --*--
# @time:10/23/25 16:52
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:coordinate_decoder.py

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency at import time
    pd = None  # type: ignore[assignment]


_LOG = logging.getLogger(__name__)
if not _LOG.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )


# =========================
# Decoding rules (strict):
# =========================
# 1) Bitstring -> turn indices (0..3) by reading bit-pairs after reversing the bitstring.
# 2) Main-chain effective bits: 2*(N-3) minus 1 bit if `fifth_bit=True`.
#    Then append trailer "0010" (turn1=(01), turn2=(00)).
#    If `fifth_bit=True`, set the 6th bit from the end of the finalized main-turn bitstring to '1'.
# 3) Side-chain bits: 2 bits per True in side_chain_hot_vector; slice is placed before the main bits.
# 4) Coordinates use a tetrahedral lattice with unit vectors scaled by 1/sqrt(3);
#    main positions are cumulative sums of (-1)^i * COORD[turn_i] over i,
#    side position at bead k (1-based counter in original) is main_pos[k-1] + (-1)^k * COORD[side_turn].


def _bit_pairs_to_ints(bitstring: str) -> List[int]:
    """Decode reversed-pair mapping: '00'->0, '01'->1, '10'->2, '11'->3."""
    s = bitstring[::-1]
    enc = {"00": 0, "01": 1, "10": 2, "11": 3}
    n = len(s) // 2
    return [enc[s[2 * i : 2 * (i + 1)]] for i in range(n)]


@dataclass(frozen=True)
class ShapeDecodeSpec:
    side_chain_hot_vector: List[bool]
    fifth_bit: bool


class _ShapeDecoder:
    """
    Internal decoder that strictly follows the provided Apache-2.0 logic.
    Produces main_vectors (List[int]) and side_vectors (List[Optional[int]]).
    """

    def __init__(self, vector_sequence: str, spec: ShapeDecodeSpec) -> None:
        self._vector_sequence = vector_sequence
        self._hot = spec.side_chain_hot_vector
        self._fifth = spec.fifth_bit
        self._N = len(self._hot)

    def _split_bits(self) -> Tuple[int, int]:
        n_main = 2 * (self._N - 3) - (1 if self._fifth else 0)
        n_side = 2 * sum(self._hot)
        return n_main, n_side

    def main_vectors(self) -> List[int]:
        n_main, _ = self._split_bits()
        # last n_main bits for main turns + fixed trailer "0010"
        main_bits = self._vector_sequence[-n_main:] + "0010"
        # set 6th bit from the end to '1' if fifth_bit is True (compatibility rule)
        if self._fifth and len(main_bits) >= 6:
            main_bits = main_bits[:-5] + "1" + main_bits[-5:]
        return _bit_pairs_to_ints(main_bits)

    def side_vectors(self) -> List[Optional[int]]:
        n_main, n_side = self._split_bits()
        side_bits = self._vector_sequence[-(n_main + n_side) : -n_main] if n_side > 0 else ""
        side_turns = _bit_pairs_to_ints(side_bits) if side_bits else []
        out: List[Optional[int]] = []
        k = 0
        for hot in self._hot:
            if hot:
                out.append(side_turns[k] if k < len(side_turns) else None)
                k += 1
            else:
                out.append(None)
        return out


class _LatticeKinematics:
    """
    Tetrahedral lattice kinematics used by the given FileGen code.
    Provides main/side positions from turn indices.
    """

    # Coordinates of the 4 edges of a tetrahedron centered at 0; normalized.
    COORD = (1.0 / np.sqrt(3.0)) * np.array(
        [[-1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, -1.0], [1.0, -1.0, 1.0]],
        dtype=float,
    )

    @staticmethod
    def main_positions(main_turns: List[int]) -> np.ndarray:
        """
        Generate main-chain positions as cumulative sum of step vectors.
        For turn i (0-based), step = (-1)^i * COORD[turn_i].
        Returns positions for N beads where N = len(main_turns) + 1.
        """
        L = len(main_turns)
        rel = np.zeros((L + 1, 3), dtype=float)
        for i, t in enumerate(main_turns):
            rel[i + 1] = ((-1.0) ** i) * _LatticeKinematics.COORD[int(t)]
        return rel.cumsum(axis=0)

    @staticmethod
    def side_positions(side_turns: List[Optional[int]], main_pos: np.ndarray) -> List[Optional[np.ndarray]]:
        """
        For bead index k (1-based counter in the original implementation),
        side position is: main_pos[k-1] + (-1)^k * COORD[side_turn].
        If side_turn is None, output None.
        """
        out: List[Optional[np.ndarray]] = []
        counter = 1
        for bead_idx, t in enumerate(side_turns):
            if t is None:
                out.append(None)
            else:
                base = main_pos[bead_idx]  # position of the main bead
                step = ((-1.0) ** counter) * _LatticeKinematics.COORD[int(t)]
                out.append(base + step)
            counter += 1
        return out


@dataclass
class CoordinateDecoderConfig:
    """
    Configuration for the batch decoder.
    """
    side_chain_hot_vector: List[bool]
    fifth_bit: bool
    output_format: str = "jsonl"  # "jsonl" or "parquet"
    output_path: str = "decoded_coordinates.jsonl"
    # Column names expected in input DataFrame:
    bitstring_col: str = "bitstring"
    sequence_col: str = "sequence"  # kept for traceability; not used for geometry
    # Behavior on bad rows:
    strict: bool = False  # if True, raise; else warn and skip
    # Optional: limit rows (for debugging)
    max_rows: Optional[int] = None


@dataclass
class CoordinateBatchDecoder:
    """
    End-to-end decoder:
      DataFrame rows (bitstrings) -> turn sequences -> 3D coordinates,
      then save one consolidated output file with one record per input row.
    """
    cfg: CoordinateDecoderConfig = field(default_factory=CoordinateDecoderConfig)

    # ---------- public API ----------

    def decode_and_save(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Decode all valid rows and save to a single file (JSONL or Parquet).
        Returns a small summary dict.
        """
        if pd is None:
            raise ImportError(
                "pandas is required for decode_and_save(), but it could not be imported in this environment."
            )
        records: List[Dict[str, Any]] = []
        processed = 0
        skipped = 0

        required_cols = [self.cfg.bitstring_col]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            msg = f"Missing required columns in input DataFrame: {missing}"
            if self.cfg.strict:
                raise ValueError(msg)
            _LOG.warning(msg + " — all rows skipped.")
            return {"written": 0, "processed": 0, "skipped": int(len(df)), "path": self.cfg.output_path}

        iterable = df.itertuples(index=False)
        for row in iterable:
            if self.cfg.max_rows is not None and processed >= self.cfg.max_rows:
                break
            # fetch fields safely by column name
            row_dict = row._asdict() if hasattr(row, "_asdict") else None
            if row_dict is None:
                # fallback for pandas namedtuples (Python <3.8)
                row_dict = {col: getattr(row, col) for col in df.columns}

            bitstring = str(row_dict[self.cfg.bitstring_col]) if row_dict.get(self.cfg.bitstring_col) is not None else None
            sequence = str(row_dict.get(self.cfg.sequence_col, "")) if self.cfg.sequence_col in df.columns else ""

            if not bitstring or not set(bitstring) <= {"0", "1"}:
                msg = "Invalid or empty bitstring encountered; row skipped."
                if self.cfg.strict:
                    raise ValueError(msg)
                _LOG.warning(msg)
                skipped += 1
                processed += 1
                continue

            rec = self._decode_one(bitstring=bitstring, sequence=sequence)
            if rec is None:
                skipped += 1
            else:
                records.append(rec)
            processed += 1

        written = self._save_records(records)
        return {"written": written, "processed": processed, "skipped": skipped, "path": self.cfg.output_path}

    # ---------- internals ----------

    def _decode_one(self, bitstring: str, sequence: str) -> Optional[Dict[str, Any]]:
        """
        Decode a single bitstring into turns and coordinates.
        Returns a JSON-serializable dict suitable for line-by-line storage.
        """
        try:
            spec = ShapeDecodeSpec(
                side_chain_hot_vector=self.cfg.side_chain_hot_vector,
                fifth_bit=self.cfg.fifth_bit,
            )
            dec = _ShapeDecoder(bitstring, spec)
            main_vec = dec.main_vectors()
            side_vec = dec.side_vectors()

            # Geometric construction on the tetrahedral lattice
            main_pos = _LatticeKinematics.main_positions(main_vec)
            side_pos = _LatticeKinematics.side_positions(side_vec, main_pos)

            # Serialize coordinates as lists for portability
            return {
                "sequence": sequence,
                "bitstring": bitstring,
                "fifth_bit": self.cfg.fifth_bit,
                "main_vectors": main_vec,
                "side_vectors": side_vec,
                "main_positions": main_pos.tolist(),                           # List[List[float]]
                "side_positions": [p.tolist() if p is not None else None for p in side_pos],
            }
        except Exception as e:
            if self.cfg.strict:
                raise
            _LOG.warning("Failed to decode row (skipped): %s", e)
            return None

    def _save_records(self, records: List[Dict[str, Any]]) -> int:
        """
        Save all decoded records into a single file.
        - JSONL: robust for nested lists, default choice.
        - Parquet: compact and fast if pyarrow/fastparquet is available.
        """
        if len(records) == 0:
            _LOG.warning("No decoded records to write. Output file will not be created.")
            return 0

        fmt = self.cfg.output_format.lower()
        path = self.cfg.output_path

        if fmt == "jsonl":
            with open(path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False))
                    f.write("\n")
            _LOG.info("Wrote %d JSONL records to %s", len(records), path)
            return len(records)

        elif fmt == "parquet":
            # Construct a DataFrame; nested lists will be stored using PyArrow if available.
            out_df = pd.DataFrame.from_records(records)
            try:
                out_df.to_parquet(path, index=False)
                _LOG.info("Wrote %d rows to Parquet: %s", len(out_df), path)
                return int(len(out_df))
            except Exception as e:
                _LOG.warning("Parquet write failed (%s). Falling back to JSONL.", e)
                # Fallback to JSONL with the same_squences path but .jsonl extension
                fallback = path.rsplit(".", 1)[0] + ".jsonl"
                with open(fallback, "w", encoding="utf-8") as f:
                    for rec in records:
                        f.write(json.dumps(rec, ensure_ascii=False))
                        f.write("\n")
                _LOG.info("Wrote %d JSONL records to %s", len(records), fallback)
                # Update configured path to reflect actual output
                self.cfg.output_path = fallback
                return len(records)

        else:
            _LOG.warning("Unknown output_format=%s. Falling back to JSONL.", fmt)
            self.cfg.output_format = "jsonl"
            return self._save_records(records)
