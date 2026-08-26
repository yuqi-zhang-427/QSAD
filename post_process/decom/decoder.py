# --*-- conding:utf-8 --*--
# @time:3/26/26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:decoder.py

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


_LOG = logging.getLogger(__name__)
if not _LOG.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )


def _bit_pairs_to_ints(bitstring: str) -> List[int]:
    """Decode reversed bit-pairs with the project encoding."""
    s = bitstring[::-1]
    encoding = {"00": 0, "01": 1, "10": 2, "11": 3}
    return [encoding[s[2 * i : 2 * (i + 1)]] for i in range(len(s) // 2)]


class DecodeError(ValueError):
    """Raised when a bitstring cannot be decoded under the configured rule."""


@dataclass(frozen=True)
class StructureDecoderConfig:
    """
    Decoder configuration following the QSAD bitstring-to-structure rule.

    `side_chain_hot_vector` defines which residues carry side-chain turns.
    """

    side_chain_hot_vector: Sequence[bool]
    fifth_bit: bool = False
    strict: bool = True

    def __post_init__(self) -> None:
        if self.sequence_length < 3:
            raise ValueError(
                "side_chain_hot_vector must describe at least 3 residues for main-chain decoding."
            )

    @property
    def sequence_length(self) -> int:
        return len(self.side_chain_hot_vector)

    @property
    def expected_main_bits(self) -> int:
        return 2 * (self.sequence_length - 3) - (1 if self.fifth_bit else 0)

    @property
    def expected_side_bits(self) -> int:
        return 2 * sum(bool(flag) for flag in self.side_chain_hot_vector)

    @property
    def expected_bitstring_length(self) -> int:
        return self.expected_main_bits + self.expected_side_bits


@dataclass
class DecodedStructure:
    """Project-native decoded structure representation."""

    bitstring: str
    sequence: str = ""
    fifth_bit: bool = False
    main_turns: List[int] = field(default_factory=list)
    side_turns: List[Optional[int]] = field(default_factory=list)
    main_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    side_positions: List[Optional[np.ndarray]] = field(default_factory=list)

    def to_record(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dict for downstream stages."""
        return {
            "sequence": self.sequence,
            "bitstring": self.bitstring,
            "fifth_bit": self.fifth_bit,
            "main_turns": list(self.main_turns),
            "side_turns": list(self.side_turns),
            "main_positions": self.main_positions.tolist(),
            "side_positions": [
                coords.tolist() if coords is not None else None for coords in self.side_positions
            ],
        }


class _EncodingRuleDecoder:
    """Pure turn decoder for QSAD tetrahedral lattice bitstring encoding."""

    def __init__(self, bitstring: str, cfg: StructureDecoderConfig) -> None:
        self._bitstring = bitstring
        self._cfg = cfg

    def decode_main_turns(self) -> List[int]:
        n_main = self._cfg.expected_main_bits
        main_bits = self._bitstring[-n_main:] + "0010"
        if self._cfg.fifth_bit and len(main_bits) >= 6:
            main_bits = main_bits[:-5] + "1" + main_bits[-5:]
        return _bit_pairs_to_ints(main_bits)

    def decode_side_turns(self) -> List[Optional[int]]:
        n_main = self._cfg.expected_main_bits
        n_side = self._cfg.expected_side_bits
        side_bits = self._bitstring[:n_side] if n_side > 0 else ""
        raw_turns = _bit_pairs_to_ints(side_bits) if side_bits else []

        turns: List[Optional[int]] = []
        cursor = 0
        for has_side_chain in self._cfg.side_chain_hot_vector:
            if has_side_chain:
                turns.append(raw_turns[cursor] if cursor < len(raw_turns) else None)
                cursor += 1
            else:
                turns.append(None)
        return turns


class _TetrahedralLattice:
    """Tetrahedral lattice geometry used by QSAD decoding."""

    COORD = (1.0 / np.sqrt(3.0)) * np.array(
        [
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
        ],
        dtype=float,
    )

    @classmethod
    def build_main_positions(cls, main_turns: Sequence[int]) -> np.ndarray:
        positions = np.zeros((len(main_turns) + 1, 3), dtype=float)
        for index, turn in enumerate(main_turns):
            positions[index + 1] = ((-1.0) ** index) * cls.COORD[int(turn)]
        return positions.cumsum(axis=0)

    @classmethod
    def build_side_positions(
        cls,
        side_turns: Sequence[Optional[int]],
        main_positions: np.ndarray,
    ) -> List[Optional[np.ndarray]]:
        positions: List[Optional[np.ndarray]] = []
        counter = 1
        for bead_index, turn in enumerate(side_turns):
            if turn is None:
                positions.append(None)
            else:
                base = main_positions[bead_index]
                step = ((-1.0) ** counter) * cls.COORD[int(turn)]
                positions.append(base + step)
            counter += 1
        return positions


@dataclass
class BinaryStructureDecoder:
    """
    Public decoder for post_process/decom.

    Designed for downstream composition:
    - `decode()` for one bitstring
    - `decode_many()` for batch use
    - `decode_to_record()` when a JSON-serializable payload is preferred
    """

    cfg: StructureDecoderConfig

    def expected_bitstring_length(self) -> int:
        return self.cfg.expected_bitstring_length

    def validate_bitstring(self, bitstring: str) -> str:
        normalized = str(bitstring or "").strip()
        if not normalized:
            raise DecodeError("Bitstring is empty.")
        if set(normalized) - {"0", "1"}:
            raise DecodeError("Bitstring must contain only '0' and '1'.")

        expected = self.cfg.expected_bitstring_length
        if len(normalized) < expected:
            raise DecodeError(
                f"Bitstring is too short for this decoder: got {len(normalized)} bits, "
                f"expected at least {expected}."
            )

        # Keep QSAD compatibility: consume the rightmost effective encoding bits.
        return normalized[-expected:]

    def decode(self, bitstring: str, sequence: str = "") -> DecodedStructure:
        effective_bits = self.validate_bitstring(bitstring)
        turn_decoder = _EncodingRuleDecoder(effective_bits, self.cfg)
        main_turns = turn_decoder.decode_main_turns()
        side_turns = turn_decoder.decode_side_turns()
        main_positions = _TetrahedralLattice.build_main_positions(main_turns)
        side_positions = _TetrahedralLattice.build_side_positions(side_turns, main_positions)
        return DecodedStructure(
            bitstring=str(bitstring).strip(),
            sequence=str(sequence or ""),
            fifth_bit=self.cfg.fifth_bit,
            main_turns=main_turns,
            side_turns=side_turns,
            main_positions=main_positions,
            side_positions=side_positions,
        )

    def decode_to_record(self, bitstring: str, sequence: str = "") -> Dict[str, Any]:
        return self.decode(bitstring=bitstring, sequence=sequence).to_record()

    def decode_many(
        self,
        bitstrings: Iterable[str],
        sequence: str = "",
    ) -> List[DecodedStructure]:
        decoded: List[DecodedStructure] = []
        for bitstring in bitstrings:
            try:
                decoded.append(self.decode(bitstring=bitstring, sequence=sequence))
            except Exception as exc:
                if self.cfg.strict:
                    raise
                _LOG.warning("Failed to decode bitstring (skipped): %s", exc)
                continue
        return decoded

    def decode_records(
        self,
        rows: Iterable[Dict[str, Any]],
        bitstring_key: str = "bitstring",
        sequence_key: str = "sequence",
    ) -> List[Dict[str, Any]]:
        decoded: List[Dict[str, Any]] = []
        for row in rows:
            bitstring = str(row.get(bitstring_key) or "").strip()
            sequence = str(row.get(sequence_key) or "").strip()
            try:
                record = self.decode_to_record(bitstring=bitstring, sequence=sequence)
            except Exception as exc:
                if self.cfg.strict:
                    raise
                _LOG.warning("Failed to decode row (skipped): %s", exc)
                continue
            decoded.append({**row, **record})
        return decoded


__all__ = [
    "BinaryStructureDecoder",
    "DecodedStructure",
    "DecodeError",
    "StructureDecoderConfig",
]
