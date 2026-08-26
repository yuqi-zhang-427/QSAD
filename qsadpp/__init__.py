# --*-- conding:utf-8 --*--
# @time:10/22/25 21:21
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:__init__.py.py

"""
QSADPP: Quantum Sampling Analysis and Decoding Post-Processing

Modules:
    io_reader.py           – Streaming CSV reader with chunking and group iteration.
    coordinate_decoder.py  – Bitstring-to-structure decoder and coordinate generator.
    energy_calculator.py   – Energy evaluation based on MJ potential and geometric terms.
    orchestrator.py        – High-level pipeline orchestrating reading, decoding, and energy evaluation.

Usage Example:
    from qsadpp import OrchestratorConfig, PipelineOrchestrator, ReaderOptions

    cfg = OrchestratorConfig(
        pdb_dir="./quantum_data/1m7y",
        reader_options=ReaderOptions(
            chunksize=100_000,
            strict=True,
            categorize_strings=True,
            include_all_csv=False,
        ),
        fifth_bit=False,
        out_dir="e_results",
        decoder_output_format="jsonl",
        aggregate_only=True,
    )

    runner = PipelineOrchestrator(cfg)
    summary = runner.run()
    print(summary)
"""

from .io_reader import ReaderOptions, SampleReader
from .coordinate_decoder import CoordinateBatchDecoder, CoordinateDecoderConfig
from .energy_calculator import LatticeEnergyCalculator, EnergyConfig
from .orchestrator import OrchestratorConfig, PipelineOrchestrator

__all__ = [
    # Reader
    "ReaderOptions",
    "SampleReader",

    # Decoder
    "CoordinateBatchDecoder",
    "CoordinateDecoderConfig",

    # Energy
    "LatticeEnergyCalculator",
    "EnergyConfig",

    # Pipeline orchestrator
    "OrchestratorConfig",
    "PipelineOrchestrator",
]
