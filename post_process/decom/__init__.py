# --*-- conding:utf-8 --*--
# @time:3/20/26 01:45
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:__init__.py.py

"""
Decoding utilities for post_process.

This subpackage exposes the project-native binary-structure decoder used to
turn QSAD bitstrings into lattice conformations for downstream post-processing.
"""

from .decoder import (
    BinaryStructureDecoder,
    DecodedStructure,
    DecodeError,
    StructureDecoderConfig,
)

__all__ = [
    "BinaryStructureDecoder",
    "DecodedStructure",
    "DecodeError",
    "StructureDecoderConfig",
]
