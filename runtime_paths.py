# --*-- conding:utf-8 --*--
# @time:4/25/26 17:20
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:runtime_paths.py

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
QSAD_DATA_ROOT_ENV = "QSAD_DATA_ROOT"
DEFAULT_QSAD_DATA_ROOT = "quantum_data"
QSAD_SAMPLING_RUN_DIR_ENV = "QSAD_SAMPLING_RUN_DIR"
DEFAULT_SAMPLING_RUN_DIR = "sampling_result_2"
QSAD_POSTPROCESS_INPUT_DIR_ENV = "QSAD_POSTPROCESS_INPUT_DIR"
QSAD_DATA_ROOT_CANDIDATES = (
    "quantum_data",
    "sampling_results",
    "Sampling_result",
)


def resolve_qsad_data_root() -> Path:
    """
    Resolve the shared on-disk root for sampling outputs and post-processing inputs.

    Priority:
    1) QSAD_DATA_ROOT environment override
    2) first existing well-known project directory
    3) default "quantum_data" for fresh runs
    """
    override = os.getenv(QSAD_DATA_ROOT_ENV)
    if override:
        return Path(override).expanduser()

    for candidate in QSAD_DATA_ROOT_CANDIDATES:
        candidate_path = PROJECT_ROOT / candidate
        if candidate_path.exists():
            return candidate_path

    return PROJECT_ROOT / DEFAULT_QSAD_DATA_ROOT


def resolve_sampling_run_root() -> Path:
    """
    Resolve the dedicated sampling workspace under the shared QSAD data root.

    This keeps fresh sampling outputs isolated from legacy roots such as
    Sampling_result/sampling_result_1 and avoids mixing CSV results with
    progress/timing metadata in the task directory.
    """
    base_root = resolve_qsad_data_root()
    run_dir = os.getenv(QSAD_SAMPLING_RUN_DIR_ENV, DEFAULT_SAMPLING_RUN_DIR).strip()
    if not run_dir:
        run_dir = DEFAULT_SAMPLING_RUN_DIR
    return base_root / run_dir


def _contains_sample_dirs(root: Path) -> bool:
    if not root.exists() or not root.is_dir():
        return False
    try:
        for child in root.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            for sample_file in child.glob("samples_*.csv"):
                if "timing" not in sample_file.name:
                    return True
    except OSError:
        return False
    return False


def _normalize_postprocess_root(root: Path) -> Path:
    samples_dir = root / "samples"
    if samples_dir.exists() and samples_dir.is_dir():
        return samples_dir
    return root


def _sampling_run_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.removeprefix("sampling_result_")
    if suffix.isdigit():
        return (int(suffix), path.name)
    return (-1, path.name)


def resolve_postprocess_input_root() -> Path:
    """
    Resolve the directory whose immediate children are per-protein sampling results.

    Supported layouts:
    - <data_root>/sampling_result_N/<pdbid>/...
    - <data_root>/sampling_result_N/samples/<pdbid>/...
    """
    override = os.getenv(QSAD_POSTPROCESS_INPUT_DIR_ENV)
    if override:
        return Path(override).expanduser()

    run_root = _normalize_postprocess_root(resolve_sampling_run_root())
    if _contains_sample_dirs(run_root):
        return run_root

    base_root = resolve_qsad_data_root()
    if base_root.exists():
        run_roots = sorted(
            (
                child for child in base_root.iterdir()
                if child.is_dir() and child.name.startswith("sampling_result_")
            ),
            key=_sampling_run_sort_key,
            reverse=True,
        )
        for candidate in run_roots:
            normalized = _normalize_postprocess_root(candidate)
            if _contains_sample_dirs(normalized):
                return normalized

        if _contains_sample_dirs(base_root):
            return base_root

    return run_root
