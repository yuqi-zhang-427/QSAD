# --*-- conding:utf-8 --*--
# @time:10/21/25 14:23
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:run_post_process.py

# process_all.py
# Batch runner: iterate through sampling-result folders and run the QSAD pipeline.
# Results are saved to ./pp_result/<pdbid>; qsad_rmsd_summary.csv/jsonl are generated at the end.

import os
import json
import csv
import traceback
import shutil
import glob
from pathlib import Path
from typing import List, Dict, Any

from qsadpp.orchestrator import OrchestratorConfig, PipelineOrchestrator
from qsadpp.io_reader import ReaderOptions
from qsadpp.feature_calculator import FeatureConfig
from runtime_paths import PROJECT_ROOT, resolve_postprocess_input_root

INPUT_ROOT = resolve_postprocess_input_root()
OUT_ROOT = PROJECT_ROOT / "pp_result"
STAGING_ROOT = OUT_ROOT / "_staging"

INCLUDE_PATTERNS = [
    "samples_*.csv",
    "*.jsonl",
    "*.pdb",
    "*.pdb.gz",
]

EXCLUDE_PATTERNS = [
    "*timing*.csv",
]

IGNORED_DIR_NAMES = {
    "tracking",
    "summary",
    "_source_extras",
    "__pycache__",
}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _is_ignored_dir(p: Path) -> bool:
    return any(part in IGNORED_DIR_NAMES for part in p.parts)


def _has_stageable_inputs(p: Path) -> bool:
    if _is_ignored_dir(p):
        return False
    try:
        for f in p.iterdir():
            if not f.is_file():
                continue
            if _match_any(f.name, EXCLUDE_PATTERNS):
                continue
            if _match_any(f.name, INCLUDE_PATTERNS):
                return True
    except Exception:
        return False
    return False


def discover_targets(base_dir: Path) -> List[Path]:
    if not base_dir.exists():
        return []

    targets: List[Path] = []
    if _has_stageable_inputs(base_dir):
        targets.append(base_dir)

    for p in sorted(base_dir.rglob("*")):
        if p.is_dir() and not _is_ignored_dir(p) and _has_stageable_inputs(p):
            targets.append(p)
    return targets


def _match_any(name: str, patterns: List[str]) -> bool:
    return any(glob.fnmatch.fnmatch(name, pat) for pat in patterns)


def stage_input(src_dir: Path, staging_dir: Path, use_symlink: bool = True) -> None:
    """
    Copy or symlink only the files matching INCLUDE_PATTERNS
    while skipping those matching EXCLUDE_PATTERNS.
    """
    ensure_dir(staging_dir)
    for f in sorted(src_dir.iterdir()):
        if not f.is_file():
            continue
        if _match_any(f.name, EXCLUDE_PATTERNS):
            continue
        if not _match_any(f.name, INCLUDE_PATTERNS):
            continue
        dst = staging_dir / f.name
        try:
            if use_symlink:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(f.resolve())
            else:
                shutil.copy2(f, dst)
        except Exception:
            shutil.copy2(f, dst)


def make_cfg(pdb_dir: Path, out_root: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        pdb_dir=str(pdb_dir),
        reader_options=ReaderOptions(
            chunksize=100_000,
            strict=True,
            categorize_strings=True,
            include_all_csv=False,
        ),
        fifth_bit=False,
        out_dir=str(out_root),
        compute_features=True,
        feature_from="decoded",
        combined_feature_name="features.jsonl",
        feature_config=FeatureConfig(
            output_format="jsonl",
        ),
    )


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("pdb_id,status,message\n", encoding="utf-8")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ensure_dir(OUT_ROOT)
    ensure_dir(STAGING_ROOT)

    targets = discover_targets(INPUT_ROOT)
    if not targets:
        print(f"[WARN] No sampling targets under {INPUT_ROOT}")
        return

    aggregate: List[Dict[str, Any]] = []

    for pdb_path in targets:
        pdb_id = pdb_path.name
        out_dir = OUT_ROOT / pdb_id
        staging_dir = STAGING_ROOT / pdb_id

        print(f"==> Processing {pdb_id}")
        try:
            # Prepare a temporary folder excluding timing files
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            stage_input(pdb_path, staging_dir, use_symlink=True)

            # Run orchestrator
            cfg = make_cfg(staging_dir, OUT_ROOT)
            runner = PipelineOrchestrator(cfg)
            summary = runner.run()

            actual_out_dir = out_dir
            if isinstance(summary, dict):
                for key in ("decoded_all", "energies_all", "features_all"):
                    path_value = summary.get(key)
                    if path_value:
                        actual_out_dir = Path(path_value).parent
                        break
            ensure_dir(actual_out_dir)

            print(f"[OK] {pdb_id}")
            row = {
                "pdb_id": pdb_id,
                "status": "ok",
                "input_dir": str(pdb_path),
                "out_dir": str(actual_out_dir),
            }
            if isinstance(summary, dict):
                for key in ("groups", "decoded_rows", "energy_rows", "feature_rows"):
                    if key in summary:
                        row[key] = summary[key]
            aggregate.append(row)

            with (actual_out_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary if isinstance(summary, dict) else {"summary": str(summary)},
                          f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[FAIL] {pdb_id}: {e}")
            ensure_dir(out_dir)
            aggregate.append({
                "pdb_id": pdb_id,
                "status": "fail",
                "message": str(e),
                "input_dir": str(pdb_path),
                "out_dir": str(out_dir),
            })
            with (out_dir / "error.log").open("w", encoding="utf-8") as f:
                f.write("".join(traceback.format_exc()))

    write_csv(OUT_ROOT / "qsad_rmsd_summary.csv", aggregate)
    write_jsonl(OUT_ROOT / "summary.jsonl", aggregate)
    print(f"\nAll done. Summary at {OUT_ROOT}/qsad_rmsd_summary.csv & summary.jsonl")


if __name__ == "__main__":
    main()
