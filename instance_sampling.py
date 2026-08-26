# --*-- conding:utf-8 --*--
# @time:10/23/25 14:47
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:instance_sampling.py

import time
import random
import json
import os
import csv
import shutil
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

from qiskit.quantum_info import SparsePauliOp

from sampling import SamplingRunner, SamplingConfig, BackendConfig
from Protein_Folding import Peptide
from Protein_Folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
from Protein_Folding.penalty_parameters import PenaltyParameters
from Protein_Folding.protein_folding_problem import ProteinFoldingProblem
from runtime_paths import resolve_qsad_data_root, resolve_sampling_run_root


IBM_CONFIG_FILE = os.getenv("IBM_CONFIG_FILE", "./ibm_config.txt")
TASKS_FILE = os.getenv("TASKS_CSV", "./task_input/tasks.csv")

# Where the circuits are executed: "simulator" (local qiskit-aer, the default)
# or "ibm" (live QPU). The default needs no credentials and no code edit, so a
# fresh checkout runs end-to-end out of the box; set QSAD_BACKEND_KIND=ibm to
# submit to real hardware instead.
BACKEND_KIND = os.getenv("QSAD_BACKEND_KIND", "simulator").strip().lower()
if BACKEND_KIND not in ("ibm", "simulator"):
    raise SystemExit(
        f"QSAD_BACKEND_KIND must be 'ibm' or 'simulator', got {BACKEND_KIND!r}"
    )
USE_IBM = BACKEND_KIND == "ibm"

# QSAD_AER_METHOD overrides Aer's simulation method; leaving it unset lets
# sampling/backends.py pick matrix_product_state for wide circuits.
# It is simulator-only and ignored on hardware.
SIM_METHOD = os.getenv("QSAD_AER_METHOD", "").strip().lower() or None

# QSAD_SIM_SEED pins the run: it seeds the Aer sampling RNG *and* derives the
# per-group run_seed that fixes the ansatz parameters, so two runs with the
# same seed agree byte for byte. The run_seed half applies on hardware too
# (identical circuits), though QPU shot noise still varies run to run.
_sim_seed_env = os.getenv("QSAD_SIM_SEED", "").strip()
SIM_SEED = int(_sim_seed_env) if _sim_seed_env else None

PENALTY_PARAMS: Tuple[int, int, int] = (10, 10, 10)
BETA_LIST: List[float] = [1.0, 2.0, 3.0, 4.0]
SEEDS: int = 3
REPS: int = 1

GROUP_COUNT = 10
SHOTS_PER_GROUP = 2000

DATA_ROOT = resolve_qsad_data_root()
SAMPLING_RUN_ROOT = resolve_sampling_run_root()
SAMPLES_ROOT = SAMPLING_RUN_ROOT / "samples"
TRACKING_ROOT = SAMPLING_RUN_ROOT / "tracking"
SUMMARY_ROOT = SAMPLING_RUN_ROOT / "summary"
RUN_MANIFEST = SAMPLING_RUN_ROOT / "run_manifest.json"
IBM_SERVICE: Optional[Any] = None  # qiskit_ibm_runtime.QiskitRuntimeService


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Failed to read progress JSON {path}: {e}")
        return {}


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_run_manifest(task_count: int) -> None:
    payload = load_json(RUN_MANIFEST)
    payload.setdefault("created_at", utc_now_iso())
    payload["updated_at"] = utc_now_iso()
    payload["data_root"] = str(DATA_ROOT)
    payload["sampling_run_root"] = str(SAMPLING_RUN_ROOT)
    payload["samples_root"] = str(SAMPLES_ROOT)
    payload["tracking_root"] = str(TRACKING_ROOT)
    payload["summary_root"] = str(SUMMARY_ROOT)
    payload["tasks_file"] = str(Path(TASKS_FILE).resolve())
    payload["task_count"] = int(task_count)
    payload["settings"] = {
        "PENALTY_PARAMS": list(PENALTY_PARAMS),
        "BETA_LIST": list(BETA_LIST),
        "SEEDS": SEEDS,
        "REPS": REPS,
        "GROUP_COUNT": GROUP_COUNT,
        "SHOTS_PER_GROUP": SHOTS_PER_GROUP,
        "IBM_BACKEND_NAME": IBM_BACKEND_NAME,
    }
    save_json(RUN_MANIFEST, payload)


def default_progress(protein_name: str, sequence: str, out_dir: Path) -> Dict[str, Any]:
    now = utc_now_iso()
    return {
        "created_at": now,
        "updated_at": now,
        "pdbid": protein_name,
        "sequence": sequence,
        "status": "pending",
        "merged_csv": str(out_dir / f"samples_{protein_name}_all_ibm.csv"),
        "settings": {
            "PENALTY_PARAMS": list(PENALTY_PARAMS),
            "BETA_LIST": list(BETA_LIST),
            "SEEDS": SEEDS,
            "REPS": REPS,
            "GROUP_COUNT": GROUP_COUNT,
            "SHOTS_PER_GROUP": SHOTS_PER_GROUP,
            "IBM_BACKEND_NAME": IBM_BACKEND_NAME,
            "SAMPLING_RUN_ROOT": str(SAMPLING_RUN_ROOT),
            "SAMPLES_ROOT": str(SAMPLES_ROOT),
            "TRACKING_ROOT": str(TRACKING_ROOT),
        },
        "groups": {},
    }


def load_timing_lookup(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read timing CSV {path}: {e}")
        return {}

    lookup: Dict[int, Dict[str, Any]] = {}
    if "group_id" not in df.columns:
        return lookup

    for _, row in df.iterrows():
        try:
            group_id = int(row["group_id"])
        except Exception:
            continue
        item: Dict[str, Any] = {}
        if "run_seed" in df.columns and pd.notna(row["run_seed"]):
            item["run_seed"] = int(row["run_seed"])
        if "rows" in df.columns and pd.notna(row["rows"]):
            item["rows"] = int(row["rows"])
        if "seconds" in df.columns and pd.notna(row["seconds"]):
            item["seconds"] = round(float(row["seconds"]), 6)
        for text_field in (
            "local_started_at",
            "local_finished_at",
            "runtime_job_id",
            "runtime_status",
            "runtime_creation_date",
            "runtime_timestamp_created",
            "runtime_timestamp_running",
            "runtime_timestamp_finished",
            "runtime_error_message",
        ):
            if text_field in df.columns and pd.notna(row[text_field]):
                item[text_field] = str(row[text_field])
        if "runtime_details_json" in df.columns and pd.notna(row["runtime_details_json"]):
            try:
                item["runtime_details"] = json.loads(str(row["runtime_details_json"]))
            except Exception:
                pass
        lookup[group_id] = item
    return lookup


def ensure_progress(
    progress_path: Path,
    protein_name: str,
    sequence: str,
    out_dir: Path,
    timing_csv: Path,
) -> Dict[str, Any]:
    progress = load_json(progress_path) or default_progress(protein_name, sequence, out_dir)
    groups = progress.get("groups")
    if not isinstance(groups, dict):
        groups = {}
    progress["groups"] = groups
    progress["pdbid"] = protein_name
    progress["sequence"] = sequence
    progress["merged_csv"] = str(out_dir / f"samples_{protein_name}_all_ibm.csv")
    progress["settings"] = default_progress(protein_name, sequence, out_dir)["settings"]
    progress.setdefault("created_at", utc_now_iso())

    timing_lookup = load_timing_lookup(timing_csv)
    changed = False

    for group_id in range(GROUP_COUNT):
        key = str(group_id)
        csv_path = out_dir / f"samples_{protein_name}_group{group_id}_ibm.csv"
        entry = groups.get(key)
        if not isinstance(entry, dict):
            entry = {}
            groups[key] = entry
            changed = True

        if entry.get("csv") != str(csv_path):
            entry["csv"] = str(csv_path)
            changed = True

        timing_item = timing_lookup.get(group_id, {})
        if csv_path.exists():
            try:
                row_count = int(len(pd.read_csv(csv_path)))
            except Exception as e:
                print(f"[Resume] existing CSV unreadable, will rerun group {group_id}: {csv_path} ({e})")
                if entry.get("status") == "done":
                    entry["status"] = "pending"
                    changed = True
                continue

            updates: Dict[str, Any] = {
                "status": "done",
                "rows": row_count,
            }
            if "local_started_at" in timing_item and "started_at" not in entry:
                updates["started_at"] = timing_item["local_started_at"]
            if "local_finished_at" in timing_item:
                updates["finished_at"] = timing_item["local_finished_at"]
            elif "finished_at" in entry:
                updates["finished_at"] = entry["finished_at"]
            if "run_seed" in timing_item:
                updates["run_seed"] = timing_item["run_seed"]
            elif "run_seed" in entry:
                updates["run_seed"] = entry["run_seed"]
            if "seconds" in timing_item:
                updates["seconds"] = timing_item["seconds"]
            if "runtime_details" in timing_item:
                updates["runtime_details"] = timing_item["runtime_details"]

            for k, v in updates.items():
                if entry.get(k) != v:
                    entry[k] = v
                    changed = True
        else:
            if entry.get("status") == "done":
                entry["status"] = "pending"
                changed = True

    done_groups = [
        gid for gid in range(GROUP_COUNT)
        if groups.get(str(gid), {}).get("status") == "done"
    ]
    progress["status"] = "done" if len(done_groups) == GROUP_COUNT else ("in_progress" if done_groups else "pending")
    progress["updated_at"] = utc_now_iso()

    if changed or not progress_path.exists():
        save_json(progress_path, progress)
    return progress


def write_timing_from_progress(progress: Dict[str, Any], protein_name: str, timing_csv: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for group_id in range(GROUP_COUNT):
        entry = progress.get("groups", {}).get(str(group_id), {})
        if entry.get("status") != "done":
            continue
        runtime_details = entry.get("runtime_details", {})
        runtime_job = runtime_details.get("job", {}) if isinstance(runtime_details, dict) else {}
        metrics = runtime_job.get("metrics", {}) if isinstance(runtime_job, dict) else {}
        timestamps = metrics.get("timestamps", {}) if isinstance(metrics, dict) else {}
        usage = runtime_job.get("usage", {}) if isinstance(runtime_job, dict) else {}
        usage_estimation = runtime_job.get("usage_estimation", {}) if isinstance(runtime_job, dict) else {}
        rows.append({
            "protein_name": protein_name,
            "group_id": group_id,
            "run_seed": entry.get("run_seed"),
            "rows": entry.get("rows"),
            "seconds": entry.get("seconds"),
            "local_started_at": entry.get("started_at"),
            "local_finished_at": entry.get("finished_at"),
            "runtime_mode": runtime_details.get("mode") if isinstance(runtime_details, dict) else None,
            "runtime_pub_count": runtime_details.get("pub_count") if isinstance(runtime_details, dict) else None,
            "runtime_job_id": runtime_job.get("job_id") if isinstance(runtime_job, dict) else None,
            "runtime_status": runtime_job.get("status") if isinstance(runtime_job, dict) else None,
            "runtime_creation_date": runtime_job.get("creation_date") if isinstance(runtime_job, dict) else None,
            "runtime_error_message": runtime_job.get("error_message") if isinstance(runtime_job, dict) else None,
            "runtime_timestamp_created": timestamps.get("created") if isinstance(timestamps, dict) else None,
            "runtime_timestamp_running": timestamps.get("running") if isinstance(timestamps, dict) else None,
            "runtime_timestamp_finished": timestamps.get("finished") if isinstance(timestamps, dict) else None,
            "runtime_usage_quantum_seconds": usage.get("quantum_seconds") if isinstance(usage, dict) else None,
            "runtime_usage_seconds": usage.get("seconds") if isinstance(usage, dict) else None,
            "runtime_estimated_quantum_seconds": (
                usage_estimation.get("quantum_seconds") if isinstance(usage_estimation, dict) else None
            ),
            "runtime_details_json": json.dumps(to_jsonable(runtime_details), ensure_ascii=False, sort_keys=True),
        })

    if not rows:
        return

    timing_df = pd.DataFrame(rows).sort_values("group_id").reset_index(drop=True)
    numeric_seconds = pd.to_numeric(timing_df["seconds"], errors="coerce").fillna(0.0)
    total_elapsed = round(float(numeric_seconds.sum()), 6)
    timing_df["total_seconds_for_protein"] = total_elapsed
    timing_df.to_csv(timing_csv, index=False)


def done_group_csvs(progress: Dict[str, Any]) -> List[Path]:
    csvs: List[Path] = []
    for group_id in range(GROUP_COUNT):
        entry = progress.get("groups", {}).get(str(group_id), {})
        csv_path = Path(entry.get("csv", ""))
        if entry.get("status") == "done" and csv_path.exists():
            csvs.append(csv_path)
    return csvs


def normalize_sequence(sequence: str) -> str:
    return str(sequence).strip().upper()


def sampling_paths(protein_name: str) -> Tuple[Path, Path, Path, Path, Path]:
    sample_dir = SAMPLES_ROOT / protein_name
    tracking_dir = TRACKING_ROOT / protein_name
    progress_path = tracking_dir / "progress.json"
    timing_csv = tracking_dir / f"{protein_name}_timing.csv"
    merged_csv = sample_dir / f"samples_{protein_name}_all_ibm.csv"
    return sample_dir, tracking_dir, progress_path, timing_csv, merged_csv


def completed_merged_path(protein_name: str, sequence: str) -> str:
    sample_dir, _, progress_path, timing_csv, merged_csv = sampling_paths(protein_name)
    if not progress_path.exists():
        return ""
    progress = ensure_progress(progress_path, protein_name, sequence, sample_dir, timing_csv)
    existing_done = done_group_csvs(progress)
    if len(existing_done) == GROUP_COUNT and merged_csv.exists():
        write_timing_from_progress(progress, protein_name, timing_csv)
        return str(merged_csv)
    return ""


def load_completed_sequence_index() -> Dict[str, str]:
    index: Dict[str, str] = {}
    if not TRACKING_ROOT.exists():
        return index

    for tracking_dir in sorted(TRACKING_ROOT.iterdir()):
        if not tracking_dir.is_dir():
            continue
        protein_name = tracking_dir.name
        progress_path = tracking_dir / "progress.json"
        merged_csv = SAMPLES_ROOT / protein_name / f"samples_{protein_name}_all_ibm.csv"
        progress = load_json(progress_path)
        if progress.get("status") != "done" or not merged_csv.exists():
            continue
        sequence = normalize_sequence(progress.get("sequence", ""))
        if sequence:
            index.setdefault(sequence, protein_name)
    return index


def rewrite_sample_copy(src_path: Path, dst_path: Path, source_name: str, target_name: str) -> None:
    with src_path.open("r", newline="", encoding="utf-8") as src_handle:
        reader = csv.DictReader(src_handle)
        fieldnames = reader.fieldnames or []
        with dst_path.open("w", newline="", encoding="utf-8") as dst_handle:
            writer = csv.DictWriter(dst_handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                if "protein" in row:
                    row["protein"] = target_name
                if "pdbid" in row:
                    row["pdbid"] = target_name
                if "label" in row and row["label"]:
                    row["label"] = row["label"].replace(source_name, target_name)
                if "kras_task" in row and row["kras_task"]:
                    row["kras_task"] = str(row["kras_task"]).replace(source_name, target_name)
                writer.writerow(row)


def rewrite_timing_copy(src_path: Path, dst_path: Path, target_name: str) -> None:
    with src_path.open("r", newline="", encoding="utf-8") as src_handle:
        reader = csv.DictReader(src_handle)
        fieldnames = reader.fieldnames or []
        with dst_path.open("w", newline="", encoding="utf-8") as dst_handle:
            writer = csv.DictWriter(dst_handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                if "protein_name" in row:
                    row["protein_name"] = target_name
                if "pdbid" in row:
                    row["pdbid"] = target_name
                writer.writerow(row)


def copy_completed_sampling_result(source_name: str, target_name: str, sequence: str) -> str:
    src_sample_dir, src_tracking_dir, src_progress_path, src_timing_csv, src_merged_csv = sampling_paths(source_name)
    dst_sample_dir, dst_tracking_dir, dst_progress_path, dst_timing_csv, dst_merged_csv = sampling_paths(target_name)

    if not src_progress_path.exists() or not src_merged_csv.exists():
        return ""

    if dst_sample_dir.exists():
        shutil.rmtree(dst_sample_dir)
    if dst_tracking_dir.exists():
        shutil.rmtree(dst_tracking_dir)
    dst_sample_dir.mkdir(parents=True, exist_ok=True)
    dst_tracking_dir.mkdir(parents=True, exist_ok=True)

    for src_path in sorted(src_sample_dir.iterdir()):
        if not src_path.is_file():
            continue
        dst_name = src_path.name.replace(source_name, target_name)
        dst_path = dst_sample_dir / dst_name
        if src_path.suffix.lower() == ".csv":
            rewrite_sample_copy(src_path, dst_path, source_name, target_name)
        else:
            shutil.copy2(src_path, dst_path)

    if src_timing_csv.exists():
        rewrite_timing_copy(src_timing_csv, dst_timing_csv, target_name)

    progress = load_json(src_progress_path) or default_progress(target_name, sequence, dst_sample_dir)
    progress["pdbid"] = target_name
    progress["sequence"] = sequence
    progress["merged_csv"] = str(dst_merged_csv)
    progress["status"] = "done"
    progress["copied_from"] = source_name
    progress["copied_at"] = utc_now_iso()
    progress["updated_at"] = utc_now_iso()
    progress.setdefault("created_at", utc_now_iso())

    groups = progress.get("groups")
    if not isinstance(groups, dict):
        groups = {}
    for group_id in range(GROUP_COUNT):
        key = str(group_id)
        entry = groups.get(key)
        if not isinstance(entry, dict):
            entry = {}
        entry["csv"] = str(dst_sample_dir / f"samples_{target_name}_group{group_id}_ibm.csv")
        if entry.get("status") != "done":
            entry["status"] = "done"
        groups[key] = entry
    progress["groups"] = groups
    save_json(dst_progress_path, progress)
    write_timing_from_progress(progress, target_name, dst_timing_csv)
    return str(dst_merged_csv)


def read_ibm_config(path: str) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    raw_path = Path(path)
    resolved_path = raw_path
    if not raw_path.exists():
        try:
            for candidate in sorted(raw_path.parent.iterdir() if raw_path.parent.exists() else []):
                if candidate.is_file() and candidate.name.strip() == raw_path.name:
                    resolved_path = candidate
                    print(f"[Config] Using fallback IBM config path: {candidate}")
                    break
        except Exception:
            resolved_path = raw_path
    try:
        with resolved_path.open("r", encoding="utf-8") as f:
            for line in f:
                if "=" not in line:
                    continue
                key, value = line.strip().split("=", 1)
                cfg[key.strip().upper()] = value.strip()
    except Exception as e:
        print(f"Failed to read IBM config: {e}")
    return cfg


# Only consult ibm_config.txt on the hardware path, so a simulator run needs
# neither the file nor a token.
cfg_data = read_ibm_config(IBM_CONFIG_FILE) if USE_IBM else {}
IBM_TOKEN = cfg_data.get("TOKEN", "")
IBM_INSTANCE = cfg_data.get("INSTANCE", None)
IBM_BACKEND_NAME = cfg_data.get("BACKEND", None)


def init_ibm_service() -> Any:
    """Create (once) the IBM Runtime service.

    qiskit_ibm_runtime is imported here rather than at module scope so that the
    default simulator path never depends on the IBM runtime stack being
    importable.
    """
    global IBM_SERVICE
    if IBM_SERVICE is not None:
        return IBM_SERVICE
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except Exception as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "QSAD_BACKEND_KIND=ibm requires qiskit-ibm-runtime; it could not be "
            f"imported ({exc}). Use the default simulator backend, or repair the "
            "installation."
        ) from exc
    if IBM_TOKEN:
        try:
            IBM_SERVICE = QiskitRuntimeService(
                channel="ibm_quantum_platform",
                token=IBM_TOKEN,
                instance=IBM_INSTANCE
            )
            return IBM_SERVICE
        except Exception:
            IBM_SERVICE = QiskitRuntimeService()
            return IBM_SERVICE
    IBM_SERVICE = QiskitRuntimeService()
    return IBM_SERVICE


def build_protein_hamiltonian(sequence: str, penalties: Tuple[int, int, int]) -> SparsePauliOp:
    side_chain_residue_sequences = ['' for _ in range(len(sequence))]
    peptide = Peptide(sequence, side_chain_residue_sequences)
    mj_interaction = MiyazawaJerniganInteraction()
    penalty_terms = PenaltyParameters(*penalties)
    problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
    H = problem.qubit_op()
    if isinstance(H, (list, tuple)) and len(H) > 0:
        H = H[0]
    if not isinstance(H, SparsePauliOp):
        H = SparsePauliOp(H)
    return H


def read_tasks(path: str) -> List[Dict[str, str]]:
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    if "protein_name" in cols:
        pn_col = df.columns[cols.index("protein_name")]
    elif "pdbid" in cols:
        pn_col = df.columns[cols.index("pdbid")]
    elif "pdb_id" in cols:
        pn_col = df.columns[cols.index("pdb_id")]
    elif "protein" in cols:
        pn_col = df.columns[cols.index("protein")]
    else:
        raise ValueError("Input must contain column: protein_name / pdbid / pdb_id / protein")

    if "main_chain_residue_seq" in cols:
        seq_col = df.columns[cols.index("main_chain_residue_seq")]
    elif "sequence" in cols:
        seq_col = df.columns[cols.index("sequence")]
    else:
        raise ValueError("Input must contain column: main_chain_residue_seq or sequence")

    tasks: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        protein_name = str(row[pn_col]).strip()
        sequence = str(row[seq_col]).strip()
        if protein_name and sequence:
            tasks.append({"protein_name": protein_name, "main_chain_residue_seq": sequence})
    return tasks


def per_example_sampling(protein_name: str, sequence: str) -> str:
    print(f"\n=== Running {protein_name} ({sequence}) ===")

    sample_dir = SAMPLES_ROOT / protein_name
    tracking_dir = TRACKING_ROOT / protein_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    tracking_dir.mkdir(parents=True, exist_ok=True)
    progress_path = tracking_dir / "progress.json"
    timing_csv = tracking_dir / f"{protein_name}_timing.csv"
    merged_csv = sample_dir / f"samples_{protein_name}_all_ibm.csv"

    progress = ensure_progress(progress_path, protein_name, sequence, sample_dir, timing_csv)
    existing_done = done_group_csvs(progress)
    if len(existing_done) == GROUP_COUNT and merged_csv.exists():
        print(f"[Resume] {protein_name}: all groups already completed, reusing existing outputs.")
        write_timing_from_progress(progress, protein_name, timing_csv)
        return str(merged_csv)

    H = None

    for group_id in range(GROUP_COUNT):
        key = str(group_id)
        csv_path = sample_dir / f"samples_{protein_name}_group{group_id}_ibm.csv"
        entry = progress.get("groups", {}).get(key, {})
        if entry.get("status") == "done" and csv_path.exists():
            print(f"[Resume] {protein_name}: skip finished group {group_id} -> {csv_path}")
            continue

        if H is None:
            H = build_protein_hamiltonian(sequence, PENALTY_PARAMS)

        if entry.get("run_seed") is not None:
            # Resumed group: reuse the seed recorded in progress.json.
            run_seed = int(entry["run_seed"])
        elif SIM_SEED is not None:
            # QSAD_SIM_SEED pins the whole run: run_seed drives the ansatz
            # parameters, so deriving it from the seed (rather than drawing it
            # at random) is what actually makes a run reproducible.
            run_seed = (SIM_SEED + group_id) % (2 ** 31 - 1) or 1
        else:
            run_seed = random.randint(1, 2**31 - 1)
        entry.update({
            "csv": str(csv_path),
            "status": "running",
            "run_seed": run_seed,
            "started_at": entry.get("started_at", utc_now_iso()),
        })
        progress["groups"][key] = entry
        progress["status"] = "in_progress"
        progress["updated_at"] = utc_now_iso()
        save_json(progress_path, progress)

        cfg = SamplingConfig(
            L=len(sequence),
            betas=list(BETA_LIST),
            seeds=SEEDS,
            seed_base=run_seed,
            reps=REPS,
            entanglement="linear",
            label=f"qsad_{BACKEND_KIND}_{protein_name}_g{group_id}",
            backend=BackendConfig(
                kind=BACKEND_KIND,
                shots=SHOTS_PER_GROUP,
                # seed_sim is unused on hardware; on the simulator it makes a
                # run reproducible when QSAD_SIM_SEED is set.
                seed_sim=None if USE_IBM else (SIM_SEED if SIM_SEED is not None else run_seed),
                ibm_backend=IBM_BACKEND_NAME if USE_IBM else None,
                sim_method=None if USE_IBM else SIM_METHOD,
            ),
            out_csv=str(csv_path),
            extra_meta={
                "protein": protein_name,
                "sequence": sequence,
                "group_id": group_id,
                "run_seed": run_seed,
            },
        )

        runner = SamplingRunner(
            cfg, H, backend_service=init_ibm_service() if USE_IBM else None
        )

        try:
            t0 = time.perf_counter()
            df = runner.run()
            t1 = time.perf_counter()
            elapsed = t1 - t0
            runtime_details = to_jsonable(runner.last_run_details)
        except Exception as e:
            entry["status"] = "failed"
            entry["error"] = str(e)
            if runner.last_run_details:
                entry["runtime_details"] = to_jsonable(runner.last_run_details)
            progress["groups"][key] = entry
            progress["status"] = "in_progress"
            progress["updated_at"] = utc_now_iso()
            save_json(progress_path, progress)
            raise

        print(f"[Group {group_id}] wrote {len(df)} rows -> {cfg.out_csv} (time: {elapsed:.2f}s)")
        entry.update({
            "status": "done",
            "rows": int(len(df)),
            "seconds": round(elapsed, 6),
            "finished_at": utc_now_iso(),
        })
        if runtime_details:
            entry["runtime_details"] = runtime_details
        entry.pop("error", None)
        progress["groups"][key] = entry
        progress["updated_at"] = utc_now_iso()
        save_json(progress_path, progress)
        write_timing_from_progress(progress, protein_name, timing_csv)

    progress = ensure_progress(progress_path, protein_name, sequence, sample_dir, timing_csv)
    write_timing_from_progress(progress, protein_name, timing_csv)
    group_csvs = [str(path) for path in done_group_csvs(progress)]

    if len(group_csvs) != GROUP_COUNT:
        progress["status"] = "in_progress" if group_csvs else "pending"
        progress["updated_at"] = utc_now_iso()
        save_json(progress_path, progress)
        print(f"[Resume] {protein_name}: {len(group_csvs)}/{GROUP_COUNT} groups completed; merged file not refreshed.")
        return ""

    timing_df = pd.read_csv(timing_csv)
    total_elapsed = pd.to_numeric(timing_df["seconds"], errors="coerce").fillna(0.0).sum()
    print(f"[Timing] -> {timing_csv} (total: {total_elapsed:.2f}s)")

    combined = []
    for fpath in group_csvs:
        try:
            df = pd.read_csv(fpath)
            combined.append(df)
        except Exception:
            pass
    if combined:
        all_df = pd.concat(combined, ignore_index=True)
        all_df.to_csv(merged_csv, index=False)
        progress["merged_csv"] = str(merged_csv)
        progress["status"] = "done"
        progress["updated_at"] = utc_now_iso()
        save_json(progress_path, progress)
        print(f"[Merged] {protein_name}: {len(all_df)} rows -> {merged_csv}")
        return str(merged_csv)
    return ""


if __name__ == "__main__":

    SAMPLES_ROOT.mkdir(parents=True, exist_ok=True)
    TRACKING_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    if USE_IBM:
        init_ibm_service()
    else:
        print(
            "[Backend] kind=simulator -> running Stage 1 on the local qiskit-aer "
            "simulator (no IBM Quantum credentials required). "
            "Set QSAD_BACKEND_KIND=ibm to submit to a real backend instead."
        )
        # Guard against a silent simulator run by someone who set up credentials
        # and expected hardware.
        if not os.getenv("QSAD_BACKEND_KIND") and os.path.exists(IBM_CONFIG_FILE):
            print(
                f"[Backend] NOTE: {IBM_CONFIG_FILE} exists but QSAD_BACKEND_KIND "
                "is unset, so this run stays on the SIMULATOR. Export "
                "QSAD_BACKEND_KIND=ibm if you meant to use the QPU."
            )

    EXAMPLES: List[Dict[str, Any]] = read_tasks(TASKS_FILE)
    write_run_manifest(len(EXAMPLES))

    all_combined = []
    failed_examples: List[Tuple[str, str]] = []
    sequence_index = load_completed_sequence_index()
    for ex in EXAMPLES:
        protein_name = ex["protein_name"]
        sequence = ex["main_chain_residue_seq"]
        sequence_key = normalize_sequence(sequence)

        merged_path = completed_merged_path(protein_name, sequence)
        if not merged_path:
            source_name = sequence_index.get(sequence_key)
            if source_name and source_name != protein_name:
                print(f"[Dedup] {protein_name}: sequence matches {source_name}, copying existing outputs.")
                merged_path = copy_completed_sampling_result(source_name, protein_name, sequence)

        try:
            if not merged_path:
                merged_path = per_example_sampling(protein_name, sequence)
        except Exception as e:
            failed_examples.append((protein_name, str(e)))
            print(f"[FAIL] {protein_name}: {e}")
            continue
        if merged_path:
            sequence_index.setdefault(sequence_key, protein_name)
            try:
                df = pd.read_csv(merged_path)
                all_combined.append(df)
            except Exception:
                pass


    if all_combined:
        all_df = pd.concat(all_combined, ignore_index=True)
        out_all = SUMMARY_ROOT / "samples_all_ibm.csv"
        all_df.to_csv(out_all, index=False)
        print(f"\n[Global merged] all proteins -> {out_all} ({len(all_df)} rows)")

    if failed_examples:
        print("\n[Summary] failed proteins:")
        for protein_name, err in failed_examples:
            print(f"  - {protein_name}: {err}")

    print("\nAll sampling runs completed.")
