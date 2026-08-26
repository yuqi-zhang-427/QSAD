#!/usr/bin/env python3
# --*-- conding:utf-8 --*--
# @time:4/25/26 17:20
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:build_fullatom_results.py
"""
build_fullatom_results.py
=========================
For each protein with Stage-2 outputs under ``pp_result/<pdbid>/``, pick the
candidate with the lowest ``E_total`` from ``energies.jsonl``, decode it,
reconstruct full-atom coordinates, apply lightweight side-chain polishing,
and save the result to ``qsad_pdb_results/``.

Pipeline (lightweight Plan B):
  1. Select best bitstring : argmin(E_total) over energies.jsonl per pdbid
  2. Decode (decom)        : bitstring -> DecodedStructure (lattice coords)
  3. CA -> full-atom       : FullAtomBuilder -> backbone + side-chain PDB
                             (skip FullAtomReranker / FastRelax)
  4. Lightweight polish    : sidechain_only_mode=True
                             backbone frozen, chi angles only, max_iter=2
                             (no MDAnnealer, no backbone perturbation)
  5. Save qsad_pdb_results/{pdbid}.pdb
     + summary  qsad_pdb_results/build_summary.csv
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# -- project root on sys.path ------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from post_process.decom import BinaryStructureDecoder, StructureDecoderConfig
from post_process.rerank.models import StructureCandidate
from post_process.ca2all import FullAtomBuilder, BuilderConfig
from runtime_paths import PROJECT_ROOT

# -- paths -------------------------------------------------------------------
PP_RESULT_ROOT = PROJECT_ROOT / "pp_result"
ENERGIES_FILENAME = "energies.jsonl"
OUTPUT_DIR  = SCRIPT_DIR / "qsad_pdb_results"

# -- optional PyRosetta-based side-chain polish ------------------------------
# Off by default: the production path emits the raw NERF + rotamer structure
# straight from FullAtomBuilder, which has zero PyRosetta dependency.
# Set the environment variable ``QSAD_ENABLE_POLISH=1`` to opt in to the
# IterativeRefiner side-chain polish stage (requires PyRosetta).
ENABLE_POLISH = os.getenv("QSAD_ENABLE_POLISH", "0").strip().lower() in {
    "1", "true", "yes", "on",
}

# -- logging -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            SCRIPT_DIR / "build_fullatom.log", mode="w", encoding="utf-8"
        ),
    ],
)
_LOG = logging.getLogger("build_fullatom")


# -- component factories -----------------------------------------------------

def _make_builder() -> FullAtomBuilder:
    """Return a FullAtomBuilder with default parameters (target CA spacing 3.8 A)."""
    return FullAtomBuilder(BuilderConfig())


def _make_refiner() -> Any:
    """
    Return a lightweight IterativeRefiner (only invoked when ``ENABLE_POLISH``
    is True; PyRosetta is imported lazily inside post_process.polish):

      - sidechain_only_mode=True : freeze backbone; auto-disables MDAnnealer
                                   and backbone FastRelax
      - max_iter=2               : at most two side-chain optimisation rounds
    """
    # Local imports so PyRosetta is not touched when polishing is disabled.
    from post_process.polish import IterativeRefiner, IterativeRefinerConfig
    cfg = IterativeRefinerConfig(
        max_iter=2,
        sidechain_only_mode=True,
    )
    return IterativeRefiner(cfg)


# -- best-candidate selection from Stage-2 output ----------------------------

def _select_best_from_energies(energies_path: Path) -> Optional[Dict[str, Any]]:
    """
    Read all rows from ``energies.jsonl`` and return the one with the lowest
    finite ``E_total``. ``None`` when the file is empty or every row is unusable.
    """
    best: Optional[Dict[str, Any]] = None
    best_e = math.inf
    with energies_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                e_total = float(rec.get("E_total"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(e_total):
                continue
            if e_total < best_e:
                best = rec
                best_e = e_total
    return best


def discover_pdbid_targets(pp_root: Path) -> List[Dict[str, Any]]:
    """
    Walk ``pp_result/<pdbid>/energies.jsonl`` and return one selected row per pdbid.

    Each entry has: ``pdbid``, ``sequence``, ``best_bitstring``, ``best_E_total``.
    Directories without a usable ``energies.jsonl`` are skipped with a warning.
    """
    if not pp_root.exists():
        _LOG.warning("Stage-2 output root not found: %s", pp_root)
        return []

    targets: List[Dict[str, Any]] = []
    for entry in sorted(pp_root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_") or entry.name.startswith("."):
            continue
        energies_path = entry / ENERGIES_FILENAME
        if not energies_path.is_file():
            _LOG.warning("[%s] missing %s; skipping", entry.name, ENERGIES_FILENAME)
            continue
        best = _select_best_from_energies(energies_path)
        if best is None:
            _LOG.warning("[%s] no usable rows in %s; skipping", entry.name, ENERGIES_FILENAME)
            continue
        bitstring = best.get("bitstring")
        sequence  = best.get("sequence")
        if not bitstring or not sequence:
            _LOG.warning(
                "[%s] selected row missing bitstring/sequence; skipping", entry.name
            )
            continue
        targets.append({
            "pdbid":          entry.name,
            "sequence":       str(sequence),
            "best_bitstring": str(bitstring),
            "best_E_total":   float(best["E_total"]),
        })
    return targets


# -- per-protein processing --------------------------------------------------

def process_one(
    row: Dict[str, Any],
    builder: FullAtomBuilder,
    refiner: Optional[Any],
) -> Dict[str, Any]:
    """
    Process one selected candidate and return a result dict.

    ``refiner`` is ``None`` when polishing is disabled (default); in that case
    the raw NERF + rotamer structure produced by :class:`FullAtomBuilder` is
    written directly and ``final_energy`` stays ``NaN``.

    Status codes:
      ok           : success
      decode_error : bitstring decoding failed
      build_error  : CA-to-full-atom build failed
      write_error  : PDB write to disk failed
      (Polish failure degrades gracefully to the raw CA2ALL result; status stays ok)
    """
    pdbid        = str(row["pdbid"])
    sequence     = str(row["sequence"])
    bitstring    = str(row["best_bitstring"])
    best_e_total = float(row["best_E_total"])

    result: Dict[str, Any] = {
        "pdbid":        pdbid,
        "sequence":     sequence,
        "length":       len(sequence),
        "best_E_total": best_e_total,
        "final_energy": float("nan"),
        "status":       "ok",
        "message":      "",
        "output_pdb":   "",
    }

    # -- step 1: decode ------------------------------------------------------
    try:
        hot_vec = [False] * len(sequence)   # coarse-grained model: no side-chain beads
        dec_cfg = StructureDecoderConfig(
            side_chain_hot_vector=hot_vec,
            fifth_bit=False,
            strict=False,
        )
        decoder = BinaryStructureDecoder(dec_cfg)
        decoded = decoder.decode(bitstring, sequence)
    except Exception as exc:
        result["status"]  = "decode_error"
        result["message"] = str(exc)
        _LOG.warning("[%s] Decode failed: %s", pdbid, exc)
        return result

    # -- step 2: CA -> full-atom ---------------------------------------------
    try:
        candidate    = StructureCandidate.from_decoded(decoded, candidate_id=pdbid)
        build_result = builder.build(candidate)
        raw_pdb      = build_result.write_pdb()
    except Exception as exc:
        result["status"]  = "build_error"
        result["message"] = str(exc)
        _LOG.warning("[%s] CA2ALL build failed: %s", pdbid, exc)
        return result

    # -- step 3: optional PyRosetta side-chain polish ------------------------
    final_pdb    = raw_pdb
    final_energy = float("nan")
    if refiner is not None:
        # Local import keeps PyRosetta-dependent symbols out of the import
        # graph entirely when ENABLE_POLISH is False.
        from post_process.ca2all.fullatom_reranker import RankedFullAtomCandidate
        try:
            ranked_cand = RankedFullAtomCandidate(
                candidate      = candidate,
                build_result   = build_result,
                fullatom_score = 0.0,   # reranker skipped; placeholder score
                fullatom_terms = {},
                rank           = 1,
                refined_pdb    = None,  # start from raw NERF structure
            )
            polished     = refiner.refine(ranked_cand)
            final_pdb    = polished.refined_pdb
            final_energy = polished.final_score
        except Exception as exc:
            _LOG.warning(
                "[%s] Polish failed, falling back to raw CA2ALL result: %s", pdbid, exc
            )

    # -- step 4: save PDB ----------------------------------------------------
    out_path = OUTPUT_DIR / f"{pdbid}.pdb"
    try:
        out_path.write_text(final_pdb, encoding="utf-8")
        result["output_pdb"]   = str(out_path)
        result["final_energy"] = final_energy
        energy_str = f"{final_energy:.3f}" if final_energy == final_energy else "N/A"
        _LOG.info("[%s] Saved -> %s  (energy=%s)", pdbid, out_path.name, energy_str)
    except Exception as exc:
        result["status"]  = "write_error"
        result["message"] = str(exc)
        _LOG.warning("[%s] PDB write failed: %s", pdbid, exc)

    return result


# -- main --------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _LOG.info("Discovering Stage-2 outputs under %s", PP_RESULT_ROOT)
    targets = discover_pdbid_targets(PP_RESULT_ROOT)
    if not targets:
        _LOG.warning("No usable pdbid targets found; nothing to do.")
        return
    _LOG.info("%d proteins to process", len(targets))

    _LOG.info("Initialising FullAtomBuilder ...")
    builder = _make_builder()

    if ENABLE_POLISH:
        _LOG.info(
            "Initialising IterativeRefiner (sidechain_only_mode=True, max_iter=2) "
            "[QSAD_ENABLE_POLISH=1, requires PyRosetta]"
        )
        refiner = _make_refiner()
    else:
        _LOG.info(
            "Polish stage disabled (default). Set QSAD_ENABLE_POLISH=1 to enable "
            "PyRosetta-based side-chain refinement."
        )
        refiner = None

    records: List[Dict[str, Any]] = []
    t_start = time.time()

    for idx, row in enumerate(targets, 1):
        pdbid = str(row["pdbid"])
        _LOG.info("--- [%d/%d] %s (E_total=%.4f) ---",
                  idx, len(targets), pdbid, row["best_E_total"])
        t1 = time.time()

        rec = process_one(row, builder, refiner)

        _LOG.info("[%s] Done in %.1f s, status: %s", pdbid, time.time() - t1, rec["status"])
        records.append(rec)

    # write summary CSV
    summary_path = OUTPUT_DIR / "build_summary.csv"
    fieldnames = [
        "pdbid", "sequence", "length", "best_E_total",
        "final_energy", "status", "message", "output_pdb",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # summary stats
    n_ok     = sum(1 for r in records if r["status"] == "ok")
    n_failed = len(records) - n_ok
    total    = time.time() - t_start

    _LOG.info("=" * 60)
    _LOG.info(
        "Finished: %d succeeded / %d failed / %d total",
        n_ok, n_failed, len(records),
    )
    _LOG.info(
        "Elapsed: %.1f s  (avg %.1f s/protein)",
        total, total / max(len(records), 1),
    )
    _LOG.info("PDB output dir : %s", OUTPUT_DIR)
    _LOG.info("Summary CSV    : %s", summary_path)

    if n_failed:
        _LOG.warning("Failed proteins:")
        for r in records:
            if r["status"] != "ok":
                _LOG.warning("  %s  [%s]  %s", r["pdbid"], r["status"], r["message"])


if __name__ == "__main__":
    main()
