# --*-- conding:utf-8 --*--
# @time:10/19/25 10:28
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:utils.py

from __future__ import annotations
import hashlib
from typing import Dict, Iterable, List, Any, Optional

import pandas as pd


def circuit_hash(qasm_text: str) -> str:
    """
    Compute a short SHA1 hash for a circuit's QASM text.
    Useful for experiment provenance in output tables.
    """
    return hashlib.sha1(qasm_text.encode()).hexdigest()[:16]


def counts_to_rows(counts: Dict[str, int], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert a counts dictionary and shared metadata into a list of flat rows.

    Parameters
    ----------
    counts : Dict[str, int]
        Mapping from bitstring to integer count.
    meta : Dict[str, Any]
        Metadata to copy into each row. Should include "shots" if you want "prob" computed.

    Returns
    -------
    List[Dict[str, Any]]
        One row per bitstring with fields: meta..., bitstring, count, prob.
    """
    rows: List[Dict[str, Any]] = []
    shots = int(meta.get("shots", 0))
    for bitstring, count in counts.items():
        row = dict(meta)
        row["bitstring"] = bitstring
        row["count"] = int(count)
        row["prob"] = (count / shots) if shots else None
        rows.append(row)
    return rows


def write_csv(
    rows: Iterable[Dict[str, Any]],
    path: str,
    write_parquet: bool = False,
    parquet_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Write rows to a CSV file (and optionally Parquet) and return a DataFrame.

    Parameters
    ----------
    rows : Iterable[Dict[str, Any]]
        Flat rows to be written.
    path : str
        Output CSV path.
    write_parquet : bool
        If True, also write a Parquet file.
    parquet_path : Optional[str]
        Optional Parquet path. If None and write_parquet=True,
        the path is derived by replacing the CSV extension with '.parquet'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing all rows.
    """
    df = pd.DataFrame(list(rows))
    df.to_csv(path, index=False)

    if write_parquet:
        pq = parquet_path
        if pq is None:
            if "." in path:
                pq = path.rsplit(".", 1)[0] + ".parquet"
            else:
                pq = path + ".parquet"
        df.to_parquet(pq, index=False)

    return df
