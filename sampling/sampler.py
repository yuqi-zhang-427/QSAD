# --*-- conding:utf-8 --*--
# @time:10/19/25 10:18
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:sampler.py

from __future__ import annotations
from typing import Any, Dict, List, Optional
from qiskit.quantum_info import SparsePauliOp

from .config import SamplingConfig
from .backends import make_backend
from .circuits import make_sampling_circuit, build_ansatz, random_params
from .utils import counts_to_rows, write_csv, circuit_hash


class SamplingRunner:
    """
    Orchestrates the full quantum sampling process.

    Parameters
    ----------
    cfg : SamplingConfig
        Configuration describing seeds, betas, backend, etc.
    H_sparse : SparsePauliOp
        The Hamiltonian defining the system.
    meta : Optional[dict]
        Optional metadata to be written into each CSV row.
    """

    def __init__(
        self,
        cfg: SamplingConfig,
        H_sparse: SparsePauliOp,
        meta: Optional[Dict[str, Any]] = None,
        backend_service: Optional[Any] = None,
    ):
        self.cfg = cfg
        self.H = H_sparse
        self.meta = meta or {}
        self.backend_service = backend_service
        self.last_run_details: Dict[str, Any] = {}
        self.cfg.validate()

    def _derived_seed(self, seed_index: int) -> int:
        if self.cfg.seed_base is None:
            return int(seed_index)
        modulus = (2 ** 31) - 1
        return int((int(self.cfg.seed_base) + int(seed_index)) % modulus)

    def run(self):
        """Run all sampling jobs and return a pandas DataFrame."""
        be = make_backend(
            kind=self.cfg.backend.kind,
            shots=self.cfg.backend.shots,
            seed_sim=self.cfg.backend.seed_sim,
            ibm_backend=self.cfg.backend.ibm_backend,
            service=self.backend_service,
            sim_method=self.cfg.backend.sim_method,
        )
        self.last_run_details = {}

        rows: List[dict] = []
        n_qubits = getattr(self.H, "num_qubits", self.cfg.n_qubits_hint or 1)

        # Pre-build ansatz template once
        ans = build_ansatz(n_qubits, reps=self.cfg.reps, entanglement=self.cfg.entanglement)
        circuits = []
        metas: List[Dict[str, Any]] = []

        for seed_index in range(self.cfg.seeds):
            actual_seed = self._derived_seed(seed_index)
            params = random_params(ans, actual_seed)
            for beta in self.cfg.betas:
                circ = make_sampling_circuit(
                    n_qubits=n_qubits,
                    H=self.H,
                    beta=beta,
                    params=params,
                    reps=self.cfg.reps,
                    entanglement=self.cfg.entanglement,
                )
                circuits.append(circ)

                meta = dict(
                    L=self.cfg.L,
                    n_qubits=n_qubits,
                    shots=self.cfg.backend.shots,
                    beta=float(beta),
                    seed=int(actual_seed),
                    seed_index=int(seed_index),
                    label=self.cfg.label,
                    backend=self.cfg.backend.kind,
                    ibm_backend=self.cfg.backend.ibm_backend,
                    circuit_hash=circuit_hash(circ.qasm()) if hasattr(circ, "qasm") else None,
                    **self.cfg.extra_meta,
                    **self.meta,
                )
                metas.append(meta)

        try:
            counts_batches = be.run_counts_batch(circuits)
        finally:
            self.last_run_details = getattr(be, "last_run_details", {}) or {}

        if len(counts_batches) != len(metas):
            raise RuntimeError(
                f"Backend returned {len(counts_batches)} count batches for {len(metas)} circuits."
            )

        runtime_row_meta: Dict[str, Any] = {}
        runtime_job = self.last_run_details.get("job") if isinstance(self.last_run_details, dict) else None
        if isinstance(runtime_job, dict):
            job_id = runtime_job.get("job_id")
            if job_id:
                runtime_row_meta["runtime_job_id"] = job_id

        for meta, counts in zip(metas, counts_batches):
            row_meta = dict(meta)
            row_meta.update(runtime_row_meta)
            rows.extend(counts_to_rows(counts, row_meta))

        df = write_csv(rows, self.cfg.out_csv, write_parquet=self.cfg.write_parquet)
        return df
