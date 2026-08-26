# --*-- coding:utf-8 --*--
# @time:10/19/25 10:12
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:beck.py

from __future__ import annotations
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from qiskit import QuantumCircuit, transpile

# Import circuit lowering helpers
from .circuits import expand_high_level, lower_for_sim

# Local simulator: AerSimulator
try:
    from qiskit_aer import AerSimulator
except Exception:
    AerSimulator = None  # type: ignore

# Optional statevector fallback
try:
    from qiskit.quantum_info import Statevector
except Exception:
    Statevector = None  # type: ignore


class SamplerBackend:
    """Abstract base for a backend producing measurement counts."""
    def __init__(self, shots: int):
        self.shots = int(shots)
        self.last_run_details: Dict[str, Any] = {}

    def run_counts(self, circuit: QuantumCircuit) -> Dict[str, int]:
        raise NotImplementedError

    def run_counts_batch(self, circuits: Sequence[QuantumCircuit]) -> List[Dict[str, int]]:
        return [self.run_counts(circuit) for circuit in circuits]


def _bit_reverse(s: str) -> str:
    """Reverse bitstring for msb/lsb alignment if needed."""
    return s[::-1]


def _quasi_to_counts(quasi: Dict[str, float], shots: int) -> Dict[str, int]:
    """Discretize a quasi-probability distribution to integer counts with total==shots."""
    raw = {k: int(round(max(0.0, float(p)) * shots)) for k, p in quasi.items()}
    return _normalize_counts_to_total(raw, shots)


def _normalize_counts_to_total(counts: Dict[str, int], shots: int) -> Dict[str, int]:
    """
    Make sure sum(counts.values()) equals shots without producing negative buckets.
    Uses proportional rescaling and drift correction over top-frequency keys.
    """
    if not counts:
        return {}
    total = sum(counts.values())
    if total == shots:
        return counts
    if total <= 0:
        scaled = {k: 0 for k in counts}
    else:
        scaled = {k: max(0, int(round(v * shots / total))) for k, v in counts.items()}
    diff = shots - sum(scaled.values())
    if diff != 0:
        keys = sorted(scaled.keys(), key=lambda k: counts[k], reverse=True)
        i = 0
        step = 1 if diff > 0 else -1
        while diff != 0 and keys:
            k = keys[i % len(keys)]
            if step < 0 and scaled[k] == 0:
                i += 1
                continue
            scaled[k] += step
            diff -= step
            i += 1
    return scaled


def _serialize_runtime_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _serialize_runtime_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_runtime_value(v) for v in value]
    return value


def _extract_counts_from_sampler_data(data: Any, shots: int) -> Dict[str, int]:
    # Path A: data.<reg>.get_counts()
    try:
        cr = getattr(data, "cr", None)
        if cr is not None and hasattr(cr, "get_counts"):
            counts = dict(cr.get_counts())
            return _normalize_counts_to_total(counts, shots)
        for name in dir(data):
            obj = getattr(data, name)
            if hasattr(obj, "get_counts"):
                counts = dict(obj.get_counts())
                return _normalize_counts_to_total(counts, shots)
    except Exception:
        pass

    # Path B: data.samples
    try:
        samples = getattr(data, "samples", None)
        if samples is not None:
            out: Dict[str, int] = {}
            for s in samples:
                key = s if isinstance(s, str) else "".join(map(str, s))
                out[key] = out.get(key, 0) + 1
            return _normalize_counts_to_total(out, shots)
    except Exception:
        pass

    # Path C: data.quasi_dists / data.quasi_dist
    try:
        quasi = getattr(data, "quasi_dists", None)
        if quasi is None:
            quasi = getattr(data, "quasi_dist", None)
        if quasi is not None:
            q0 = dict(quasi[0]) if isinstance(quasi, list) else dict(quasi)
            return _normalize_counts_to_total(_quasi_to_counts(q0, shots), shots)
    except Exception:
        pass

    # Path D: data.meas.get_counts()
    try:
        meas = getattr(data, "meas", None)
        if meas is not None and hasattr(meas, "get_counts"):
            counts = dict(meas.get_counts())
            return _normalize_counts_to_total(counts, shots)
    except Exception:
        pass

    raise RuntimeError("Unsupported SamplerV2 result format; cannot extract counts.")


# A dense statevector needs 2**n complex amplitudes (16 B each), so it stops
# fitting in RAM well before the widths used here: 2**29 = 8.6 GB, 2**33 = 137 GB.
# Circuits wider than this threshold are therefore simulated with the
# matrix-product-state method, which is exact and inexpensive for the QSAD
# ansatz (linear entanglement, reps=1).
AUTO_MPS_QUBIT_THRESHOLD = int(os.getenv("QSAD_AER_MPS_THRESHOLD", "28"))

# Hard ceiling for the Aer-less Statevector fallback path.
STATEVECTOR_QUBIT_LIMIT = 30


class LocalSimulatorBackend(SamplerBackend):
    """
    Local simulator using AerSimulator with transpile->run->get_counts flow.
    Falls back to Statevector to derive probabilities when Aer is unavailable.

    The Aer simulation method is selected per circuit: ``method`` when one is
    given explicitly (or via the ``QSAD_AER_METHOD`` environment variable),
    otherwise ``matrix_product_state`` above ``AUTO_MPS_QUBIT_THRESHOLD``
    qubits and Aer's own ``automatic`` choice below it.
    """
    def __init__(
        self,
        shots: int,
        seed: Optional[int] = None,
        method: Optional[str] = None,
    ):
        super().__init__(shots)
        self.seed = seed
        self.method = (method or os.getenv("QSAD_AER_METHOD") or "").strip().lower() or None
        if AerSimulator is None and Statevector is None:
            raise RuntimeError(
                "No local simulation path available. Install qiskit-aer or ensure qiskit.quantum_info.Statevector is present."
            )
        self._mode = "aer" if AerSimulator is not None else "statevector"
        self._sims: Dict[str, Any] = {}
        self._seed_kw = {"seed_simulator": int(self.seed)} if self.seed is not None else {}

    def resolve_method(self, num_qubits: int) -> str:
        """Return the Aer simulation method to use for a circuit of this width."""
        if self.method:
            return self.method
        if num_qubits > AUTO_MPS_QUBIT_THRESHOLD:
            return "matrix_product_state"
        return "automatic"

    def _simulator(self, num_qubits: int) -> Any:
        method = self.resolve_method(num_qubits)
        if method not in self._sims:
            self._sims[method] = AerSimulator(method=method)
        return self._sims[method]

    def run_counts(self, circuit: QuantumCircuit) -> Dict[str, int]:
        if self._mode == "statevector":
            if circuit.num_qubits > STATEVECTOR_QUBIT_LIMIT:
                raise RuntimeError(
                    f"Circuit has {circuit.num_qubits} qubits, which exceeds the "
                    f"{STATEVECTOR_QUBIT_LIMIT}-qubit ceiling of the dense Statevector "
                    "fallback (it would need "
                    f"{(2 ** circuit.num_qubits) * 16 / 1024 ** 3:.0f} GB of RAM). "
                    "Install qiskit-aer so the matrix-product-state method can be used."
                )
            # Expand library gates and remove measurements
            unitary = expand_high_level(circuit)
            try:
                unitary = unitary.remove_final_measurements(inplace=False)
            except Exception:
                qc = QuantumCircuit(circuit.num_qubits)
                for instr, qargs, cargs in unitary.data:
                    if instr.name.lower() == "measure":
                        continue
                    qc.append(instr, qargs, cargs)
                unitary = qc
            sv = Statevector.from_instruction(unitary)
            probs = sv.probabilities_dict()
            counts = {b: int(round(float(p) * self.shots)) for b, p in probs.items() if float(p) > 0.0}
            return _normalize_counts_to_total(counts, self.shots)

        # Aer path: lower to a common basis without routing
        sim = self._simulator(circuit.num_qubits)
        lowered = lower_for_sim(circuit)
        tqc = transpile(
            lowered,
            sim,
            optimization_level=1,
            seed_transpiler=self.seed if self.seed is not None else None,
        )
        job = sim.run(tqc, shots=self.shots, **self._seed_kw)
        res = job.result()
        counts = res.get_counts(tqc)
        if isinstance(counts, list):
            counts = counts[0]
        return _normalize_counts_to_total(dict(counts), self.shots)

    def run_counts_batch(self, circuits: Sequence[QuantumCircuit]) -> List[Dict[str, int]]:
        self.last_run_details = {
            "mode": "local_batch",
            "pub_count": int(len(circuits)),
            "method": (
                self.resolve_method(circuits[0].num_qubits)
                if self._mode == "aer" and circuits else self._mode
            ),
        }
        return [self.run_counts(circuit) for circuit in circuits]


class IBMSamplerBackend(SamplerBackend):
    """
    IBM Runtime SamplerV2 backend in job mode with explicit lowering and routing
    to the target backend.
    """
    def __init__(self, shots: int, backend_name: Optional[str], service: Optional[Any] = None):
        super().__init__(shots)
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        except Exception as e:
            raise RuntimeError("qiskit-ibm-runtime must be installed to use IBM backends.") from e

        self._service = service or QiskitRuntimeService()
        if backend_name:
            backend = self._service.backend(backend_name)
        else:
            backend = self._service.least_busy(operational=True, simulator=False)
        self._backend = backend
        try:
            self._sampler = SamplerV2(
                mode=self._backend,
                options={
                    "default_shots": shots,
                },
            )
        except TypeError:
            # Compatibility for older qiskit-ibm-runtime releases that still
            # accepted backend=... as a dedicated keyword.
            self._sampler = SamplerV2(
                backend=self._backend,
                options={
                    "default_shots": shots,
                },
            )

    def _backend_name(self) -> Optional[str]:
        name = getattr(self._backend, "name", None)
        return name() if callable(name) else name

    def _capture_job_details(self, job: Any, *, error: Optional[BaseException] = None) -> Dict[str, Any]:
        details: Dict[str, Any] = {
            "backend_name": self._backend_name(),
            "shots": int(self.shots),
        }

        for key, getter in (
            ("job_id", lambda: job.job_id()),
            ("status", lambda: job.status()),
            ("creation_date", lambda: getattr(job, "creation_date", None)),
            ("usage_estimation", lambda: getattr(job, "usage_estimation", None)),
            ("metrics", lambda: job.metrics()),
            ("error_message", lambda: job.error_message()),
            ("session_id", lambda: getattr(job, "session_id", None)),
        ):
            try:
                details[key] = _serialize_runtime_value(getter())
            except Exception:
                continue

        if error is not None:
            details["exception"] = str(error)

        return details

    def run_counts(self, circuit: QuantumCircuit) -> Dict[str, int]:
        counts_list = self.run_counts_batch([circuit])
        return counts_list[0]

    def run_counts_batch(self, circuits: Sequence[QuantumCircuit]) -> List[Dict[str, int]]:
        if not circuits:
            self.last_run_details = {
                "mode": "ibm_job_batch",
                "pub_count": 0,
                "job": None,
            }
            return []

        lowered = [expand_high_level(circuit) for circuit in circuits]
        transpiled = transpile(lowered, backend=self._backend, optimization_level=3)
        if isinstance(transpiled, QuantumCircuit):
            transpiled_circuits = [transpiled]
        else:
            transpiled_circuits = list(transpiled)

        job = self._sampler.run(transpiled_circuits, shots=self.shots)
        base_details = {
            "mode": "ibm_job_batch",
            "pub_count": int(len(transpiled_circuits)),
            "job": {},
        }

        try:
            result = job.result()
        except Exception as exc:
            base_details["job"] = self._capture_job_details(job, error=exc)
            self.last_run_details = base_details
            raise

        base_details["job"] = self._capture_job_details(job)
        self.last_run_details = base_details

        counts_list: List[Dict[str, int]] = []
        for pub in result:
            counts_list.append(_extract_counts_from_sampler_data(pub.data, self.shots))

        if len(counts_list) != len(transpiled_circuits):
            raise RuntimeError(
                f"Sampler returned {len(counts_list)} pubs for {len(transpiled_circuits)} circuits."
            )

        return counts_list


def make_backend(
    kind: str,
    shots: int,
    seed_sim: Optional[int],
    ibm_backend: Optional[str],
    service: Optional[Any] = None,
    sim_method: Optional[str] = None,
) -> SamplerBackend:
    kind = (kind or "simulator").lower()
    if kind == "simulator":
        return LocalSimulatorBackend(shots=shots, seed=seed_sim, method=sim_method)
    elif kind == "ibm":
        return IBMSamplerBackend(shots=shots, backend_name=ibm_backend, service=service)
    else:
        raise ValueError(f"Unknown backend kind: {kind}")
