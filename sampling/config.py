# --*-- coding:utf-8 --*--
# @time:10/19/25 10:10
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:config.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any, Union, Sequence


@dataclass
class BackendConfig:
    """
    Configuration for the quantum backend.
    """
    kind: Literal["simulator", "ibm"] = "simulator"
    shots: int = 1024
    seed_sim: Optional[int] = None
    ibm_backend: Optional[str] = None
    # Aer simulation method ("statevector", "matrix_product_state", ...).
    # None lets LocalSimulatorBackend pick one from the circuit width.
    sim_method: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "shots": int(self.shots),
            "seed_sim": self.seed_sim,
            "ibm_backend": self.ibm_backend,
            "sim_method": self.sim_method,
        }


@dataclass
class SamplingConfig:
    """
    Configuration for quantum sampling.
    """
    L: Optional[int]
    betas: List[float]
    seeds: int = 8
    reps: int = 1
    entanglement: Union[str, Sequence[Sequence[int]]] = "linear"
    label: str = "default"
    backend: BackendConfig = field(default_factory=BackendConfig)
    out_csv: str = "samples.csv"
    write_parquet: bool = False
    extra_meta: Dict[str, Any] = field(default_factory=dict)
    seed_base: Optional[int] = None  # reproducibility anchor for derived seeds

    @property
    def n_qubits_hint(self) -> Optional[int]:
        """
        Optional hint for number of qubits when using tetrahedral-like encodings.
        """
        if self.L is None:
            return None
        return max(1, 2 * (self.L - 1))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "L": self.L,
            "betas": list(self.betas),
            "seeds": int(self.seeds),
            "reps": int(self.reps),
            "entanglement": self.entanglement,
            "label": self.label,
            "out_csv": self.out_csv,
            "write_parquet": bool(self.write_parquet),
            "backend": self.backend.as_dict(),
            "extra_meta": dict(self.extra_meta),
            "n_qubits_hint": self.n_qubits_hint,
            "seed_base": self.seed_base,
        }

    def validate(self) -> None:
        if not self.betas:
            raise ValueError("betas must be a non-empty list, e.g. [0.0, 0.5, 1.0].")
        if self.seeds <= 0:
            raise ValueError("seeds must be > 0.")
        if self.backend.shots <= 0:
            raise ValueError("backend.shots must be > 0.")
        if isinstance(self.entanglement, str):
            if not self.entanglement:
                raise ValueError("entanglement must be a non-empty string.")
        else:
            # Basic structural check for connectivity lists
            for item in self.entanglement:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    raise ValueError("entanglement connectivity must be sequences of length >= 2.")
