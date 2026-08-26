# --*-- conding:utf-8 --*--
# @time:10/19/25 10:15
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:circuits.py

from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit import transpile

# EfficientSU2 import
try:
    from qiskit.circuit.library import EfficientSU2
except Exception:
    from qiskit.circuit.library.n_local import EfficientSU2  # type: ignore

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter


def build_ansatz(n_qubits: int, reps: int = 1, entanglement: str = "linear") -> EfficientSU2:
    return EfficientSU2(n_qubits, entanglement=entanglement, reps=reps)


def random_params(ansatz: EfficientSU2, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 2 * np.pi * rng.random(ansatz.num_parameters)


def apply_beta_layer(circ: QuantumCircuit, H: SparsePauliOp, beta: float, trotter: int = 1) -> QuantumCircuit:
    if beta == 0:
        return circ
    evo = PauliEvolutionGate(H, time=beta, synthesis=SuzukiTrotter(order=2, reps=trotter))
    circ.append(evo, range(circ.num_qubits))
    # QAOA-style transverse-field mixer: turns the diagonal phase kick e^{-iβH}
    # into constructive/destructive interference on Z-basis measurement
    # probabilities. Without this layer, β is a no-op because H is diagonal.
    for q in range(circ.num_qubits):
        circ.rx(2.0 * beta, q)
    return circ


def make_sampling_circuit(
    n_qubits: int,
    H: SparsePauliOp,
    beta: float,
    params: np.ndarray,
    reps: int = 1,
    entanglement: str = "linear",
    trotter: int = 1
) -> QuantumCircuit:
    ans = build_ansatz(n_qubits, reps=reps, entanglement=entanglement)

    # Map parameters using ansatz's parameter order
    param_list = list(ans.parameters)
    if len(param_list) != len(params):
        raise ValueError(f"Parameter size mismatch: expected {len(param_list)}, got {len(params)}")
    mapping = {p: float(v) for p, v in zip(param_list, params)}

    # Compose + assign
    circ = QuantumCircuit(n_qubits)
    circ.compose(ans, inplace=True)
    circ = circ.assign_parameters(mapping, inplace=False)

    # Optional: a lightweight decompose; transpile will also unroll this.
    # circ = circ.decompose(reps=1)

    # β evolution and measurement
    circ = apply_beta_layer(circ, H, beta, trotter=trotter)
    circ.measure_all()
    return circ

def expand_high_level(circ: QuantumCircuit) -> QuantumCircuit:
    """
    Expand library gates (e.g., EfficientSU2, PauliEvolutionGate) into basic gates.
    Multiple reps helps descend nested library gates.
    """
    return circ.decompose(reps=3)

def lower_for_sim(circ: QuantumCircuit) -> QuantumCircuit:
    """
    Lower to a common IBM-like basis for local simulators without routing.
    """
    circ = expand_high_level(circ)
    return transpile(
        circ,
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=1,
        coupling_map=None,
    )
