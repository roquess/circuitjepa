"""Transverse-field Ising dataset: sampling, circuit templates, exact reference."""

from dataclasses import dataclass

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector

from qjepa.config import H_FIELD, J, N_DATA, N_QUBITS, STEPS

A = ParameterVector("a", 3)
B = ParameterVector("b", 3)


@dataclass(frozen=True)
class Sample:
    a: tuple[float, float, float]
    b: tuple[float, float, float]
    t0: int
    s: int


def make_dataset(
    n: int, seed: int, t0_max: int = 3, steps: tuple[int, ...] = STEPS
) -> list[Sample]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        a = tuple(float(x) for x in rng.uniform(0.0, np.pi, 3))
        b = tuple(float(x) for x in rng.uniform(0.0, 2 * np.pi, 3))
        t0 = int(rng.integers(0, t0_max + 1))
        s = int(steps[rng.integers(0, len(steps))])
        out.append(Sample(a, b, t0, s))
    return out


def prep_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS)
    for i in range(N_DATA):
        qc.ry(A[i], i)
        qc.rz(B[i], i)
    return qc


def trotter_step(delta: float) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS)
    qc.rzz(-2 * J * delta, 0, 1)
    qc.rzz(-2 * J * delta, 1, 2)
    for i in range(N_DATA):
        qc.rx(-2 * H_FIELD * delta, i)
    return qc


def data_values(samples: list[Sample]) -> dict[Parameter, torch.Tensor]:
    a = torch.tensor([s.a for s in samples], dtype=torch.float64)
    b = torch.tensor([s.b for s in samples], dtype=torch.float64)
    vals: dict[Parameter, torch.Tensor] = {}
    for i in range(N_DATA):
        vals[A[i]] = a[:, i]
        vals[B[i]] = b[:, i]
    return vals


def magnetisation(sample: Sample, steps: int, delta: float) -> np.ndarray:
    qc = prep_circuit().assign_parameters(
        {**dict(zip(A, sample.a)), **dict(zip(B, sample.b))}
    )
    step = trotter_step(delta)
    for _ in range(steps):
        qc = qc.compose(step)
    probs = Statevector(qc).probabilities(range(N_DATA))
    idx = np.arange(2**N_DATA)
    return np.array(
        [float(np.sum(probs * (1 - 2 * ((idx >> i) & 1)))) for i in range(N_DATA)]
    )
