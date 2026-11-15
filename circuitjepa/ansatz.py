"""Parameterised encoder and predictor ansatz templates."""

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from circuitjepa.config import LATENT, N_QUBITS

ENC_LAYERS = 3
PRED_LAYERS = 2
N_ENC_PARAMS = ENC_LAYERS * N_QUBITS * 2
N_PRED_PARAMS = PRED_LAYERS * len(LATENT) * 2

THETA = ParameterVector("theta", N_ENC_PARAMS)
PHI = ParameterVector("phi", N_PRED_PARAMS)


def encoder() -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS)
    k = 0
    for _ in range(ENC_LAYERS):
        for q in range(N_QUBITS):
            qc.ry(THETA[k], q)
            qc.rz(THETA[k + 1], q)
            k += 2
        for q in range(N_QUBITS - 1):
            qc.cz(q, q + 1)
    return qc


def predictor() -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS)
    k = 0
    for _ in range(PRED_LAYERS):
        for q in LATENT:
            qc.ry(PHI[k], q)
            qc.rz(PHI[k + 1], q)
            k += 2
        qc.cz(*LATENT)
    return qc
