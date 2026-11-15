"""Two-qubit Pauli embeddings (Bloch-style vectors of 4x4 density matrices)."""

import numpy as np
from qiskit.quantum_info import Pauli

# Qiskit label convention: the rightmost character acts on latent qubit 0.
PAULI_LABELS: list[str] = [a + b for a in "IXYZ" for b in "IXYZ"][1:]
PAULI_MATS: np.ndarray = np.stack([Pauli(label).to_matrix() for label in PAULI_LABELS])


def bloch_vector(rho: np.ndarray) -> np.ndarray:
    """Map rho (..., 4, 4) to its real Pauli expectations Tr(rho P), shape (..., 15)."""
    return np.einsum("...ij,pji->...p", rho, PAULI_MATS).real


def cosine(r: np.ndarray, s: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Cosine similarity along the last axis, smoothed so zero vectors stay finite."""
    num = np.sum(r * s, axis=-1)
    nr = np.sqrt(np.sum(r * r, axis=-1) + eps**2)
    ns = np.sqrt(np.sum(s * s, axis=-1) + eps**2)
    return num / (nr * ns)
