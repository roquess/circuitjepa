import numpy as np
import torch

from circuitjepa.paulis import PAULI_LABELS, PAULI_MATS, bloch_vector, cosine
from circuitjepa.sim import cosine_torch


def _random_rho(rng):
    a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho = a @ a.conj().T
    return rho / np.trace(rho).real


def test_labels_and_mats_shape():
    assert len(PAULI_LABELS) == 15
    assert PAULI_LABELS[0] == "IX" and PAULI_LABELS[-1] == "ZZ"
    assert PAULI_MATS.shape == (15, 4, 4)


def test_bloch_of_basis_states():
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = 1.0
    b = bloch_vector(rho)
    for label, v in zip(PAULI_LABELS, b):
        assert np.isclose(v, 1.0 if label in ("IZ", "ZI", "ZZ") else 0.0)


def test_bloch_maximally_mixed_is_zero():
    assert np.allclose(bloch_vector(np.eye(4) / 4), 0.0)


def test_overlap_identity():
    rng = np.random.default_rng(0)
    for _ in range(5):
        rho, sigma = _random_rho(rng), _random_rho(rng)
        lhs = np.trace(rho @ sigma).real
        rhs = (1 + bloch_vector(rho) @ bloch_vector(sigma)) / 4
        assert np.isclose(lhs, rhs, atol=1e-10)


def test_cosine_zero_vector_is_finite():
    c = cosine(np.zeros(15), np.ones(15))
    assert np.isfinite(c) and c == 0
    r = torch.zeros(2, 15, dtype=torch.float64, requires_grad=True)
    s = torch.ones(2, 15, dtype=torch.float64)
    cosine_torch(r, s).sum().backward()
    assert torch.isfinite(r.grad).all()
