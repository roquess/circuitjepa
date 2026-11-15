import numpy as np
import torch
from qiskit.quantum_info import Statevector, partial_trace

from qjepa.ansatz import N_ENC_PARAMS, N_PRED_PARAMS, encoder, predictor
from qjepa.data import Sample, make_dataset
from qjepa.model import bound_circuit, embed
from qjepa.paulis import bloch_vector

DELTA = 0.3


def _params(seed=0):
    g = torch.Generator().manual_seed(seed)
    theta = (torch.rand(N_ENC_PARAMS, generator=g, dtype=torch.float64) * 6).requires_grad_()
    phi = (torch.rand(N_PRED_PARAMS, generator=g, dtype=torch.float64) * 6).requires_grad_()
    return theta, phi


def test_param_counts():
    assert encoder().num_parameters == 30
    assert predictor().num_parameters == 8
    used = {predictor().find_bit(q).index for inst in predictor().data for q in inst.qubits}
    assert used == {0, 1}


def test_embed_matches_bound_circuit():
    samples = make_dataset(6, seed=1)
    samples[0] = Sample(samples[0].a, samples[0].b, 0, 1)
    samples[1] = Sample(samples[1].a, samples[1].b, 2, 3)
    theta, phi = _params()
    for kind in ("context", "target", "unadvanced"):
        out = embed(samples, kind, theta, phi, DELTA).detach().numpy()
        for row, smp in zip(out, samples):
            qc = bound_circuit(smp, kind, theta.detach().numpy(), phi.detach().numpy(), DELTA)
            rho = partial_trace(Statevector(qc), [2, 3, 4]).data
            np.testing.assert_allclose(row, bloch_vector(rho), atol=1e-8)


def test_embed_preserves_order():
    samples = make_dataset(8, seed=2)
    theta, phi = _params()
    perm = [3, 0, 7, 1, 6, 2, 5, 4]
    base = embed(samples, "context", theta, phi, DELTA)
    shuf = embed([samples[i] for i in perm], "context", theta, phi, DELTA)
    assert torch.allclose(base[perm], shuf, atol=1e-10)


def test_no_predictor_equals_unadvanced():
    samples = make_dataset(5, seed=3)
    theta, _ = _params()
    a = embed(samples, "context", theta, None, DELTA, use_predictor=False)
    b = embed(samples, "unadvanced", theta, None, DELTA)
    assert torch.allclose(a, b, atol=1e-12)


def test_gradients_flow():
    samples = make_dataset(6, seed=4)
    theta, phi = _params()
    embed(samples, "context", theta, phi, DELTA).sum().backward()
    for g in (theta.grad, phi.grad):
        assert torch.isfinite(g).all()
        assert g.abs().sum() > 0
