import numpy as np
import pytest
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, partial_trace

from circuitjepa.sim import latent_rho, simulate


def _random_circuit(seed, n=5, n_gates=40):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    values = {}
    for i in range(n_gates):
        kind = rng.choice(["rx", "ry", "rz", "rzz", "cz", "h"])
        if kind in ("cz", "rzz"):
            q = [int(x) for x in rng.choice(n, 2, replace=False)]
        else:
            q = [int(rng.integers(n))]
        if kind in ("rx", "ry", "rz", "rzz"):
            ang = float(rng.uniform(-np.pi, np.pi))
            if rng.random() < 0.5:
                p = Parameter(f"p{i}")
                values[p] = torch.tensor(ang, dtype=torch.float64)
                getattr(qc, kind)(p, *q)
            else:
                getattr(qc, kind)(ang, *q)
        else:
            getattr(qc, kind)(*q)
    return qc, values


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_simulate_matches_qiskit(seed):
    qc, values = _random_circuit(seed)
    out = simulate(qc, values, batch=1)
    bound = qc.assign_parameters({p: float(v) for p, v in values.items()})
    ref = Statevector(bound).data
    assert out.shape == (1, 32)
    assert np.allclose(out[0].numpy(), ref, atol=1e-10)


def test_batched_angles():
    th = Parameter("th")
    qc = QuantumCircuit(5)
    qc.h(0)
    qc.h(3)
    qc.ry(th, 3)
    qc.rzz(th, 0, 3)
    qc.cz(3, 4)
    angles = torch.tensor([0.3, -1.1, 2.0], dtype=torch.float64)
    batched = simulate(qc, {th: angles}, batch=3)
    for i in range(3):
        single = simulate(qc, {th: angles[i]}, batch=1)
        assert torch.allclose(batched[i], single[0], atol=1e-12)


def test_latent_rho_matches_partial_trace():
    qc, values = _random_circuit(7)
    state = simulate(qc, values, batch=1)
    rho = latent_rho(state)
    bound = qc.assign_parameters({p: float(v) for p, v in values.items()})
    ref = partial_trace(Statevector(bound), [2, 3, 4]).data
    assert rho.shape == (1, 4, 4)
    assert np.allclose(rho[0].numpy(), ref, atol=1e-10)


def test_unsupported_gate_raises():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    with pytest.raises(NotImplementedError):
        simulate(qc, {}, batch=1)
