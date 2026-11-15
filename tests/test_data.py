import numpy as np
import scipy.linalg
from qiskit.quantum_info import SparsePauliOp, Statevector

from circuitjepa.config import H_FIELD, J, STEPS
from circuitjepa.data import (
    A,
    B,
    Sample,
    data_values,
    magnetisation,
    make_dataset,
    prep_circuit,
    trotter_step,
)


def test_dataset_deterministic():
    assert make_dataset(20, 1) == make_dataset(20, 1)
    assert make_dataset(20, 1) != make_dataset(20, 2)


def test_dataset_ranges():
    ds = make_dataset(300, 0, t0_max=3)
    assert len(ds) == 300
    assert {x.t0 for x in ds} == {0, 1, 2, 3}
    assert {x.s for x in ds} == set(STEPS)
    assert all(0 <= x.t0 <= 3 for x in ds)


def test_trotter_matches_exact_evolution():
    s = Sample(a=(0.4, 1.1, 2.0), b=(0.3, 4.0, 5.5), t0=0, s=1)
    delta, n = 0.01, 10
    qc = prep_circuit().assign_parameters(
        {**dict(zip(A, s.a)), **dict(zip(B, s.b))}
    )
    step = trotter_step(delta)
    for _ in range(n):
        qc = qc.compose(step)
    got = Statevector(qc).data.reshape(4, 8)  # qubits 3,4 are |00>
    psi_t = got[0]
    assert np.isclose(np.linalg.norm(psi_t), 1.0)

    h = SparsePauliOp.from_sparse_list(
        [("ZZ", [0, 1], -J), ("ZZ", [1, 2], -J)]
        + [("X", [i], -H_FIELD) for i in range(3)],
        num_qubits=3,
    ).to_matrix()
    prep = prep_circuit().assign_parameters(
        {**dict(zip(A, s.a)), **dict(zip(B, s.b))}
    )
    psi0 = Statevector(prep).data.reshape(4, 8)[0]
    exact = scipy.linalg.expm(-1j * h * delta * n) @ psi0
    assert abs(np.vdot(exact, psi_t)) ** 2 > 0.999


def test_magnetisation_product_state():
    z = Sample(a=(0.0, 0.0, 0.0), b=(0.0, 0.0, 0.0), t0=0, s=1)
    p = Sample(a=(np.pi,) * 3, b=(0.0, 0.0, 0.0), t0=0, s=1)
    assert np.allclose(magnetisation(z, 0, 0.1), [1, 1, 1])
    assert np.allclose(magnetisation(p, 0, 0.1), [-1, -1, -1])


def test_data_values_shapes():
    ds = make_dataset(5, 0)
    v = data_values(ds)
    assert v[A[0]].shape == (5,) and str(v[B[2]].dtype) == "torch.float64"
