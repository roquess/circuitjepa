import json
from unittest import mock

import numpy as np
import pytest
import torch
from qiskit.quantum_info import random_density_matrix

from circuitjepa import hardware
from circuitjepa.ansatz import N_ENC_PARAMS, N_PRED_PARAMS
from circuitjepa.data import make_dataset
from circuitjepa.hardware import (
    BASES,
    bloch_from_counts,
    estimate_qpu_seconds,
    get_backend,
    measure_embeddings,
    submit,
    tomography_circuits,
)
from circuitjepa.model import bound_circuit, embed
from circuitjepa.paulis import bloch_vector
from circuitjepa.train import RunResult, TrainConfig


def test_bases():
    assert BASES == ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"]


def test_bloch_from_exact_counts():
    rho = random_density_matrix(4, seed=3).data
    counts = []
    for basis in BASES:
        # rotate rho into the measurement basis; rightmost char = qubit 0
        from qiskit.quantum_info import DensityMatrix, Operator
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        for k in range(2):
            b = basis[1 - k]
            if b == "X":
                qc.h(k)
            elif b == "Y":
                qc.sdg(k)
                qc.h(k)
        probs = DensityMatrix(rho).evolve(Operator(qc)).probabilities()
        counts.append({format(i, "02b"): int(round(p * 1e6)) for i, p in enumerate(probs)})
    np.testing.assert_allclose(bloch_from_counts(counts), bloch_vector(rho), atol=1e-3)


def _run(seed=0):
    g = np.random.default_rng(seed)
    cfg = TrainConfig(variant="full", seed=seed)
    return RunResult(
        config=cfg,
        theta=list(g.uniform(0, 6, N_ENC_PARAMS)),
        theta_bar=list(g.uniform(0, 6, N_ENC_PARAMS)),
        phi=list(g.uniform(0, 6, N_PRED_PARAMS)),
        history=[],
        wall_seconds=0.0,
    )


def test_tomography_circuit_structure():
    smp = make_dataset(1, seed=1)[0]
    circs = tomography_circuits(bound_circuit(smp, "unadvanced", np.zeros(N_ENC_PARAMS), None, 0.25))
    assert len(circs) == 9
    assert all(c.num_clbits == 2 for c in circs)


def test_tomography_matches_exact_on_aer():
    run = _run()
    samples = make_dataset(3, seed=5)
    for kind in ("context", "target", "unadvanced"):
        got = measure_embeddings(samples, run, kind, get_backend("aer"), 20000)
        theta = torch.tensor(run.theta_bar if kind == "target" else run.theta, dtype=torch.float64)
        phi = torch.tensor(run.phi, dtype=torch.float64)
        exact = embed(samples, kind, theta, phi, run.config.delta).detach().numpy()
        assert np.abs(got - exact).max() < 0.05


def test_noisy_backend_offline():
    with mock.patch.object(hardware, "QiskitRuntimeService", side_effect=AssertionError):
        be = get_backend("fake_torino")
    out = measure_embeddings(make_dataset(1, seed=2), _run(), "unadvanced", be, 500)
    assert out.shape == (1, 15) and np.isfinite(out).all()


def test_budget_estimate():
    assert estimate_qpu_seconds(576, 1000) == pytest.approx(144)


def test_missing_credentials_message():
    with mock.patch.object(hardware, "QiskitRuntimeService", side_effect=Exception("no account")):
        with pytest.raises(RuntimeError, match="save_account"):
            get_backend("ibm_anything")


def test_submit_writes_job_before_wait(tmp_path):
    smp = make_dataset(1, seed=1)[0]
    circs = tomography_circuits(bound_circuit(smp, "unadvanced", np.zeros(N_ENC_PARAMS), None, 0.25))
    job = mock.Mock()
    job.job_id.return_value = "job123"
    job.result.side_effect = RuntimeError("boom")
    sampler = mock.Mock()
    sampler.run.return_value = job
    jobs = tmp_path / "sub" / "jobs.jsonl"
    with mock.patch.object(hardware, "SamplerV2", return_value=sampler):
        jid = submit(circs, get_backend("aer"), 100, jobs, "tagA")
    assert jid == "job123"
    rec = json.loads(jobs.read_text().splitlines()[-1])
    assert rec["job_id"] == "job123" and rec["tag"] == "tagA"
    assert rec["shots"] == 100 and rec["n_circuits"] == 9
    assert {"backend", "submitted_at"} <= set(rec)
