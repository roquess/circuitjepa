"""QJEPA circuit templates, batched differentiable embedding, bound circuits."""

from functools import lru_cache
from typing import Literal

import numpy as np
import torch
from qiskit import QuantumCircuit

from qjepa.ansatz import PHI, THETA, encoder, predictor
from qjepa.config import LATENT
from qjepa.data import A, B, Sample, data_values, prep_circuit, trotter_step
from qjepa.sim import bloch_torch, latent_rho, simulate

Kind = Literal["context", "target", "unadvanced"]


@lru_cache(maxsize=None)
def template(
    kind: Kind, t0: int, s: int, delta: float, use_predictor: bool = True
) -> QuantumCircuit:
    steps = t0 + s if kind == "target" else t0
    qc = prep_circuit()
    step = trotter_step(delta)
    for _ in range(steps):
        qc = qc.compose(step)
    qc = qc.compose(encoder())
    if kind == "context" and use_predictor:
        pred = predictor()
        for _ in range(s):
            qc = qc.compose(pred)
    return qc


def _uses_phi(kind: Kind, use_predictor: bool) -> bool:
    return kind == "context" and use_predictor


def embed(
    samples: list[Sample],
    kind: Kind,
    theta: torch.Tensor,
    phi: torch.Tensor | None,
    delta: float,
    use_predictor: bool = True,
) -> torch.Tensor:
    groups: dict[tuple[int, int], list[int]] = {}
    for i, smp in enumerate(samples):
        groups.setdefault((smp.t0, smp.s), []).append(i)
    rows, order = [], []
    for (t0, s), idx in groups.items():
        sub = [samples[i] for i in idx]
        qc = template(kind, t0, s, delta, use_predictor)
        values = data_values(sub)
        for k in range(len(THETA)):
            values[THETA[k]] = theta[k]
        if _uses_phi(kind, use_predictor):
            for k in range(len(PHI)):
                values[PHI[k]] = phi[k]
        state = simulate(qc, values, len(sub))
        rows.append(bloch_torch(latent_rho(state, LATENT)))
        order.extend(idx)
    out = torch.cat(rows, 0)
    inv = torch.empty(len(order), dtype=torch.long)
    inv[torch.as_tensor(order)] = torch.arange(len(order))
    return out[inv]


def bound_circuit(
    sample: Sample,
    kind: Kind,
    theta: np.ndarray,
    phi: np.ndarray | None,
    delta: float,
    use_predictor: bool = True,
) -> QuantumCircuit:
    qc = template(kind, sample.t0, sample.s, delta, use_predictor)
    binding = {**dict(zip(A, sample.a)), **dict(zip(B, sample.b))}
    binding.update({THETA[k]: float(theta[k]) for k in range(len(THETA))})
    if _uses_phi(kind, use_predictor):
        binding.update({PHI[k]: float(phi[k]) for k in range(len(PHI))})
    return qc.assign_parameters(binding)
