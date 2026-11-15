"""Differentiable statevector interpreter for Qiskit circuits (torch, complex128).

Qubit ordering matches Qiskit: qubit q is bit q of the basis index, so qubit 0
is the least-significant bit. The state is held as a tensor of shape
(batch, 2, ..., 2) where the axis of qubit q is ``n - q``.
"""

import numpy as np
import torch
from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit

from qjepa.config import LATENT
from qjepa.paulis import PAULI_MATS

_CDTYPE = torch.complex128
_RDTYPE = torch.float64
_PAULI_T = torch.as_tensor(PAULI_MATS, dtype=_CDTYPE)
_SUPPORTED = {"rx", "ry", "rz", "rzz", "cz", "h"}


def _angle(param, values, batch) -> torch.Tensor:
    if isinstance(param, Parameter):
        if param not in values:
            raise KeyError(f"unbound parameter {param.name}")
        theta = torch.as_tensor(values[param], dtype=_RDTYPE)
    elif isinstance(param, ParameterExpression):
        if param.parameters:
            raise NotImplementedError("parameter expressions are not supported")
        theta = torch.as_tensor(float(param), dtype=_RDTYPE)
    else:
        theta = torch.as_tensor(float(param), dtype=_RDTYPE)
    return theta.expand(batch) if theta.dim() == 0 else theta


def _apply_1q(state, mat, ax):
    # mat: (B, 2, 2); contract its column index with the target axis.
    s = state.movedim(ax, -1)
    s = torch.einsum("b...j,bij->b...i", s, mat)
    return s.movedim(-1, ax)


def _apply_diag(state, diag, axes, n):
    # diag: (B, 2, 2) over the two target axes (in the order given by `axes`).
    shape = [1] * (n + 1)
    shape[0] = diag.shape[0]
    for a in axes:
        shape[a] = 2
    if axes[0] > axes[1]:
        diag = diag.transpose(1, 2)
    return state * diag.reshape(shape)


def simulate(circuit: QuantumCircuit, values: dict, batch: int) -> torch.Tensor:
    """Run ``circuit`` from |0...0> and return the (batch, 2**n) statevector."""
    n = circuit.num_qubits
    state = torch.zeros((batch,) + (2,) * n, dtype=_CDTYPE)
    state[(slice(None),) + (0,) * n] = 1.0
    for inst in circuit.data:
        op = inst.operation
        name = op.name
        if name not in _SUPPORTED:
            raise NotImplementedError(name)
        qs = [circuit.find_bit(q).index for q in inst.qubits]
        axes = [n - q for q in qs]
        if name == "h":
            h = torch.tensor([[1, 1], [1, -1]], dtype=_CDTYPE) / np.sqrt(2)
            state = _apply_1q(state, h.expand(batch, 2, 2), axes[0])
        elif name == "cz":
            d = torch.tensor([[1, 1], [1, -1]], dtype=_CDTYPE).expand(batch, 2, 2)
            state = _apply_diag(state, d, axes, n)
        else:
            th = _angle(op.params[0], values, batch)
            if name == "rzz":
                sign = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=_RDTYPE)
                d = torch.exp(-0.5j * th[:, None, None] * sign)
                state = _apply_diag(state, d, axes, n)
                continue
            c, s = torch.cos(th / 2), torch.sin(th / 2)
            z = torch.zeros_like(c)
            if name == "rx":
                m = torch.stack([c + 0j, -1j * s, -1j * s, c + 0j], -1)
            elif name == "ry":
                m = torch.stack([c + 0j, -s + 0j, s + 0j, c + 0j], -1)
            else:  # rz
                m = torch.stack([torch.exp(-0.5j * th), z + 0j, z + 0j, torch.exp(0.5j * th)], -1)
            state = _apply_1q(state, m.reshape(batch, 2, 2), axes[0])
    return state.reshape(batch, 2**n)


def latent_rho(state: torch.Tensor, latent: tuple = LATENT, n: int = 5) -> torch.Tensor:
    """Reduced density matrix (batch, 4, 4) of the latent qubits, latent[0] least significant."""
    batch = state.shape[0]
    t = state.reshape((batch,) + (2,) * n)
    ax = lambda q: n - q  # noqa: E731
    rest = [ax(q) for q in range(n - 1, -1, -1) if q not in latent]
    order = [0] + rest + [ax(latent[1]), ax(latent[0])]
    psi = t.permute(order).reshape(batch, -1, 4)
    return torch.einsum("bra,brc->bac", psi, psi.conj())


def bloch_torch(rho: torch.Tensor) -> torch.Tensor:
    """Real Pauli expectations Tr(rho P), shape (batch, 15)."""
    return torch.einsum("bij,pji->bp", rho, _PAULI_T.to(rho.dtype)).real


def cosine_torch(r: torch.Tensor, s: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Torch twin of ``qjepa.paulis.cosine`` (same smoothed norm)."""
    num = (r * s).sum(-1)
    nr = torch.sqrt((r * r).sum(-1) + eps**2)
    ns = torch.sqrt((s * s).sum(-1) + eps**2)
    return num / (nr * ns)
