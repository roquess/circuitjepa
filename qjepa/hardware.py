"""Pauli-basis tomography of the latent qubits on Aer, fake and IBM backends."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from qjepa.config import LATENT
from qjepa.model import Kind, bound_circuit
from qjepa.paulis import PAULI_LABELS

# Qiskit convention: the rightmost character is latent qubit 0.
BASES: list[str] = [a + b for a in "XYZ" for b in "XYZ"]
_REG = "meas"


def tomography_circuits(
    circ: QuantumCircuit, latent: tuple[int, int] = LATENT
) -> list[QuantumCircuit]:
    out = []
    for basis in BASES:
        qc = circ.copy()
        creg = ClassicalRegister(2, _REG)
        qc.add_register(creg)
        for k in range(2):
            b = basis[1 - k]  # rightmost char = latent[0]
            if b == "Y":
                qc.sdg(latent[k])
            if b in "XY":
                qc.h(latent[k])
        for k in range(2):
            qc.measure(latent[k], creg[k])
        out.append(qc)
    return out


def bloch_from_counts(counts: list[dict[str, int]]) -> np.ndarray:
    out = np.zeros(len(PAULI_LABELS))
    for i, label in enumerate(PAULI_LABELS):
        vals = []
        for basis, cnt in zip(BASES, counts):
            if any(p != "I" and p != b for p, b in zip(label, basis)):
                continue
            total = sum(cnt.values())
            acc = 0.0
            for bits, n in cnt.items():
                bits = bits.replace(" ", "")
                sign = 1
                for p, bit in zip(label, bits):
                    if p != "I" and bit == "1":
                        sign = -sign
                acc += sign * n
            vals.append(acc / total)
        out[i] = float(np.mean(vals))
    return out


def estimate_qpu_seconds(n_circuits: int, shots: int, per_shot_s: float = 250e-6) -> float:
    return n_circuits * shots * per_shot_s


def get_backend(name: str):
    if name == "aer":
        return AerSimulator()
    if name.startswith("fake_"):
        from qiskit_ibm_runtime import fake_provider

        cls = getattr(fake_provider, "Fake" + name[5:].capitalize(), None)
        if cls is None:
            raise ValueError(f"unknown fake backend: {name}")
        return AerSimulator.from_backend(cls())
    try:
        return QiskitRuntimeService().backend(name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not open IBM backend {name!r} ({exc}). Save credentials once with "
            "QiskitRuntimeService.save_account(token=..., channel='ibm_quantum_platform')."
        ) from exc


def _isa(
    circuits: list[QuantumCircuit], backend, seed_transpiler: int | None = 0
) -> list[QuantumCircuit]:
    return generate_preset_pass_manager(
        optimization_level=3, backend=backend, seed_transpiler=seed_transpiler
    ).run(circuits)


def transpile(circuits: list[QuantumCircuit], backend) -> list[QuantumCircuit]:
    """Deterministic ISA transpilation (optimisation level 3, fixed seed)."""
    return _isa(circuits, backend)


def _two_qubit_count(qc: QuantumCircuit) -> int:
    return sum(
        1
        for inst in qc.data
        if len(inst.qubits) == 2 and inst.operation.name not in ("barrier", "measure")
    )


def circuit_stats(isa: list[QuantumCircuit]) -> dict:
    """Depth / two-qubit-gate statistics of transpiled circuits, plus first layout."""
    depths = [qc.depth() for qc in isa]
    twoq = [_two_qubit_count(qc) for qc in isa]
    layout = None
    if isa and isa[0].layout is not None:
        lay = isa[0].layout
        layout = {
            "initial": [int(q) for q in lay.initial_index_layout(filter_ancillas=True)],
            "final": [int(q) for q in lay.final_index_layout(filter_ancillas=True)],
        }
    return {
        "n_circuits": len(isa),
        "depth_mean": float(np.mean(depths)),
        "depth_max": int(np.max(depths)),
        "twoq_mean": float(np.mean(twoq)),
        "twoq_max": int(np.max(twoq)),
        "first_layout": layout,
    }


def _counts(result) -> list[dict[str, int]]:
    return [dict(pub.data.__getattribute__(_REG).get_counts()) for pub in result]


def run_counts(
    circuits: list[QuantumCircuit], backend, shots: int, transpiled: bool = False
) -> list[dict[str, int]]:
    sampler = SamplerV2(mode=backend)
    isa = circuits if transpiled else _isa(circuits, backend)
    return _counts(sampler.run(isa, shots=shots).result())


def submit(
    circuits, backend, shots: int, jobs_file: Path, tag: str, transpiled: bool = False
) -> str:
    sampler = SamplerV2(mode=backend)
    job = sampler.run(circuits if transpiled else _isa(circuits, backend), shots=shots)
    job_id = job.job_id()
    jobs_file = Path(jobs_file)
    jobs_file.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "tag": tag,
        "job_id": job_id,
        "backend": getattr(backend, "name", str(backend)),
        "shots": shots,
        "n_circuits": len(circuits),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    with jobs_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return job_id


def fetch(job_id: str) -> list[dict[str, int]]:
    return _counts(QiskitRuntimeService().job(job_id).result())


def measure_embeddings(samples, run, kind: Kind, backend, shots: int) -> np.ndarray:
    cfg = run.config
    theta = np.asarray(run.theta_bar if kind == "target" else run.theta, dtype=float)
    use_predictor = cfg.variant != "no_predictor"
    phi = np.asarray(run.phi, dtype=float) if len(run.phi) else None
    circuits = []
    for smp in samples:
        circuits += tomography_circuits(
            bound_circuit(smp, kind, theta, phi, cfg.delta, use_predictor)
        )
    counts = run_counts(circuits, backend, shots)
    return np.stack([bloch_from_counts(counts[i * 9 : (i + 1) * 9]) for i in range(len(samples))])
