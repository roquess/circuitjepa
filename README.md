# CircuitJEPA

CircuitJEPA is a Joint-Embedding Predictive Architecture (JEPA) in which the
encoder, the exponential-moving-average (EMA) target encoder and the
action-conditioned predictor are all parametrized quantum circuits. The data are
states of a 3-qubit transverse-field Ising chain evolved by Trotter steps. From a
context state, the model learns, without reconstruction or negative pairs, to
predict the latent embedding of the state reached 1, 2 or 3 steps later.
Embeddings are the 15 two-qubit Pauli expectation values of a 2-qubit latent
register, compared with a cosine similarity. Training is done in exact
simulation; the trained circuits are evaluated on noiseless and noisy simulators
and on the IBM Heron QPU `ibm_marrakesh`.

This is a feasibility demonstration at 5 qubits. It makes no claim of quantum
advantage or scalability.

## Paper

- [paper/circuitjepa.md](paper/circuitjepa.md) (source)
- [paper/circuitjepa.pdf](paper/circuitjepa.pdf)

## Headline results

Held-out retrieval top-1 accuracy and diagonal gap. Exact simulation: 64 test
samples (chance 1/64), mean ± sample std over seeds 0, 1, 2. QPU: `ibm_marrakesh`,
seed 0 only, 16 test samples (chance 1/16), 1000 shots, no error mitigation.

| setting | variant | top-1 | diag. gap |
|---|---|---|---|
| exact, 3 seeds | full | 0.469 ± 0.402 | 0.632 ± 0.210 |
| exact, 3 seeds | no_ema | 0.391 ± 0.327 | 0.573 ± 0.219 |
| exact, 3 seeds | no_predictor | 0.031 ± 0.054 | 0.175 ± 0.204 |
| exact, 3 seeds | untrained | 0.000 ± 0.000 | -0.030 ± 0.051 |
| QPU, seed 0 | full | 0.938 | 0.827 |
| QPU, seed 0 | no_ema | 0.438 | 0.455 |
| QPU, seed 0 | untrained | 0.062 | 0.033 |

Pre-registered QPU test (`full` vs `untrained`, seed 0): met. `full` retrieves 15
of 16 targets; the diagonal-gap sign-flip test gives p = 1.0e-4 and the top-1
binomial test gives p = 1.3e-17.

Caveats: seed variance is large (`full` exact top-1 is 0.922, 0.328, 0.156 for
seeds 0, 1, 2); the QPU row is the best seed, designated in advance; removing the
EMA and the stop-gradient did not cause collapse; the linear dynamics probe does
not separate `full` from `no_predictor`, so it does not show learned dynamics.
See the paper's Limitations section.

## Install

Python 3.12 is required.

```
python -m venv .venv
.venv/Scripts/activate        # Windows; on Linux/macOS: source .venv/bin/activate
pip install -e .[dev]
```

## Reproduce

Everything below is run from the repository root. Results are written to
`results/`.

```
# 1. Train 4 variants x 3 seeds (about 9 min with 12 processes; reported config uses delta 0.5)
circuitjepa train-all --jobs 12 --delta 0.5

# 2. Evaluate on every backend (m = 16 test samples, 1000 shots for shot-based backends)
circuitjepa evaluate --backend exact          # seconds
circuitjepa evaluate --backend aer            # tens of seconds
circuitjepa evaluate --backend fake_fez       # tens of seconds
circuitjepa evaluate --backend fake_marrakesh # tens of seconds

# 3. Figures
circuitjepa figures

# 4. QPU (needs an IBM Quantum account, see below)
circuitjepa hw-submit --backend ibm_marrakesh --dry-run   # estimate only, nothing submitted
circuitjepa hw-submit --backend ibm_marrakesh             # default runs: full_seed0 untrained_seed0
circuitjepa hw-fetch                                      # fetch the latest submission and evaluate
python scripts/merge_qpu_eval.py                          # merge the QPU submissions into results/eval/qpu.json

# 5. Checks
python scripts/check_paper_numbers.py                     # paper numbers vs results/
pytest
```

Notes:

- Reported timings (one machine): `train-all --jobs 12` took 523 s; the exact,
  aer, fake_fez and fake_marrakesh evaluations took about 6, 19, 26 and 39 s.
  Each QPU job was charged 118 s of QPU time.
- `hw-submit --runs <tag> ...` selects other runs; the `no_ema_seed0` QPU run was
  a second submission, which `scripts/merge_qpu_eval.py` merges with the first.
- `scripts/check_paper_numbers.py --skip-transpile` skips the transpilation-based checks.
- The repository already contains the trained runs, evaluations, raw QPU counts
  and figures behind the paper, so steps 1-4 can be skipped.

## IBM account

QPU runs need your own IBM Quantum account. Configure it once, yourself; no
token is stored in this repository:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="<your token>", instance="<your instance CRN>")
```

Budget guard: `hw-submit` estimates the QPU time first and refuses to submit
above 240 s unless `--force` is given.

## Repository layout

- `circuitjepa/` - library and CLI (ansatz, model, training, simulation, metrics, hardware, figures)
- `tests/` - pytest suite
- `scripts/` - `merge_qpu_eval.py`, `check_paper_numbers.py`
- `results/` - trained runs, evaluations per backend, raw QPU counts and metadata, figures, `SUMMARY.md`, `TUNING.md`
- `paper/` - paper source, bibliography, figures and PDF

## Citation

```bibtex
@misc{roques2026circuitjepa,
  author       = {Roques, Steve},
  title        = {{CircuitJEPA}: a fully circuit-based Joint-Embedding Predictive Architecture for quantum dynamics},
  year         = {2026},
  howpublished = {\url{https://github.com/roquess/circuitjepa}}
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).
