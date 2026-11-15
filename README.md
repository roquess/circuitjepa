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
samples (chance 1/64), mean ± sample std over seeds 0-9 (seeds 3-9 were added
after the first results, with the same configuration and no re-tuning). QPU:
`ibm_marrakesh`, 16 test samples (chance 1/16), no error mitigation.

| setting | variant | top-1 | diag. gap |
|---|---|---|---|
| exact, 10 seeds | full | 0.539 ± 0.341 | 0.665 ± 0.186 |
| exact, 10 seeds | no_ema | 0.369 ± 0.332 | 0.486 ± 0.286 |
| exact, 10 seeds | no_predictor | 0.031 ± 0.040 | 0.116 ± 0.118 |
| exact, 10 seeds | untrained | 0.008 ± 0.013 | -0.021 ± 0.047 |
| QPU, seed 0, 1000 shots (pre-registered) | full | 15/16 | 0.827 |
| QPU, seed 0, 1000 shots (pre-registered) | untrained | 1/16 | 0.033 |
| QPU, seed 0, 1000 shots | no_ema | 7/16 | 0.455 |
| QPU, seed 1, 500 shots (exploratory) | full | 10/16 | 0.428 |
| QPU, seed 2, 500 shots (exploratory) | full | 6/16 | 0.512 |
| QPU, seed 0, 450 shots (exploratory) | no_predictor | 2/16 | 0.151 |
| QPU, seed 1, 450 shots (exploratory) | untrained | 1/16 | -0.021 |

Pre-registered QPU test (`full` vs `untrained`, seed 0): met. `full` retrieves 15
of 16 targets; the diagonal-gap sign-flip test gives p = 1.0e-4 and the top-1
binomial test gives p = 1.3e-17. The other QPU runs are exploratory (not
pre-registered); computed the same way, `full` seed 1 vs `untrained` seed 1
gives p = 5.0e-4 / 5.1e-9, and `full` seed 2 vs `untrained` seed 0 (no seed-2
baseline was run on the QPU) gives p = 1.6e-3 / 2.8e-4. The seven QPU jobs used
584 of the 600 s free-tier budget.

Caveats: seed variance is large (`full` exact top-1 from 5/64 to 62/64 across
ten seeds); seed 0, the pre-registered hardware seed, was the best of the first
three seeds and ties for second of ten; the configuration was chosen on test
metrics; removing the EMA and the stop-gradient did not cause flagged collapse,
but 2 of 10 `no_ema` seeds ended near-collapsed; the linear dynamics probe does
not separate `full` from `no_predictor` (median MSE 0.0480 vs 0.0473), and for
`no_predictor` seed 5 it blows up (MSE 18.59) because the online and EMA
encoders differ on two near-constant embedding components. See the paper's
Results and Limitations sections.

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
# 1. Train 4 variants x 10 seeds (reported config uses delta 0.5)
circuitjepa train-all --jobs 12 --delta 0.5                              # seeds 0 1 2
circuitjepa train-all --jobs 14 --delta 0.5 --seeds 3 4 5 6 7 8 9      # seeds 3-9

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

- Reported timings (one machine): `train-all --jobs 12` (seeds 0-2) took 523 s and
  `train-all --jobs 14` for seeds 3-9 took 757.8 s. QPU charge: 118 s per job at
  1000 shots, 60 s at 500 shots, 55 s at 450 shots (584 s in total).
- `hw-submit --runs <tag> ...` selects other runs. The QPU data come from four
  submissions (the pre-registered pair, `no_ema_seed0`, and two exploratory ones);
  `scripts/merge_qpu_eval.py` merges all seven runs with their per-run shots.
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
