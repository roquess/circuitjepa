# CircuitJEPA

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24259038.svg)](https://doi.org/10.5281/zenodo.24259038)
[![Paper DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24259981.svg)](https://doi.org/10.5281/zenodo.24259981)

Paper (preprint, PDF): [zenodo.org/records/24259981](https://zenodo.org/records/24259981/files/circuitjepa.pdf)

CircuitJEPA is a Joint-Embedding Predictive Architecture (JEPA) whose three
parametrized components - encoder, exponential-moving-average (EMA) target
encoder and horizon-conditioned predictor - are parametrized quantum circuits.
The predictor is conditioned on the prediction horizon (number of Trotter steps),
not on a control action. Training is classical: exact statevector simulation with
PyTorch autodiff, a classically computed cosine loss and Adam; embedding readout
from Pauli measurements is classical post-processing. The data are
states of a 3-qubit transverse-field Ising chain evolved by Trotter steps. From a
context state, the model learns, without reconstruction or negative pairs, to
predict the latent embedding of the state reached 1, 2 or 3 steps later.
Embeddings are the 15 two-qubit Pauli expectation values of a 2-qubit latent
register, compared with a cosine similarity. The trained circuits are evaluated
on noiseless and noisy simulators and on the IBM Heron QPU `ibm_marrakesh`.

This is a feasibility demonstration at 5 qubits. It makes no claim of quantum
advantage or scalability.

## Paper

- [paper/circuitjepa.md](paper/circuitjepa.md) (source)
- [paper/circuitjepa.pdf](paper/circuitjepa.pdf)

## Headline results

Retrieval top-1 accuracy and diagonal gap on test samples. Exact simulation: 64
test samples (chance 1/64), mean ± sample std over seeds 0-9 (seeds 3-9 were added
after the first results, with the same configuration and no re-tuning), on the
original test set (data seed 5678, also used to select the configuration) and on
the independent held-out test set (data seed 424242, never used for any choice).
QPU: `ibm_marrakesh`, first 16 samples of the original test set (chance 1/16), no
error mitigation.

| setting | variant | top-1 | diag. gap |
|---|---|---|---|
| exact, original test set | full | 0.539 ± 0.341 | 0.665 ± 0.186 |
| exact, original test set | no_ema | 0.369 ± 0.332 | 0.486 ± 0.286 |
| exact, original test set | no_predictor | 0.031 ± 0.040 | 0.116 ± 0.118 |
| exact, original test set | untrained | 0.008 ± 0.013 | -0.021 ± 0.047 |
| exact, held-out test set | full | 0.530 ± 0.365 | 0.651 ± 0.203 |
| exact, held-out test set | no_ema | 0.366 ± 0.350 | 0.503 ± 0.273 |
| exact, held-out test set | no_predictor | 0.053 ± 0.046 | 0.132 ± 0.123 |
| exact, held-out test set | untrained | 0.016 ± 0.019 | 0.004 ± 0.034 |
| QPU, seed 0, 1000 shots (pre-registered) | full | 15/16 | 0.827 |
| QPU, seed 0, 1000 shots (pre-registered) | untrained | 1/16 | 0.033 |
| QPU, seed 0, 1000 shots | no_ema | 7/16 | 0.455 |
| QPU, seed 1, 500 shots (exploratory) | full | 10/16 | 0.428 |
| QPU, seed 2, 500 shots (exploratory) | full | 6/16 | 0.512 |
| QPU, seed 0, 450 shots (exploratory) | no_predictor | 2/16 | 0.151 |
| QPU, seed 1, 450 shots (exploratory) | untrained | 1/16 | -0.021 |

Pre-registered QPU test (`full` vs `untrained`, seed 0): met. `full` retrieves 15
of 16 targets; the diagonal-gap sign-flip test gives p = 1.0e-4 (the floor with
10,000 resamples) and the top-1
binomial test gives p = 1.3e-17. The other QPU runs are exploratory (not
pre-registered); computed the same way, `full` seed 1 vs `untrained` seed 1
gives p = 5.0e-4 / 5.1e-9, and `full` seed 2 vs `untrained` seed 0 (no seed-2
baseline was run on the QPU) gives p = 1.6e-3 / 2.8e-4. The seven QPU jobs used
584 of the 600 s free-tier budget. Only the seed-0 pair on the QPU was
pre-registered; the same tests applied to other seeds or in simulation are
retrospective and exploratory. A label-permutation test, added after the fact as a
robustness check (the binomial test assumes independent rows, although rows share
the same targets), gives p = 1.0e-4 (the floor) for the pre-registered pair
(`results/permutation_tests.json`).

Caveats: seed variance is large (`full` exact top-1 from 5/64 to 62/64 across
ten seeds); all seeds share the same training and test data, so the spread reflects
initialization and optimization only; seed 0, designated for the hardware test
before training, turned out to be the best of the first three seeds and ties for
second of ten; the configuration was chosen on the original test set (the held-out
set addresses the reuse of those samples, not the selection itself, and the QPU
runs remain on the original set); without the EMA and the stop-gradient, 2 of 10
seeds end with near-zero retrieval (1/64), not flagged by the formal collapse
criterion; the linear dynamics probe does
not separate `full` from `no_predictor` (median MSE 0.0480 vs 0.0473), and for
`no_predictor` seed 5 it blows up (MSE 18.59) because the online and EMA
encoders differ on two near-constant embedding components. See the paper's
Results and Limitations sections.

Independent held-out test set: the configuration was selected on the original test
set (data seed 5678). A second test set of 64 samples with data seed 424242
(`HELDOUT_TEST_DATA_SEED` in `circuitjepa/config.py`), never used for any choice,
was fixed before any evaluation on it; the trained runs are evaluated on it without
retraining.

## Install

Python 3.12 is required. Tested with Python 3.12.10, qiskit 2.6.2, qiskit-aer 0.18.2,
qiskit-ibm-runtime 0.52.1, torch 2.15.0 (CPU), numpy 2.5.4, scipy 1.18.1 and
matplotlib 3.11.2 (`results/ENVIRONMENT.json`). `constraints.txt` pins these versions:
bit-exact agreement with the committed results needs them, and other versions should
give statistically similar but not identical numbers.

```
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\Activate.ps1      # Windows PowerShell
# .venv\Scripts\activate.bat      # Windows cmd
pip install -e ".[dev]" -c constraints.txt
```

## Reproduce

Everything below is run from the repository root. Results are written to
`results/`.

```
# 1. Train 4 variants x 10 seeds (delta 0.5 is the default; given here for explicitness)
circuitjepa train-all --jobs 12 --delta 0.5                              # seeds 0 1 2
circuitjepa train-all --jobs 14 --delta 0.5 --seeds 3 4 5 6 7 8 9      # seeds 3-9

# 2. Evaluate every run on the local simulators (m = 16 test samples, 1000 shots for
#    shot-based backends); `evaluate` refuses real IBM backends, see step 4
circuitjepa evaluate --backend exact --jobs 12
circuitjepa evaluate --backend aer --jobs 12
circuitjepa evaluate --backend fake_fez --jobs 12
circuitjepa evaluate --backend fake_marrakesh --jobs 12
# Independent held-out test set (data seed 424242); the probe is still fitted on the
# training set; writes results/eval/<backend>_heldout.json
circuitjepa evaluate --backend exact --test-data-seed 424242 --tag heldout
circuitjepa evaluate --backend aer --test-data-seed 424242 --tag heldout --jobs 12

# 3. Figures
circuitjepa figures

# 4. QPU (needs an IBM Quantum account, see below)
circuitjepa hw-submit --backend ibm_marrakesh --dry-run   # estimate only, nothing submitted
circuitjepa hw-submit --backend ibm_marrakesh             # default runs: full_seed0 untrained_seed0
circuitjepa hw-fetch                                      # fetch the latest submission and evaluate
python scripts/merge_qpu_eval.py                          # rebuild the paper's merged qpu.json from the committed counts
python scripts/record_hw_metadata.py                      # job usage + calibration snapshot (reads only)
python scripts/permutation_tests.py                       # label-permutation tests -> results/permutation_tests.json

# 5. Checks
python scripts/check_paper_numbers.py                     # paper numbers vs results/
pytest
```

Notes:

- Timings on one machine (24 logical CPUs): training took about 9 minutes for seeds 0-2
  and about 13 minutes for seeds 3-9 with the commands above (per-run times are in
  `results/runs/*.json`, `wall_seconds`). The evaluations above took 18, 41, 83 and
  83 s (`wall_seconds` in `results/eval/<backend>.json`); with the default `--jobs 1`
  the shot-based ones take several minutes each. QPU charge: 118 s per job at 1000
  shots, 60 s at 500 shots, 55 s at 450 shots (584 s in total, from
  `results/hardware/job_metrics.json`); this is about 9 % above the `hw-submit`
  estimate (250 µs per shot).
- `hw-submit --runs <tag> ...` selects other runs. The QPU data come from four
  submissions (the pre-registered pair, `no_ema_seed0`, and two exploratory ones);
  `scripts/merge_qpu_eval.py` merges all seven runs with their per-run shots.
- `scripts/check_paper_numbers.py --skip-transpile` skips the transpilation-based checks.
- The repository already contains the trained runs, evaluations, raw QPU counts
  and figures behind the paper, so steps 1-4 can be skipped. `hw-fetch` refuses to
  overwrite a committed `counts_<tag>.json` from a different job unless `--force` is
  given, so your own QPU runs do not silently replace the paper's raw data.

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
- `scripts/` - `merge_qpu_eval.py`, `permutation_tests.py`, `check_paper_numbers.py`, `record_hw_metadata.py`
- `results/` - trained runs, evaluations per backend (`*_heldout.json`: independent held-out test set), `permutation_tests.json`, raw QPU counts and metadata (`hardware/jobs.json` is JSON Lines, one job per line), figures, `SUMMARY.md`, `TUNING.md`
- `paper/` - paper source, bibliography, figures and PDF

## Citation

Paper (preprint): [10.5281/zenodo.24259981](https://doi.org/10.5281/zenodo.24259981). Software and results archived on Zenodo: [10.5281/zenodo.24259038](https://doi.org/10.5281/zenodo.24259038) (concept DOI, always resolves to the latest version; v1.0.0 is [10.5281/zenodo.24259039](https://doi.org/10.5281/zenodo.24259039)).

```bibtex
@misc{roques2026circuitjepa,
  author       = {Roques, Steve},
  title        = {{CircuitJEPA}: a fully circuit-based Joint-Embedding Predictive Architecture for quantum dynamics},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.24259038},
  howpublished = {\url{https://github.com/roquess/circuitjepa}}
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).
