# Tuning log (sanity gate)

Gate (task brief, Step 6): `full` exact test top-1 averaged over seeds ≥ 0.3 (chance 1/64)
and no `full` seed collapsed. Only builder-level knobs (δ, T₀, layers, lr, epochs, τ, batch)
may change, identically for all variants. At most 4 attempts.

Attempts 2–4 were screened on `full` (+ `untrained`, needed for the collapse baseline) only,
so ablation results could not influence the choice. The chosen setting was then re-run for
all four variants from scratch (`full` reproduced attempt 3 bit-for-bit: training is
deterministic given the seed).

## Chronology and selection rule

1. Attempt 1 (defaults, all four variants) ran first, on its own. It failed the gate.
2. With attempt 1's numbers in hand, the settings of attempts 2, 3 and 4 were all chosen
   up front. The three were then started together as one batch: concurrent processes from
   a single command, each with its own results directory. Their numbers are labels, not a
   time order. No attempt was started after attempt 3's outcome was known. In particular,
   attempt 4 did not run after attempt 3 had passed. All three finished, and their exact
   evaluations were read together.
3. Selection rule, fixed before the batch was read: keep a setting that passes the gate.
   Attempt 3 was the only one that passed, so no tie-break was needed. No further attempt
   was made after that.

Exact (noiseless, m = 64) test metrics of `full`, seeds 0 / 1 / 2:

| # | change vs defaults | top-1 per seed | top-1 mean ± std | diag gap mean | collapsed | gate |
|---|---|---|---|---|---|---|
| 1 | none (δ = 0.25, T₀ = 3, 3 enc. layers, lr = 0.02, 150 epochs, τ = 0.99, batch 32) | 0.125 / 0.422 / 0.188 | 0.245 ± 0.157 | 0.471 | 0/3 | fail |
| 2 | τ = 0.996, 200 epochs | 0.141 / 0.156 / 0.516 | 0.271 ± 0.212 | 0.451 | 0/3 | fail |
| 3 | δ = 0.5 | 0.922 / 0.328 / 0.156 | 0.469 ± 0.402 | 0.632 | 0/3 | **pass** |
| 4 | lr = 0.05, batch 16 | 0.141 / 0.125 / 0.375 | 0.214 ± 0.140 | 0.450 | 0/3 | fail |

Attempt 1 also trained all variants (exact top-1 mean ± std: no_ema 0.302 ± 0.048,
no_predictor 0.500 ± 0.204, untrained 0.031 ± 0.041). With δ = 0.25 a single Trotter
step changes the state little, so the no-predictor shortcut (target ≈ context) was
competitive; this is reported, not tuned against.

## Final configuration

δ = 0.5, all other values at their defaults: T₀ = 3, encoder 3 layers, predictor 2 layers,
lr = 0.02 (Adam), 150 epochs, τ = 0.99, batch 32, n_train = 256, n_test = 64,
train data seed 1234, test data seed 5678, model seeds 0, 1, 2.

Command: `circuitjepa train-all --jobs 12 --delta 0.5`.

## Caveats

- The gate passes on the mean, but seed variance is large (0.92 / 0.33 / 0.16): one seed
  dominates the mean. The median (0.33) also clears 0.3.
- Selection used the test split (the gate is defined on it); there is no separate
  validation split, so the `full` exact numbers carry a mild selection bias over 4 settings.
- Raw artefacts of every attempt (run JSONs and exact evaluation) are in
  `results/tuning/attempt{1,2,3,4}/`.
