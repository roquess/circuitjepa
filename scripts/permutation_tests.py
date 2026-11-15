"""Label-permutation tests on the similarity matrices G of the saved evaluations.

A complementary robustness check, added after the fact (not pre-registered). The
pre-registered criterion (paired sign-flip test on row gaps, binomial top-1 test) and
its verdict are unchanged. The binomial test assumes independent rows with equal
success probability under H0, although all rows share the same m targets; this test
does not. Under H0 the target (column) order of G is permuted, and top-1 and the
diagonal gap are recomputed; for a paired comparison the same permutation is applied
to the model and baseline matrices and the statistics are the differences. One-sided
p-values with add-one smoothing, 10,000 resamples, RNG seed 0 for every test
(circuitjepa.metrics.permutation_p); the floor is 1/10,001.

Covered: every run and every full-vs-untrained pair of the QPU evaluation (the
pre-registered seed-0 pair and the two exploratory pairs), and of the exact
evaluations on the original test set and on the independent held-out set.

Usage: python scripts/permutation_tests.py   (writes results/permutation_tests.json)
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from circuitjepa import experiments as ex  # noqa: E402
from circuitjepa.metrics import permutation_p  # noqa: E402
from circuitjepa.train import TrainConfig  # noqa: E402

N_RESAMPLES = 10000
SEED = 0
EVALS = ("exact", "exact_heldout", "qpu")
QPU_PAIRS = [("full_seed0", "untrained_seed0"), ("full_seed1", "untrained_seed1"), ("full_seed2", "untrained_seed0")]


def pairs_for(name: str, runs: dict) -> list[tuple[str, str]]:
    if name == "qpu":
        return QPU_PAIRS
    seeds = sorted(r["seed"] for r in runs.values() if r["variant"] == "full")
    return [(f"full_seed{s}", f"untrained_seed{s}") for s in seeds if f"untrained_seed{s}" in runs]


def main() -> None:
    out = {
        "note": (
            "Label-permutation test, complementary robustness check added after the fact; "
            "not pre-registered. One-sided, add-one smoothing; floor 1/(n_resamples+1)."
        ),
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "evals": {},
    }
    for name in EVALS:
        path = ex.results_dir() / "eval" / f"{name}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        runs = data["runs"]
        G = {n: np.asarray(r["G"]) for n, r in runs.items()}
        single = {n: permutation_p(G[n], n_resamples=N_RESAMPLES, seed=SEED) for n in runs}
        paired = {}
        for f, u in pairs_for(name, runs):
            r = permutation_p(G[f], G[u], n_resamples=N_RESAMPLES, seed=SEED)
            r["pre_registered_pair"] = name == "qpu" and (f, u) == QPU_PAIRS[0]
            paired[f"{f}_vs_{u}"] = r
        out["evals"][name] = {"m": int(data["m"]), "test_data_seed": data.get("test_data_seed", TrainConfig("full", 0).test_data_seed),
                              "single": single, "paired": paired}
        print(f"{name}: {len(single)} runs, {len(paired)} pairs")
        for pair, r in paired.items():
            print(f"  {pair}: p_top1={r['p_top1']:.2g} p_gap={r['p_gap']:.2g}")
    path = ex.results_dir() / "permutation_tests.json"
    ex.write_json(path, out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
