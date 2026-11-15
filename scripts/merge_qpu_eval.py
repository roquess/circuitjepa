"""Rebuild results/eval/qpu.json from the saved raw QPU counts of all submissions.

The QPU data were collected in four submissions on ibm_marrakesh:

1. the pre-registered pair (full_seed0, untrained_seed0), 1000 shots;
2. no_ema_seed0, 1000 shots;
3. full_seed1 and full_seed2, 500 shots (exploratory, not pre-registered);
4. no_predictor_seed0 and untrained_seed1, 450 shots (exploratory, not pre-registered).

`circuitjepa hw-fetch` evaluates one submission at a time, so this script merges
them from results/hardware/counts_<tag>.json. Each run's parameters are checked
against the hash stored at submission time. Shots differ between submissions and
are stored per run; the Bloch reconstruction normalises by each circuit's own
total count, so it does not assume a fixed shot number.

Tests: the pre-registered test (full_seed0 vs untrained_seed0) is computed exactly
as before. Exploratory tests, computed the same way and marked
`pre_registered: false`, are added for full_seed1 vs untrained_seed1 and for
full_seed2 vs untrained_seed0 (untrained_seed2 was not run on the QPU; both use
the same 16 test samples). A per-run comparison with the shot-based simulators
(aer, fake_fez, fake_marrakesh; 1000 shots, same 16 samples) is also stored.

Usage: python scripts/merge_qpu_eval.py
"""

import json

import numpy as np

from circuitjepa import experiments as ex
from circuitjepa.metrics import row_gaps, signflip_p, top1_binom_p

TAGS = [
    "full_seed0",
    "untrained_seed0",
    "no_ema_seed0",
    "full_seed1",
    "full_seed2",
    "no_predictor_seed0",
    "untrained_seed1",
]
PRE_REGISTERED = ("full_seed0", "untrained_seed0")
# Exploratory pairs (model, baseline); computed like the pre-registered test.
EXPLORATORY = [("full_seed1", "untrained_seed1"), ("full_seed2", "untrained_seed0")]
SIMULATORS = ("aer", "fake_fez", "fake_marrakesh")


def pair_test(rec: dict, base: dict, pre_registered: bool) -> dict:
    Gf, Gu = np.asarray(rec["G"]), np.asarray(base["G"])
    m = Gf.shape[0]
    k = int(round(rec["top1"] * m))
    p_gap = signflip_p(row_gaps(Gf), row_gaps(Gu))
    p_top1 = top1_binom_p(k, m)
    return {
        "m": m,
        "signflip_p_gap": p_gap,
        "top1_hits": k,
        "top1_binom_p": p_top1,
        "pass_gap": bool(p_gap < ex.ALPHA_GAP),
        "pass_top1": bool(p_top1 < ex.ALPHA_TOP1),
        "pre_registered": pre_registered,
    }


def main() -> None:
    hw = ex.results_dir() / "hardware"
    raw = {t: json.loads((hw / f"counts_{t}.json").read_text()) for t in TAGS}
    runs = ex.load_runs(TAGS)
    for t in TAGS:
        if ex.params_hash(runs[t]) != raw[t]["params_hash"]:
            raise SystemExit(f"{t}: params_hash mismatch with the submitted job")
    counts = {
        t: [c["counts"] for c in sorted(raw[t]["circuits"], key=lambda c: c["index"])]
        for t in TAGS
    }
    for t in TAGS:  # every circuit of a run must carry the run's shot number
        totals = {sum(c.values()) for c in counts[t]}
        if totals != {raw[t]["shots"]}:
            raise SystemExit(f"{t}: per-circuit totals {totals} != shots {raw[t]['shots']}")
    data = ex.evaluate_from_counts(runs, counts, "qpu", {"job_ids": {t: raw[t]["job_id"] for t in TAGS}})
    per_run = data["runs"]
    for t in TAGS:
        per_run[t]["shots"] = raw[t]["shots"]
        per_run[t]["submission_id"] = raw[t]["submission_id"]

    # Tests: keep only the pre-registered pair from the generic seed pairing, then add
    # the exploratory pairs explicitly.
    tests = {}
    f, u = PRE_REGISTERED
    tests[f"{f}_vs_{u}"] = pair_test(per_run[f], per_run[u], pre_registered=True)
    for f, u in EXPLORATORY:
        tests[f"{f}_vs_{u}"] = pair_test(per_run[f], per_run[u], pre_registered=False)
    data["tests"] = tests

    # Same run on the shot-based simulators (first m test samples, 1000 shots).
    sims = {be: json.loads((ex.results_dir() / "eval" / f"{be}.json").read_text()) for be in SIMULATORS}
    data["sim_comparison"] = {
        t: {
            be: {k: sims[be]["runs"][t][k] for k in ("top1", "diag_gap", "total_var", "mean_norm")}
            for be in SIMULATORS
        }
        for t in TAGS
    }

    data.update(
        backend_name=raw[TAGS[0]]["backend"],
        shots={t: raw[t]["shots"] for t in TAGS},
        m=raw[TAGS[0]]["m"],
        submission_id={t: raw[t]["submission_id"] for t in TAGS},
        job_ids={t: raw[t]["job_id"] for t in TAGS},
        params_hash={t: raw[t]["params_hash"] for t in TAGS},
        submitted_at={t: raw[t]["submitted_at"] for t in TAGS},
        note=(
            "Merged from four submissions on ibm_marrakesh: the pre-registered pair "
            "(full_seed0, untrained_seed0; 1000 shots), no_ema_seed0 (1000 shots), and two "
            "exploratory, not pre-registered submissions: full_seed1 and full_seed2 (500 shots), "
            "no_predictor_seed0 and untrained_seed1 (450 shots). Exploratory test full_seed2 is "
            "paired with untrained_seed0 because untrained_seed2 was not run on the QPU. Rebuilt "
            "from results/hardware/counts_*.json by scripts/merge_qpu_eval.py."
        ),
    )
    print(f"wrote {ex.save_eval('qpu', data)}")


if __name__ == "__main__":
    main()
