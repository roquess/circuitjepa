"""Rebuild results/eval/qpu.json from the saved raw QPU counts of several submissions.

The QPU data were collected in two submissions on ibm_marrakesh: the
pre-registered pair (full_seed0, untrained_seed0) and a later no_ema_seed0 job.
`circuitjepa hw-fetch` evaluates one submission at a time, so this script merges
them from results/hardware/counts_<tag>.json. Each run's parameters are checked
against the hash stored at submission time.

Usage: python scripts/merge_qpu_eval.py
"""

import json

from circuitjepa import experiments as ex

TAGS = ["full_seed0", "untrained_seed0", "no_ema_seed0"]


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
    data = ex.evaluate_from_counts(runs, counts, "qpu", {"job_ids": {t: raw[t]["job_id"] for t in TAGS}})
    data.update(
        backend_name=raw[TAGS[0]]["backend"],
        shots=raw[TAGS[0]]["shots"],
        m=raw[TAGS[0]]["m"],
        submission_id={t: raw[t]["submission_id"] for t in TAGS},
        job_ids={t: raw[t]["job_id"] for t in TAGS},
        params_hash={t: raw[t]["params_hash"] for t in TAGS},
        submitted_at={t: raw[t]["submitted_at"] for t in TAGS},
        note=(
            "Merged from two submissions on ibm_marrakesh: the pre-registered pair "
            "(full_seed0, untrained_seed0) and a later no_ema_seed0 job; rebuilt from "
            "results/hardware/counts_*.json by scripts/merge_qpu_eval.py."
        ),
    )
    print(f"wrote {ex.save_eval('qpu', data)}")


if __name__ == "__main__":
    main()
