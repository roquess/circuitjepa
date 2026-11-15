"""Command-line interface: `circuitjepa train | train-all | evaluate | hw-submit | hw-fetch | figures`."""

import argparse
import json
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

from circuitjepa import experiments as ex
from circuitjepa.config import HELDOUT_TEST_DATA_SEED
from circuitjepa.hardware import estimate_qpu_seconds

QPU_BUDGET_S = 240.0
DEFAULT_HW_RUNS = ["full_seed0", "untrained_seed0"]
TUNABLE = ("epochs", "delta", "t0_max", "lr", "tau", "batch_size")


def _add_tunables(p: argparse.ArgumentParser) -> None:
    p.add_argument("--epochs", type=int)
    p.add_argument("--delta", type=float)
    p.add_argument("--t0-max", dest="t0_max", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--tau", type=float)
    p.add_argument("--batch-size", dest="batch_size", type=int)


def _overrides(args) -> dict:
    return {k: getattr(args, k) for k in TUNABLE if getattr(args, k) is not None}


def cmd_train(args) -> int:
    print(ex.train_one(args.variant, args.seed, _overrides(args)))
    return 0


def cmd_train_all(args) -> int:
    ex.write_environment()
    jobs = [(v, s) for v in args.variants for s in args.seeds]
    ov = _overrides(args)
    t0 = time.perf_counter()
    if args.jobs <= 1:
        for v, s in jobs:
            print(ex.train_one(v, s, ov), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futs = [pool.submit(ex.train_one, v, s, ov, 1) for v, s in jobs]
            for f in as_completed(futs):
                print(f.result(), flush=True)
    print(f"train-all wall time: {time.perf_counter() - t0:.1f} s")
    return 0


def cmd_evaluate(args) -> int:
    if not (args.backend == "exact" or _is_simulated(args.backend)):
        print(
            f"Refusing to evaluate on {args.backend!r}: `evaluate` only runs local simulators "
            "(exact, aer, fake_<name>). Real IBM backends spend QPU time; use `circuitjepa hw-submit` "
            "(budget guard, dry run, job ids) and then `circuitjepa hw-fetch`.",
            file=sys.stderr,
        )
        return 2
    if (args.test_data_seed is None) != (args.tag is None):
        print(
            "--test-data-seed and --tag go together: an evaluation on another test set is saved "
            "as results/eval/<backend>_<tag>.json and must not overwrite <backend>.json",
            file=sys.stderr,
        )
        return 2
    runs = ex.load_runs()
    if not runs:
        print("no runs found under results/runs", file=sys.stderr)
        return 1
    ex.write_environment()
    t0 = time.perf_counter()
    if args.backend == "exact":
        data = ex.evaluate_exact(runs, args.test_data_seed)
    else:
        data = ex.evaluate_shots(runs, args.backend, args.m, args.shots, args.jobs, args.test_data_seed)
    data["wall_seconds"] = time.perf_counter() - t0
    if args.tag is not None:
        data["tag"] = args.tag
    path = ex.save_eval(args.backend if args.tag is None else f"{args.backend}_{args.tag}", data)
    print(f"wrote {path} ({data['wall_seconds']:.1f} s)")
    for name, rec in data["runs"].items():
        print(
            f"  {name:22s} top1={rec['top1']:.3f} gap={rec['diag_gap']:.3f} "
            f"collapsed={rec['collapsed']} probe pred/unadv={rec['probe_mse_pred']:.4f}/"
            f"{rec['probe_mse_unadv']:.4f}"
        )
    return 0


def _is_simulated(name: str) -> bool:
    return name == "aer" or name.startswith("fake_")


def cmd_hw_submit(args) -> int:
    runs = ex.load_runs(args.runs)
    n = len(runs) * len(ex.KINDS) * args.m * ex.N_BASES
    est = estimate_qpu_seconds(n, args.shots)
    print(f"backend: {args.backend}")
    print(f"runs: {', '.join(runs)}")
    print(f"circuits: {n} ({len(runs)} runs x {len(ex.KINDS)} kinds x {args.m} samples x {ex.N_BASES} bases)")
    print(f"shots per circuit: {args.shots}")
    print(f"estimated QPU seconds: {est:.1f} (budget {QPU_BUDGET_S:.0f} s)")
    if est > QPU_BUDGET_S and not args.force:
        print(
            f"Refusing to submit: estimate {est:.1f} s exceeds the {QPU_BUDGET_S:.0f} s budget. "
            "Re-run with --force to override."
        )
        return 2

    from circuitjepa.hardware import get_backend, submit, transpile

    stats, isa = {}, {}
    if _is_simulated(args.backend) or not args.dry_run:
        backend = get_backend(args.backend)
        for name, run in runs.items():
            circs = ex.run_circuits(run, ex.first_test_samples(run.config, args.m))
            isa[name] = transpile(circs, backend)
            stats[name] = ex.stats_by_kind(isa[name], args.m)
        print("circuit_stats:")
        print(json.dumps(stats, indent=1))
    else:
        print("circuit_stats: skipped (real backend in dry run; use a fake_* backend for a transpiled estimate)")

    record = {
        "backend": args.backend,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": list(runs),
        "m": args.m,
        "shots": args.shots,
        "n_circuits": n,
        "estimated_qpu_seconds": est,
        "circuit_stats": stats,
    }
    hw = ex.results_dir() / "hardware"
    if args.dry_run:
        ex.write_json(hw / f"dry_run_{args.backend}.json", record)
        print("Dry run: nothing submitted.")
        return 0

    jobs_file = hw / "jobs.json"
    stamp = record["date"].replace(":", "").replace("-", "")
    submission_id = f"{args.backend}_{stamp}_{uuid.uuid4().hex[:8]}"
    record["submission_id"] = submission_id
    record["params_hash"] = {name: ex.params_hash(run) for name, run in runs.items()}
    record["job_ids"] = {}
    for name in runs:
        extra = {
            "submission_id": submission_id,
            "params_hash": record["params_hash"][name],
            "m": args.m,
            "submission_runs": list(runs),
        }
        jid = submit(isa[name], backend, args.shots, jobs_file, tag=name, transpiled=True, extra=extra)
        record["job_ids"][name] = jid
        print(f"submitted {name}: job {jid}")
    ex.write_json(hw / f"submission_{submission_id}.json", record)
    print(f"submission id: {submission_id}")
    return 0


class FetchRefused(Exception):
    pass


FAILED_STATES = ("ERROR", "CANCELLED")


def _job_status(job_id: str) -> str:
    from circuitjepa.hardware import job_status

    return job_status(job_id)


def _fetch_counts(job_id: str) -> list[dict]:
    from circuitjepa.hardware import fetch

    return fetch(job_id)


def _submission_records(submission_id: str | None) -> tuple[str, dict[str, dict]]:
    """Job records of one submission (default: the most recent one), checked for consistency."""
    path = ex.results_dir() / "hardware" / "jobs.json"
    if not path.exists():
        raise FetchRefused("no jobs recorded in results/hardware/jobs.json")
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    recs = [r for r in recs if r.get("submission_id")]
    if not recs:
        raise FetchRefused("no job record carries a submission_id (submitted by an older circuitjepa?)")
    if submission_id is None:
        submission_id = recs[-1]["submission_id"]
    chosen = [r for r in recs if r["submission_id"] == submission_id]
    if not chosen:
        raise FetchRefused(f"unknown submission id {submission_id!r}")
    by_tag: dict[str, dict] = {}
    for r in chosen:
        if r["tag"] in by_tag:
            raise FetchRefused(f"submission {submission_id} has two jobs for {r['tag']}")
        by_tag[r["tag"]] = r
    expected = set(chosen[0].get("submission_runs", []))
    if expected != set(by_tag):
        raise FetchRefused(
            f"submission {submission_id} is incomplete: expected {sorted(expected)}, found {sorted(by_tag)}"
        )
    for key in ("backend", "shots", "m", "submission_runs"):
        values = {json.dumps(r.get(key)) for r in chosen}
        if len(values) != 1:
            raise FetchRefused(f"jobs of submission {submission_id} differ in {key}: {sorted(values)}")
    return submission_id, by_tag


def _check_overwrite(tag: str, rec: dict, force: bool) -> None:
    """Refuse to replace raw counts saved from a different job unless forced."""
    path = ex.results_dir() / "hardware" / f"counts_{tag}.json"
    if path.exists() and not force:
        old = json.loads(path.read_text(encoding="utf-8")).get("job_id")
        if old != rec["job_id"]:
            raise FetchRefused(
                f"{path} holds counts of job {old}, not {rec['job_id']}; it is not overwritten. "
                "Move it away or re-run with --force."
            )


def _save_raw(tag: str, rec: dict, counts: list[dict], submission_id: str) -> None:
    m = rec["m"]
    index = ex.circuit_index(m)
    if len(counts) != len(index):
        raise FetchRefused(f"{tag}: got {len(counts)} results, expected {len(index)}")
    bloch = ex.blochs_from_counts(counts, m)
    payload = {
        "tag": tag,
        "submission_id": submission_id,
        "job_id": rec["job_id"],
        "backend": rec["backend"],
        "shots": rec["shots"],
        "m": m,
        "params_hash": rec["params_hash"],
        "submitted_at": rec.get("submitted_at"),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "circuits": [{**idx, "counts": c} for idx, c in zip(index, counts)],
        "bloch": {kind: v.tolist() for kind, v in bloch.items()},
    }
    ex.write_json(ex.results_dir() / "hardware" / f"counts_{tag}.json", payload, indent=None)


def cmd_hw_fetch(args) -> int:
    try:
        submission_id, recs = _submission_records(args.submission)
    except FetchRefused as exc:
        print(f"Refusing to fetch: {exc}")
        return 1
    print(f"submission: {submission_id}")
    try:
        for tag, rec in recs.items():
            _check_overwrite(tag, rec, args.force)
    except FetchRefused as exc:
        print(f"Refusing to fetch: {exc}")
        return 1
    pending, failed = [], []
    for tag, rec in recs.items():
        status = _job_status(rec["job_id"])
        print(f"{tag}: job {rec['job_id']} {status}")
        if status in FAILED_STATES:
            failed.append(tag)
            from circuitjepa.hardware import job_error

            print(f"  {status}: {job_error(rec['job_id'])}")
        elif status != "DONE":
            pending.append(tag)
    if failed:
        print(f"job(s) ended without results: {', '.join(failed)}; nothing was evaluated")
        return 6
    if pending:
        print(f"not finished yet: {', '.join(pending)}; try again later")
        return 3

    # Persist raw counts and Bloch vectors first: IBM keeps results only for a limited time.
    counts = {}
    try:
        for tag, rec in recs.items():
            counts[tag] = _fetch_counts(rec["job_id"])
            _save_raw(tag, rec, counts[tag], submission_id)
            print(f"saved raw counts for {tag}")
    except FetchRefused as exc:
        print(f"Refusing to evaluate: {exc}")
        return 4

    runs = ex.load_runs(list(recs))
    bad = [t for t, r in recs.items() if ex.params_hash(runs[t]) != r["params_hash"]]
    if bad:
        print(
            "Refusing to evaluate: parameters of "
            + ", ".join(bad)
            + " differ from those submitted (params_hash mismatch). Raw counts were saved."
        )
        return 5

    first = next(iter(recs.values()))
    data = ex.evaluate_from_counts(runs, counts, "qpu", {"job_ids": {t: r["job_id"] for t, r in recs.items()}})
    data["backend_name"] = first["backend"]
    data["shots"] = first["shots"]
    data["m"] = first["m"]
    data["submission_id"] = submission_id
    data["job_ids"] = {t: r["job_id"] for t, r in recs.items()}
    data["params_hash"] = {t: r["params_hash"] for t, r in recs.items()}
    data["submitted_at"] = {t: r["submitted_at"] for t, r in recs.items()}
    path = ex.save_eval("qpu", data)
    print(f"wrote {path}")
    print(json.dumps(data["tests"], indent=1))
    return 0


def cmd_figures(args) -> int:
    from circuitjepa.figures import make_all

    for p in make_all():
        print(f"wrote {p}")
    return 0


def cmd_env(args) -> int:
    print(f"wrote {ex.write_environment()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="circuitjepa", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train", help="train one run")
    p.add_argument("--variant", required=True, choices=ex.VARIANTS)
    p.add_argument("--seed", type=int, required=True)
    _add_tunables(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("train-all", help="train every variant x seed")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--variants", nargs="+", default=list(ex.VARIANTS), choices=ex.VARIANTS)
    p.add_argument("--jobs", type=int, default=1, help="parallel processes (1 thread each)")
    _add_tunables(p)
    p.set_defaults(func=cmd_train_all)

    p = sub.add_parser("evaluate", help="evaluate all runs on a backend")
    p.add_argument("--backend", required=True,
                   help="exact, aer or fake_<name> (e.g. fake_fez); real backends: use hw-submit")
    p.add_argument("--m", type=int, default=16, help="test samples for shot-based backends")
    p.add_argument("--shots", type=int, default=1000)
    p.add_argument("--jobs", type=int, default=1, help="parallel processes over runs")
    p.add_argument("--test-data-seed", dest="test_data_seed", type=int,
                   help=f"evaluate on another test set (held-out set: {HELDOUT_TEST_DATA_SEED}); "
                        "the probe is still fitted on the training set; needs --tag")
    p.add_argument("--tag", choices=[ex.HELDOUT_TAG],
                   help="save as results/eval/<backend>_<tag>.json (needs --test-data-seed)")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("hw-submit", help="submit tomography circuits to an IBM backend")
    p.add_argument("--backend", required=True)
    p.add_argument("--shots", type=int, default=1000)
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--runs", nargs="+", default=DEFAULT_HW_RUNS)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="allow exceeding the QPU budget")
    p.set_defaults(func=cmd_hw_submit)

    p = sub.add_parser("hw-fetch", help="fetch finished hardware jobs and evaluate")
    p.add_argument("--submission", help="submission id (default: the most recent one)")
    p.add_argument("--force", action="store_true", help="overwrite counts saved from a different job")
    p.set_defaults(func=cmd_hw_fetch)

    p = sub.add_parser("figures", help="render figures from results/")
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("env", help="write results/ENVIRONMENT.json")
    p.set_defaults(func=cmd_env)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, RuntimeError) as exc:  # unknown run names, missing IBM credentials
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
