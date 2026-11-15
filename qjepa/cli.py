"""Command-line interface: `qjepa train | train-all | evaluate | hw-submit | hw-fetch | figures`."""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

from qjepa import experiments as ex
from qjepa.hardware import estimate_qpu_seconds

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
    runs = ex.load_runs()
    if not runs:
        print("no runs found under results/runs", file=sys.stderr)
        return 1
    ex.write_environment()
    t0 = time.perf_counter()
    if args.backend == "exact":
        data = ex.evaluate_exact(runs)
    else:
        data = ex.evaluate_shots(runs, args.backend, args.m, args.shots, args.jobs)
    data["wall_seconds"] = time.perf_counter() - t0
    path = ex.save_eval(args.backend, data)
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

    from qjepa.hardware import get_backend, submit, transpile

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
    record["job_ids"] = {}
    for name in runs:
        jid = submit(isa[name], backend, args.shots, jobs_file, tag=name, transpiled=True)
        record["job_ids"][name] = jid
        print(f"submitted {name}: job {jid}")
    stamp = record["date"].replace(":", "").replace("-", "")
    ex.write_json(hw / f"submission_{args.backend}_{stamp}.json", record)
    return 0


def _job_records() -> dict[str, dict]:
    path = ex.results_dir() / "hardware" / "jobs.json"
    if not path.exists():
        return {}
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            latest[rec["tag"]] = rec  # later submissions supersede earlier ones
    return latest


def cmd_hw_fetch(args) -> int:
    recs = _job_records()
    if not recs:
        print("no jobs recorded in results/hardware/jobs.json")
        return 1
    from qiskit_ibm_runtime import QiskitRuntimeService

    from qjepa.hardware import fetch

    service = QiskitRuntimeService()
    pending = []
    for tag, rec in recs.items():
        status = service.job(rec["job_id"]).status()
        status = getattr(status, "name", str(status))
        print(f"{tag}: job {rec['job_id']} {status}")
        if status != "DONE":
            pending.append(tag)
    if pending:
        print(f"not finished yet: {', '.join(pending)}; try again later")
        return 3
    runs = ex.load_runs(list(recs))
    counts = {tag: fetch(rec["job_id"]) for tag, rec in recs.items()}
    backends = sorted({rec["backend"] for rec in recs.values()})
    data = ex.evaluate_from_counts(
        runs,
        counts,
        "qpu",
        {"job_ids": {t: r["job_id"] for t, r in recs.items()}},
    )
    data["backend_name"] = ", ".join(backends)
    data["shots"] = sorted({rec["shots"] for rec in recs.values()})[0]
    data["m"] = max(r["m"] for r in data["runs"].values())
    data["job_ids"] = {t: r["job_id"] for t, r in recs.items()}
    data["submitted_at"] = {t: r["submitted_at"] for t, r in recs.items()}
    path = ex.save_eval("qpu", data)
    print(f"wrote {path}")
    print(json.dumps(data["tests"], indent=1))
    return 0


def cmd_figures(args) -> int:
    from qjepa.figures import make_all

    for p in make_all():
        print(f"wrote {p}")
    return 0


def cmd_env(args) -> int:
    print(f"wrote {ex.write_environment()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="qjepa", description=__doc__)
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
    p.add_argument("--backend", required=True, help="exact, aer or fake_<name> (e.g. fake_fez)")
    p.add_argument("--m", type=int, default=16, help="test samples for shot-based backends")
    p.add_argument("--shots", type=int, default=1000)
    p.add_argument("--jobs", type=int, default=1, help="parallel processes over runs")
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
    p.set_defaults(func=cmd_hw_fetch)

    p = sub.add_parser("figures", help="render figures from results/")
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("env", help="write results/ENVIRONMENT.json")
    p.set_defaults(func=cmd_env)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
