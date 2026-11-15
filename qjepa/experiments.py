"""Experiment orchestration: training runs, backend evaluation, summaries."""

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

from qjepa.data import Sample, magnetisation, make_dataset
from qjepa.hardware import (
    bloch_from_counts,
    circuit_stats,
    get_backend,
    run_counts,
    tomography_circuits,
    transpile,
)
from qjepa.metrics import (
    evaluate_embeddings,
    predictor_distance,
    row_gaps,
    signflip_p,
    similarity_matrix,
    top1_binom_p,
    total_variance,
)
from qjepa.model import bound_circuit, embed
from qjepa.train import RunResult, TrainConfig, load_run, save_run, train

VARIANTS = ("full", "no_ema", "no_predictor", "untrained")
KINDS = ("context", "target", "unadvanced")
N_BASES = 9
# Pre-registered hardware criterion (design spec, "Hardware protocol").
ALPHA_GAP = 0.01
ALPHA_TOP1 = 0.05


def results_dir() -> Path:
    return Path(os.environ.get("QJEPA_RESULTS", "results"))


def is_smoke() -> bool:
    return os.environ.get("QJEPA_SMOKE", "") == "1"


def run_name(variant: str, seed: int) -> str:
    return f"{variant}_seed{seed}"


def run_path(name: str) -> Path:
    return results_dir() / "runs" / f"{name}.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")


# --------------------------------------------------------------------- training


def make_config(variant: str, seed: int, overrides: dict | None = None) -> TrainConfig:
    kw = {k: v for k, v in (overrides or {}).items() if v is not None}
    if is_smoke():
        kw.setdefault("n_train", 16)
        kw.setdefault("n_test", 8)
    return TrainConfig(variant=variant, seed=seed, **kw)


def train_one(variant: str, seed: int, overrides: dict | None = None, threads: int | None = None) -> str:
    if threads is not None:
        torch.set_num_threads(threads)
    cfg = make_config(variant, seed, overrides)
    res = train(cfg)
    path = run_path(run_name(variant, seed))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_run(res, path)
    return f"{run_name(variant, seed)}: final loss {res.history[-1] if res.history else float('nan'):.4f}, {res.wall_seconds:.1f} s"


def load_runs(names: list[str] | None = None) -> dict[str, RunResult]:
    if names is None:
        paths = sorted((results_dir() / "runs").glob("*.json"))
    else:
        paths = [run_path(n) for n in names]
    return {p.stem: load_run(p) for p in paths}


# ------------------------------------------------------------------- embeddings


def datasets(cfg: TrainConfig) -> tuple[list[Sample], list[Sample]]:
    train_set = make_dataset(cfg.n_train, cfg.train_data_seed, cfg.t0_max)
    test_set = make_dataset(cfg.n_test, cfg.test_data_seed, cfg.t0_max)
    return train_set, test_set


def first_test_samples(cfg: TrainConfig, m: int) -> list[Sample]:
    """First m test samples (the sample stream does not depend on n_test)."""
    return make_dataset(m, cfg.test_data_seed, cfg.t0_max)


def magnetisations(samples: list[Sample], delta: float) -> np.ndarray:
    return np.stack([magnetisation(s, s.t0 + s.s, delta) for s in samples])


def _params(run: RunResult):
    theta = torch.tensor(run.theta, dtype=torch.float64)
    theta_bar = torch.tensor(run.theta_bar, dtype=torch.float64)
    phi = torch.tensor(run.phi, dtype=torch.float64) if run.phi else None
    return theta, theta_bar, phi


def exact_embeddings(run: RunResult, samples: list[Sample], kind: str) -> np.ndarray:
    cfg = run.config
    theta, theta_bar, phi = _params(run)
    use_pred = cfg.variant != "no_predictor"
    th = theta_bar if kind == "target" else theta
    with torch.no_grad():
        out = embed(samples, kind, th, phi if kind == "context" else None, cfg.delta, use_pred)
    return out.numpy()


def run_circuits(run: RunResult, samples: list[Sample]) -> list:
    """Tomography circuits ordered kind-major, then sample, then basis."""
    cfg = run.config
    use_pred = cfg.variant != "no_predictor"
    phi = np.asarray(run.phi, float) if run.phi else None
    circs = []
    for kind in KINDS:
        theta = np.asarray(run.theta_bar if kind == "target" else run.theta, float)
        for smp in samples:
            circs += tomography_circuits(bound_circuit(smp, kind, theta, phi, cfg.delta, use_pred))
    return circs


def stats_by_kind(isa: list, m: int) -> dict:
    per = m * N_BASES
    return {kind: circuit_stats(isa[i * per : (i + 1) * per]) for i, kind in enumerate(KINDS)}


def blochs_from_counts(counts: list[dict], m: int) -> dict[str, np.ndarray]:
    per = m * N_BASES
    out = {}
    for i, kind in enumerate(KINDS):
        block = counts[i * per : (i + 1) * per]
        out[kind] = np.stack(
            [bloch_from_counts(block[j * N_BASES : (j + 1) * N_BASES]) for j in range(m)]
        )
    return out


# ------------------------------------------------------------------- evaluation


def _probe_inputs(run: RunResult):
    """Probe fitted on exact target embeddings of the training set (all backends)."""
    train_set, _ = datasets(run.config)
    S_train = exact_embeddings(run, train_set, "target")
    Z_train = magnetisations(train_set, run.config.delta)
    return S_train, Z_train


def _metrics(run: RunResult, emb: dict, test: list[Sample], baseline_tv: float | None, probe) -> dict:
    S_train, Z_train = probe
    Z_test = magnetisations(test, run.config.delta)
    out = evaluate_embeddings(
        emb["context"],
        emb["target"],
        emb["unadvanced"],
        S_train,
        Z_train,
        Z_test,
        [s.s for s in test],
        baseline_tv,
    )
    out["predictor_distance"] = predictor_distance(np.asarray(run.phi)) if run.phi else 0.0
    out["G"] = similarity_matrix(emb["context"], emb["target"]).tolist()
    return out


def hypothesis_tests(per_run: dict) -> dict:
    """Pre-registered tests: full vs untrained, paired by seed (seed 0 is the registered one)."""
    tests = {}
    for name, rec in per_run.items():
        if not name.startswith("full_seed"):
            continue
        seed = name[len("full_seed") :]
        base = per_run.get(f"untrained_seed{seed}")
        if base is None:
            continue
        Gf, Gu = np.asarray(rec["G"]), np.asarray(base["G"])
        m = Gf.shape[0]
        k = int(round(rec["top1"] * m))
        p_gap = signflip_p(row_gaps(Gf), row_gaps(Gu))
        p_top1 = top1_binom_p(k, m)
        tests[f"full_seed{seed}_vs_untrained_seed{seed}"] = {
            "m": m,
            "signflip_p_gap": p_gap,
            "top1_hits": k,
            "top1_binom_p": p_top1,
            "pass_gap": bool(p_gap < ALPHA_GAP),
            "pass_top1": bool(p_top1 < ALPHA_TOP1),
            "pre_registered": seed == "0",
        }
    return tests


def _assemble(backend: str, runs: dict[str, RunResult], embs: dict, tests_m: dict, extra: dict) -> dict:
    """Compute metrics for every run once all embeddings exist (baselines need untrained)."""
    per_run = {}
    for name, run in runs.items():
        emb, test = embs[name], tests_m[name]
        base = f"untrained_seed{run.config.seed}"
        baseline_tv = total_variance(embs[base]["target"]) if base in embs else None
        t0 = time.perf_counter()
        rec = _metrics(run, emb, test, baseline_tv, _probe_inputs(run))
        rec["variant"] = run.config.variant
        rec["seed"] = run.config.seed
        rec["m"] = len(test)
        rec["wall_seconds"] = extra.get("wall", {}).get(name, 0.0) + time.perf_counter() - t0
        if name in extra.get("stats", {}):
            rec["circuit_stats"] = extra["stats"][name]
        if name in extra.get("job_ids", {}):
            rec["job_id"] = extra["job_ids"][name]
        per_run[name] = rec
    return {
        "backend": backend,
        "date": _now(),
        "probe_note": "linear probe fitted on exact (noiseless) target embeddings of the training set",
        "runs": per_run,
        "tests": hypothesis_tests(per_run),
    }


def evaluate_exact(runs: dict[str, RunResult]) -> dict:
    embs, tests_m, wall = {}, {}, {}
    for name, run in runs.items():
        t0 = time.perf_counter()
        _, test = datasets(run.config)
        embs[name] = {k: exact_embeddings(run, test, k) for k in KINDS}
        tests_m[name] = test
        wall[name] = time.perf_counter() - t0
    out = _assemble("exact", runs, embs, tests_m, {"wall": wall})
    out["m"] = max(len(t) for t in tests_m.values())
    return out


def evaluate_shots(runs: dict[str, RunResult], backend_name: str, m: int, shots: int) -> dict:
    backend = get_backend(backend_name)
    embs, tests_m, wall, stats = {}, {}, {}, {}
    for name, run in runs.items():
        t0 = time.perf_counter()
        test = first_test_samples(run.config, m)
        isa = transpile(run_circuits(run, test), backend)
        stats[name] = stats_by_kind(isa, len(test))
        counts = run_counts(isa, backend, shots, transpiled=True)
        embs[name] = blochs_from_counts(counts, len(test))
        tests_m[name] = test
        wall[name] = time.perf_counter() - t0
        print(f"  {name}: {wall[name]:.1f} s", flush=True)
    out = _assemble(backend_name, runs, embs, tests_m, {"wall": wall, "stats": stats})
    out["m"] = m
    out["shots"] = shots
    return out


def evaluate_from_counts(
    runs: dict[str, RunResult], counts: dict[str, list[dict]], backend: str, extra: dict
) -> dict:
    """Same schema as evaluate_shots, from counts fetched later (QPU jobs)."""
    embs, tests_m = {}, {}
    for name, run in runs.items():
        m = len(counts[name]) // (len(KINDS) * N_BASES)
        tests_m[name] = first_test_samples(run.config, m)
        embs[name] = blochs_from_counts(counts[name], m)
    return _assemble(backend, runs, embs, tests_m, extra)


# --------------------------------------------------------------------- summary

SCALARS = ("top1", "diag_gap", "total_var", "mean_norm", "probe_mse_pred", "probe_mse_unadv", "predictor_distance")
BY_S = ("top1_by_s", "probe_mse_pred_by_s", "probe_mse_unadv_by_s")


def _flat(rec: dict) -> dict[str, float]:
    out = {k: float(rec[k]) for k in SCALARS}
    for k in BY_S:
        for s, v in rec[k].items():
            out[f"{k}[{s}]"] = float(v)
    return out


def _agg(values: list[float]) -> dict:
    a = np.asarray(values, float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "n": int(a.size),
    }


def summarise(evals: dict[str, dict]) -> dict:
    backends = {}
    for be, data in evals.items():
        by_var: dict[str, list[dict]] = {}
        for rec in data["runs"].values():
            by_var.setdefault(rec["variant"], []).append(rec)
        backends[be] = {}
        for var in VARIANTS:
            recs = by_var.get(var)
            if not recs:
                continue
            flats = [_flat(r) for r in recs]
            row = {k: _agg([f[k] for f in flats]) for k in flats[0]}
            row["collapsed_count"] = sum(1 for r in recs if r["collapsed"] is True)
            row["seeds"] = sorted(r["seed"] for r in recs)
            row["wall_seconds_total"] = float(sum(r["wall_seconds"] for r in recs))
            backends[be][var] = row
        backends[be]["_meta"] = {
            k: data[k] for k in ("m", "shots", "date") if k in data
        }
    return {
        "std_note": "sample standard deviation over seeds (ddof=1)",
        "backends": backends,
        "tests": {be: d.get("tests", {}) for be, d in evals.items()},
    }


BACKEND_ORDER = ("exact", "aer", "qpu")


def _sort_backends(names):
    def key(n):
        if n in BACKEND_ORDER:
            return (BACKEND_ORDER.index(n), n)
        return (1.5, n)

    return sorted(names, key=key)


def _pm(cell: dict, fmt: str = "{:.3f}") -> str:
    return f"{fmt.format(cell['mean'])} ± {fmt.format(cell['std'])}"


def summary_markdown(summary: dict) -> str:
    lines = [
        "# QJEPA results summary",
        "",
        "Mean ± sample std (ddof=1) over seeds. `collapsed` = number of seeds flagged collapsed",
        "(diag gap < 0.05 and target total variance < 5 % of the same-seed untrained baseline).",
        "Probe: linear map fitted on exact target embeddings of the training set, evaluated",
        "on predicted (`pred`) vs un-advanced context (`unadv`) embeddings; target = ⟨Z_i⟩(t₀+s).",
        "",
    ]
    for be in _sort_backends(summary["backends"]):
        rows = summary["backends"][be]
        meta = rows.get("_meta", {})
        desc = f"m = {meta.get('m')}" + (f", shots = {meta['shots']}" if meta.get("shots") else "")
        lines += [f"## Backend: {be} ({desc})", ""]
        lines += [
            "| variant | top-1 | diag gap | total var | mean ‖r‖ | collapsed | probe pred | probe unadv | ‖P−I‖ |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for var in VARIANTS:
            if var not in rows:
                continue
            r = rows[var]
            lines.append(
                f"| {var} | {_pm(r['top1'])} | {_pm(r['diag_gap'])} | {_pm(r['total_var'])} | "
                f"{_pm(r['mean_norm'])} | {r['collapsed_count']}/{len(r['seeds'])} | "
                f"{_pm(r['probe_mse_pred'], '{:.4f}')} | {_pm(r['probe_mse_unadv'], '{:.4f}')} | "
                f"{_pm(r['predictor_distance'])} |"
            )
        lines += ["", "Per time step s:", ""]
        lines += [
            "| variant | s | top-1 | probe pred | probe unadv |",
            "|---|---|---|---|---|",
        ]
        for var in VARIANTS:
            if var not in rows:
                continue
            r = rows[var]
            for s in sorted({k.split("[")[1].rstrip("]") for k in r if k.startswith("top1_by_s[")}):
                lines.append(
                    f"| {var} | {s} | {_pm(r[f'top1_by_s[{s}]'])} | "
                    f"{_pm(r[f'probe_mse_pred_by_s[{s}]'], '{:.4f}')} | "
                    f"{_pm(r[f'probe_mse_unadv_by_s[{s}]'], '{:.4f}')} |"
                )
        tests = summary["tests"].get(be, {})
        if tests:
            lines += [
                "",
                f"Pre-registered tests (full vs untrained; gap: paired sign-flip, p < {ALPHA_GAP}; "
                f"top-1: binomial vs chance 1/m, p < {ALPHA_TOP1}). Seed 0 is the registered pair.",
                "",
                "| pair | m | gap p | top-1 hits | top-1 p | pass gap | pass top-1 |",
                "|---|---|---|---|---|---|---|",
            ]
            for pair, t in tests.items():
                mark = " (registered)" if t["pre_registered"] else ""
                lines.append(
                    f"| {pair}{mark} | {t['m']} | {t['signflip_p_gap']:.2g} | {t['top1_hits']}/{t['m']} | "
                    f"{t['top1_binom_p']:.2g} | {t['pass_gap']} | {t['pass_top1']} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def load_evals() -> dict[str, dict]:
    d = results_dir() / "eval"
    out = {}
    for p in sorted(d.glob("*.json")):
        if p.stem == "summary":
            continue
        out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return out


def write_summary() -> dict:
    summary = summarise(load_evals())
    write_json(results_dir() / "eval" / "summary.json", summary)
    (results_dir() / "SUMMARY.md").write_text(summary_markdown(summary), encoding="utf-8")
    return summary


def save_eval(name: str, data: dict) -> Path:
    path = results_dir() / "eval" / f"{name}.json"
    write_json(path, data)
    write_summary()
    return path


# ------------------------------------------------------------------ environment


def environment() -> dict:
    def ver(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None

    return {
        "python": sys.version.split()[0],
        "packages": {
            p: ver(p) for p in ("qiskit", "qiskit-aer", "qiskit-ibm-runtime", "torch", "numpy", "scipy", "matplotlib")
        },
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "date": _now(),
    }


def write_environment() -> Path:
    path = results_dir() / "ENVIRONMENT.json"
    write_json(path, environment())
    return path

