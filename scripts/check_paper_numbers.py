"""Check every number quoted in paper/circuitjepa.md against the files in results/.

Each claim is (text as written in the paper, value recomputed from a results file,
source description). A claim passes when the value, formatted as in the paper,
equals the text AND the text occurs in the paper. The script also lists decimal
numbers in the paper that no claim covers, for manual review.

Usage: python scripts/check_paper_numbers.py [--skip-transpile]  (exit code 1 on any mismatch)
--skip-transpile skips the FakeMarrakesh re-transpilation used for the layout claims.
"""

import json
import re
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
PAPER = (ROOT / "paper" / "circuitjepa.md").read_text(encoding="utf-8")


def load(rel: str):
    return json.loads((RES / rel).read_text(encoding="utf-8"))


SUMMARY = load("eval/summary.json")
EVAL = {b: load(f"eval/{b}.json") for b in ("exact", "aer", "fake_fez", "fake_marrakesh", "qpu", "exact_heldout", "aer_heldout")}
JOBS = [json.loads(l) for l in (RES / "hardware/jobs.json").read_text().splitlines() if l.strip()]
METRICS = load("hardware/job_metrics.json")
CAL = load("hardware/calibration_ibm_marrakesh.json")
ENV = load("ENVIRONMENT.json")
RUN = {p.stem: json.loads(p.read_text()) for p in (RES / "runs").glob("*.json")}

claims: list[tuple[str, str, str]] = []


def f(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}"


def pm(cell: dict, d: int = 3) -> str:
    return f"{f(cell['mean'], d)} ± {f(cell['std'], d)}"


def sci(p: float) -> str:
    """1.3e-17 -> '1.3 × 10^-17^' (pandoc superscript)."""
    m, e = f"{p:.1e}".split("e")
    return f"{m} × 10^{int(e)}^"


def add(text: str, value: str, src: str) -> None:
    claims.append((text, value, src))


def S(be: str, var: str, key: str) -> dict:
    return SUMMARY["backends"][be][var][key]


def R(be: str, run: str, key: str):
    return EVAL[be]["runs"][run][key]


# ---------------------------------------------------------------- Table 1 (exact)
for var in ("full", "no_ema", "no_predictor", "untrained"):
    row = "| " + var + " | " + " | ".join(
        [pm(S("exact", var, k)) for k in ("top1", "diag_gap", "total_var", "mean_norm")]
        + [pm(S("exact", var, k), 4) for k in ("probe_mse_pred", "probe_mse_unadv")]
        + [pm(S("exact", var, "predictor_distance"))]
    ) + " |"
    add(row, row, f"eval/summary.json backends.exact.{var}.*")
    assert S("exact", var, "top1")["n"] == 10
add("No run was flagged as collapsed", "No run was flagged as collapsed"
        if all(r["collapsed"] is not True for b in EVAL.values() for r in b["runs"].values()) else "?",
    "eval/*.json runs.*.collapsed (None = no same-seed baseline, QPU full_seed2)")
from circuitjepa.ansatz import N_ENC_PARAMS, N_PRED_PARAMS  # noqa: E402

add("for 30 parameters in total", f"for {N_ENC_PARAMS} parameters in total", "circuitjepa/ansatz.py N_ENC_PARAMS")
add("8 parameters)", f"{N_PRED_PARAMS} parameters)", "circuitjepa/ansatz.py N_PRED_PARAMS")

# ---------------------------------------------------------------- per-seed exact (10 seeds)
SEEDS = range(10)
for v in ("full", "no_ema", "no_predictor", "untrained"):
    for s in SEEDS:
        k = EVAL["exact"]["runs"][f"{v}_seed{s}"]["top1"] * 64
        assert abs(k - round(k)) < 1e-9
hits = lambda be, run, m: round(R(be, run, "top1") * m)
add("59, 21, 10, 5, 28, 36, 50, 59, 15 and 62 out of 64",
    ", ".join(str(hits("exact", f"full_seed{s}", 64)) for s in range(9)) + f" and {hits('exact', 'full_seed9', 64)} out of 64",
    "eval/exact.json runs.full_seed*.top1")
add("20, 48, 7, 11, 31, 44, 12, 1, 61 and 1 out of 64",
    ", ".join(str(hits("exact", f"no_ema_seed{s}", 64)) for s in range(9)) + f" and {hits('exact', 'no_ema_seed9', 64)} out of 64",
    "eval/exact.json runs.no_ema_seed*.top1")
ft = [R("exact", f"full_seed{s}", "top1") for s in SEEDS]
add("(median 0.5)", f"(median {st.median(ft):g})", "eval/exact.json full top1 median")
fl = [RUN[f"full_seed{s}"]["history"][-1] for s in SEEDS]
add("range from 0.105 to 0.532", f"range from {f(min(fl))} to {f(max(fl))}", "runs/full_seed*.json history[-1]")
late = [R("exact", f"full_seed{s}", "top1") for s in range(3, 10)]
early = [R("exact", f"full_seed{s}", "top1") for s in range(3)]
add("average 0.569 ± 0.341, against 0.469 ± 0.402",
    f"average {f(np.mean(late))} ± {f(np.std(late, ddof=1))}, against {f(np.mean(early))} ± {f(np.std(early, ddof=1))}",
    "eval/exact.json full seeds 3-9 vs 0-2")
T = EVAL["exact"]["tests"]
assert all(T[f"full_seed{s}_vs_untrained_seed{s}"]["pass_gap"] and T[f"full_seed{s}_vs_untrained_seed{s}"]["pass_top1"] for s in SEEDS)
assert all(abs(t["signflip_p_gap"] - 1 / 10001) < 1e-15 for t in T.values())  # all at the resampling floor
add("sign-flip p = 1.0 × 10^-4^, the resampling floor, for each",
    "sign-flip p = " + (sci(T["full_seed0_vs_untrained_seed0"]["signflip_p_gap"])
                         if len({t["signflip_p_gap"] for t in T.values()}) == 1 else "?") + ", the resampling floor, for each",
    "eval/exact.json tests.*.signflip_p_gap")
bp = {s: T[f"full_seed{s}_vs_untrained_seed{s}"]["top1_binom_p"] for s in SEEDS}
smax, smin = max(bp, key=bp.get), min(bp, key=bp.get)
add("binomial p between 3.3 × 10^-3^ for seed 3 and 2.0 × 10^-109^ for seed 9",
    f"binomial p between {sci(bp[smax])} for seed {smax} and {sci(bp[smin])} for seed {smin}", "eval/exact.json tests.*.top1_binom_p")
add("0.539 ± 0.341 (chance 1/64)", pm(S("exact", "full", "top1")) + " (chance 1/64)", "summary exact.full.top1")
assert (min(hits('exact', f'full_seed{s}', 64) for s in SEEDS), max(hits('exact', f'full_seed{s}', 64) for s in SEEDS)) == (5, 62)
add("top-1 ranges from 0.078 to 0.969", f"top-1 ranges from {f(min(ft))} to {f(max(ft))}", "eval/exact.json full_seed*.top1")
add("1/64 ≈ 0.016", f"1/64 ≈ {f(1/64)}", "chance level")
order = sorted(SEEDS, key=lambda s: -ft[s])
assert order[0] == 9 and ft[0] == ft[7] and ft[0] < ft[9] and sum(x > ft[0] for x in ft) == 1
add("ties for second (seed 9 reaches 62/64)", f"ties for second (seed {order[0]} reaches {hits('exact', f'full_seed{order[0]}', 64)}/64)",
    "eval/exact.json full top1 ranking")
add("it turned out to be the best of the three tuning seeds", "it turned out to be the best of the three tuning seeds"
    if max(range(3), key=lambda s_: ft[s_]) == 0 else "?", "eval/exact.json full_seed0-2.top1")

# ---------------------------------------------------------------- ablation prose
add("top-1 retrieval is 0.031 ± 0.040", "top-1 retrieval is " + pm(S("exact", "no_predictor", "top1")), "summary exact.no_predictor.top1")
add("diagonal gap is 0.116 ± 0.118", "diagonal gap is " + pm(S("exact", "no_predictor", "diag_gap")), "summary exact.no_predictor.diag_gap")
add("`no_ema` reaches 0.369 ± 0.332 top-1 and 0.486 ± 0.286 diagonal gap",
    f"`no_ema` reaches {pm(S('exact','no_ema','top1'))} top-1 and {pm(S('exact','no_ema','diag_gap'))} diagonal gap", "summary exact.no_ema")
add("target variance is 1.288 ± 0.515 (`full`: 1.552 ± 0.370)",
    f"target variance is {pm(S('exact','no_ema','total_var'))} (`full`: {pm(S('exact','full','total_var'))})", "summary exact total_var")
ratio = lambda v, s: R("exact", f"{v}_seed{s}", "total_var") / R("exact", f"untrained_seed{s}", "total_var")
add("seeds 7 and 9 retrieve 1 of 64 with gaps of 0.062 and 0.053 and target variances at 41% and 51% of their untrained baselines",
    f"seeds 7 and 9 retrieve {hits('exact','no_ema_seed7',64)} of 64 with gaps of {f(R('exact','no_ema_seed7','diag_gap'))} and "
    f"{f(R('exact','no_ema_seed9','diag_gap'))} and target variances at {round(100*ratio('no_ema',7))}% and {round(100*ratio('no_ema',9))}% of their untrained baselines",
    "eval/exact.json no_ema_seed7/9")
assert hits("exact", "no_ema_seed9", 64) == 1
nef = sum(1 for s in SEEDS if R("exact", f"no_ema_seed{s}", "diag_gap") < 0.07)
add("2 of 10 `no_ema` seeds ended", f"{nef} of 10 `no_ema` seeds ended", "eval/exact.json no_ema gap < 0.07")
add("2 of 10 seeds end with near-zero retrieval (1/64), not flagged by the formal collapse criterion",
    f"{nef} of 10 seeds end with near-zero retrieval ({max(hits('exact', f'no_ema_seed{s}', 64) for s in (7, 9))}/64), not flagged by the formal collapse criterion"
    if all(hits('exact', f'no_ema_seed{s}', 64) == 1 and R('exact', f'no_ema_seed{s}', 'collapsed') is False for s in (7, 9)) else "?",
    "eval/exact.json no_ema_seed7/9 (abstract)")

# tuning attempts (results/tuning/attempt*/exact.json)
def tuning(att: int, var: str) -> str:
    d = load(f"tuning/attempt{att}/exact.json")["runs"]
    vals = [d[f"{var}_seed{s}"]["top1"] for s in range(3)]
    return f"{f(np.mean(vals))} ± {f(np.std(vals, ddof=1))}"


add("(`full` top-1 0.245 ± 0.157)", f"(`full` top-1 {tuning(1, 'full')})", "tuning/attempt1/exact.json")
add("(0.271 ± 0.212, fail)", f"({tuning(2, 'full')}, fail)", "tuning/attempt2/exact.json")
add("(0.469 ± 0.402, pass)", f"({tuning(3, 'full')}, pass)", "tuning/attempt3/exact.json")
add("(0.214 ± 0.140, fail)", f"({tuning(4, 'full')}, fail)", "tuning/attempt4/exact.json")
add("exact top-1 of 0.500 ± 0.204, against 0.245 ± 0.157",
    f"exact top-1 of {tuning(1, 'no_predictor')}, against {tuning(1, 'full')}", "tuning/attempt1/exact.json")
cfg = RUN["full_seed0"]["config"]
for att, txt in ((2, "$\\tau = {tau}$ and {epochs} epochs"),
                 (4, "learning rate {lr} with batch size {batch_size}")):
    c2 = load(f"tuning/attempt{att}/runs/full_seed0.json")["config"]
    add(txt.format(tau=0.996, epochs=200, lr=0.05, batch_size=16), txt.format(**c2), f"tuning/attempt{att}/runs/full_seed0.json config")
c3 = load("tuning/attempt1/runs/full_seed0.json")["config"]
add("Attempt 1 used the defaults ($\\delta = 0.25$)", f"Attempt 1 used the defaults ($\\delta = {c3['delta']}$)", "tuning/attempt1/runs/full_seed0.json config")
add("$\\delta = 0.5$, $t_0 \\le 3$, 3 encoder layers, 2 predictor layers, learning rate 0.02 (Adam), 150 epochs, $\\tau = 0.99$, batch size 32, 256 training and 64 test samples",
    f"$\\delta = {cfg['delta']}$, $t_0 \\le {cfg['t0_max']}$, 3 encoder layers, 2 predictor layers, learning rate {cfg['lr']} (Adam), "
    f"{cfg['epochs']} epochs, $\\tau = {cfg['tau']}$, batch size {cfg['batch_size']}, {cfg['n_train']} training and {cfg['n_test']} test samples",
    "runs/full_seed0.json config (layers: circuitjepa/ansatz.py)")
# every run (seeds 0-9) uses exactly the final configuration: no re-tuning for seeds 3-9
assert len(RUN) == 40 and all(r["config"] == {**cfg, "variant": r["config"]["variant"], "seed": r["config"]["seed"]} for r in RUN.values())
add("Four variants are trained with seeds 0–9", "Four variants are trained with seeds " + f"{min(r['config']['seed'] for r in RUN.values())}–{max(r['config']['seed'] for r in RUN.values())}",
    "runs/*.json config.seed")
add("256 training samples (data seed 1234) and 64 test samples (data seed 5678)",
    f"{cfg['n_train']} training samples (data seed {cfg['train_data_seed']}) and {cfg['n_test']} test samples (data seed {cfg['test_data_seed']})",
    "runs/*.json config")
from circuitjepa.data import make_dataset  # noqa: E402

cnt = [sum(1 for x in make_dataset(64, 5678, 3) if x.s == s) for s in (1, 2, 3)]
add("18, 23 and 23 samples", f"{cnt[0]}, {cnt[1]} and {cnt[2]} samples", "test set (seed 5678)")
cnt16 = [sum(1 for x in make_dataset(16, 5678, 3) if x.s == s) for s in (1, 2, 3)]
add("(5, 5 and 6 samples)", f"({cnt16[0]}, {cnt16[1]} and {cnt16[2]} samples)", "first 16 test samples")
import subprocess  # noqa: E402

# chronology: seeds 3-9 were committed after the first committed results (seeds 0-2)
first_add = lambda path: subprocess.run(["git", "log", "--diff-filter=A", "--format=%ct", "--", path], cwd=ROOT,
                                        capture_output=True, text=True).stdout.split()
t_s0, t_s3 = first_add("results/runs/full_seed0.json"), first_add("results/runs/full_seed3.json")
assert t_s0 and (not t_s3 or int(t_s3[-1]) > int(t_s0[-1])), "seeds 3-9 must be added after seeds 0-2"

# ---------------------------------------------------------------- collapse
add("exact diagonal gap of 0.0457", f"exact diagonal gap of {f(R('exact','no_predictor_seed2','diag_gap'),4)}", "eval/exact.json no_predictor_seed2.diag_gap")
assert [s for s in SEEDS for v in ("full", "no_ema", "no_predictor") if R("exact", f"{v}_seed{s}", "diag_gap") < 0.05] == [2]
add("its target total variance is 0.720, against 1.158",
    f"its target total variance is {f(R('exact','no_predictor_seed2','total_var'))}, against {f(R('exact','untrained_seed2','total_var'))}",
    "eval/exact.json no_predictor_seed2 / untrained_seed2 total_var")
add("With 0/64 retrieved", f"With {hits('exact','no_predictor_seed2',64)}/64 retrieved", "eval/exact.json")
others = [s for s in SEEDS if s != 2 and 0.05 <= R("exact", f"no_predictor_seed{s}", "diag_gap") < 0.07]
og = [R("exact", f"no_predictor_seed{s}", "diag_gap") for s in others]
add("Six other `no_predictor` seeds are near-collapsed, with gaps between 0.059 and 0.068 and at most 3 of 64 retrieved",
    f"{['Zero','One','Two','Three','Four','Five','Six','Seven'][len(others)]} other `no_predictor` seeds are near-collapsed, with gaps between {f(min(og))} and {f(max(og))} "
    f"and at most {max(hits('exact', f'no_predictor_seed{s}', 64) for s in others)} of 64 retrieved", "eval/exact.json no_predictor gaps")

# three categories (Section 4): collapse / near-collapse / retrieval failure, exact original test set
def near(v, s_):
    r, b = EVAL["exact"]["runs"][f"{v}_seed{s_}"], EVAL["exact"]["runs"][f"untrained_seed{s_}"]
    return r["collapsed"] is False and r["diag_gap"] < 0.07 and r["total_var"] < 0.75 * b["total_var"]
NEAR = {v: [s_ for s_ in SEEDS if near(v, s_)] for v in ("full", "no_ema", "no_predictor")}
assert NEAR == {"full": [], "no_ema": [7, 9], "no_predictor": sorted([2] + others)}, NEAR
assert all(hits("exact", f"{v}_seed{s_}", 64) <= 3 for v in NEAR for s_ in NEAR[v])  # near-collapsed runs all fail at retrieval
add("a gap below 0.07 and a target total variance below 75% of the baseline", "a gap below 0.07 and a target total variance below 75% of the baseline",
    "definition used in near() above")
add("2 of 10 `no_ema` seeds and seven `no_predictor` seeds are near-collapsed",
    f"{len(NEAR['no_ema'])} of 10 `no_ema` seeds and {['zero','one','two','three','four','five','six','seven','eight'][len(NEAR['no_predictor'])]} `no_predictor` seeds are near-collapsed",
    "near-collapse definition on eval/exact.json")
add("`no_predictor` seed 0 (0/64, gap 0.068)", f"`no_predictor` seed 0 ({hits('exact','no_predictor_seed0',64)}/64, gap {f(R('exact','no_predictor_seed0','diag_gap'))})",
    "eval/exact.json no_predictor_seed0")
norms = [r["mean_norm"] for r in EVAL["exact"]["runs"].values()]
add("between 0.93 and 1.48", f"between {np.floor(min(norms)*100)/100:.2f} and {np.ceil(max(norms)*100)/100:.2f}", "eval/exact.json runs.*.mean_norm (min floored, max ceiled)")

# ---------------------------------------------------------------- probe
add("(0.0483 ± 0.0058 against 0.0715 ± 0.0258)",
    f"({pm(S('exact','full','probe_mse_pred'),4)} against {pm(S('exact','full','probe_mse_unadv'),4)})", "summary exact.full.probe")
bys = lambda k: ", ".join(f(S("exact", "full", f"{k}[{s}]")["mean"], 4) for s in (1, 2, 3))
add("(0.0424, 0.0471, 0.0541 against 0.0566, 0.0831, 0.0716",
    f"({bys('probe_mse_pred_by_s')} against {bys('probe_mse_unadv_by_s')}", "summary exact.full.probe_*_by_s")
npu = sum(R("exact", f"full_seed{s}", "probe_mse_pred") < R("exact", f"full_seed{s}", "probe_mse_unadv") for s in SEEDS)
add("and in 8 of 10 seeds", f"and in {npu} of 10 seeds", "eval/exact.json full probe pred < unadv")
pmed = lambda v: st.median(R("exact", f"{v}_seed{s}", "probe_mse_pred") for s in SEEDS)
add("the median probe MSE is 0.0473, against 0.0480 for `full`", f"the median probe MSE is {f(pmed('no_predictor'),4)}, against {f(pmed('full'),4)} for `full`",
    "eval/exact.json probe_mse_pred medians")
add("$\\lVert P - I \\rVert$ of 0.461 ± 0.259", f"$\\lVert P - I \\rVert$ of {pm(S('exact','full','predictor_distance'))}", "summary exact.full.predictor_distance")
npp = {s: R("exact", f"no_predictor_seed{s}", "probe_mse_pred") for s in SEEDS}
worst = max(npp, key=npp.get)
rest = [v for s, v in npp.items() if s != worst]
add("The `no_predictor` mean of 1.9033 ± 5.8633 comes from seed 5 alone, whose probe MSE is 18.5904; without seed 5 the mean is 0.0492 ± 0.0064",
    f"The `no_predictor` mean of {pm(S('exact','no_predictor','probe_mse_pred'),4)} comes from seed {worst} alone, whose probe MSE is {f(npp[worst],4)}; "
    f"without seed {worst} the mean is {f(np.mean(rest),4)} ± {f(np.std(rest, ddof=1),4)}", "eval/exact.json no_predictor probe")
assert max(rest) < 0.1
# anomaly mechanism, recomputed from the run's exact embeddings
from circuitjepa import experiments as ex  # noqa: E402
from circuitjepa.metrics import _design, fit_probe  # noqa: E402
from circuitjepa.paulis import PAULI_LABELS  # noqa: E402

_r5 = ex.load_runs(["no_predictor_seed5"])["no_predictor_seed5"]
_tr, _te = ex.datasets(_r5.config)
_S = ex.exact_embeddings(_r5, _tr, "target")
_P = ex.exact_embeddings(_r5, _te, "context")
_W = fit_probe(_S, ex.magnetisations(_tr, _r5.config.delta))
_Zte = ex.magnetisations(_te, _r5.config.delta)
_ix, _iy = PAULI_LABELS.index("XZ"), PAULI_LABELS.index("YZ")
_low = set(np.argsort(_S.std(0))[:2])
assert _low == {_ix, _iy}
add("(standard deviations 0.0072 and 0.0055)", f"(standard deviations {f(_S[:, _ix].std(), 4)} and {f(_S[:, _iy].std(), 4)})", "no_predictor_seed5 train-target std XZ, YZ")
add("weights of up to 33.08", f"weights of up to {np.abs(_W[:-1]).max():.2f}", "no_predictor_seed5 ridge probe weights")
add("shift the means of these components by 0.082 and 0.151",
    f"shift the means of these components by {f(_P[:, _ix].mean() - _S[:, _ix].mean())} and {f(_P[:, _iy].mean() - _S[:, _iy].mean())}",
    "no_predictor_seed5 test-pred mean - train-target mean")
_err = np.mean((_design(_P) @ _W - _Zte) ** 2, axis=1)
add("so all 64 test samples have errors above 1", f"so all {int((_err > 1).sum())} test samples have errors above 1" if len(_err) == 64 else "?", "no_predictor_seed5 per-sample probe error")
_St = ex.exact_embeddings(_r5, _te, "target")
add("the probe's test MSE is 0.0295", f"the probe's test MSE is {f(np.mean((_design(_St) @ _W - _Zte) ** 2), 4)}", "no_predictor_seed5 probe on test targets")
add("`no_predictor` seed 0 (0.4694)", f"`no_predictor` seed 0 ({f(R('qpu','no_predictor_seed0','probe_mse_pred'),4)})", "eval/qpu.json no_predictor_seed0")
diff = lambda be, run: abs(R(be, run, "probe_mse_pred") - R(be, run, "probe_mse_unadv"))
mx = {be: max((diff(be, f"no_predictor_seed{s}"), s) for s in SEEDS if s != worst) for be in ("aer", "fake_fez", "fake_marrakesh")}
add("Excluding seed 5, they reach 0.1035 on aer, 0.0905 on fake_fez and 0.0742 on fake_marrakesh (seed 5: 5.160, 3.888 and 4.767)",
    f"Excluding seed {worst}, they reach {f(mx['aer'][0],4)} on aer, {f(mx['fake_fez'][0],4)} on fake_fez and {f(mx['fake_marrakesh'][0],4)} on fake_marrakesh "
    f"(seed {worst}: {f(diff('aer', f'no_predictor_seed{worst}'))}, {f(diff('fake_fez', f'no_predictor_seed{worst}'))} and {f(diff('fake_marrakesh', f'no_predictor_seed{worst}'))})",
    "eval/{aer,fake_fez,fake_marrakesh}.json no_predictor |pred-unadv|")
fullmax = max(R(be, f"full_seed{s}", "probe_mse_unadv") - R(be, f"full_seed{s}", "probe_mse_pred") for be in ("aer", "fake_fez", "fake_marrakesh") for s in SEEDS)
add("The largest pred-versus-unadvanced difference of `full` on these backends is 0.0723", f"The largest pred-versus-unadvanced difference of `full` on these backends is {f(fullmax,4)}",
    "eval/{aer,fake_fez,fake_marrakesh}.json full unadv-pred")
assert fullmax < min(v[0] for v in mx.values()) or fullmax < max(v[0] for v in mx.values())

# ---------------------------------------------------------------- Table 2 (shot sims)
for be, label in (("aer", "aer (noiseless, 1000 shots)"), ("fake_fez", "fake_fez"), ("fake_marrakesh", "fake_marrakesh")):
    row = f"| {label} | " + " | ".join(pm(S(be, "full", k)) for k in ("top1", "diag_gap", "total_var", "mean_norm")) + " |"
    add(row, row, f"eval/summary.json backends.{be}.full")
    assert SUMMARY["backends"][be]["_meta"]["shots"] == 1000 and SUMMARY["backends"][be]["_meta"]["m"] == 16
    assert S(be, "full", "top1")["n"] == 10
add("`no_ema` gives 0.494 ± 0.330, 0.481 ± 0.331 and 0.494 ± 0.330",
    "`no_ema` gives " + ", ".join(pm(S(be, "no_ema", "top1")) for be in ("aer", "fake_fez")) + " and " + pm(S("fake_marrakesh", "no_ema", "top1")),
    "summary *.no_ema.top1")
same = {pm(S(be, "untrained", "top1")) for be in ("aer", "fake_fez", "fake_marrakesh")}
add("`untrained` gives 0.037 ± 0.044 on all three", f"`untrained` gives {same.pop() if len(same) == 1 else '?'} on all three", "summary *.untrained.top1")

# ---------------------------------------------------------------- Table 3 (all QPU runs vs simulators)
QR = EVAL["qpu"]["runs"]
assert len(QR) == 7
labels = {"full_seed0": "full seed 0 (pre-reg.)", "untrained_seed0": "untrained seed 0 (pre-reg.)", "no_ema_seed0": "no_ema seed 0",
          "full_seed1": "full seed 1 (expl.)", "full_seed2": "full seed 2 (expl.)", "no_predictor_seed0": "no_predictor seed 0 (expl.)",
          "untrained_seed1": "untrained seed 1 (expl.)"}
for run, lab in labels.items():
    row = (f"| {lab} | {QR[run]['shots']} | {hits('qpu', run, 16)}/16 | {f(R('qpu', run, 'diag_gap'))} | "
           f"{hits('aer', run, 16)}/16 | {f(R('aer', run, 'diag_gap'))} | {hits('fake_marrakesh', run, 16)}/16 |")
    add(row, row, f"eval/qpu.json, aer.json, fake_marrakesh.json runs.{run}")
    assert EVAL["qpu"]["sim_comparison"][run]["aer"]["top1"] == R("aer", run, "top1")
assert max(abs(hits("qpu", r, 16) - hits("aer", r, 16)) for r in QR) == 2
add("QPU top-1 is within two hits of the aer value", "QPU top-1 is within two hits of the aer value", "eval/qpu.json vs aer.json")
add("from 0.847 to 0.827", f"from {f(R('aer','full_seed0','diag_gap'))} to {f(R('qpu','full_seed0','diag_gap'))}", "eval/aer,qpu full_seed0.diag_gap")
add("from 1.823 to 1.218 and the mean norm from 1.441 to 1.241",
    f"from {f(R('aer','full_seed0','total_var'))} to {f(R('qpu','full_seed0','total_var'))} and the mean norm from "
    f"{f(R('aer','full_seed0','mean_norm'))} to {f(R('qpu','full_seed0','mean_norm'))}", "eval/aer,qpu full_seed0")
add("15 of 16 targets on the QPU, as on the simulators, against 1 of 16",
    f"{hits('qpu','full_seed0',16)} of 16 targets on the QPU, as on the simulators, against {hits('qpu','untrained_seed0',16)} of 16", "eval/qpu.json")
QT = EVAL["qpu"]["tests"]["full_seed0_vs_untrained_seed0"]
assert QT["pre_registered"] and QT["pass_gap"] and QT["pass_top1"] and QT["m"] == 16 and QT["top1_hits"] == 15
assert [k for k, t in EVAL["qpu"]["tests"].items() if t["pre_registered"]] == ["full_seed0_vs_untrained_seed0"]
assert not any(t["pre_registered"] for be in ("exact", "aer", "fake_fez", "fake_marrakesh") for t in EVAL[be]["tests"].values())
add("The diagonal-gap test gives p = 1.0 × 10^-4^", f"The diagonal-gap test gives p = {sci(QT['signflip_p_gap'])}", "eval/qpu.json tests")
add("the binomial top-1 test gives p = 1.3 × 10^-17^", f"the binomial top-1 test gives p = {sci(QT['top1_binom_p'])}", "eval/qpu.json tests")
add("(sign-flip p = 1.0 × 10^-4^, the floor of 10,000 resamples; binomial p = 1.3 × 10^-17^); in exploratory",
    f"(sign-flip p = {sci(QT['signflip_p_gap'])}, the floor of 10,000 resamples; binomial p = {sci(QT['top1_binom_p'])}); in exploratory",
    "eval/qpu.json tests (abstract)")
add("$1/10{,}001 \\approx 1.0 \\times 10^{-4}$", "$1/10{,}001 \\approx " + f"{1/10001:.1e}".replace("e-04", " \\times 10^{-4}") + "$", "metrics.signflip_p, n_resamples=10000")
assert abs(QT["signflip_p_gap"] - 1 / 10001) < 1e-15
q = QR["full_seed0"]["top1_by_s"]
add("1.000, 1.000 and 0.833", f"{f(q['1'])}, {f(q['2'])} and {f(q['3'])}", "eval/qpu.json full_seed0.top1_by_s")
X1 = EVAL["qpu"]["tests"]["full_seed1_vs_untrained_seed1"]
X2 = EVAL["qpu"]["tests"]["full_seed2_vs_untrained_seed0"]
assert not X1["pre_registered"] and not X2["pre_registered"] and X1["pass_gap"] and X1["pass_top1"] and X2["pass_gap"] and X2["pass_top1"]
add("`full` seeds 1 and 2 retrieve 10/16 and 6/16 on the QPU, against 8/16 and 6/16 on aer",
    f"`full` seeds 1 and 2 retrieve {hits('qpu','full_seed1',16)}/16 and {hits('qpu','full_seed2',16)}/16 on the QPU, against "
    f"{hits('aer','full_seed1',16)}/16 and {hits('aer','full_seed2',16)}/16 on aer", "eval/qpu.json, aer.json")
add("gives sign-flip p = 5.0 × 10^-4^ and binomial p = 5.1 × 10^-9^", f"gives sign-flip p = {sci(X1['signflip_p_gap'])} and binomial p = {sci(X1['top1_binom_p'])}", "eval/qpu.json tests full_seed1")
add("sign-flip p = 1.6 × 10^-3^ and binomial p = 2.8 × 10^-4^", f"sign-flip p = {sci(X2['signflip_p_gap'])} and binomial p = {sci(X2['top1_binom_p'])}", "eval/qpu.json tests full_seed2")
assert "untrained_seed2" not in QR and "full_seed3" not in QR
add("retrieves 2/16 on hardware (1/16 on aer) and `untrained` seed 1 retrieves 1/16",
    f"retrieves {hits('qpu','no_predictor_seed0',16)}/16 on hardware ({hits('aer','no_predictor_seed0',16)}/16 on aer) and `untrained` seed 1 retrieves {hits('qpu','untrained_seed1',16)}/16",
    "eval/qpu.json")
add("two further seeds retrieve 10/16 and 6/16", f"two further seeds retrieve {hits('qpu','full_seed1',16)}/16 and {hits('qpu','full_seed2',16)}/16", "eval/qpu.json (abstract)")
add("retrieved 10/16 and 6/16", f"retrieved {hits('qpu','full_seed1',16)}/16 and {hits('qpu','full_seed2',16)}/16", "eval/qpu.json (conclusion)")

# ---------------------------------------------------------------- hardware facts
jid = {j["tag"]: j for j in JOBS}
SUBS = {d["submission_id"]: d for d in (load(f"hardware/{p.name}") for p in (RES / "hardware").glob("submission_*.json"))}
assert len(SUBS) == 4 and len(JOBS) == 7
for tag in jid:
    add(f"`{jid[tag]['job_id']}`", f"`{EVAL['qpu']['job_ids'][tag]}`", f"jobs.json vs eval/qpu.json job id {tag}")
for sid in SUBS:
    add(f"`{sid}`", f"`{SUBS[sid]['submission_id']}`", "submission file")
for tag, txt in (("full_seed0", "10:07 UTC"), ("no_ema_seed0", "10:16 UTC"), ("full_seed1", "11:10 UTC"), ("no_predictor_seed0", "11:15 UTC")):
    add(txt, jid[tag]["submitted_at"][11:16] + " UTC", f"jobs.json submitted_at {tag}")
assert all(j["backend"] == "ibm_marrakesh" and j["n_circuits"] == 432 and j["m"] == 16 for j in JOBS)
assert all(j["submitted_at"].startswith("2026-11-15") for j in JOBS)
add("on 2026-11-15 on `ibm_marrakesh`", f"on {JOBS[0]['submitted_at'][:10]} on `{JOBS[0]['backend']}`", "jobs.json")
add("i.e. 432 circuits per model", f"i.e. {JOBS[0]['n_circuits']} circuits per model", "jobs.json")
charge = {t: METRICS[t]["usage"]["qpu_charge_time_seconds"] for t in jid}
shots = {t: jid[t]["shots"] for t in jid}
add("10:07 UTC, 1000 shots", f"10:07 UTC, {shots['full_seed0']} shots", "jobs.json shots")
add("each charged 118 s of QPU time", f"each charged {charge['full_seed0']} s of QPU time" if charge["full_seed0"] == charge["untrained_seed0"] else "?", "job_metrics.json")
add("1000 shots, job `dbsod0gk09pc73b0ebpg`, 118 s", f"{shots['no_ema_seed0']} shots, job `{jid['no_ema_seed0']['job_id']}`, {charge['no_ema_seed0']} s", "jobs/job_metrics no_ema")
add("500 shots, jobs `dbsp6b8k09pc73b0fb80` and `dbsp6bok09pc73b0fb90`, 60 s each",
    f"{shots['full_seed1']} shots, jobs `{jid['full_seed1']['job_id']}` and `{jid['full_seed2']['job_id']}`, {charge['full_seed1']} s each"
    if (shots["full_seed1"], charge["full_seed1"]) == (shots["full_seed2"], charge["full_seed2"]) else "?", "jobs/job_metrics submission 3")
add("450 shots, jobs `dbsp8e7v5hds73dj8png` and `dbsp8emjjaos73akfiog`, 55 s each",
    f"{shots['no_predictor_seed0']} shots, jobs `{jid['no_predictor_seed0']['job_id']}` and `{jid['untrained_seed1']['job_id']}`, {charge['no_predictor_seed0']} s each"
    if (shots["no_predictor_seed0"], charge["no_predictor_seed0"]) == (shots["untrained_seed1"], charge["untrained_seed1"]) else "?", "jobs/job_metrics submission 4")
acct = METRICS["account_usage_after"]
assert sum(charge.values()) == acct["usage_consumed_seconds"]
add("584 of 600 s were used", f"{acct['usage_consumed_seconds']} of {acct['usage_limit_seconds']} s were used", "job_metrics.json account_usage_after")
add("1000, 500 or 450 shots depending on the submission",
    ", ".join(str(s) for s in sorted(set(shots.values()), reverse=True)[:-1]) + f" or {min(shots.values())} shots depending on the submission", "jobs.json shots")
for t in jid:
    assert EVAL["qpu"]["runs"][t]["shots"] == shots[t]
assert jid["no_ema_seed0"]["submitted_at"] > max(METRICS[t]["timestamps"]["finished"].replace("Z", "") for t in ("full_seed0", "untrained_seed0"))
SUB1 = SUBS["ibm_marrakesh_20261115T100708+0000_6cc5cc3c"]
cs = SUB1["circuit_stats"]["full_seed0"]
add("mean depth of 62.6 (maximum 84) and a mean of 21.4 two-qubit gates (maximum 27)",
    f"mean depth of {cs['context']['depth_mean']:.1f} (maximum {cs['context']['depth_max']}) and a mean of {cs['context']['twoq_mean']:.1f} two-qubit gates (maximum {cs['context']['twoq_max']})",
    "submission_..6cc5cc3c circuit_stats.full_seed0.context")
add("the mean depth is 77.8 (maximum 109) with a mean of 27.0 two-qubit gates (maximum 36)",
    f"the mean depth is {cs['target']['depth_mean']:.1f} (maximum {cs['target']['depth_max']}) with a mean of {cs['target']['twoq_mean']:.1f} two-qubit gates (maximum {cs['target']['twoq_max']})",
    "circuit_stats.full_seed0.target")
add("48.2 (maximum 68) with a mean of 18.75 two-qubit gates (maximum 24)",
    f"{cs['unadvanced']['depth_mean']:.1f} (maximum {cs['unadvanced']['depth_max']}) with a mean of {cs['unadvanced']['twoq_mean']:.2f} two-qubit gates (maximum {cs['unadvanced']['twoq_max']})",
    "circuit_stats.full_seed0.unadvanced")
layouts = {tuple(st_["first_layout"]["initial"]) for sub in SUBS.values() for runs in sub["circuit_stats"].values() for st_ in runs.values()}
assert len(layouts) == 1 and sum(len(sub["circuit_stats"]) for sub in SUBS.values()) == 7
layout = list(layouts.pop())
add("[55, 54, 53, 39, 33]", str(layout), "submission_*.json circuit_stats.*.first_layout.initial")
add("the same for all seven runs", "the same for all seven runs", "submission_*.json (asserted above)")

# calibration on the layout qubits
qp = {q: {p["name"]: p["value"] for p in CAL["qubits"][q]} for q in layout}
add("median T1 of 176.4 µs", f"median T1 of {st.median(qp[q]['T1'] for q in layout):.1f} µs", "calibration qubits[layout].T1")
add("median T2 of 184.1 µs", f"median T2 of {st.median(qp[q]['T2'] for q in layout):.1f} µs", "calibration qubits[layout].T2")
add("median readout error of 0.0132", f"median readout error of {st.median(qp[q]['readout_error'] for q in layout):.4f}", "calibration qubits[layout].readout_error")
add("(0.0015 and 0.0039 on qubits 55 and 54)",
    f"({qp[layout[0]]['readout_error']:.4f} and {qp[layout[1]]['readout_error']:.4f} on qubits {layout[0]} and {layout[1]})", "calibration qubits[55,54]")
edges = {frozenset(e) for e in zip(layout, layout[1:])}
cz = {}
for g in CAL["gates"]:
    if g["gate"] == "cz" and frozenset(g["qubits"]) in edges:
        cz[frozenset(g["qubits"])] = next(p["value"] for p in g["parameters"] if p["name"] == "gate_error")
assert len(cz) == 4
add("median CZ error on the four couplers of this chain is 0.00176", f"median CZ error on the four couplers of this chain is {st.median(cz.values()):.5f}", "calibration gates cz on layout chain")
add("`last_update_date` is 2026-11-13 18:22:56+01:00", f"`last_update_date` is {CAL['last_update_date']}", "calibration last_update_date")
t_first = min(datetime.fromisoformat(j["submitted_at"]) for j in JOBS)
lag = t_first - datetime.fromisoformat(CAL["last_update_date"])
add("about 41 hours before the first job", f"about {round(lag.total_seconds() / 3600)} hours before the first job", "jobs.json vs calibration date")
val_dates = [datetime.fromisoformat(p["date"]) for q in layout for p in CAL["qubits"][q]
             if p["name"] in ("T1", "T2", "readout_error")]
val_dates += [datetime.fromisoformat(p["date"]) for g in CAL["gates"]
              if g["gate"] == "cz" and frozenset(g["qubits"]) in edges
              for p in g["parameters"] if p["name"] == "gate_error"]
lo, hi = (t_first - max(val_dates)).total_seconds() / 3600, (t_first - min(val_dates)).total_seconds() / 3600
add("carry dates between 54 and 57 hours before the first job", f"carry dates between {int(np.floor(lo))} and {int(np.ceil(hi))} hours before the first job",
    "calibration per-parameter dates vs jobs.json")
added = subprocess.run(["git", "log", "--diff-filter=A", "--format=%cI", "--", "results/hardware/calibration_ibm_marrakesh.json"],
                       cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()[-1]
t_cal = datetime.fromisoformat(added)
pair_done = max(datetime.fromisoformat(METRICS[t]["timestamps"]["finished"].replace("Z", "+00:00")) for t in ("full_seed0", "untrained_seed0"))
assert pair_done < t_cal < datetime.fromisoformat(jid["no_ema_seed0"]["submitted_at"])
# Upper bound only: the commit time of the file. The file mtime is not used because git does not
# preserve it (a fresh clone gets the checkout time), so it is not reliable evidence.
add("at about 10:13 UTC", f"at about {t_cal.astimezone(t_first.tzinfo):%H:%M} UTC",
    "UPPER BOUND on retrieval time: commit time of the git commit adding calibration_ibm_marrakesh.json")
gap_h = (datetime.fromisoformat(jid["full_seed1"]["submitted_at"]) - t_cal).total_seconds() / 3600
add("about an hour before submissions 3 and 4", "about an hour before submissions 3 and 4" if 0.5 < gap_h < 1.5 else "?", "jobs.json vs calibration commit time")
show = subprocess.run(["git", "show", "a473602"], cwd=ROOT, capture_output=True, text=True).stdout
assert "ALPHA_GAP = 0.01" in show and "ALPHA_TOP1 = 0.05" in show and 'pre_registered": seed == "0"' in show
add("commit `a473602`", "commit `a473602`", "git show a473602")
anc = subprocess.run(["git", "merge-base", "--is-ancestor", "a473602", "190699a"], cwd=ROOT).returncode
ct = lambda c: int(subprocess.run(["git", "show", "-s", "--format=%ct", c], cwd=ROOT, capture_output=True, text=True).stdout)
assert anc == 0 and ct("a473602") < ct("190699a"), "pre-registration commit is not before the first results commit"
tq = [cs[b]["twoq_mean"] for b in ("context", "target", "unadvanced")]
add("on average 19–27 CZ gates after transpilation for `ibm_marrakesh`", f"on average {round(min(tq))}–{round(max(tq))} CZ gates after transpilation for `{SUB1['backend']}`", "circuit_stats.full_seed0.*.twoq_mean")
if "--skip-transpile" not in sys.argv:
    import collections  # noqa: E402

    from circuitjepa.hardware import get_backend, transpile  # noqa: E402

    _run = ex.load_runs(["full_seed0"])["full_seed0"]
    _isa = transpile(ex.run_circuits(_run, ex.first_test_samples(_run.config, 16)), get_backend("fake_marrakesh"))
    _lays = collections.Counter(tuple(qc.layout.initial_index_layout(filter_ancillas=True)) for qc in _isa).most_common()
    _ops = {i.operation.name for qc in _isa for i in qc.data if len(i.qubits) == 2}
    add("404 used one layout and 28 another", f"{_lays[0][1]} used one layout and {_lays[1][1]} another" if len(_lays) == 2 else "?",
        "re-transpilation of full_seed0 circuits against FakeMarrakesh")
    add("CZ was the only two-qubit gate", "CZ was the only two-qubit gate" if _ops == {"cz"} else "?", "same re-transpilation")
    add("the 432 `full` seed-0 circuits", f"the {len(_isa)} `full` seed-0 circuits", "same re-transpilation")

# environment
pk = ENV["packages"]
add(f"Python 3.12.10, Qiskit 2.6.2, Qiskit Aer 0.18.2, qiskit-ibm-runtime 0.52.1, PyTorch 2.15.0",
    f"Python {ENV['python']}, Qiskit {pk['qiskit']}, Qiskit Aer {pk['qiskit-aer']}, qiskit-ibm-runtime {pk['qiskit-ibm-runtime']}, PyTorch {pk['torch'].split('+')[0]}",
    "ENVIRONMENT.json")
add("`qiskit-ibm-runtime` 0.52.1", f"`qiskit-ibm-runtime` {pk['qiskit-ibm-runtime']}", "ENVIRONMENT.json")

# conclusion
add("retrieved 15 of 16 targets and met the pre-registered criterion",
    f"retrieved {QT['top1_hits']} of {QT['m']} targets and "
    + ("met" if QT["pass_gap"] and QT["pass_top1"] else "?") + " the pre-registered criterion", "eval/qpu.json tests")
add("(sign-flip p = 1.0 × 10^-4^, the floor of 10,000 resamples; binomial p = 1.3 × 10^-17^); two",
    f"(sign-flip p = {sci(QT['signflip_p_gap'])}, the floor of 10,000 resamples; binomial p = {sci(QT['top1_binom_p'])}); two",
    "eval/qpu.json tests (conclusion)")

# ---------------------------------------------------------------- held-out test set (exact_heldout, aer_heldout)
from circuitjepa.config import HELDOUT_TEST_DATA_SEED  # noqa: E402

HO = EVAL["exact_heldout"]
assert HO["test_data_seed"] == HELDOUT_TEST_DATA_SEED == EVAL["aer_heldout"]["test_data_seed"] and HO["m"] == 64
assert all(r["config"]["test_data_seed"] != HELDOUT_TEST_DATA_SEED and r["config"]["train_data_seed"] != HELDOUT_TEST_DATA_SEED for r in RUN.values())
add("with a new data seed, 424242", f"with a new data seed, {HELDOUT_TEST_DATA_SEED}", "circuitjepa/config.py HELDOUT_TEST_DATA_SEED")
add("--test-data-seed 424242 --tag heldout\ncircuitjepa evaluate --backend aer --test-data-seed 424242",
    f"--test-data-seed {HELDOUT_TEST_DATA_SEED} --tag heldout\ncircuitjepa evaluate --backend aer --test-data-seed {HELDOUT_TEST_DATA_SEED}", "config (reproducibility)")
# chronology: the held-out seed was committed before any held-out evaluation file
t_seed = subprocess.run(["git", "log", "-S", "HELDOUT_TEST_DATA_SEED = 424242", "--format=%H %ct", "--reverse", "--", "circuitjepa/config.py"],
                        cwd=ROOT, capture_output=True, text=True).stdout.split()
t_ho = first_add("results/eval/exact_heldout.json")
assert t_seed and t_ho and int(t_ho[-1]) >= int(t_seed[1])
assert subprocess.run(["git", "merge-base", "--is-ancestor", t_seed[0], "HEAD"], cwd=ROOT).returncode == 0
_ho_add = subprocess.run(["git", "log", "--diff-filter=A", "--format=%H", "--", "results/eval/exact_heldout.json"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()[-1]
assert subprocess.run(["git", "merge-base", "--is-ancestor", t_seed[0], _ho_add + "~1"], cwd=ROOT).returncode == 0, "seed commit must precede held-out results"
cnt_h = [sum(1 for x in make_dataset(64, HELDOUT_TEST_DATA_SEED, 3) if x.s == s_) for s_ in (1, 2, 3)]
add("It has 21, 23 and 20 samples", f"It has {cnt_h[0]}, {cnt_h[1]} and {cnt_h[2]} samples", "held-out test set")
for var in ("full", "no_ema", "no_predictor", "untrained"):
    row = (f"| {var} | {pm(S('exact', var, 'top1'))} | {pm(S('exact_heldout', var, 'top1'))} | "
           f"{pm(S('exact', var, 'diag_gap'))} | {pm(S('exact_heldout', var, 'diag_gap'))} |")
    add(row, row, f"summary exact / exact_heldout {var}")
    assert S("exact_heldout", var, "top1")["n"] == 10
assert all(r["collapsed"] is False for r in HO["runs"].values())
add("and 0.530 ± 0.365 on an independent held-out test set never used for any choice (untrained baseline: 0.016 ± 0.019)",
    f"and {pm(S('exact_heldout','full','top1'))} on an independent held-out test set never used for any choice (untrained baseline: {pm(S('exact_heldout','untrained','top1'))})",
    "summary exact_heldout (abstract)")
hh = lambda v, s_: hits("exact_heldout", f"{v}_seed{s_}", 64)
add("(from 4/64 to 63/64 on the held-out set)", f"(from {min(hh('full', s_) for s_ in SEEDS)}/64 to {max(hh('full', s_) for s_ in SEEDS)}/64 on the held-out set)",
    "eval/exact_heldout.json full range")
add("`full` retrieves 58, 22, 7, 4, 31, 34, 50, 62, 8 and 63 of the 64 held-out targets",
    "`full` retrieves " + ", ".join(str(hh("full", s_)) for s_ in range(9)) + f" and {hh('full', 9)} of the 64 held-out targets", "eval/exact_heldout.json")
dif = {s_: hh("full", s_) - hits("exact", f"full_seed{s_}", 64) for s_ in SEEDS}
sw = max(dif, key=lambda k: abs(dif[k]))
add("within 7 hits of its original-set value for every seed (seed 8: 15 to 8)",
    f"within {abs(dif[sw])} hits of its original-set value for every seed (seed {sw}: {hits('exact', f'full_seed{sw}', 64)} to {hh('full', sw)})",
    "eval/exact.json vs exact_heldout.json")
add("`no_ema` seeds 7 and 9 retrieve 1 and 0 of 64", f"`no_ema` seeds 7 and 9 retrieve {hh('no_ema', 7)} and {hh('no_ema', 9)} of 64", "eval/exact_heldout.json")
assert SUMMARY["backends"]["aer_heldout"]["_meta"]["m"] == 16 and SUMMARY["backends"]["aer_heldout"]["_meta"]["shots"] == 1000
add("`full` reaches 0.719 ± 0.291 (chance 1/16)", f"`full` reaches {pm(S('aer_heldout','full','top1'))} (chance 1/16)", "summary aer_heldout.full.top1")
HT = HO["tests"]
assert all(t["pass_gap"] and t["pass_top1"] and not t["pre_registered"] and abs(t["signflip_p_gap"] - 1 / 10001) < 1e-15 for t in HT.values()) and len(HT) == 10
hbp = {s_: HT[f"full_seed{s_}_vs_untrained_seed{s_}"]["top1_binom_p"] for s_ in SEEDS}
hmax, hmin = max(hbp, key=hbp.get), min(hbp, key=hbp.get)
add("binomial p between 1.8 × 10^-2^ for seed 3 and 1.0 × 10^-112^ for seed 9",
    f"binomial p between {sci(hbp[hmax])} for seed {hmax} and {sci(hbp[hmin])} for seed {hmin}", "eval/exact_heldout.json tests")
hpm = lambda k: pm(S("exact_heldout", "full", k), 4)
hpu = sum(R("exact_heldout", f"full_seed{s_}", "probe_mse_pred") < R("exact_heldout", f"full_seed{s_}", "probe_mse_unadv") for s_ in SEEDS)
hmed = lambda v: st.median(R("exact_heldout", f"{v}_seed{s_}", "probe_mse_pred") for s_ in SEEDS)
add("for `full`, 0.0549 ± 0.0066 from the predicted against 0.0760 ± 0.0224 from the un-advanced embedding, lower in 9 of 10 seeds, and medians of 0.0566 for `no_predictor` against 0.0560 for `full`",
    f"for `full`, {hpm('probe_mse_pred')} from the predicted against {hpm('probe_mse_unadv')} from the un-advanced embedding, lower in {hpu} of 10 seeds, "
    f"and medians of {f(hmed('no_predictor'), 4)} for `no_predictor` against {f(hmed('full'), 4)} for `full`", "eval/exact_heldout.json probe")
add("close to the 0.539 ± 0.341 of the original test set", f"close to the {pm(S('exact','full','top1'))} of the original test set", "summary exact.full.top1 (conclusion)")
add("the ten `full` seeds reach a top-1 of 0.530 ± 0.365", f"the ten `full` seeds reach a top-1 of {pm(S('exact_heldout','full','top1'))}", "summary exact_heldout (conclusion)")

# ---------------------------------------------------------------- permutation tests, recomputed from the G matrices
from circuitjepa.metrics import permutation_p  # noqa: E402

PERM = load("permutation_tests.json")
assert PERM["n_resamples"] == 10000 and PERM["seed"] == 0
G = lambda be, run: np.asarray(EVAL[be]["runs"][run]["G"])
def perm(be, a, b=None):
    r = permutation_p(G(be, a), None if b is None else G(be, b), n_resamples=10000, seed=0)
    stored = PERM["evals"][be]["paired" if b else "single"][f"{a}_vs_{b}" if b else a]
    assert r["p_top1"] == stored["p_top1"] and r["p_gap"] == stored["p_gap"], (be, a, b)
    return r
FLOOR = 1 / 10001
pre = perm("qpu", "full_seed0", "untrained_seed0")
assert PERM["evals"]["qpu"]["paired"]["full_seed0_vs_untrained_seed0"]["pre_registered_pair"]
add("gives p = 1.0 × 10^-4^ for both the top-1 and the gap difference, the floor",
    f"gives p = {sci(pre['p_top1'])} for both the top-1 and the gap difference, the floor" if pre["p_top1"] == pre["p_gap"] == FLOOR else "?", "permutation qpu pre-registered pair")
x1 = perm("qpu", "full_seed1", "untrained_seed1")
assert x1["p_top1"] == x1["p_gap"] == FLOOR
x2 = perm("qpu", "full_seed2", "untrained_seed0")
add("`full` seed 2 against `untrained` seed 0 gives p = 6.0 × 10^-4^ (top-1) and 2.0 × 10^-4^ (gap)",
    f"`full` seed 2 against `untrained` seed 0 gives p = {sci(x2['p_top1'])} (top-1) and {sci(x2['p_gap'])} (gap)", "permutation qpu full_seed2 vs untrained_seed0")
u0, u1 = perm("qpu", "untrained_seed0"), perm("qpu", "untrained_seed1")
add("give top-1 p = 0.64 and 0.65", f"give top-1 p = {u0['p_top1']:.2f} and {u1['p_top1']:.2f}", "permutation qpu untrained single runs")
EP = {be: {s_: perm(be, f"full_seed{s_}", f"untrained_seed{s_}") for s_ in SEEDS} for be in ("exact", "exact_heldout")}
assert all(r["p_gap"] == FLOOR for be in EP for r in EP[be].values())
above = {be: [s_ for s_ in SEEDS if EP[be][s_]["p_top1"] > FLOOR] for be in EP}
assert above == {"exact": [3], "exact_heldout": [2, 3]}, above
add("seed 3 on the original set (p = 8.8 × 10^-3^)", f"seed 3 on the original set (p = {sci(EP['exact'][3]['p_top1'])})", "permutation exact full_seed3")
add("seeds 2 and 3 on the held-out set (7.9 × 10^-3^ and 3.3 × 10^-2^)",
    f"seeds 2 and 3 on the held-out set ({sci(EP['exact_heldout'][2]['p_top1'])} and {sci(EP['exact_heldout'][3]['p_top1'])})", "permutation exact_heldout")
add("7.9 × 10^-3^ against 6.5 × 10^-5^ for seed 2 on the held-out set",
    f"{sci(EP['exact_heldout'][2]['p_top1'])} against {sci(hbp[2])} for seed 2 on the held-out set", "permutation vs binomial, exact_heldout seed 2")
# every above-floor permutation p is larger than the binomial p of the same pair, and below 0.05
cmp_ = [(EP["exact"][3]["p_top1"], bp[3]), (EP["exact_heldout"][2]["p_top1"], hbp[2]), (EP["exact_heldout"][3]["p_top1"], hbp[3]),
        (x2["p_top1"], EVAL["qpu"]["tests"]["full_seed2_vs_untrained_seed0"]["top1_binom_p"])]
assert all(pp > bb and pp < 0.05 for pp, bb in cmp_)
add("but all remain below 0.05", "but all remain below 0.05", "asserted above")

# reproducibility: pinned versions
add("numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2", f"numpy {pk['numpy']}, scipy {pk['scipy']}, matplotlib {pk['matplotlib']}", "ENVIRONMENT.json")
cons = (ROOT / "constraints.txt").read_text()
assert all(f"{k}=={v.split('+')[0]}" in cons for k, v in pk.items())
rc = re.search(r"results at commit `([0-9a-f]+)`", PAPER).group(1)
assert subprocess.run(["git", "merge-base", "--is-ancestor", rc, "HEAD"], cwd=ROOT).returncode == 0
assert not subprocess.run(["git", "diff", "--stat", rc, "HEAD", "--", "results"], cwd=ROOT, capture_output=True, text=True).stdout.strip(), \
    "results/ changed after the commit quoted in the paper"
assert "IQM" not in PAPER and "iqm" not in PAPER.lower()

# ---------------------------------------------------------------- run checks
bad = 0
for text, value, src in claims:
    ok_val = text == value
    ok_in = text in PAPER
    if not (ok_val and ok_in):
        bad += 1
        print(f"FAIL [{src}]\n  paper text : {text}\n  from files : {value}\n  in paper   : {ok_in}")
print(f"{len(claims)} claims checked, {len(claims) - bad} OK, {bad} failed")

# coverage: decimal numbers in the paper not inside any claim text
covered = " ".join(t for t, _, _ in claims)
nums = set(re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", PAPER))
uncovered = sorted(n for n in nums if n not in covered)
print("decimal numbers not covered by a claim (manual review):", ", ".join(uncovered) or "none")
sys.exit(1 if bad else 0)
