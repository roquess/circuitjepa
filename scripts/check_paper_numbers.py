"""Check every number quoted in paper/circuitjepa.md against the files in results/.

Each claim is (text as written in the paper, value recomputed from a results file,
source description). A claim passes when the value, formatted as in the paper,
equals the text AND the text occurs in the paper. The script also lists decimal
numbers in the paper that no claim covers, for manual review.

Usage: python scripts/check_paper_numbers.py  (exit code 1 on any mismatch)
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
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PAPER = (ROOT / "paper" / "circuitjepa.md").read_text(encoding="utf-8")


def load(rel: str):
    return json.loads((RES / rel).read_text(encoding="utf-8"))


SUMMARY = load("eval/summary.json")
EVAL = {b: load(f"eval/{b}.json") for b in ("exact", "aer", "fake_fez", "fake_marrakesh", "qpu")}
JOBS = [json.loads(l) for l in (RES / "hardware/jobs.json").read_text().splitlines() if l.strip()]
METRICS = load("hardware/job_metrics.json")
SUB1 = load("hardware/submission_ibm_marrakesh_20261115T100708+0000_6cc5cc3c.json")
SUB2 = load("hardware/submission_ibm_marrakesh_20261115T101632+0000_a8e12a16.json")
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
    assert S("exact", var, "top1")["n"] == 3
add("No run was flagged as collapsed", "No run was flagged as collapsed"
        if all(r["collapsed"] is False for b in EVAL.values() for r in b["runs"].values()) else "?",
    "eval/*.json runs.*.collapsed")
from circuitjepa.ansatz import N_ENC_PARAMS, N_PRED_PARAMS  # noqa: E402

add("for 30 parameters in total", f"for {N_ENC_PARAMS} parameters in total", "circuitjepa/ansatz.py N_ENC_PARAMS")
add("8 parameters)", f"{N_PRED_PARAMS} parameters)", "circuitjepa/ansatz.py N_PRED_PARAMS")

# ---------------------------------------------------------------- per-seed exact
for v in ("full", "no_ema"):
    for s in range(3):
        k = EVAL["exact"]["runs"][f"{v}_seed{s}"]["top1"] * 64
        assert abs(k - round(k)) < 1e-9
for s, txt in enumerate(["59/64 (0.922)", "21/64 (0.328)", "10/64 (0.156)"]):
    t = R("exact", f"full_seed{s}", "top1")
    add(txt, f"{round(t * 64)}/64 ({f(t)})", f"eval/exact.json runs.full_seed{s}.top1")
add("0.872, 0.536 and 0.486",
    ", ".join(f(R("exact", f"full_seed{s}", "diag_gap")) for s in (0, 1)) + " and " + f(R("exact", "full_seed2", "diag_gap")),
    "eval/exact.json runs.full_seed*.diag_gap")
fl = [RUN[f"full_seed{s}"]["history"][-1] for s in range(3)]
add("0.117, 0.442 and 0.466", f"{f(fl[0])}, {f(fl[1])} and {f(fl[2])}", "runs/full_seed*.json history[-1]")
add("(0.922, 0.328, 0.156)", "(" + ", ".join(f(R("exact", f"full_seed{s}", "top1")) for s in range(3)) + ")",
    "eval/exact.json runs.full_seed*.top1")
add("20/64, 48/64 and 7/64",
    f"{round(R('exact','no_ema_seed0','top1')*64)}/64, {round(R('exact','no_ema_seed1','top1')*64)}/64 and {round(R('exact','no_ema_seed2','top1')*64)}/64",
    "eval/exact.json runs.no_ema_seed*.top1")
T = EVAL["exact"]["tests"]
add("sign-flip p = 1.0 × 10^-4^ for each",
    "sign-flip p = " + (sci(T["full_seed0_vs_untrained_seed0"]["signflip_p_gap"])
                         if len({t["signflip_p_gap"] for t in T.values()}) == 1 else "?") + " for each",
    "eval/exact.json tests.*.signflip_p_gap")
add("binomial p = 1.9 × 10^-100^, 2.5 × 10^-22^ and 6.1 × 10^-8^",
    "binomial p = " + ", ".join(sci(T[f"full_seed{s}_vs_untrained_seed{s}"]["top1_binom_p"]) for s in (0, 1))
    + " and " + sci(T["full_seed2_vs_untrained_seed2"]["top1_binom_p"]),
    "eval/exact.json tests.*.top1_binom_p")
add("0.469 ± 0.402 (chance 1/64)", pm(S("exact", "full", "top1")) + " (chance 1/64)", "summary exact.full.top1")
add("top-1 ranges from 0.156 to 0.922",
    f"top-1 ranges from {f(min(R('exact', f'full_seed{s}', 'top1') for s in range(3)))} to "
    f"{f(max(R('exact', f'full_seed{s}', 'top1') for s in range(3)))}", "eval/exact.json full_seed*.top1")
add("1/64 ≈ 0.016", f"1/64 ≈ {f(1/64)}", "chance level")

# ---------------------------------------------------------------- ablation prose
add("top-1 retrieval is 0.031 ± 0.054", "top-1 retrieval is " + pm(S("exact", "no_predictor", "top1")), "summary exact.no_predictor.top1")
add("diagonal gap is 0.175 ± 0.204", "diagonal gap is " + pm(S("exact", "no_predictor", "diag_gap")), "summary exact.no_predictor.diag_gap")
add("`no_ema` reaches 0.391 ± 0.327 top-1 and 0.573 ± 0.219 diagonal gap",
    f"`no_ema` reaches {pm(S('exact','no_ema','top1'))} top-1 and {pm(S('exact','no_ema','diag_gap'))} diagonal gap",
    "summary exact.no_ema")
add("(1.527 ± 0.522)", f"({pm(S('exact','no_ema','total_var'))})", "summary exact.no_ema.total_var")
add("(1.562 ± 0.479)", f"({pm(S('exact','full','total_var'))})", "summary exact.full.total_var")

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
for att, txt in ((2, "attempt 2 used $\\tau = {tau}$ and {epochs} epochs"),
                 (4, "attempt 4 used learning rate {lr} and batch size {batch_size}")):
    c2 = load(f"tuning/attempt{att}/runs/full_seed0.json")["config"]
    add(txt.format(tau=0.996, epochs=200, lr=0.05, batch_size=16), txt.format(**c2), f"tuning/attempt{att}/runs/full_seed0.json config")
c3 = load("tuning/attempt1/runs/full_seed0.json")["config"]
add("Attempt 1 used the defaults ($\\delta = 0.25$)", f"Attempt 1 used the defaults ($\\delta = {c3['delta']}$)", "tuning/attempt1/runs/full_seed0.json config")
add("$\\delta = 0.5$, $t_0 \\le 3$, 3 encoder layers, 2 predictor layers, learning rate 0.02 (Adam), 150 epochs, $\\tau = 0.99$, batch size 32, 256 training and 64 test samples",
    f"$\\delta = {cfg['delta']}$, $t_0 \\le {cfg['t0_max']}$, 3 encoder layers, 2 predictor layers, learning rate {cfg['lr']} (Adam), "
    f"{cfg['epochs']} epochs, $\\tau = {cfg['tau']}$, batch size {cfg['batch_size']}, {cfg['n_train']} training and {cfg['n_test']} test samples",
    "runs/full_seed0.json config (layers: circuitjepa/ansatz.py)")
assert all(r["config"] == {**cfg, "variant": r["config"]["variant"], "seed": r["config"]["seed"]} for r in RUN.values())
add("256 training samples (data seed 1234) and 64 test samples (data seed 5678)",
    f"{cfg['n_train']} training samples (data seed {cfg['train_data_seed']}) and {cfg['n_test']} test samples (data seed {cfg['test_data_seed']})",
    "runs/*.json config")
sys.path.insert(0, str(ROOT))
from circuitjepa.data import make_dataset  # noqa: E402

cnt = [sum(1 for x in make_dataset(64, 5678, 3) if x.s == s) for s in (1, 2, 3)]
add("18, 23 and 23 samples", f"{cnt[0]}, {cnt[1]} and {cnt[2]} samples", "test set (seed 5678)")
cnt16 = [sum(1 for x in make_dataset(16, 5678, 3) if x.s == s) for s in (1, 2, 3)]
add("(5, 5 and 6 samples)", f"({cnt16[0]}, {cnt16[1]} and {cnt16[2]} samples)", "first 16 test samples")

# ---------------------------------------------------------------- collapse
add("exact diagonal gap of 0.0457", f"exact diagonal gap of {f(R('exact','no_predictor_seed2','diag_gap'),4)}", "eval/exact.json no_predictor_seed2.diag_gap")
add("its target total variance is 0.720, against 1.158",
    f"its target total variance is {f(R('exact','no_predictor_seed2','total_var'))}, against {f(R('exact','untrained_seed2','total_var'))}",
    "eval/exact.json no_predictor_seed2 / untrained_seed2 total_var")
add("Its retrieval is 0/64", f"Its retrieval is {round(R('exact','no_predictor_seed2','top1')*64)}/64", "eval/exact.json")
norms = [r["mean_norm"] for r in EVAL["exact"]["runs"].values()]
add("between 0.95 and 1.48", f"between {np.floor(min(norms)*100)/100:.2f} and {np.ceil(max(norms)*100)/100:.2f}", "eval/exact.json runs.*.mean_norm (min floored, max ceiled)")

# ---------------------------------------------------------------- probe
add("(0.0498 ± 0.0082 against 0.0830 ± 0.0289)",
    f"({pm(S('exact','full','probe_mse_pred'),4)} against {pm(S('exact','full','probe_mse_unadv'),4)})", "summary exact.full.probe")
bys = lambda k: ", ".join(f(S("exact", "full", f"{k}[{s}]")["mean"], 4) for s in (1, 2, 3))
add("(0.0425, 0.0470, 0.0582 against 0.0614, 0.0992, 0.0838",
    f"({bys('probe_mse_pred_by_s')} against {bys('probe_mse_unadv_by_s')}", "summary exact.full.probe_*_by_s")
add("(0.0414 against 0.0936, and 0.0579 against 0.1052)",
    f"({f(R('exact','full_seed0','probe_mse_pred'),4)} against {f(R('exact','full_seed0','probe_mse_unadv'),4)}, and "
    f"{f(R('exact','full_seed1','probe_mse_pred'),4)} against {f(R('exact','full_seed1','probe_mse_unadv'),4)})", "eval/exact.json full_seed0/1")
add("(0.0500 against 0.0504)",
    f"({f(R('exact','full_seed2','probe_mse_pred'),4)} against {f(R('exact','full_seed2','probe_mse_unadv'),4)})", "eval/exact.json full_seed2")
add("probe MSE of 0.0474 ± 0.0012", f"probe MSE of {pm(S('exact','no_predictor','probe_mse_pred'),4)}", "summary exact.no_predictor.probe_mse_pred")
add("$\\lVert P - I \\rVert$ of 0.453 ± 0.330", f"$\\lVert P - I \\rVert$ of {pm(S('exact','full','predictor_distance'))}", "summary exact.full.predictor_distance")
diff = lambda be, run: abs(R(be, run, "probe_mse_pred") - R(be, run, "probe_mse_unadv"))
add("reach 0.0519 on aer, 0.0905 on fake_fez and 0.0742 on fake_marrakesh",
    f"reach {f(diff('aer','no_predictor_seed0'),4)} on aer, {f(diff('fake_fez','no_predictor_seed0'),4)} on fake_fez and {f(diff('fake_marrakesh','no_predictor_seed0'),4)} on fake_marrakesh",
    "eval/{aer,fake_fez,fake_marrakesh}.json no_predictor_seed0 |pred-unadv|")
for be in ("aer", "fake_fez", "fake_marrakesh"):  # seed 0 is the maximum on each backend
    assert max(diff(be, f"no_predictor_seed{s}") for s in range(3)) == diff(be, "no_predictor_seed0")
add("(for example 0.0475 on aer for seed 0)", f"(for example {f(diff('aer','full_seed0'),4)} on aer for seed 0)", "eval/aer.json full_seed0")
for be in ("aer", "fake_fez", "fake_marrakesh", "qpu"):
    full_runs = [n for n in EVAL[be]["runs"] if n.startswith("full")]
    assert max(diff(be, n) for n in full_runs) <= diff("fake_fez", "no_predictor_seed0")

# ---------------------------------------------------------------- Table 2 (shot sims)
for be, label in (("aer", "aer (noiseless, 1000 shots)"), ("fake_fez", "fake_fez"), ("fake_marrakesh", "fake_marrakesh")):
    row = f"| {label} | " + " | ".join(pm(S(be, "full", k)) for k in ("top1", "diag_gap", "total_var", "mean_norm")) + " |"
    add(row, row, f"eval/summary.json backends.{be}.full")
    assert SUMMARY["backends"][be]["_meta"]["shots"] == 1000 and SUMMARY["backends"][be]["_meta"]["m"] == 16
same = {pm(S(be, "no_ema", "top1")) for be in ("aer", "fake_fez", "fake_marrakesh")}
add("`no_ema` gives 0.521 ± 0.321 top-1 on all three simulators",
    f"`no_ema` gives {same.pop() if len(same) == 1 else '?'} top-1 on all three simulators", "summary *.no_ema.top1")
same = {pm(S(be, "untrained", "top1")) for be in ("aer", "fake_fez", "fake_marrakesh")}
add("`untrained` gives 0.021 ± 0.036", f"`untrained` gives {same.pop() if len(same) == 1 else '?'}", "summary *.untrained.top1")
k = {round(R(be, "full_seed0", "top1") * 16) for be in ("aer", "fake_fez", "fake_marrakesh")}
add("for seed 0 the top-1 is 15/16 on all three", f"for seed 0 the top-1 is {k.pop() if len(k) == 1 else '?'}/16 on all three", "eval/*.json full_seed0.top1")
k = {round(R(be, "no_ema_seed0", "top1") * 16) for be in ("aer", "fake_fez", "fake_marrakesh")}
add("(also 7/16 on all three shot-based simulators)", f"(also {k.pop() if len(k) == 1 else '?'}/16 on all three shot-based simulators)", "eval/*.json no_ema_seed0.top1")

# ---------------------------------------------------------------- Table 3 (QPU)
for be, run, label in (("aer", "full_seed0", "full, aer"), ("fake_marrakesh", "full_seed0", "full, fake_marrakesh"),
                       ("qpu", "full_seed0", "full, ibm_marrakesh"), ("qpu", "no_ema_seed0", "no_ema, ibm_marrakesh"),
                       ("qpu", "untrained_seed0", "untrained, ibm_marrakesh")):
    row = f"| {label} | " + " | ".join(f(R(be, run, k)) for k in ("top1", "diag_gap", "total_var", "mean_norm")) + " |"
    add(row, row, f"eval/{be}.json runs.{run}")
add("from 0.847 to 0.827", f"from {f(R('aer','full_seed0','diag_gap'))} to {f(R('qpu','full_seed0','diag_gap'))}", "eval/aer,qpu full_seed0.diag_gap")
add("from 1.823 to 1.218 and the mean norm from 1.441 to 1.241",
    f"from {f(R('aer','full_seed0','total_var'))} to {f(R('qpu','full_seed0','total_var'))} and the mean norm from "
    f"{f(R('aer','full_seed0','mean_norm'))} to {f(R('qpu','full_seed0','mean_norm'))}", "eval/aer,qpu full_seed0")
add("gives 0.438 top-1 (7/16)", f"gives {f(R('qpu','no_ema_seed0','top1'))} top-1 ({round(R('qpu','no_ema_seed0','top1')*16)}/16)", "eval/qpu.json no_ema_seed0")
add("15 of 16 targets on the QPU, as on the simulators, against 1 of 16",
    f"{round(R('qpu','full_seed0','top1')*16)} of 16 targets on the QPU, as on the simulators, against {round(R('qpu','untrained_seed0','top1')*16)} of 16",
    "eval/qpu.json")
QT = EVAL["qpu"]["tests"]["full_seed0_vs_untrained_seed0"]
assert QT["pre_registered"] and QT["pass_gap"] and QT["pass_top1"] and QT["m"] == 16 and QT["top1_hits"] == 15
add("The diagonal-gap test gives p = 1.0 × 10^-4^", f"The diagonal-gap test gives p = {sci(QT['signflip_p_gap'])}", "eval/qpu.json tests")
add("The binomial top-1 test gives p = 1.3 × 10^-17^", f"The binomial top-1 test gives p = {sci(QT['top1_binom_p'])}", "eval/qpu.json tests")
add("(sign-flip p = 1.0 × 10^-4^, binomial p = 1.3 × 10^-17^)", f"(sign-flip p = {sci(QT['signflip_p_gap'])}, binomial p = {sci(QT['top1_binom_p'])})", "eval/qpu.json tests (abstract)")
add("$1/10{,}001 \\approx 1.0 \\times 10^{-4}$", "$1/10{,}001 \\approx " + f"{1/10001:.1e}".replace("e-04", " \\times 10^{-4}") + "$", "metrics.signflip_p, n_resamples=10000")
assert abs(QT["signflip_p_gap"] - 1 / 10001) < 1e-15
q = SUMMARY["backends"]["qpu"]["full"]
add("1.000, 1.000 and 0.833", f"{f(q['top1_by_s[1]']['mean'])}, {f(q['top1_by_s[2]']['mean'])} and {f(q['top1_by_s[3]']['mean'])}", "summary qpu.full.top1_by_s")

# ---------------------------------------------------------------- hardware facts
jid = {j["tag"]: j for j in JOBS}
add("job `dbso8j9abdes73frdplg`", f"job `{jid['full_seed0']['job_id']}`", "hardware/jobs.json")
add("job `dbso8jujjaos73aked3g`", f"job `{jid['untrained_seed0']['job_id']}`", "hardware/jobs.json")
add("job `dbsod0gk09pc73b0ebpg`", f"job `{jid['no_ema_seed0']['job_id']}`", "hardware/jobs.json")
add("`ibm_marrakesh_20261115T100708+0000_6cc5cc3c`", f"`{SUB1['submission_id']}`", "submission file")
add("`ibm_marrakesh_20261115T101632+0000_a8e12a16`", f"`{SUB2['submission_id']}`", "submission file")
add("10:07 UTC", jid["full_seed0"]["submitted_at"][11:16] + " UTC", "jobs.json submitted_at")
add("10:16 UTC", jid["no_ema_seed0"]["submitted_at"][11:16] + " UTC", "jobs.json submitted_at")
assert all(j["backend"] == "ibm_marrakesh" and j["shots"] == 1000 and j["n_circuits"] == 432 and j["m"] == 16 for j in JOBS)
assert all(j["submitted_at"].startswith("2026-11-15") for j in JOBS)
add("on 2026-11-15 on `ibm_marrakesh`", f"on {JOBS[0]['submitted_at'][:10]} on `{JOBS[0]['backend']}`", "jobs.json")
add("1000 shots, which gives 432 circuits per model", f"{JOBS[0]['shots']} shots, which gives {JOBS[0]['n_circuits']} circuits per model", "jobs.json")
assert jid["no_ema_seed0"]["submitted_at"] > max(METRICS[t]["timestamps"]["finished"].replace("Z", "") for t in ("full_seed0", "untrained_seed0"))
charge = {METRICS[t]["usage"]["qpu_charge_time_seconds"] for t in ("full_seed0", "untrained_seed0", "no_ema_seed0")}
add("charged 118 s of QPU time", f"charged {charge.pop() if len(charge) == 1 else '?'} s of QPU time", "hardware/job_metrics.json usage.qpu_charge_time_seconds")
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
layouts = {tuple(st_["first_layout"]["initial"]) for sub in (SUB1, SUB2) for runs in sub["circuit_stats"].values() for st_ in runs.values()}
assert len(layouts) == 1
layout = list(layouts.pop())
add("[55, 54, 53, 39, 33]", str(layout), "submission_*.json circuit_stats.*.first_layout.initial")

# calibration on the layout qubits
qp = {q: {p["name"]: p["value"] for p in CAL["qubits"][q]} for q in layout}
add("median T1 of 176.4 µs", f"median T1 of {st.median(qp[q]['T1'] for q in layout):.1f} µs", "calibration qubits[layout].T1")
add("median T2 of 184.1 µs", f"median T2 of {st.median(qp[q]['T2'] for q in layout):.1f} µs", "calibration qubits[layout].T2")
add("median readout error of 0.0132", f"median readout error of {st.median(qp[q]['readout_error'] for q in layout):.4f}", "calibration qubits[layout].readout_error")
add("(0.0015 and 0.0039 on the two measured latent qubits)",
    f"({qp[layout[0]]['readout_error']:.4f} and {qp[layout[1]]['readout_error']:.4f} on the two measured latent qubits)", "calibration qubits[55,54]")
edges = {frozenset(e) for e in zip(layout, layout[1:])}
cz = {}
for g in CAL["gates"]:
    if g["gate"] == "cz" and frozenset(g["qubits"]) in edges:
        cz[frozenset(g["qubits"])] = next(p["value"] for p in g["parameters"] if p["name"] == "gate_error")
assert len(cz) == 4
add("median CZ error on the four couplers of the chain is 0.00176", f"median CZ error on the four couplers of the chain is {st.median(cz.values()):.5f}", "calibration gates cz on layout chain")
add("`last_update_date` 2026-11-13 18:22:56+01:00", f"`last_update_date` {CAL['last_update_date']}", "calibration last_update_date")
lag = datetime.fromisoformat(JOBS[0]["submitted_at"]) - datetime.fromisoformat(CAL["last_update_date"])
add("about 41 hours before the first job", f"about {round(lag.total_seconds() / 3600)} hours before the first job", "jobs.json vs calibration date")
add("predate the runs by about 41 hours", f"predate the runs by about {round(lag.total_seconds() / 3600)} hours", "jobs.json vs calibration date")

# environment
pk = ENV["packages"]
add(f"Python 3.12.10, Qiskit 2.6.2, Qiskit Aer 0.18.2, qiskit-ibm-runtime 0.52.1, PyTorch 2.15.0",
    f"Python {ENV['python']}, Qiskit {pk['qiskit']}, Qiskit Aer {pk['qiskit-aer']}, qiskit-ibm-runtime {pk['qiskit-ibm-runtime']}, PyTorch {pk['torch'].split('+')[0]}",
    "ENVIRONMENT.json")
add("`qiskit-ibm-runtime` 0.52.1", f"`qiskit-ibm-runtime` {pk['qiskit-ibm-runtime']}", "ENVIRONMENT.json")

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
