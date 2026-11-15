import numpy as np

from circuitjepa import experiments as ex
from circuitjepa.ansatz import N_ENC_PARAMS, N_PRED_PARAMS
from circuitjepa.hardware import get_backend, run_counts
from circuitjepa.train import RunResult, TrainConfig


def _run(variant, seed):
    g = np.random.default_rng(seed + 10 * len(variant))
    cfg = TrainConfig(variant=variant, seed=seed, n_train=16, n_test=8)
    th = list(g.uniform(0, 6, N_ENC_PARAMS))
    return RunResult(cfg, th, th, list(g.uniform(0, 6, N_PRED_PARAMS)), [], 0.0)


def test_evaluate_from_counts_matches_schema_and_tests():
    runs = {"full_seed0": _run("full", 0), "untrained_seed0": _run("untrained", 0)}
    m, be = 3, get_backend("aer")
    counts = {}
    for name, run in runs.items():
        circs = ex.run_circuits(run, ex.first_test_samples(run.config, m))
        assert len(circs) == 3 * m * 9
        counts[name] = run_counts(circs, be, 4000, seed=1)
    data = ex.evaluate_from_counts(runs, counts, "qpu", {"job_ids": {"full_seed0": "j1"}})
    rec = data["runs"]["full_seed0"]
    assert rec["m"] == m and rec["job_id"] == "j1"
    assert np.asarray(rec["G"]).shape == (m, m)
    assert rec["collapsed"] in (True, False)
    t = data["tests"]["full_seed0_vs_untrained_seed0"]
    assert t["pre_registered"] and 0 < t["signflip_p_gap"] <= 1 and 0 < t["top1_binom_p"] <= 1

    # shot-based Bloch vectors agree with the exact embedding
    exact = ex.exact_embeddings(runs["full_seed0"], ex.first_test_samples(runs["full_seed0"].config, m), "context")
    got = ex.blochs_from_counts(counts["full_seed0"], m)["context"]
    assert np.abs(got - exact).max() < 0.06


def test_first_test_samples_is_prefix_of_test_set():
    cfg = TrainConfig("full", 0)
    _, test = ex.datasets(cfg)
    assert ex.first_test_samples(cfg, 16) == test[:16]


def test_summary_mean_std_and_markdown():
    def rec(variant, seed, top1):
        by = {"1": top1, "2": top1, "3": top1}
        return {
            "variant": variant, "seed": seed, "top1": top1, "diag_gap": 0.1, "total_var": 1.0,
            "mean_norm": 0.5, "collapsed": False, "probe_mse_pred": 0.1, "probe_mse_unadv": 0.2,
            "predictor_distance": 0.3, "top1_by_s": by, "probe_mse_pred_by_s": by,
            "probe_mse_unadv_by_s": by, "wall_seconds": 1.0,
        }

    evals = {"exact": {"m": 64, "runs": {f"full_seed{k}": rec("full", k, v) for k, v in enumerate([0.2, 0.4, 0.6])}}}
    s = ex.summarise(evals)
    cell = s["backends"]["exact"]["full"]["top1"]
    assert abs(cell["mean"] - 0.4) < 1e-12 and abs(cell["std"] - 0.2) < 1e-12 and cell["n"] == 3
    assert s["backends"]["exact"]["full"]["collapsed_count"] == 0
    md = ex.summary_markdown(s)
    assert "0.400 ± 0.200" in md
