import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from circuitjepa import cli
from circuitjepa import experiments as ex
from circuitjepa.ansatz import N_ENC_PARAMS, N_PRED_PARAMS
from circuitjepa.hardware import get_backend, run_counts
from circuitjepa.train import RunResult, TrainConfig, load_run, save_run

PY = sys.executable


def _cli(args, cwd, smoke=True):
    env = dict(os.environ)
    if smoke:
        env["CIRCUITJEPA_SMOKE"] = "1"
    # Make sure no IBM account could ever be reached from a test.
    env["QISKIT_IBM_TOKEN"] = ""
    return subprocess.run(
        [PY, "-m", "circuitjepa.cli", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    out = _cli(["train", "--variant", "full", "--seed", "0", "--epochs", "2"], d)
    assert out.returncode == 0, out.stderr
    return d


def test_cli_train_smoke(workdir):
    path = Path(workdir) / "results" / "runs" / "full_seed0.json"
    assert path.exists()
    run = load_run(path)
    assert run.config.variant == "full"
    assert run.config.epochs == 2
    assert run.config.n_train == 16 and run.config.n_test == 8
    assert len(run.history) == 2


def test_cli_hw_submit_dry_run(workdir):
    out = _cli(
        ["hw-submit", "--backend", "fake_torino", "--runs", "full_seed0", "--dry-run"], workdir
    )
    assert out.returncode == 0, out.stderr
    assert "circuits: 432" in out.stdout
    assert "estimated QPU seconds" in out.stdout
    assert "dry run" in out.stdout.lower()
    assert not (Path(workdir) / "results" / "hardware" / "jobs.json").exists()


def test_cli_budget_refusal(workdir):
    out = _cli(
        [
            "hw-submit",
            "--backend",
            "fake_torino",
            "--runs",
            "full_seed0",
            "--shots",
            "100000",
            "--dry-run",
        ],
        workdir,
    )
    assert out.returncode == 2
    assert "--force" in out.stdout + out.stderr
    assert "refus" in (out.stdout + out.stderr).lower()
    assert not (Path(workdir) / "results" / "hardware" / "jobs.json").exists()


@pytest.fixture(scope="module")
def evaluated(workdir):
    out = _cli(["train", "--variant", "untrained", "--seed", "0"], workdir)
    assert out.returncode == 0, out.stderr
    out = _cli(["evaluate", "--backend", "exact"], workdir)
    assert out.returncode == 0, out.stderr
    return workdir


def test_cli_evaluate_exact_smoke(evaluated):
    data = json.loads((Path(evaluated) / "results" / "eval" / "exact.json").read_text())
    full = data["runs"]["full_seed0"]
    assert set(full) >= {"top1", "diag_gap", "collapsed", "predictor_distance", "G"}
    assert full["collapsed"] in (True, False)
    summary = json.loads((Path(evaluated) / "results" / "eval" / "summary.json").read_text())
    assert "exact" in summary["backends"]


def test_cli_evaluate_heldout_keeps_original(evaluated):
    ev = Path(evaluated) / "results" / "eval"
    before = json.loads((ev / "exact.json").read_text())
    out = _cli(["evaluate", "--backend", "exact", "--test-data-seed", "424242", "--tag", "heldout"], evaluated)
    assert out.returncode == 0, out.stderr
    held = json.loads((ev / "exact_heldout.json").read_text())
    after = json.loads((ev / "exact.json").read_text())
    assert after == before  # the original evaluation is not touched
    assert held["test_data_seed"] == 424242 and held["tag"] == "heldout"
    assert "test_data_seed" not in before and "tag" not in before
    assert held["runs"]["full_seed0"]["G"] != before["runs"]["full_seed0"]["G"]
    assert not any(t["pre_registered"] for t in held["tests"].values())
    summary = json.loads((ev / "summary.json").read_text())
    assert {"exact", "exact_heldout"} <= set(summary["backends"])


def test_cli_evaluate_test_seed_needs_tag(tmp_path):
    out = _cli(["evaluate", "--backend", "exact", "--test-data-seed", "424242"], tmp_path)
    assert out.returncode == 2
    assert "--tag" in out.stderr
    assert not (tmp_path / "results").exists()


def test_heldout_samples_differ_from_original():
    from circuitjepa.config import HELDOUT_TEST_DATA_SEED

    cfg = TrainConfig("full", 0)
    train_set, test_set = ex.datasets(cfg)
    _, held = ex.datasets(cfg, HELDOUT_TEST_DATA_SEED)
    assert len(held) == len(test_set) == 64
    assert not {s.a for s in held} & ({s.a for s in test_set} | {s.a for s in train_set})
    assert ex.first_test_samples(cfg, 16, HELDOUT_TEST_DATA_SEED) == held[:16]


def test_sort_backends_puts_heldout_after_its_backend():
    names = ["qpu", "fake_fez", "exact_heldout", "aer_heldout", "aer", "exact"]
    assert ex._sort_backends(names) == ["exact", "exact_heldout", "aer", "aer_heldout", "fake_fez", "qpu"]


def test_cli_figures_smoke(evaluated):
    out = _cli(["figures"], evaluated)
    assert out.returncode == 0, out.stderr
    figs = Path(evaluated) / "results" / "figures"
    for name in ("loss_curves", "similarity_matrices", "ablation", "dynamics_probe"):
        assert (figs / f"{name}.png").stat().st_size > 1000
    assert (Path(evaluated) / "results" / "SUMMARY.md").exists()


# ---------------------------------------------------------------- hw-fetch (no IBM access)

M_HW = 2


def _synthetic_run(variant, seed=0):
    g = np.random.default_rng(len(variant))
    cfg = TrainConfig(variant=variant, seed=seed, n_train=16, n_test=8)
    th = list(g.uniform(0, 6, N_ENC_PARAMS))
    return RunResult(cfg, th, th, list(g.uniform(0, 6, N_PRED_PARAMS)), [], 0.0)


@pytest.fixture
def hw(tmp_path, monkeypatch):
    """Two runs submitted to local Aer; IBM status/fetch replaced by Aer counts."""
    monkeypatch.setenv("CIRCUITJEPA_RESULTS", str(tmp_path / "results"))
    for variant in ("full", "untrained"):
        path = ex.run_path(f"{variant}_seed0")
        path.parent.mkdir(parents=True, exist_ok=True)
        save_run(_synthetic_run(variant), path)
    rc = cli.main(
        ["hw-submit", "--backend", "aer", "--m", str(M_HW), "--shots", "500",
         "--runs", "full_seed0", "untrained_seed0"]
    )
    assert rc == 0
    jobs = tmp_path / "results" / "hardware" / "jobs.json"
    recs = [json.loads(line) for line in jobs.read_text().splitlines()]
    by_job = {r["job_id"]: r["tag"] for r in recs}
    be = get_backend("aer")

    def fake_counts(job_id):
        run = ex.load_runs([by_job[job_id]])[by_job[job_id]]
        circs = ex.run_circuits(run, ex.first_test_samples(run.config, M_HW))
        return run_counts(circs, be, 4000, seed=3)

    monkeypatch.setattr(cli, "_job_status", lambda job_id: "DONE")
    monkeypatch.setattr(cli, "_fetch_counts", fake_counts)
    return tmp_path / "results", jobs, recs


def test_hw_submit_records_submission_and_hash(hw):
    _, _, recs = hw
    assert len(recs) == 2
    assert len({r["submission_id"] for r in recs}) == 1
    for r in recs:
        run = ex.load_runs([r["tag"]])[r["tag"]]
        assert r["params_hash"] == ex.params_hash(run)
        assert r["m"] == M_HW and r["n_circuits"] == 3 * M_HW * 9


def test_hw_fetch_saves_raw_counts_then_evaluates(hw):
    results, _, _ = hw
    assert cli.main(["hw-fetch"]) == 0
    raw = json.loads((results / "hardware" / "counts_full_seed0.json").read_text())
    assert len(raw["circuits"]) == 3 * M_HW * 9
    c = raw["circuits"][9 * M_HW + 9 + 2]  # target kind, sample 1, basis index 2
    assert (c["kind"], c["sample"], c["basis"]) == ("target", 1, "XZ")
    assert sum(c["counts"].values()) == 4000
    assert np.asarray(raw["bloch"]["context"]).shape == (M_HW, 15)
    qpu = json.loads((results / "eval" / "qpu.json").read_text())
    assert qpu["submission_id"] == raw["submission_id"]
    assert qpu["shots"] == 500 and qpu["m"] == M_HW
    assert "full_seed0_vs_untrained_seed0" in qpu["tests"]


def test_hw_fetch_refuses_param_mismatch(hw, capsys):
    results, _, _ = hw
    run = _synthetic_run("full")
    run.theta[0] += 0.1
    save_run(run, ex.run_path("full_seed0"))
    assert cli.main(["hw-fetch"]) != 0
    assert "params_hash mismatch" in capsys.readouterr().out
    assert (results / "hardware" / "counts_full_seed0.json").exists()  # raw data kept
    assert not (results / "eval" / "qpu.json").exists()


def _rewrite(jobs, recs):
    jobs.write_text("".join(json.dumps(r) + chr(10) for r in recs))


@pytest.mark.parametrize("key,value", [("backend", "ibm_other"), ("shots", 999)])
def test_hw_fetch_refuses_mixed_submission(hw, capsys, key, value):
    results, jobs, recs = hw
    recs[1][key] = value
    _rewrite(jobs, recs)
    assert cli.main(["hw-fetch"]) != 0
    assert f"differ in {key}" in capsys.readouterr().out
    assert not (results / "eval" / "qpu.json").exists()


def test_hw_fetch_uses_only_latest_submission(hw, capsys):
    results, jobs, recs = hw
    # an older submission's job for the same tag must not be paired with the new one
    old = dict(recs[0], submission_id="older", job_id="old-job", shots=123)
    _rewrite(jobs, [old] + recs)
    assert cli.main(["hw-fetch"]) == 0
    qpu = json.loads((results / "eval" / "qpu.json").read_text())
    assert qpu["shots"] == 500 and "old-job" not in qpu["job_ids"].values()
    capsys.readouterr()
    assert cli.main(["hw-fetch", "--submission", "older"]) != 0  # no untrained job in it
    assert "incomplete" in capsys.readouterr().out


def test_hw_fetch_refuses_to_overwrite_counts_of_another_job(hw, capsys):
    results, _, _ = hw
    assert cli.main(["hw-fetch"]) == 0
    path = results / "hardware" / "counts_full_seed0.json"
    raw = json.loads(path.read_text())
    raw["job_id"] = "some-other-job"
    path.write_text(json.dumps(raw))
    capsys.readouterr()
    assert cli.main(["hw-fetch"]) != 0
    assert "not overwritten" in capsys.readouterr().out
    assert json.loads(path.read_text())["job_id"] == "some-other-job"
    assert cli.main(["hw-fetch", "--force"]) == 0
    assert json.loads(path.read_text())["job_id"] != "some-other-job"


@pytest.mark.parametrize("status", ["ERROR", "CANCELLED"])
def test_hw_fetch_reports_failed_jobs(hw, capsys, monkeypatch, status):
    import circuitjepa.hardware as hwmod

    monkeypatch.setattr(cli, "_job_status", lambda job_id: status)
    monkeypatch.setattr(hwmod, "job_error", lambda job_id: "boom")
    assert cli.main(["hw-fetch"]) == 6
    out = capsys.readouterr().out
    assert f"{status}: boom" in out and "without results" in out


# ---------------------------------------------------------------- guards and errors


@pytest.mark.parametrize("backend", ["ibm_marrakesh", "ibm_fez", "qpu"])
def test_evaluate_refuses_real_backends(tmp_path, monkeypatch, capsys, backend):
    monkeypatch.setenv("CIRCUITJEPA_RESULTS", str(tmp_path / "results"))

    def boom(*a, **k):
        raise AssertionError("must not open a backend")

    monkeypatch.setattr(ex, "evaluate_shots", boom)
    assert cli.main(["evaluate", "--backend", backend]) == 2
    assert "hw-submit" in capsys.readouterr().err


def test_unknown_run_name_is_a_clean_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CIRCUITJEPA_RESULTS", str(tmp_path / "results"))
    rc = cli.main(["hw-submit", "--backend", "fake_fez", "--runs", "nosuch_seed0", "--dry-run"])
    assert rc == 1
    assert "unknown run(s): nosuch_seed0" in capsys.readouterr().err


def test_train_config_default_is_final_configuration():
    assert TrainConfig("full", 0).delta == 0.5
