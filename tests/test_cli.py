import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qjepa.train import load_run

PY = sys.executable


def _cli(args, cwd, smoke=True):
    env = dict(os.environ)
    if smoke:
        env["QJEPA_SMOKE"] = "1"
    # Make sure no IBM account could ever be reached from a test.
    env["QISKIT_IBM_TOKEN"] = ""
    return subprocess.run(
        [PY, "-m", "qjepa.cli", *args],
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
    assert "--force" in out.stdout + out.stderr
    assert "refus" in (out.stdout + out.stderr).lower()
    assert not (Path(workdir) / "results" / "hardware" / "jobs.json").exists()


def test_cli_evaluate_exact_smoke(workdir):
    for variant in ("untrained",):
        out = _cli(["train", "--variant", variant, "--seed", "0"], workdir)
        assert out.returncode == 0, out.stderr
    out = _cli(["evaluate", "--backend", "exact"], workdir)
    assert out.returncode == 0, out.stderr
    data = json.loads((Path(workdir) / "results" / "eval" / "exact.json").read_text())
    full = data["runs"]["full_seed0"]
    assert set(full) >= {"top1", "diag_gap", "collapsed", "predictor_distance", "G"}
    assert full["collapsed"] in (True, False)
    summary = json.loads((Path(workdir) / "results" / "eval" / "summary.json").read_text())
    assert "exact" in summary["backends"]


def test_cli_figures_smoke(workdir):
    # relies on the exact evaluation written by test_cli_evaluate_exact_smoke
    out = _cli(["figures"], workdir)
    assert out.returncode == 0, out.stderr
    figs = Path(workdir) / "results" / "figures"
    for name in ("loss_curves", "similarity_matrices", "ablation", "dynamics_probe"):
        assert (figs / f"{name}.png").stat().st_size > 1000
    assert (Path(workdir) / "results" / "SUMMARY.md").exists()
