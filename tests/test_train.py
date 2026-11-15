import numpy as np
import torch

from circuitjepa.train import RunResult, TrainConfig, ema_update, load_run, loss_fn, save_run, train


def test_loss_perfect_and_opposite():
    t = torch.randn(5, 15, dtype=torch.float64)
    assert loss_fn(t, t).item() < 1e-9
    assert abs(loss_fn(-t, t).item() - 2.0) < 1e-9


def test_ema_update():
    out = ema_update(torch.zeros(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64), 0.99)
    assert torch.allclose(out, torch.full((3,), 0.01, dtype=torch.float64))


def test_full_training_reduces_loss():
    r = train(TrainConfig("full", 0, n_train=16, n_test=8, epochs=20, batch_size=8))
    assert r.history[-1] < r.history[0]
    assert len(r.theta) == 30 and len(r.phi) == 8


def test_no_ema_target_gets_gradient():
    kw = dict(seed=0, n_train=8, n_test=8, epochs=1, batch_size=8)
    a = train(TrainConfig("full", **kw))
    b = train(TrainConfig("no_ema", **kw))
    assert not np.allclose(a.theta, b.theta)
    assert b.theta_bar == b.theta


def test_untrained_has_no_history():
    r = train(TrainConfig("untrained", 0, n_train=8, n_test=8))
    assert r.history == []
    assert len(r.theta) == 30


def test_no_predictor_has_empty_phi():
    r = train(TrainConfig("no_predictor", 0, n_train=8, n_test=8, epochs=1, batch_size=8))
    assert r.phi == [] and len(r.history) == 1


def test_save_load_roundtrip(tmp_path):
    r = train(TrainConfig("untrained", 3, n_train=8, n_test=8))
    save_run(r, tmp_path / "r.json")
    r2 = load_run(tmp_path / "r.json")
    assert isinstance(r2, RunResult)
    assert r2 == r
