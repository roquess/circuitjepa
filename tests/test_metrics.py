import numpy as np

from circuitjepa.metrics import (
    diag_gap,
    fit_probe,
    is_collapsed,
    predictor_distance,
    probe_mse,
    signflip_p,
    similarity_matrix,
    top1,
    top1_binom_p,
)


def test_metrics_identity():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 15))
    G = similarity_matrix(X, X)
    assert top1(G) == 1.0
    assert diag_gap(G) > 0.5


def test_metrics_collapsed():
    X = np.tile(np.arange(1.0, 16.0), (8, 1))
    G = similarity_matrix(X, X)
    assert abs(diag_gap(G)) < 1e-12
    assert is_collapsed(0.0, 0.0, 1.0)
    assert not is_collapsed(0.5, 0.0, 1.0)


def test_probe_recovers_linear_map():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(64, 15))
    W = rng.normal(size=(15, 3))
    b = rng.normal(size=(3,))
    Y = X @ W + b
    P = fit_probe(X, Y)
    assert P.shape == (16, 3)
    assert probe_mse(P, X, Y) < 1e-6


def test_predictor_distance_identity():
    assert predictor_distance(np.zeros(8)) < 1e-12
    rng = np.random.default_rng(2)
    d = predictor_distance(rng.uniform(0, 2 * np.pi, 8))
    assert 0 < d <= 1


def test_signflip_and_binom():
    rng = np.random.default_rng(3)
    d_base = rng.normal(size=16)
    assert signflip_p(d_base + 1, d_base) < 0.01
    assert signflip_p(d_base, d_base) > 0.3
    assert top1_binom_p(16, 16) < 1e-10
