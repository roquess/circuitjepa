import numpy as np

from circuitjepa.metrics import (
    diag_gap,
    fit_probe,
    is_collapsed,
    permutation_p,
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


def test_permutation_perfect_diagonal_is_at_floor():
    G = np.eye(16) * 2 - 1  # diagonal 1, off-diagonal -1
    r = permutation_p(G, n_resamples=2000)
    assert r["top1"] == 1.0
    assert r["p_top1"] == r["p_gap"] == 1 / 2001


def test_permutation_noise_is_not_significant():
    rng = np.random.default_rng(4)
    G = rng.uniform(-1, 1, size=(16, 16))
    r = permutation_p(G, n_resamples=2000)
    assert 0.05 < r["p_gap"] < 1
    assert r["p_top1"] > 0.05


def test_permutation_known_partial_retrieval():
    # Rows 0-7 peak on their own target, rows 8-15 on target 0: top-1 = 8/16.
    G = np.full((16, 16), -0.5)
    G[np.arange(8), np.arange(8)] = 1.0
    G[8:, 0] = 1.0
    r = permutation_p(G, n_resamples=2000)
    assert r["top1"] == 0.5
    assert r["p_top1"] < 0.01 and r["p_gap"] < 0.01


def test_permutation_collapsed_rows_share_one_target():
    # Every row peaks on the same column: at most one hit is possible, under H0 as observed.
    G = np.zeros((16, 16))
    G[:, 3] = 1.0
    r = permutation_p(G, n_resamples=2000)
    assert r["top1"] == 1 / 16
    assert r["p_top1"] == 1.0  # every permutation also gives exactly one hit
    assert r["p_gap"] == 1.0  # the gap is identical under every permutation


def test_permutation_paired_identical_matrices_gives_one():
    rng = np.random.default_rng(5)
    G = rng.uniform(-1, 1, size=(12, 12))
    r = permutation_p(G, G.copy(), n_resamples=500)
    assert r["paired"] and r["top1"] == 0 and r["diag_gap"] == 0
    assert r["p_top1"] == r["p_gap"] == 1.0


def test_permutation_paired_model_beats_baseline():
    rng = np.random.default_rng(6)
    base = rng.uniform(-1, 1, size=(16, 16))
    model = base + 3 * np.eye(16)
    r = permutation_p(model, base, n_resamples=2000)
    assert r["p_top1"] == r["p_gap"] == 1 / 2001
