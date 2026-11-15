"""Evaluation metrics for QJEPA embeddings."""

import numpy as np
from qiskit.quantum_info import Operator
from scipy import stats

from qjepa.ansatz import PHI, predictor
from qjepa.paulis import cosine


def similarity_matrix(R: np.ndarray, S: np.ndarray) -> np.ndarray:
    return cosine(R[:, None, :], S[None, :, :])


def top1(G: np.ndarray) -> float:
    return float(np.mean(np.argmax(G, axis=1) == np.arange(G.shape[0])))


def row_gaps(G: np.ndarray) -> np.ndarray:
    m = G.shape[0]
    diag = np.diag(G)
    off = (G.sum(axis=1) - diag) / (m - 1)
    return diag - off


def diag_gap(G: np.ndarray) -> float:
    return float(np.mean(row_gaps(G)))


def total_variance(S: np.ndarray) -> float:
    return float(np.sum(np.var(S, axis=0)))


def is_collapsed(gap: float, total_var: float, baseline_total_var: float) -> bool:
    return bool(gap < 0.05 and total_var < 0.05 * baseline_total_var)


def _design(X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((X.shape[0], 1))])


def fit_probe(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Ridge least squares with an unpenalised bias row; returns (d+1, k)."""
    A = _design(X)
    reg = ridge * np.eye(A.shape[1])
    reg[-1, -1] = 0.0
    return np.linalg.solve(A.T @ A + reg, A.T @ Y)


def probe_mse(W: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
    return float(np.mean((_design(X) @ W - Y) ** 2))


def predictor_distance(phi: np.ndarray) -> float:
    """1 - |Tr U|/4 for the predictor unitary on the two latent qubits.

    The 5-qubit predictor acts as I (x) U with U on qubits 0, 1, so
    Tr(full) = 2**3 * Tr(U) and |Tr U|/4 = |Tr full| / 2**5.
    """
    bound = predictor().assign_parameters({PHI[k]: float(phi[k]) for k in range(len(PHI))})
    full = Operator(bound).data
    return float(1 - abs(np.trace(full)) / full.shape[0])


def signflip_p(
    d_model: np.ndarray, d_base: np.ndarray, n_resamples: int = 10000, seed: int = 0
) -> float:
    """Paired one-sided sign-flip test that model > base (mean difference)."""
    d = np.asarray(d_model, float) - np.asarray(d_base, float)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, d.size))
    null = (signs * d).mean(axis=1)
    obs = d.mean()
    return float((1 + np.sum(null >= obs - 1e-12)) / (n_resamples + 1))


def top1_binom_p(k: int, m: int) -> float:
    return float(stats.binomtest(k, m, 1 / m, alternative="greater").pvalue)



def evaluate_embeddings(
    R_pred, S_tgt, U_unadv, S_train, Z_train, Z_test, steps_test, baseline_total_var
) -> dict:
    steps_test = np.asarray(steps_test)
    G = similarity_matrix(R_pred, S_tgt)
    gap = diag_gap(G)
    tv = total_variance(S_tgt)
    W = fit_probe(S_train, Z_train)
    hit = np.argmax(G, axis=1) == np.arange(G.shape[0])
    err_pred = np.mean((_design(R_pred) @ W - Z_test) ** 2, axis=1)
    err_unadv = np.mean((_design(U_unadv) @ W - Z_test) ** 2, axis=1)
    by = lambda v: {int(s): float(np.mean(v[steps_test == s])) for s in np.unique(steps_test)}
    return {
        "top1": top1(G),
        "diag_gap": gap,
        "total_var": tv,
        "mean_norm": float(np.mean(np.linalg.norm(R_pred, axis=1))),
        "collapsed": None if baseline_total_var is None else is_collapsed(gap, tv, baseline_total_var),
        "top1_by_s": by(hit.astype(float)),
        "probe_mse_pred": float(np.mean(err_pred)),
        "probe_mse_unadv": float(np.mean(err_unadv)),
        "probe_mse_pred_by_s": by(err_pred),
        "probe_mse_unadv_by_s": by(err_unadv),
    }
