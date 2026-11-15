"""Figures from results/: loss curves, similarity matrices, ablation, dynamics probe."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from circuitjepa import experiments as ex  # noqa: E402

# Fixed categorical order: colour follows the variant everywhere.
COLORS = {
    "full": "#2a78d6",
    "no_ema": "#eb6834",
    "no_predictor": "#1baf7a",
    "untrained": "#eda100",
}
LABELS = {
    "full": "full",
    "no_ema": "no EMA",
    "no_predictor": "no predictor",
    "untrained": "untrained",
}
INK = "#333333"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": "#888888",
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e5e5e5",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "savefig.dpi": 160,
    }
)


def _fig_dir() -> Path:
    d = ex.results_dir() / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _backend_label(be: str, data: dict) -> str:
    if be == "exact":
        return f"exact (m={data.get('m')})"
    if be == "qpu":
        return f"QPU {data.get('backend_name', '')} (m={data.get('m')}, {ex.shots_label(data.get('shots'))} shots)"
    return f"{be} (m={data.get('m')}, {data.get('shots')} shots)"


def loss_curves(runs: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for var in ("full", "no_ema", "no_predictor"):
        hs = [r.history for r in runs.values() if r.config.variant == var and r.history]
        if not hs:
            continue
        n = min(len(h) for h in hs)
        H = np.array([h[:n] for h in hs])
        x = np.arange(1, n + 1)
        mu, sd = H.mean(0), H.std(0)
        ax.plot(x, mu, color=COLORS[var], lw=2, label=f"{LABELS[var]} (n={len(hs)} seeds)")
        ax.fill_between(x, mu - sd, mu + sd, color=COLORS[var], alpha=0.18, lw=0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss  1 − mean cos(r_pred, s_target)")
    ax.set_title("Training loss (mean ± std over seeds)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = _fig_dir() / "loss_curves.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# Exploratory QPU runs (not seed 0), shown in an extra row of the similarity figure.
QPU_EXTRA = ("full_seed1", "full_seed2", "no_predictor_seed0", "untrained_seed1")


def _sim_rows(evals: dict) -> list[tuple[str, list[str | None], str]]:
    """(backend, run name per column, row label) for each row of the similarity figure."""
    rows = []
    for be in ex._sort_backends(evals):
        # Exploratory QPU runs go to their own row, not next to the pre-registered pair.
        names = [None if be == "qpu" and f"{v}_seed0" in QPU_EXTRA else f"{v}_seed0" for v in ex.VARIANTS]
        label = _backend_label(be, evals[be])
        shots = evals[be].get("shots")
        if isinstance(shots, dict):  # QPU: shots of the runs actually shown in this row
            sh = sorted({shots[n] for n in names if n and n in shots})
            label = f"QPU {evals[be].get('backend_name', '')} (m={evals[be].get('m')}, {'/'.join(map(str, sh))} shots)"
        rows.append((be, names, label))
    if "qpu" in evals and any(n in evals["qpu"]["runs"] for n in QPU_EXTRA):
        shots = evals["qpu"].get("shots", {})
        names = [n if n in evals["qpu"]["runs"] else None for n in QPU_EXTRA]
        sh = sorted({shots[n] for n in names if n and isinstance(shots, dict)})
        label = f"QPU exploratory (m={evals['qpu'].get('m')}, {'/'.join(map(str, sh))} shots)"
        rows.append(("qpu", names, label))
    return rows


SIM_DPI = 200


def similarity_matrices(evals: dict) -> Path:
    rows_spec = _sim_rows(evals)
    fig, axes = plt.subplots(
        len(rows_spec), len(ex.VARIANTS), figsize=(3.9 * len(ex.VARIANTS), 3.5 * len(rows_spec) + 0.4),
        squeeze=False,
    )
    fig.subplots_adjust(top=1 - 1.7 / fig.get_figheight(), hspace=0.6, wspace=0.12)
    im = None
    for i, (be, names, row_label) in enumerate(rows_spec):
        for j in range(len(ex.VARIANTS)):
            ax = axes[i, j]
            ax.grid(False)
            name = names[j]
            rec = evals[be]["runs"].get(name) if name else None
            if rec is None:
                ax.axis("off")
                continue
            var, seed = name.rsplit("_seed", 1)
            G = np.asarray(rec["G"])
            im = ax.imshow(G, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            ax.set_title(f"{LABELS[var]}" + (f" (seed {seed})" if seed != "0" else "")
                         + f"\ntop-1 {rec['top1']:.2f}, gap {rec['diag_gap']:.2f}", fontsize=17)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(row_label.replace(" (", "\n(", 1) + "\nprediction i", fontsize=16)
            if i == len(rows_spec) - 1:
                ax.set_xlabel("target j", fontsize=16)
    if im is not None:
        cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
        cb.set_label("G[i, j] = cos(r_pred,i , s_target,j)", fontsize=17)
        cb.ax.tick_params(labelsize=15)
    fig.suptitle("Similarity matrices, seed 0 unless noted (diagonal = matching pairs)", fontsize=20)
    path = _fig_dir() / "similarity_matrices.png"
    fig.savefig(path, bbox_inches="tight", dpi=SIM_DPI)
    plt.close(fig)
    return path


def ablation(summary: dict, evals: dict) -> Path:
    bes = ex._sort_backends(summary["backends"])
    metrics = [("top1", "retrieval top-1 accuracy"), ("diag_gap", "diagonal gap"),
               ("total_var", "target total Bloch variance")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.2))
    width = 0.8 / len(ex.VARIANTS)
    x = np.arange(len(bes))
    for ax, (key, title) in zip(axes, metrics):
        for k, var in enumerate(ex.VARIANTS):
            mu = [summary["backends"][be].get(var, {}).get(key, {}).get("mean", np.nan) for be in bes]
            sd = [summary["backends"][be].get(var, {}).get(key, {}).get("std", 0.0) for be in bes]
            ax.bar(x + (k - 1.5) * width, mu, width * 0.92, yerr=sd, color=COLORS[var],
                   label=LABELS[var], capsize=2, error_kw={"lw": 1, "ecolor": INK})
        if key == "top1":
            for i, be in enumerate(bes):
                m = summary["backends"][be]["_meta"].get("m")
                if m:
                    ax.hlines(1 / m, i - 0.45, i + 0.45, colors=INK, linestyles="--", lw=1,
                              label="chance 1/m" if i == 0 else None)
        ax.set_xticks(x, [_backend_label(be, evals[be]).split(" (")[0] for be in bes],
                      rotation=25, ha="right", rotation_mode="anchor")
        ax.set_title(title)
        ax.axhline(0, color="#888888", lw=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Ablation by backend (mean ± std over seeds)")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = _fig_dir() / "ablation.png"
    fig.savefig(path)
    plt.close(fig)
    return path


PROBE_YMAX = 0.2


PROBE_COLS = 3
PROBE_FONT = 16


def dynamics_probe(summary: dict, evals: dict) -> Path:
    """One panel per backend on a grid (PROBE_COLS wide); the spare cell holds the legend."""
    from matplotlib.lines import Line2D

    bes = ex._sort_backends(summary["backends"])
    off_scale: list[str] = []
    n_rows = -(-(len(bes) + 1) // PROBE_COLS)
    fig, axes = plt.subplots(n_rows, PROBE_COLS, figsize=(4.2 * PROBE_COLS, 3.9 * n_rows),
                             squeeze=False, sharey=True)
    flat = axes.ravel()
    for ax, be in zip(flat, bes):
        rows = summary["backends"][be]
        for vi, var in enumerate(ex.VARIANTS):
            if var not in rows:
                continue
            r = rows[var]
            dx = (vi - 1.5) * 0.06  # small horizontal offset so error bars do not overlap
            ss = sorted(int(k.split("[")[1].rstrip("]")) for k in r if k.startswith("probe_mse_pred_by_s["))
            for kind, ls, mk in (("pred", "-", "o"), ("unadv", "--", "s")):
                mu = [r[f"probe_mse_{kind}_by_s[{s}]"]["mean"] for s in ss]
                sd = [r[f"probe_mse_{kind}_by_s[{s}]"]["std"] for s in ss]
                ax.errorbar(np.array(ss) + dx, mu, yerr=sd, color=COLORS[var], ls=ls, marker=mk, ms=6, lw=1.8,
                            capsize=2)
            for k in r:  # mark means above the plotted range
                if k.startswith("probe_mse_") and "_by_s[" in k and r[k]["mean"] > PROBE_YMAX:
                    s = int(k.split("[")[1].rstrip("]"))
                    ax.plot(s + dx, PROBE_YMAX * 0.97, marker="^", ms=9, color=COLORS[var],
                            ls="none", clip_on=False)
                    off_scale.append(f"{be}: {LABELS[var]}, s={s}, {k.split('_by_s')[0][10:]} {r[k]['mean']:.2f}")
        ax.set_ylim(0, PROBE_YMAX)
        ax.set_xticks([1, 2, 3])
        ax.tick_params(labelsize=PROBE_FONT - 1)
        ax.set_xlabel("time step s", fontsize=PROBE_FONT)
        ax.set_title(_backend_label(be, evals[be]).replace(" (", "\n(", 1), fontsize=PROBE_FONT)
    for ax in axes[:, 0]:
        ax.set_ylabel("probe MSE on ⟨Z_i⟩(t₀+s)", fontsize=PROBE_FONT)
    for ax in flat[len(bes):]:
        ax.axis("off")
    # Legend: colour = variant, line style = embedding.
    handles = [Line2D([], [], color=COLORS[v], lw=3, label=LABELS[v]) for v in ex.VARIANTS]
    handles += [Line2D([], [], color=INK, ls="-", marker="o", label="predicted embedding"),
                Line2D([], [], color=INK, ls="--", marker="s", label="un-advanced context")]
    if off_scale:
        handles.append(Line2D([], [], color=INK, ls="none", marker="^", ms=9, label=f"mean above {PROBE_YMAX}"))
    flat[len(bes)].legend(handles=handles, loc="center", fontsize=PROBE_FONT, frameon=False)
    fig.suptitle("Dynamics probe by time step (mean ± std over seeds)", fontsize=PROBE_FONT + 2)
    fig.tight_layout()
    path = _fig_dir() / "dynamics_probe.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def make_all() -> list[Path]:
    runs = ex.load_runs()
    summary = ex.write_summary()
    # Figures show the original test set; held-out evaluations are reported in tables.
    evals = {k: v for k, v in ex.load_evals().items() if not k.endswith(f"_{ex.HELDOUT_TAG}")}
    summary = {**summary, "backends": {k: v for k, v in summary["backends"].items() if k in evals}}
    out = [loss_curves(runs)]
    if evals:
        out += [similarity_matrices(evals), ablation(summary, evals), dynamics_probe(summary, evals)]
    return out
