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


def similarity_matrices(evals: dict) -> Path:
    rows_spec = _sim_rows(evals)
    fig, axes = plt.subplots(
        len(rows_spec), len(ex.VARIANTS), figsize=(3.0 * len(ex.VARIANTS), 3.0 * len(rows_spec) + 0.4),
        squeeze=False,
    )
    fig.subplots_adjust(top=1 - 0.9 / fig.get_figheight(), hspace=0.45, wspace=0.15)
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
                         + f"\ntop-1 {rec['top1']:.2f}, gap {rec['diag_gap']:.2f}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(row_label.replace(" (", "\n(", 1) + "\nprediction i", fontsize=8)
            if i == len(rows_spec) - 1:
                ax.set_xlabel("target j", fontsize=9)
    if im is not None:
        cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
        cb.set_label("G[i, j] = cos(r_pred,i , s_target,j)")
    fig.suptitle("Similarity matrices, seed 0 unless noted (diagonal = matching pairs)")
    path = _fig_dir() / "similarity_matrices.png"
    fig.savefig(path, bbox_inches="tight")
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


def dynamics_probe(summary: dict, evals: dict) -> Path:
    bes = ex._sort_backends(summary["backends"])
    off_scale: list[str] = []
    fig, axes = plt.subplots(1, len(bes), figsize=(4.4 * len(bes), 4.0), squeeze=False, sharey=True)
    for ax, be in zip(axes[0], bes):
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
                ax.errorbar(np.array(ss) + dx, mu, yerr=sd, color=COLORS[var], ls=ls, marker=mk, ms=6, lw=1.6,
                            capsize=2, label=f"{LABELS[var]} — {'predicted' if kind == 'pred' else 'un-advanced'}")
        for vi, var in enumerate(ex.VARIANTS):  # mark means above the plotted range
            if var not in rows:
                continue
            for k in rows[var]:
                if k.startswith("probe_mse_") and "_by_s[" in k and rows[var][k]["mean"] > PROBE_YMAX:
                    s = int(k.split("[")[1].rstrip("]"))
                    ax.plot(s + (vi - 1.5) * 0.06, PROBE_YMAX * 0.97, marker="^", ms=7, color=COLORS[var],
                            ls="none", clip_on=False)
                    off_scale.append(f"{be}: {LABELS[var]}, s={s}, {k.split('_by_s')[0][10:]} "
                                     f"{rows[var][k]['mean']:.2f}")
        ax.set_ylim(0, PROBE_YMAX)
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("time step s")
        ax.set_title(_backend_label(be, evals[be]), fontsize=9)
    axes[0, 0].set_ylabel("probe MSE on ⟨Z_i⟩(t₀+s)")
    # Collect entries from every panel: no_predictor has no QPU run, so the last panel lacks it.
    entries = {}
    for ax in axes[0]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            entries.setdefault(lab, h)
    axes[0, -1].legend(list(entries.values()), list(entries.keys()), fontsize=7,
                       loc="upper left", bbox_to_anchor=(1.0, 1.0))
    title = "Dynamics probe: solid = predicted embedding, dashed = un-advanced context"
    if off_scale:
        title += f"\n▲ = mean above {PROBE_YMAX} (off scale; see results/SUMMARY.md)"
    fig.suptitle(title)
    fig.tight_layout()
    path = _fig_dir() / "dynamics_probe.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def make_all() -> list[Path]:
    runs = ex.load_runs()
    evals = ex.load_evals()
    summary = ex.write_summary()
    out = [loss_curves(runs)]
    if evals:
        out += [similarity_matrices(evals), ablation(summary, evals), dynamics_probe(summary, evals)]
    return out
