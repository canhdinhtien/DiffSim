"""Figures for BASELINE_REPORT.md from results/base (run after `python -m src.base.run`).

    uv run python -m src.base.figures
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.base.common import DOW, days
from src.data.realdata_v1 import FEATURES, UNITS, load_s1
from utils.evaluate import METRICS

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "base")
FIG = os.path.join(OUT, "figures")
METHODS = ["M2_var", "M3_copula_ar"]
LABEL = {"REAL": "Real (test)", "M2_var": "M2 pooled VAR",
         "M3_copula_ar": "M3 copula + AR", "REAL_train_windows": "Real train windows (floor)"}
COLOR = {"REAL": "#3d3d3a", "M2_var": "#eb6834", "M3_copula_ar": "#1baf7a",
         "REAL_train_windows": "#a3a29b"}
METRIC_LABEL = {"discriminative": "Discriminative |0.5 - acc|", "predictive": "Predictive MAE (TSTR)",
                "context_fid": "Context-FID", "correlational": "Correlational score"}
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3de", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984",
                     "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "legend.frameon": False, "lines.linewidth": 1.6,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=130)
    plt.close(fig)


def acf(x, max_lag=14):
    """Pooled ACF over windows (global mean, so between-driver variance counts, as in the EDA)."""
    z = x - x.mean()
    den = (z ** 2).sum()
    return np.array([(z[:, :x.shape[1] - k] * z[:, k:]).sum() / den * x.shape[1] / (x.shape[1] - k)
                     for k in range(max_lag + 1)])


def main():
    os.makedirs(FIG, exist_ok=True)
    d = load_s1()
    real = d.splits["test"].windows
    dd = days(d.splits["test"].start, d.window)
    S = {"REAL": real, **{m: np.load(os.path.join(OUT, "samples", f"{m}_seed0.npy")).astype(float) for m in METHODS}}
    N = {k: d.normalize(v) for k, v in S.items()}
    rng = np.random.default_rng(0)

    # 1. PCA + t-SNE
    k = 800
    idx = {m: rng.choice(len(N[m]), k, replace=False) for m in S}
    flat = {m: N[m][idx[m]].reshape(k, -1) for m in S}
    fig, ax = plt.subplots(2, len(METHODS), figsize=(5 * len(METHODS), 8))
    pca = PCA(2).fit(flat["REAL"])
    for j, m in enumerate(METHODS):
        for row, emb in enumerate(["PCA", "t-SNE"]):
            if emb == "PCA":
                r, f = pca.transform(flat["REAL"]), pca.transform(flat[m])
            else:
                e = TSNE(2, perplexity=40, init="pca", random_state=0).fit_transform(np.vstack([flat["REAL"], flat[m]]))
                r, f = e[:k], e[k:]
            a = ax[row, j]
            a.scatter(*r.T, s=8, c=COLOR["REAL"], alpha=0.45, label=LABEL["REAL"], linewidths=0)
            a.scatter(*f.T, s=8, c=COLOR[m], alpha=0.45, label=LABEL[m], linewidths=0)
            a.set_title(f"{emb}: real vs {LABEL[m]}")
            a.set_xlabel(f"{emb} 1 (normalized window, 168-d)"); a.set_ylabel(f"{emb} 2")
            a.legend(loc="upper right", markerscale=2)
    fig.suptitle("Embedding of 28-day windows, 800 real test vs 800 synthetic (seed 0)")
    save(fig, "1_tsne_pca.png")

    # 2. marginals
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.5))
    for f, a in enumerate(ax.flat):
        hi = np.quantile(real[..., f], 0.999)
        bins = np.arange(-0.5, hi + 1.5) if f != 3 else np.linspace(0, 24, 49)
        for m in S:
            a.hist(S[m][..., f].ravel(), bins=bins, density=True, histtype="step", color=COLOR[m], label=LABEL[m],
                   linewidth=2.2 if m == "REAL" else 1.4)
        a.set_title(FEATURES[f]); a.set_xlabel(UNITS[FEATURES[f]]); a.set_ylabel("density")
    ax[0, 0].legend()
    fig.suptitle("Per-feature marginal density per driver-day, real test vs synthetic")
    save(fig, "2_marginals.png")

    # 3. ACF with 95% bands from resampling windows with replacement
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.5))
    lags = np.arange(15)
    for f, a in enumerate(ax.flat):
        for m in S:
            x = S[m][..., f]
            boot = np.array([acf(x[rng.integers(0, len(x), len(x))]) for _ in range(100)])
            lo, hi = np.percentile(boot, [2.5, 97.5], 0)
            a.fill_between(lags, lo, hi, color=COLOR[m], alpha=0.2, linewidth=0)
            a.plot(lags, acf(x), color=COLOR[m], label=LABEL[m], marker="o", markersize=3)
        a.set_title(FEATURES[f]); a.set_xlabel("lag (days)"); a.set_ylabel("autocorrelation"); a.set_xticks(lags[::7])
    ax[0, 0].legend()
    fig.suptitle("Pooled ACF (lags 0-14 days) with 95% resampling bands over windows")
    save(fig, "3_acf.png")

    # 4. cross-correlation heatmaps
    def corr(x):
        return np.corrcoef(x.reshape(-1, x.shape[-1]), rowvar=False)
    short = ["compl", "canc", "offer", "online", "stops", "rush"]
    fig, ax = plt.subplots(len(METHODS), 3, figsize=(11, 3.4 * len(METHODS)))
    for i, m in enumerate(METHODS):
        mats = [(corr(real), "real", "RdBu_r", (-1, 1)), (corr(S[m]), LABEL[m], "RdBu_r", (-1, 1)),
                (np.abs(corr(real) - corr(S[m])), "|real - synthetic|", "Greys", (0, 0.5))]
        for j, (M, t, cmap, lim) in enumerate(mats):
            a = ax[i, j]
            im = a.imshow(M, cmap=cmap, vmin=lim[0], vmax=lim[1])
            a.grid(False); a.set_xticks(range(6), short, rotation=45); a.set_yticks(range(6), short)
            for (r, c), v in np.ndenumerate(M):
                a.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                       color="white" if abs(v) > (0.6 if j < 2 else 0.3) else "#262624")
            a.set_title(t)
            fig.colorbar(im, ax=a, fraction=0.046, label="Pearson r" if j < 2 else "abs. difference")
    fig.suptitle("Lag-0 cross-feature correlation (driver-days pooled): real, synthetic, absolute difference")
    save(fig, "4_crosscorr.png")

    # 5. trajectories: examples + mean and 10-90% band by day of window
    fig, ax = plt.subplots(2, len(S), figsize=(4 * len(S), 6), sharey="row")
    for j, m in enumerate(S):
        ex = rng.choice(len(S[m]), 3, replace=False)
        for e in ex:
            ax[0, j].plot(S[m][e, :, 0], color=COLOR[m], alpha=0.85)
            ax[1, j].plot(S[m][e, :, 3], color=COLOR[m], alpha=0.85)
        ax[0, j].set_title(f"{LABEL[m]}: 3 random windows")
        ax[0, j].set_ylabel("n_completed (trips/day)"); ax[1, j].set_ylabel("online_time_h (h/day)")
        ax[1, j].set_xlabel("day of window")
    fig.suptitle("Example 28-day trajectories")
    save(fig, "5a_trajectories_examples.png")
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.5))
    t = np.arange(d.window)
    for f, a in enumerate(ax.flat):
        for m in S:
            x = S[m][..., f]
            a.fill_between(t, *np.percentile(x, [10, 90], 0), color=COLOR[m], alpha=0.12, linewidth=0)
            a.plot(t, x.mean(0), color=COLOR[m], label=LABEL[m])
        a.set_title(FEATURES[f]); a.set_xlabel("day of window (all windows start on a Wednesday)")
        a.set_ylabel(UNITS[FEATURES[f]])
    ax[0, 0].legend()
    fig.suptitle("Mean (line) and 10-90% band (shade) by day of window")
    save(fig, "5b_trajectories_mean_band.png")

    # 6. metric bars
    fm = json.load(open(os.path.join(OUT, "final_metrics.json")))["methods"]
    order = METHODS + ["REAL_train_windows"]
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.8))
    for a, k in zip(ax, METRICS):
        mean = [fm[m][k]["mean"] for m in order]; std = [fm[m][k]["std"] for m in order]
        a.bar(range(len(order)), mean, yerr=std, color=[COLOR[m] for m in order], width=0.7, capsize=3,
              edgecolor="#fcfcfb", linewidth=2)
        for i, v in enumerate(mean):
            a.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color="#262624")
        a.set_xticks(range(len(order)), [m.split("_")[0] for m in METHODS] + ["Real\nfloor"]); a.set_title(METRIC_LABEL[k] + " (lower better)")
        a.set_ylabel("score")
    fig.suptitle("Final test metrics, mean +/- std over 5 seeds (Real floor = real train windows vs test)")
    save(fig, "6_metric_bars.png")

    # 7. hyperparameter sweeps (validation)
    tr = pd.read_csv(os.path.join(OUT, "trials.csv"))
    tr = tr[(tr.split == "val") & tr.method.isin(METHODS)].drop_duplicates(["method", "config"], keep="last")
    fig, ax = plt.subplots(len(METHODS), 4, figsize=(14, 3 * len(METHODS)))
    for i, m in enumerate(METHODS):
        g = tr[tr.method == m].copy()
        g["mean_rank"] = g[METRICS].rank().mean(axis=1)
        cfg = pd.DataFrame([json.loads(c) for c in g.config], index=g.index)
        for j, hp in enumerate(cfg.columns[:4]):
            a = ax[i, j]
            vals = sorted(cfg[hp].astype(str).unique())
            for xi, v in enumerate(vals):
                y = g.mean_rank[cfg[hp].astype(str) == v]
                a.scatter(np.full(len(y), xi) + rng.uniform(-0.12, 0.12, len(y)), y, s=14, color=COLOR[m], alpha=0.8)
                a.plot([xi - 0.25, xi + 0.25], [y.median()] * 2, color="#262624", linewidth=2)
            a.set_xticks(range(len(vals)), vals); a.set_xlabel(hp); a.set_ylabel("mean rank of 4 val metrics")
            a.set_title(f"{LABEL[m]}: {hp}")
        for j in range(len(cfg.columns), 4):
            ax[i, j].axis("off")
    fig.suptitle("Validation sweep: each dot one config (selection criterion, lower better; bar = median)")
    save(fig, "7a_sweeps.png")
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, (x, y) in zip(ax, [("discriminative", "context_fid"), ("discriminative", "predictive"),
                              ("context_fid", "correlational")]):
        for m in METHODS:
            g = tr[tr.method == m]
            a.scatter(g[x], g[y], s=16, color=COLOR[m], label=LABEL[m], alpha=0.85)
        a.set_xlabel(METRIC_LABEL[x]); a.set_ylabel(METRIC_LABEL[y]); a.set_xscale("log"); a.set_yscale("log")
    ax[0].legend()
    fig.suptitle("All validation trials, metric vs metric (log scales)")
    save(fig, "7b_sweep_scatter.png")

    # 8. weekly profile
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.5))
    dw = DOW[dd]
    for f, a in enumerate(ax.flat):
        for m in S:
            a.plot(range(7), [S[m][..., f][dw == k].mean() for k in range(7)], color=COLOR[m], label=LABEL[m],
                   marker="o", markersize=4)
        a.set_xticks(range(7), DOW_NAMES); a.set_title(FEATURES[f]); a.set_ylabel(f"mean {UNITS[FEATURES[f]]}")
    ax[0, 0].legend()
    fig.suptitle("Weekly seasonality: mean by day of week")
    save(fig, "8_weekly_profile.png")

    # 9. memorization
    fig, ax = plt.subplots(1, len(METHODS), figsize=(6 * len(METHODS), 3.8), sharey=True)
    for a, m in zip(ax, METHODS):
        z = np.load(os.path.join(OUT, "samples", f"{m}_nn_dist.npz"))
        bins = np.linspace(0, max(z["real"].max(), z["fake"].max()), 50)
        a.hist(z["real"], bins=bins, density=True, histtype="step", color=COLOR["REAL"], linewidth=2,
               label="real test -> train")
        a.hist(z["fake"], bins=bins, density=True, histtype="stepfilled", alpha=0.5, color=COLOR[m],
               label=f"{LABEL[m]} -> train")
        dup = (z["fake"] < 1e-6).mean()  # same tolerance as common.memorization_report
        a.set_title(f"{LABEL[m]}\nexact copies of a train window: {dup:.1%}")
        a.set_xlabel("distance to nearest train window (normalized, L2)"); a.set_ylabel("density")
        a.legend(loc="upper right")
    fig.suptitle("Memorization diagnostic: nearest-train-window distance, synthetic vs real test")
    save(fig, "9_memorization.png")
    print("figures written to", os.path.abspath(FIG))


if __name__ == "__main__":
    main()
