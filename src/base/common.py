"""Shared pieces of the statistical baselines (M2, M3): interface, post-processing, constraint and memorization checks."""
import numpy as np

from src.data.realdata_v1 import COUNT_FEATURES, FEATURES, N_DAYS, calendar

I_COMP, I_CANC, I_OFF, I_ONLINE, I_STOP, I_RUSH = range(len(FEATURES))
DOW, HOL = calendar()


class Generator:
    """fit(data: S1Data) on TRAIN drivers only; sample(n, cond, seed) -> raw-unit windows (n, L, F).
    cond = {"group": (n,), "start": (n,)} (vehicle_model x fleet id, start day index); None draws it from train."""

    def fit(self, data):
        self.L = data.window
        tr = data.splits["train"]
        self._train_cond = (tr.group, tr.start)
        self.X = data.X[data.drivers["train"]]
        self.groups = data.groups[data.drivers["train"]]
        self.n_groups = len(data.group_names)
        self._fit()
        return self

    def sample(self, n, cond=None, seed=0, postprocess=True):
        rng = np.random.default_rng(seed)
        if cond is None:
            i = rng.integers(0, len(self._train_cond[0]), n)
            cond = {"group": self._train_cond[0][i], "start": self._train_cond[1][i]}
        assert len(cond["group"]) == n
        x = self._sample(np.asarray(cond["group"]), np.asarray(cond["start"]), rng)
        return postprocess_counts(x) if postprocess else x

    def _fit(self):
        raise NotImplementedError

    def _sample(self, group, start, rng):
        raise NotImplementedError


def days(start, L):
    """(n, L) calendar day indices of each window; wraps past day 122 onto the weekly cycle."""
    d = np.asarray(start)[:, None] + np.arange(L)[None]
    return np.where(d < N_DAYS, d, d - 7 * ((d - N_DAYS) // 7 + 1))


def postprocess_counts(x):
    """Domain post-processing: counts >= 0 and integer, online_time in [0, 24] h, rush <= total stops,
    offered >= completed + cancelled (holds on 100% of real driver-days)."""
    x = x.copy()
    x[..., COUNT_FEATURES] = np.round(np.clip(x[..., COUNT_FEATURES], 0, None))
    x[..., I_ONLINE] = np.clip(x[..., I_ONLINE], 0, 24)
    x[..., I_RUSH] = np.minimum(x[..., I_RUSH], x[..., I_STOP])
    x[..., I_OFF] = np.maximum(x[..., I_OFF], x[..., I_COMP] + x[..., I_CANC])
    return x


def constraint_report(x, ratio_band):
    """Share of synthetic driver-days satisfying each domain constraint. ratio_band = real-train [p0.5, p99.5]
    of n_completed / online_time_h."""
    c = x[..., COUNT_FEATURES]
    ratio = x[..., I_COMP] / np.maximum(x[..., I_ONLINE], 1e-6)
    return {
        "online_le_24h": float((x[..., I_ONLINE] <= 24).mean()),
        "counts_nonneg": float((c >= 0).all(-1).mean()),
        "counts_integer": float((np.abs(c - np.round(c)) < 1e-9).all(-1).mean()),
        "rush_le_total_stops": float((x[..., I_RUSH] <= x[..., I_STOP]).mean()),
        "offered_ge_completed_plus_cancelled": float((x[..., I_OFF] >= x[..., I_COMP] + x[..., I_CANC]).mean()),
        "completed_per_online_h_in_real_band": float(((ratio >= ratio_band[0]) & (ratio <= ratio_band[1])).mean()),
        "corr_completed_online": float(np.corrcoef(x[..., I_COMP].ravel(), x[..., I_ONLINE].ravel())[0, 1]),
        "mean_completed_per_online_h": float(np.median(ratio)),
    }


def nn_dist(a, b, chunk=512):
    """Euclidean distance of each row of a (n, d) to its nearest row of b (m, d)."""
    import torch
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    return np.concatenate([torch.cdist(a[i:i + chunk], b, compute_mode="donot_use_mm_for_euclid_dist").min(1).values.numpy() for i in range(0, len(a), chunk)])


def memorization_report(fake_norm, train_norm, test_norm):
    """Nearest-TRAIN-window distance of synthetic windows vs of real test windows (normalized, flattened).
    Exact copy: distance < 1e-6 (float32 round-off; real test windows sit ~1.0 from train)."""
    f, tr, te = (x.reshape(len(x), -1) for x in (fake_norm, train_norm, test_norm))
    d_fake, d_real = nn_dist(f, tr), nn_dist(te, tr)
    return {"nn_train_dist_fake_median": float(np.median(d_fake)),
            "nn_train_dist_real_test_median": float(np.median(d_real)),
            "exact_duplicate_rate": float((d_fake < 1e-6).mean()),
            "share_fake_closer_than_real_p5": float((d_fake < np.quantile(d_real, 0.05)).mean())}, d_fake, d_real
