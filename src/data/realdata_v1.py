"""Shared S1 (driver x day) dataset for realdata-v1. Reused by all 8 generation methods.

Read-only on data/mock/realdata-v1. Driver ids are replaced by an integer index (sorted order)
and never leave this module.

Protocol (see src/base/README.md):
- features: FEATURES below, raw units; online_time clipped to <= 24 h (spec FIX: 59 source rows exceed 24 h).
- split by driver, stratified by group (vehicle_model x fleet), fixed seed, 70/10/20.
- train windows: stride `train_stride` (default 7); val/test windows: non-overlapping (stride = window).
- min-max scaler fit on train windows only.
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = os.environ.get("REALDATA_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "data", "mock", "realdata-v1"))
CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "derived", "s1-panel-v1.parquet")

FEATURES = ["n_completed", "n_cancelled", "n_offered", "online_time_h", "total_stoppoints", "stoppoints_rush"]
COUNT_FEATURES = [0, 1, 2, 4, 5]  # indices of integer count features
UNITS = {"n_completed": "trips/day", "n_cancelled": "trips/day", "n_offered": "requests/day",
         "online_time_h": "h/day", "total_stoppoints": "stops/day", "stoppoints_rush": "stops/day"}
START_DATE = pd.Timestamp("2026-07-01")
N_DAYS = 123
# assumption: National Day holiday = 09-02 and 09-03 (the two highest-volume days around it, EDA 4.3)
HOLIDAYS = [pd.Timestamp("2026-09-02"), pd.Timestamp("2026-09-03")]


def load_panel(refresh=False):
    """Dense panel: X (n_drivers, 123, 6) raw, groups (n_drivers,) int, group_names list."""
    if refresh or not os.path.exists(CACHE):
        import duckdb
        root = os.path.abspath(ROOT).replace("\\", "/")
        q = f"""
        select dense_rank() over (order by s.driver_id) - 1 as drv, s.local_date,
               s.completed_count as n_completed, s.cancelled_count as n_cancelled,
               s.total_request_calculate_accept as n_offered, least(o.online_time, 24.0) as online_time_h,
               b.total_stoppoints, b.total_stoppoints_rush_hour as stoppoints_rush,
               p.vehicle_model || ' x ' || p.fleet as grp
        from read_parquet('{root}/driver_statistic_daily.parquet') s
        join read_parquet('{root}/driver_online_hours_sap_id.parquet') o using (driver_id, local_date)
        join read_parquet('{root}/driver_bike_stoppoints.parquet') b using (driver_id, local_date)
        join read_parquet('{root}/driver_profile.parquet') p using (driver_id)
        order by drv, local_date"""
        df = duckdb.sql(q).df()
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        df.to_parquet(CACHE, index=False)
    df = pd.read_parquet(CACHE)
    n = df.drv.max() + 1
    assert len(df) == n * N_DAYS, "panel must be dense driver x 123 days"
    X = df[FEATURES].to_numpy(np.float64).reshape(n, N_DAYS, len(FEATURES))
    grp_per_driver = df.grp.to_numpy().reshape(n, N_DAYS)[:, 0]
    names = sorted(set(grp_per_driver))
    groups = np.array([names.index(g) for g in grp_per_driver])
    return X, groups, names


def calendar():
    """Per day index 0..122: day of week (0=Mon) and holiday flag."""
    dates = pd.date_range(START_DATE, periods=N_DAYS)
    return dates.dayofweek.to_numpy(), dates.isin(HOLIDAYS).astype(int)


@dataclass
class Split:
    windows: np.ndarray   # (N, L, F) raw units
    driver: np.ndarray    # (N,) driver index (internal, never printed)
    group: np.ndarray     # (N,) group id
    start: np.ndarray     # (N,) start day index into the 123-day calendar


@dataclass
class S1Data:
    X: np.ndarray            # full panel (n_drivers, 123, F)
    groups: np.ndarray       # (n_drivers,)
    group_names: list
    drivers: dict            # split -> driver indices
    splits: dict             # split -> Split
    x_min: np.ndarray
    x_max: np.ndarray
    window: int

    def normalize(self, x):
        return (x - self.x_min) / (self.x_max - self.x_min)

    def denormalize(self, x):
        return x * (self.x_max - self.x_min) + self.x_min

    def cond(self, split):
        """Conditions a method may use to sample windows matching `split`: (group, start day)."""
        s = self.splits[split]
        return {"group": s.group.copy(), "start": s.start.copy()}


def _windows(X, groups, drv, L, stride):
    starts = np.arange(0, N_DAYS - L + 1, stride)
    w = np.stack([X[d, s:s + L] for d in drv for s in starts])
    return Split(w, np.repeat(drv, len(starts)), np.repeat(groups[drv], len(starts)), np.tile(starts, len(drv)))


def load_s1(window=28, train_stride=7, split_seed=0, fracs=(0.7, 0.1, 0.2)):
    X, groups, names = load_panel()
    rng = np.random.default_rng(split_seed)
    parts = {"train": [], "val": [], "test": []}
    for g in range(len(names)):  # stratify by group so every split has every group
        d = rng.permutation(np.flatnonzero(groups == g))
        a, b = int(round(fracs[0] * len(d))), int(round((fracs[0] + fracs[1]) * len(d)))
        parts["train"] += list(d[:a]); parts["val"] += list(d[a:b]); parts["test"] += list(d[b:])
    drivers = {k: np.sort(v) for k, v in parts.items()}
    assert not (set(drivers["train"]) & set(drivers["val"]) or set(drivers["train"]) & set(drivers["test"])
                or set(drivers["val"]) & set(drivers["test"]))
    splits = {"train": _windows(X, groups, drivers["train"], window, train_stride),
              "val": _windows(X, groups, drivers["val"], window, window),
              "test": _windows(X, groups, drivers["test"], window, window)}
    tw = splits["train"].windows
    return S1Data(X, groups, names, drivers, splits, tw.min((0, 1)), tw.max((0, 1)), window)
