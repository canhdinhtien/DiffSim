"""Search (validation) -> select -> final (test, 5 seeds) for the statistical baselines (M2, M3) on S1.

    uv run python -m src.base.run --stage all       # then: uv run python -m src.base.figures

Selection: per method, mean rank of the 4 validation metrics over its trials (lower is better; ties broken by
discriminative score). Test is used only for the final 5-seed evaluation of each selected config.
"""
import argparse
import importlib
import itertools
import json
import os
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml

from src.base.common import constraint_report, memorization_report
from src.data.realdata_v1 import load_s1
from utils.evaluate import METRICS, evaluate_all

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT = os.path.join(ROOT, "results", "base")
CFG = os.path.join(os.path.dirname(__file__), "configs")
TRIALS = os.path.join(OUT, "trials.csv")
SEEDS = [0, 1, 2, 3, 4]
REAL = "REAL_train_windows"  # real-vs-real floor: train-driver windows evaluated against test

_data = None


def configs():
    out = {}
    for f in sorted(os.listdir(CFG)):
        c = yaml.safe_load(open(os.path.join(CFG, f)))
        grid = [dict(zip(c["search"], v)) for v in itertools.product(*c["search"].values())]
        out[c["method"]] = (c["class"], grid)
    return out


def _init(threads):
    import torch
    torch.set_num_threads(threads)


def build(cls_path, params, seed):
    mod, cls = cls_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(mod), cls)
    if "seed" in cls.__init__.__code__.co_varnames:
        params = {**params, "seed": seed}
    return cls(**params)


def trial(method, cls_path, params, split, seed, save=False):
    global _data
    if _data is None:
        _data = load_s1()
    d = _data
    real = d.splits[split].windows
    tracemalloc.start()
    t0 = time.perf_counter()
    if method == REAL:  # equal-size random set of train windows with the same start days as `split`
        tr = d.splits["train"]
        rng = np.random.default_rng(seed)
        fake = np.stack([tr.windows[rng.choice(np.flatnonzero(tr.start == s))] for s in d.splits[split].start])
        model, t1 = None, time.perf_counter()
    else:
        model = build(cls_path, params, seed).fit(d)
        t1 = time.perf_counter()
        fake = model.sample(len(real), d.cond(split), seed=seed)
    t2 = time.perf_counter()
    peak = tracemalloc.get_traced_memory()[1] / 2 ** 20
    tracemalloc.stop()
    row = {"method": method, "config": json.dumps(params, sort_keys=True), "split": split, "seed": seed,
           **evaluate_all(d.normalize(real), d.normalize(fake), seed=seed),
           "fit_s": t1 - t0, "sample_s": t2 - t1, "peak_mem_mb": peak, "n_fake": len(fake)}
    extra = None
    if save:
        os.makedirs(os.path.join(OUT, "samples"), exist_ok=True)
        np.save(os.path.join(OUT, "samples", f"{method}_seed{seed}.npy"), fake.astype(np.float32))
        if seed == 0:
            ratio = d.splits["train"].windows[..., 0] / d.splits["train"].windows[..., 3]
            band = np.quantile(ratio, [0.005, 0.995])
            raw = model.sample(len(real), d.cond(split), seed=seed, postprocess=False) if model else fake
            memo, d_fake, d_real = memorization_report(d.normalize(fake), d.normalize(d.splits["train"].windows),
                                                       d.normalize(real))
            np.savez(os.path.join(OUT, "samples", f"{method}_nn_dist.npz"), fake=d_fake, real=d_real)
            extra = {"constraints_raw_model_output": constraint_report(raw, band),
                     "constraints_after_postprocess": constraint_report(fake, band),
                     "constraints_real_test": constraint_report(real, band),
                     "memorization": memo}
    return row, extra


def run_parallel(jobs, workers, threads):
    os.makedirs(OUT, exist_ok=True)
    extras = {}
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(threads,)) as ex:
        futs = {ex.submit(trial, *j): j for j in jobs}
        for f in as_completed(futs):
            row, extra = f.result()
            pd.DataFrame([row]).to_csv(TRIALS, mode="a", header=not os.path.exists(TRIALS), index=False)
            print(f"{row['method']:22s} {row['split']} s{row['seed']} {row['config']} "
                  + " ".join(f"{m}={row[m]:.4f}" for m in METRICS), flush=True)
            if extra:
                extras[row["method"]] = extra
    return extras


def done(split):
    if not os.path.exists(TRIALS):
        return set()
    t = pd.read_csv(TRIALS)
    t = t[t.split == split]
    return set(zip(t.method, t.config, t.seed))


def search(workers, threads):
    have = done("val")
    jobs = [(m, cls, p, "val", 0) for m, (cls, grid) in configs().items() for p in grid
            if (m, json.dumps(p, sort_keys=True), 0) not in have]
    print(f"{len(jobs)} validation trials to run")
    run_parallel(jobs, workers, threads)


def select():
    t = pd.read_csv(TRIALS)
    v = t[(t.split == "val") & (t.method != REAL)].copy()
    best = {}
    for m, g in v.groupby("method"):
        g = g.assign(mean_rank=g[METRICS].rank().mean(axis=1)).sort_values(["mean_rank", "discriminative"])
        r = g.iloc[0]
        best[m] = {"params": json.loads(r.config), "val_mean_rank_within_method": float(r.mean_rank),
                   "val_metrics": {k: float(r[k]) for k in METRICS}, "n_trials": int(len(g))}
    bv = pd.DataFrame({m: b["val_metrics"] for m, b in best.items()}).T
    rank = bv.rank().mean(axis=1).sort_values()
    res = {"selection_criterion": "per method: mean of ranks of the 4 validation metrics over its trials "
                                  "(ties: discriminative); overall: same over the selected configs on validation",
           "methods": best, "overall_val_mean_rank": rank.to_dict(), "overall_best": rank.index[0]}
    yaml.safe_dump(res, open(os.path.join(OUT, "best_params.yaml"), "w"), sort_keys=False)
    return res


def final(workers, threads):
    best = select()
    cls = {m: c for m, (c, _) in configs().items()}
    jobs = [(m, cls[m], b["params"], "test", s, True) for m, b in best["methods"].items() for s in SEEDS]
    jobs += [(REAL, None, {}, "test", s, True) for s in SEEDS]
    extras = run_parallel(jobs, workers, threads)
    t = pd.read_csv(TRIALS)
    t = t[t.split == "test"]
    res = {"protocol": {"real": "test windows (600 drivers x 4 non-overlapping 28-day windows)", "seeds": SEEDS,
                        "n_samples": int(t.n_fake.iloc[0])}, "methods": {}}
    for m in list(best["methods"]) + [REAL]:
        params = json.dumps(best["methods"][m]["params"], sort_keys=True) if m in best["methods"] else "{}"
        g = t[(t.method == m) & (t.config == params)].drop_duplicates("seed", keep="last").sort_values("seed")
        res["methods"][m] = {
            "params": json.loads(params),
            **{k: {"mean": float(g[k].mean()), "std": float(g[k].std(ddof=1)), "per_seed": g[k].round(6).tolist()}
               for k in METRICS},
            "runtime": {k: float(g[k].mean()) for k in ["fit_s", "sample_s", "peak_mem_mb", "metric_s"]},
            **(extras.get(m) or {})}
    json.dump(res, open(os.path.join(OUT, "final_metrics.json"), "w"), indent=1)
    print(json.dumps({m: {k: round(v[k]["mean"], 4) for k in METRICS} for m, v in res["methods"].items()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["search", "final", "all"], default="all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.stage in ("search", "all"):
        search(a.workers, a.threads)
    if a.stage in ("final", "all"):
        final(a.workers, a.threads)
