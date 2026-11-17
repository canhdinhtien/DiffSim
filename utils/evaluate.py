"""Single entry point for the 4 benchmark metrics; every method must call this (same code, same settings).
Inputs are min-max normalized arrays (N, L, F) (scaler fit on train, see src/data/realdata_v1.py)."""
import time

import numpy as np
import torch

from utils.catext_fid import Context_FID
from utils.cross_correation import correlational_score
from utils.discriminative_metric import discriminative_score_metrics
from utils.predictive_metric import predictive_score_metrics

METRICS = ["discriminative", "predictive", "context_fid", "correlational"]


def evaluate_all(real, fake, seed=0):
    real, fake = np.asarray(real, np.float32), np.asarray(fake, np.float32)
    assert real.ndim == 3 and fake.shape[1:] == real.shape[1:] and np.isfinite(fake).all()
    t = time.perf_counter()
    out = {"discriminative": discriminative_score_metrics(real, fake, seed=seed),
           "predictive": predictive_score_metrics(real, fake, seed=seed),
           "context_fid": Context_FID(real, fake, seed=seed),
           "correlational": correlational_score(real, fake, seed=seed)}
    out["metric_s"] = time.perf_counter() - t
    return out
