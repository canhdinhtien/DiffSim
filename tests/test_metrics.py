"""Sanity checks for the shared metrics on S1: real vs real is near zero, real vs noise is clearly worse."""
import numpy as np
import pytest

from src.data.realdata_v1 import load_s1
from utils.evaluate import evaluate_all


@pytest.fixture(scope="module")
def scores():
    d = load_s1()
    test = d.normalize(d.splits["test"].windows)
    drv = d.splits["test"].driver
    a = np.isin(drv, d.drivers["test"][::2])  # two disjoint driver halves of the test set
    ra, rb = test[a], test[~a]
    noise = np.random.default_rng(0).uniform(0, 1, rb.shape)
    return evaluate_all(ra, rb), evaluate_all(ra, noise), evaluate_all(ra, ra.mean((0, 1)) + 0 * rb)


def test_real_vs_real_near_zero(scores):
    rr, _, _ = scores
    assert rr["discriminative"] < 0.08
    assert rr["context_fid"] < 0.5
    assert rr["correlational"] < 0.05


def test_noise_and_constant_clearly_bad(scores):
    rr, noise, const = scores
    assert noise["discriminative"] > 0.4
    assert noise["predictive"] > rr["predictive"] * 1.5
    assert noise["context_fid"] > 10 * max(rr["context_fid"], 0.05)
    assert noise["correlational"] > 5 * rr["correlational"]
    # a collapsed constant sequence fools the small GRU discriminator and TSTR far less than noise
    # (measured 0.106 and 1.37x); Context-FID is the metric that flags it clearly
    assert const["discriminative"] > 3 * rr["discriminative"]
    assert const["predictive"] > rr["predictive"] * 1.2
    assert const["context_fid"] > 10 * max(rr["context_fid"], 0.05)
