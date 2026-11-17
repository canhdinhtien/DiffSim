"""Smoke tests for the TF1 -> TF2 (tf.compat.v1 + tf-keras) port. Run: uv run pytest src/TimeGAN"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from time_gan import timegan, tf  # noqa: E402
from utils import LayerNormBasicLSTMCell  # noqa: E402


def _ln(x, eps=1e-12):
    return (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + eps)


def test_layernorm_lstm_cell_matches_numpy_reference():
    tf.reset_default_graph()
    rng = np.random.default_rng(0)
    x, c0, h0 = rng.normal(size=(4, 3)), rng.normal(size=(4, 5)), rng.normal(size=(4, 5))
    cell = LayerNormBasicLSTMCell(5)
    out, (c1, h1) = cell(tf.constant(x, tf.float32), tf.nn.rnn_cell.LSTMStateTuple(
        tf.constant(c0, tf.float32), tf.constant(h0, tf.float32)))
    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        out, c1, k = s.run([out, c1, cell._kernel])
    i, j, f, o = np.split(np.concatenate([x, h0], 1) @ k, 4, axis=1)
    sig = lambda z: 1 / (1 + np.exp(-z))
    c_ref = _ln(c0 * sig(_ln(f) + 1.0) + sig(_ln(i)) * np.tanh(_ln(j)))
    np.testing.assert_allclose(c1, c_ref, atol=1e-4)
    np.testing.assert_allclose(out, np.tanh(c_ref) * sig(_ln(o)), atol=1e-4)


@pytest.mark.parametrize("module", ["gru", "lstm", "lstmLN"])
def test_timegan_runs_and_returns_original_shape_and_range(module):
    np.random.seed(0)
    t = np.linspace(0, 6, 12)
    data = [np.stack([np.sin(t + p), 5 + np.cos(t * 2 + p)], 1) for p in np.random.uniform(0, 3, 40)]
    params = {"hidden_dim": 8, "num_layer": 2, "iterations": 3, "batch_size": 16, "module": module}
    gen = np.asarray(timegan(data, params))
    assert gen.shape == (40, 12, 2) and np.isfinite(gen).all()
    lo, hi = np.min(data, (0, 1)), np.max(data, (0, 1))  # recovery is sigmoid -> renormalized into data range
    assert (gen >= lo - 1e-6).all() and (gen <= hi + 1e-6).all()
