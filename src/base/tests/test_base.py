import numpy as np
import pytest

from src.base.common import I_COMP, I_ONLINE, I_RUSH, I_STOP, days, postprocess_counts
from src.base.copula_ar import GaussianCopulaAR
from src.base.var_sarima import PooledVAR
from src.data.realdata_v1 import COUNT_FEATURES, FEATURES, load_s1


@pytest.fixture(scope="module")
def data():
    return load_s1()


MODELS = [lambda: PooledVAR(2, True, 0.5, "mean_scale"), lambda: PooledVAR(1, False, 0.0, "off"),
          lambda: GaussianCopulaAR(7, "parametric", 0.5, "randomized"), lambda: GaussianCopulaAR(1, "empirical")]


@pytest.mark.parametrize("make", MODELS)
def test_sample_shape_range_determinism(data, make):
    m = make().fit(data)
    cond = data.cond("val")
    n = len(cond["group"])
    x = m.sample(n, cond, seed=3)
    assert x.shape == (n, data.window, len(FEATURES)) and not np.isnan(x).any()
    c = x[..., COUNT_FEATURES]
    assert (c >= 0).all() and (c == np.round(c)).all()
    assert (x[..., I_ONLINE] <= 24).all() and (x[..., I_RUSH] <= x[..., I_STOP]).all()
    np.testing.assert_array_equal(x, m.sample(n, cond, seed=3))
    assert not np.array_equal(x, m.sample(n, cond, seed=4))
    assert m.sample(5).shape == (5, data.window, len(FEATURES))  # unconditional sampling
    # normalized values stay in a sane range around [0, 1] after the train-fitted min-max scaler
    z = data.normalize(x)
    assert z.min() > -0.5 and z.max() < 3


def test_split_is_by_driver_and_scaler_train_only(data):
    s = data.splits
    assert not set(s["train"].driver) & set(s["test"].driver)
    assert np.allclose(data.x_min, s["train"].windows.min((0, 1)))
    assert s["val"].windows.shape[1:] == (28, 6)


def test_postprocess_and_calendar():
    x = np.array([[[3.4, -1, 1, 30, 2, 5]]])
    y = postprocess_counts(x)
    assert y[0, 0].tolist() == [3, 0, 3, 24, 2, 2]
    assert days(np.array([110]), 28).max() < 123


def test_var_seasonality_recovers_weekly_profile(data):
    m = PooledVAR(1, True, 0.0, "mean_scale").fit(data)
    tr = data.splits["train"]
    x = m.sample(len(tr.group), {"group": tr.group, "start": tr.start}, seed=0)
    real_dow = np.array([tr.windows[:, k::7, I_COMP].mean() for k in range(7)])
    syn_dow = np.array([x[:, k::7, I_COMP].mean() for k in range(7)])
    assert np.corrcoef(real_dow, syn_dow)[0, 1] > 0.8
