import numpy as np
import pandas as pd
import pytest

from nowcast.models import Linear, RandomWalk, model_zoo


def _design(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2003-01", periods=n, freq="M")
    d = pd.DataFrame(index=idx)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.1 + 0.6 * y[t - 1] + rng.normal(0, 0.1)
    d["y"] = y
    d["y_l1"] = np.r_[0, y[:-1]]
    d["y_l2"] = np.r_[0, 0, y[:-2]]
    d["y_l12"] = 0.0
    d["exp_l1"] = 2.5
    for m in range(2, 13):
        d[f"m{m}"] = (idx.month == m).astype(float)
    d["wti_cum"] = rng.normal(size=n)
    d["wti_w1"] = rng.normal(size=n)
    d["wti_a0"] = rng.normal(size=n)
    return d.iloc[12:]


def test_random_walk_returns_last_value():
    d = _design()
    assert np.array_equal(RandomWalk().fit(d, d["y"]).predict(d), d["y_l1"].to_numpy())


def test_linear_recovers_ar_coefficient():
    d = _design(n=600)
    m = Linear(["y_l1"]).fit(d, d["y"].to_numpy())
    assert m.est.coef_[0] == pytest.approx(0.6, abs=0.08)


def test_zoo_models_fit_and_predict_finite():
    d = _design()
    X = d.drop(columns="y")
    zoo = model_zoo(d)
    assert set(zoo) == {"RW", "AR", "AR+exp", "U-MIDAS", "MIDAS-Almon", "Ridge", "LASSO", "XGBoost"}
    for name, m in zoo.items():
        p = m.fit(X.iloc[:-1], d["y"].iloc[:-1].to_numpy()).predict(X.iloc[[-1]])
        assert p.shape == (1,) and np.isfinite(p).all(), name


def test_ar_exp_benchmark_adds_only_expectations():
    # Every high-frequency model also sees exp_l1, so the value of daily data is measured against AR+exp.
    zoo = model_zoo(_design())
    assert zoo["AR+exp"].cols == zoo["AR"].cols + ["exp_l1"]
