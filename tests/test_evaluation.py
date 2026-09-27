from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from nowcast.evaluation import diebold_mariano, dm_table, error_table, expanding_nowcasts


class Spy:
    seen: ClassVar[list] = []

    def fit(self, X, y):
        Spy.seen.append(X.index.max())
        self.mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean)


def _design():
    idx = pd.period_range("2010-01", "2016-12", freq="M")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"y": rng.normal(size=len(idx)), "y_l1": rng.normal(size=len(idx))}, index=idx)


def _factory(design):
    return {name: Spy() for name in ["AR", "U-MIDAS", "MIDAS-Almon", "Ridge", "LASSO", "XGBoost"]}


def test_expanding_never_trains_on_target_month():
    Spy.seen = []
    nc = expanding_nowcasts(_design(), _factory, eval_start="2015-01")
    months = sorted(nc["month"].unique())
    assert months[0] == pd.Period("2015-01", "M") and len(months) == 24
    first_fits = Spy.seen[:6]
    assert all(m == pd.Period("2014-12", "M") for m in first_fits)      # trained strictly before 2015-01


def test_combination_is_mean_of_combo_models():
    nc = expanding_nowcasts(_design(), _factory, eval_start="2016-06")
    w = nc.pivot(index="month", columns="model", values="yhat")
    combo = w[["U-MIDAS", "MIDAS-Almon", "Ridge", "LASSO", "XGBoost"]].mean(axis=1)
    assert np.allclose(w["Combination"], combo)


def test_error_table_relative_to_ar():
    nc = pd.DataFrame({"month": [1, 2, 1, 2], "model": ["AR", "AR", "X", "X"], "y": [0, 0, 0, 0],
                       "yhat": [1.0, -1.0, 0.5, -0.5]})
    t = error_table(nc).set_index("model")
    assert t.loc["AR", "rel_rmse"] == 1.0 and t.loc["X", "rel_rmse"] == pytest.approx(0.5)


def test_dm_identical_errors_is_nan():
    e = np.array([0.1, -0.2, 0.3])
    stat, p = diebold_mariano(e, e)
    assert np.isnan(stat) and np.isnan(p)


def test_dm_detects_clearly_better_model():
    rng = np.random.default_rng(1)
    bench = rng.normal(0, 1.0, 150)
    better = rng.normal(0, 0.5, 150)
    stat, p = diebold_mariano(better, bench)
    assert stat < 0 and p < 0.01


def test_dm_table_excludes_benchmark():
    rng = np.random.default_rng(2)
    nc = pd.DataFrame({"month": list(range(50)) * 2, "model": ["AR"] * 50 + ["X"] * 50, "y": 0.0,
                       "yhat": np.r_[rng.normal(0, 1, 50), rng.normal(0, 1, 50)]})
    assert dm_table(nc)["model"].tolist() == ["X"]
