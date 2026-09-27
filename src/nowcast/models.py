import numpy as np
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from nowcast import config
from nowcast.features import almon_cols, ar_cols, feature_cols, weekly_cols


class RandomWalk:
    def fit(self, X, y):
        return self

    def predict(self, X) -> np.ndarray:
        return X["y_l1"].to_numpy()


class Linear:
    """Any regressor restricted to a column subset (OLS by default)."""

    def __init__(self, cols: list[str], estimator=None):
        self.cols = cols
        self.est = estimator if estimator is not None else LinearRegression()

    def fit(self, X, y) -> "Linear":
        self.est.fit(X[self.cols], y)
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.est.predict(X[self.cols]), dtype=float)


def model_zoo(design) -> dict:
    ar = ar_cols()
    umidas = ar + ["exp_l1"] + weekly_cols(design)
    almon = ar + ["exp_l1"] + almon_cols(design)
    ml = feature_cols(design)
    return {
        "RW": RandomWalk(),
        "AR": Linear(ar),
        "AR+exp": Linear(ar + ["exp_l1"]),
        "U-MIDAS": Linear(umidas),
        "MIDAS-Almon": Linear(almon),
        "Ridge": Linear(ml, make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))),
        # eps=1e-2 keeps the smallest penalty away from the near-OLS region, where coordinate descent stalls
        "LASSO": Linear(ml, make_pipeline(StandardScaler(),
                                          LassoCV(alphas=30, eps=1e-2, cv=TimeSeriesSplit(5), max_iter=50_000,
                                                  random_state=config.SEED))),
        # n_jobs=1: with ~100-300 rows, thread start-up costs more than the trees themselves
        "XGBoost": Linear(ml, XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8,
                                           colsample_bytree=0.8, n_jobs=1, random_state=config.SEED)),
    }
