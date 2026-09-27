import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from nowcast import config
from nowcast.models import model_zoo


def expanding_nowcasts(design: pd.DataFrame, zoo_factory=model_zoo,
                       eval_start: str = config.EVAL_START) -> pd.DataFrame:
    x_cols = [c for c in design.columns if c != "y"]
    rows = []
    for month in design.index[design.index >= pd.Period(eval_start, "M")]:
        train, test = design[design.index < month], design.loc[[month]]
        preds = {}
        for name, model in zoo_factory(design).items():
            model.fit(train[x_cols], train["y"].to_numpy())
            preds[name] = float(model.predict(test[x_cols])[0])
        preds["Combination"] = float(np.mean([preds[k] for k in config.COMBO_MODELS]))
        y = float(test["y"].iloc[0])
        rows += [{"month": month, "model": k, "y": y, "yhat": v} for k, v in preds.items()]
    return pd.DataFrame(rows)


def error_table(nowcasts: pd.DataFrame, benchmark: str = "AR") -> pd.DataFrame:
    err = nowcasts.assign(e=nowcasts["yhat"] - nowcasts["y"])
    t = err.groupby("model")["e"].agg(rmse=lambda s: float(np.sqrt(np.mean(s ** 2))),
                                      mae=lambda s: float(np.mean(np.abs(s)))).reset_index()
    t["rel_rmse"] = t["rmse"] / t.loc[t["model"] == benchmark, "rmse"].item()
    return t.sort_values("rel_rmse").reset_index(drop=True)


def diebold_mariano(e_model, e_bench, h: int = 1) -> tuple[float, float]:
    """DM test on squared-error loss with the Harvey-Leybourne-Newbold small-sample correction."""
    d = np.asarray(e_model, dtype=float) ** 2 - np.asarray(e_bench, dtype=float) ** 2
    n = len(d)
    var = float(np.mean((d - d.mean()) ** 2))
    if n < 3 or var <= 0:
        return float("nan"), float("nan")
    stat = d.mean() / np.sqrt(var / n)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat), float(2 * student_t.sf(abs(stat), df=n - 1))


def dm_table(nowcasts: pd.DataFrame, benchmark: str = "AR") -> pd.DataFrame:
    err = nowcasts.assign(e=nowcasts["yhat"] - nowcasts["y"]).pivot(index="month", columns="model", values="e")
    rows = []
    for model in [m for m in err.columns if m != benchmark]:
        stat, p = diebold_mariano(err[model], err[benchmark])
        rows.append({"model": model, "dm_stat": stat, "dm_pvalue": p})
    return pd.DataFrame(rows)
