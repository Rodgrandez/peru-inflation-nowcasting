import re

import numpy as np
import pandas as pd

from nowcast import config

BLOCKS = {1: (1, 7), 2: (8, 14), 3: (15, 21), 4: (22, 31)}


def _change(cur_mean: float, prev_mean: float, level: bool) -> float:
    if np.isnan(cur_mean) or np.isnan(prev_mean):
        return np.nan
    return cur_mean - prev_mean if level else 100 * np.log(cur_mean / prev_mean)


def _mean(s: pd.Series) -> float:
    return float(s.mean()) if len(s) else np.nan


def clean_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Drop non-positive prices and isolated glitches, using only past observations (no look-ahead)."""
    out = daily.copy()
    for name in out.columns:
        if name in config.LEVEL_SERIES:
            continue
        s = out[name].where(out[name] > 0)
        if name in config.OUTLIER_SERIES:
            obs = s.dropna()
            med = obs.rolling(config.OUTLIER_WINDOW, min_periods=3).median().shift(1)
            bad = obs.index[(np.log(obs / med)).abs() > config.OUTLIER_LOG_MAX]
            s.loc[bad] = np.nan
        out[name] = s
    return out


def hf_table(daily: pd.DataFrame, week: int, months: pd.PeriodIndex) -> pd.DataFrame:
    """High-frequency information available at the end of `week` of each month (no look-ahead)."""
    cutoff = config.WEEK_CUTOFF[week]
    rows = {}
    for name in daily.columns:
        s = daily[name].dropna()
        per, day = s.index.to_period("M"), s.index.day
        level = name in config.LEVEL_SERIES
        for month in months:
            prev = s[per == month - 1]
            cur = s[(per == month) & (day <= cutoff)]
            cur_day = cur.index.day
            prev_mean = _mean(prev)
            row = rows.setdefault(month, {})
            row[f"{name}_cum"] = _change(_mean(cur), prev_mean, level)
            for k in range(1, week + 1):
                lo, hi = BLOCKS[k]
                row[f"{name}_w{k}"] = _change(_mean(cur[(cur_day >= lo) & (cur_day <= hi)]), prev_mean, level)
            path = np.r_[prev.iloc[-1] if len(prev) else np.nan, cur.to_numpy(dtype=float)]
            steps = np.diff(path) if level else 100 * np.diff(np.log(path))
            pos = np.arange(1, len(steps) + 1) / 22
            for j in range(config.ALMON_DEGREE + 1):
                row[f"{name}_a{j}"] = float(np.nansum(steps * pos ** j)) if len(steps) else np.nan
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index = pd.PeriodIndex(out.index, freq="M")
    return out.sort_index()


def build_design(monthly: pd.DataFrame, hf: pd.DataFrame, target: str) -> pd.DataFrame:
    y = monthly[target]
    d = pd.DataFrame(index=monthly.index)
    d["y"] = y
    d["y_l1"], d["y_l2"], d["y_l12"] = y.shift(1), y.shift(2), y.shift(12)
    d["exp_l1"] = monthly["expect"].shift(1)
    for m in range(2, 13):
        d[f"m{m}"] = (d.index.month == m).astype(float)
    d = d.join(hf, how="left")
    hf_cols = list(hf.columns)
    if hf_cols:
        d[hf_cols] = d[hf_cols].fillna(0.0)      # no high-frequency news = no change
    d = d[d.index >= pd.Period(config.DESIGN_START, "M")]
    return d.dropna(subset=["y", "y_l1", "y_l2", "y_l12", "exp_l1"])


def ar_cols() -> list[str]:
    return ["y_l1", "y_l2", "y_l12"] + [f"m{m}" for m in range(2, 13)]


def weekly_cols(design: pd.DataFrame) -> list[str]:
    return [c for c in design.columns if re.search(r"_w\d$", c)]


def almon_cols(design: pd.DataFrame) -> list[str]:
    return [c for c in design.columns if re.search(r"_a\d$", c)]


def feature_cols(design: pd.DataFrame) -> list[str]:
    return [c for c in design.columns if c != "y"]
