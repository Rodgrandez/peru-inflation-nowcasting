import numpy as np
import pandas as pd
import pytest

from nowcast.features import (
    almon_cols,
    ar_cols,
    build_design,
    clean_daily,
    feature_cols,
    hf_table,
    weekly_cols,
)


def _daily(values_by_day: dict, month="2020-03", prev_value=1.0, col="wti"):
    prev = pd.bdate_range("2020-02-01", "2020-02-29")
    cur = pd.DatetimeIndex([pd.Timestamp(f"{month}-{d:02d}") for d in values_by_day], name="date")
    s = pd.concat([pd.Series(prev_value, index=prev), pd.Series(list(values_by_day.values()), index=cur)])
    return s.to_frame(col)


def test_hf_uses_only_days_up_to_cutoff():
    d = _daily({2: 1.0, 5: 1.0, 9: 100.0, 20: 100.0})
    t = hf_table(d, week=1, months=pd.period_range("2020-03", "2020-03", freq="M"))
    assert t.loc[pd.Period("2020-03", "M"), "wti_cum"] == pytest.approx(0.0)
    assert t.loc[pd.Period("2020-03", "M"), "wti_w1"] == pytest.approx(0.0)
    assert "wti_w2" not in t.columns


def test_level_series_uses_difference():
    d = _daily({2: 4.0, 3: 4.0}, prev_value=3.0, col="rate")
    t = hf_table(d, week=1, months=pd.period_range("2020-03", "2020-03", freq="M"))
    assert t.iloc[0]["rate_cum"] == pytest.approx(1.0)


def test_week4_block_includes_day_31():
    d = _daily({2: 1.0, 31: np.e})
    t = hf_table(d, week=4, months=pd.period_range("2020-03", "2020-03", freq="M"))
    assert t.iloc[0]["wti_w4"] == pytest.approx(100.0)


def test_almon_a0_is_cumulative_change_from_last_previous_obs():
    d = _daily({2: 1.1, 3: 1.21}, prev_value=1.0)
    t = hf_table(d, week=1, months=pd.period_range("2020-03", "2020-03", freq="M"))
    assert t.iloc[0]["wti_a0"] == pytest.approx(100 * np.log(1.21))
    assert {"wti_a1", "wti_a2"} <= set(t.columns)


def test_empty_block_is_nan_and_design_fills_zero():
    d = _daily({20: 2.0})                           # no data in week 1
    months = pd.period_range("2020-03", "2020-03", freq="M")
    t = hf_table(d, week=1, months=months)
    assert np.isnan(t.iloc[0]["wti_w1"])
    monthly = pd.DataFrame({"headline": 0.1, "core": 0.1, "expect": 2.0},
                           index=pd.period_range("2019-01", "2020-03", freq="M"))
    design = build_design(monthly, t, "headline")
    assert not design.isna().any().any() and design.loc[months[0], "wti_w1"] == 0.0


def test_design_lags_exclude_current_month():
    idx = pd.period_range("2002-01", "2004-12", freq="M")
    monthly = pd.DataFrame({"headline": np.arange(len(idx), dtype=float), "core": 0.0,
                            "expect": np.arange(len(idx)) * 10.0}, index=idx)
    design = build_design(monthly, pd.DataFrame(index=idx), "headline")
    t = pd.Period("2004-06", "M")
    assert design.loc[t, "y_l1"] == monthly.loc[t - 1, "headline"]
    assert design.loc[t, "exp_l1"] == monthly.loc[t - 1, "expect"]
    assert design.loc[t, "y_l12"] == monthly.loc[t - 12, "headline"]
    assert design.index.min() == pd.Period("2003-01", "M")


def test_column_groups():
    idx = pd.period_range("2002-01", "2004-12", freq="M")
    monthly = pd.DataFrame({"headline": 0.1, "core": 0.1, "expect": 2.0}, index=idx)
    hf = pd.DataFrame({"wti_cum": 0.0, "wti_w1": 0.0, "wti_w2": 0.0, "wti_a0": 0.0, "wti_a1": 0.0}, index=idx)
    design = build_design(monthly, hf, "headline")
    assert weekly_cols(design) == ["wti_w1", "wti_w2"] and almon_cols(design) == ["wti_a0", "wti_a1"]
    assert "y" not in feature_cols(design) and set(ar_cols()) <= set(feature_cols(design))


def test_clean_daily_drops_isolated_glitches_only():
    idx = pd.bdate_range("2005-03-01", periods=20, name="date")
    wheat = np.full(20, 146.0)
    wheat[12] = 0.46                                   # BCRP glitch, e.g. 2005-03-17
    wheat[15:] = 150.0                                 # ordinary move is kept
    wti = np.linspace(20, 10, 20)
    wti[5] = -36.98                                    # negative price: log undefined
    wti[10] = 6.0                                      # large genuine oil move: WTI is not filtered
    d = pd.DataFrame({"wheat": wheat, "wti": wti, "rate": np.full(20, 4.0)}, index=idx)
    d.loc[idx[3], "rate"] = 12.0                       # rate is a level series: never filtered
    c = clean_daily(d)
    assert np.isnan(c.loc[idx[12], "wheat"]) and c["wheat"].isna().sum() == 1
    assert np.isnan(c.loc[idx[5], "wti"]) and c.loc[idx[10], "wti"] == 6.0
    assert c["rate"].equals(d["rate"])
