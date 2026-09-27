import numpy as np
import pandas as pd
from conftest import SAMPLE_PAYLOAD

from nowcast import bcrp, config


def test_parse_period_monthly():
    assert bcrp.parse_period("Ene.2002") == pd.Timestamp(2002, 1, 1)
    assert bcrp.parse_period("Dic.2025") == pd.Timestamp(2025, 12, 1)


def test_parse_period_daily_set_and_sep():
    assert bcrp.parse_period("03.Set.24") == pd.Timestamp(2024, 9, 3)
    assert bcrp.parse_period("15.Sep.24") == pd.Timestamp(2024, 9, 15)
    assert bcrp.parse_period("04.Ene.00") == pd.Timestamp(2000, 1, 4)


def test_parse_response_nd_is_nan():
    df = bcrp.parse_response(SAMPLE_PAYLOAD, ["fx", "rate"])
    assert list(df.columns) == ["fx", "rate"] and df.index.name == "date"
    assert np.isnan(df.loc["2024-08-30", "rate"]) and df.loc["2024-09-02", "rate"] == 5.25


def test_config_codes():
    assert config.MONTHLY == {"headline": "PN01271PM", "core": "PN01276PM", "expect": "PD12912AM"}
    assert set(config.DAILY) == {"fx", "rate", "wti", "wheat", "maize", "soyoil"}
    assert config.WEEK_CUTOFF == {1: 7, 2: 14, 3: 21, 4: 31}


def test_download_and_load_roundtrip(tmp_path, monkeypatch):
    from conftest import make_daily, make_monthly

    from nowcast import data
    daily, monthly = make_daily(), make_monthly()
    monthly_ts = monthly.copy()
    monthly_ts.index = monthly_ts.index.to_timestamp()
    calls = []

    def fake_fetch(series, start, end):
        calls.append(tuple(series))
        return daily if "fx" in series else monthly_ts

    monkeypatch.setattr(data, "fetch", fake_fetch)
    data.download(tmp_path)
    d, m = data.load(tmp_path)
    assert calls == [tuple(config.DAILY), tuple(config.MONTHLY)]
    assert isinstance(m.index, pd.PeriodIndex) and m.index.freqstr == "M"
    assert d.shape == daily.shape and list(m.columns) == ["headline", "core", "expect"]


def test_fetch_maps_each_code_to_its_own_name(monkeypatch):
    # The API returns multi-series requests sorted by code, not in request order, so names must never
    # be assigned by position across codes.
    values = {"PN01271PM": "0.10", "PN01276PM": "0.20", "PD12912AM": "2.50"}
    requested = []

    def fake_get(codes, start, end):
        requested.append(codes)
        return {"config": {"series": [{"name": codes}]}, "periods": [{"name": "Ene.2020", "values": [values[codes]]}]}

    monkeypatch.setattr(bcrp, "_get_json", fake_get)
    df = bcrp.fetch({"headline": "PN01271PM", "core": "PN01276PM", "expect": "PD12912AM"}, "2020-1", "2020-1")
    assert sorted(requested) == sorted(values)
    assert df.loc["2020-01-01"].to_dict() == {"headline": 0.10, "core": 0.20, "expect": 2.50}
