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
