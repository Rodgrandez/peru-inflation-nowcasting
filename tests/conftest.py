import numpy as np
import pandas as pd

SAMPLE_PAYLOAD = {
    "config": {"title": "x", "series": [{"name": "A", "dec": "2"}, {"name": "B", "dec": "2"}]},
    "periods": [{"name": "30.Ago.24", "values": ["3.74", "n.d."]},
                {"name": "02.Set.24", "values": ["3.75", "5.25"]}],
}


def make_daily(start="2010-01-01", end="2016-12-31", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end, name="date")
    walk = lambda s: np.exp(np.cumsum(rng.normal(0, s, len(idx))))  # noqa: E731
    return pd.DataFrame({"fx": 3 * walk(0.003), "rate": 4 + np.cumsum(rng.normal(0, 0.01, len(idx))),
                         "wti": 60 * walk(0.02), "wheat": 500 * walk(0.015), "maize": 400 * walk(0.015),
                         "soyoil": 40 * walk(0.015)}, index=idx)


def make_monthly(start="2009-01", end="2016-12", seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start, end, freq="M")
    head = 0.2 + 0.1 * np.sin(np.arange(len(idx)) * 2 * np.pi / 12) + rng.normal(0, 0.15, len(idx))
    core = 0.2 + rng.normal(0, 0.08, len(idx))
    return pd.DataFrame({"headline": head, "core": core, "expect": 2.5 + rng.normal(0, 0.2, len(idx))}, index=idx)
