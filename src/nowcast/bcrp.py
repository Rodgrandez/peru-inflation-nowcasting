"""Minimal client for the public BCRP statistics API (estadisticas.bcrp.gob.pe)."""
import json
import urllib.request

import numpy as np
import pandas as pd

API = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api/{codes}/json/{start}/{end}"
MONTHS = {"Ene": 1, "Feb": 2, "Mar": 3, "Abr": 4, "May": 5, "Jun": 6, "Jul": 7, "Ago": 8, "Set": 9, "Sep": 9,
          "Oct": 10, "Nov": 11, "Dic": 12}


def parse_period(name: str) -> pd.Timestamp:
    """'Ene.2002' (monthly) or '03.Set.24' (daily, two-digit year; daily data start in 2000)."""
    parts = name.split(".")
    if len(parts) == 2:
        month, year = parts
        return pd.Timestamp(int(year), MONTHS[month], 1)
    day, month, year = parts
    return pd.Timestamp(2000 + int(year), MONTHS[month], int(day))


def parse_response(payload: dict, names: list[str]) -> pd.DataFrame:
    index = pd.DatetimeIndex([parse_period(p["name"]) for p in payload["periods"]], name="date")
    values = [[np.nan if v in ("n.d.", "") else float(v) for v in p["values"]] for p in payload["periods"]]
    return pd.DataFrame(values, index=index, columns=names)


def _get_json(codes: str, start: str, end: str) -> dict:
    url = API.format(codes=codes, start=start, end=end)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch(series: dict[str, str], start: str, end: str) -> pd.DataFrame:
    """One request per series: multi-series responses come back sorted by code, not in request order."""
    frames = [parse_response(_get_json(code, start, end), [name]) for name, code in series.items()]
    return pd.concat(frames, axis=1).sort_index()
