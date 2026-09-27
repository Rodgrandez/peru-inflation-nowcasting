from pathlib import Path

import pandas as pd

from nowcast import config
from nowcast.bcrp import fetch


def download(dest: Path = config.DATA_RAW) -> tuple[Path, Path]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    daily = fetch(config.DAILY, config.DAILY_START, config.DAILY_END)
    monthly = fetch(config.MONTHLY, config.MONTHLY_START, config.MONTHLY_END)
    daily.to_parquet(dest / "daily.parquet")
    monthly.to_parquet(dest / "monthly.parquet")
    return dest / "daily.parquet", dest / "monthly.parquet"


def load(dest: Path = config.DATA_RAW) -> tuple[pd.DataFrame, pd.DataFrame]:
    dest = Path(dest)
    daily = pd.read_parquet(dest / "daily.parquet")
    monthly = pd.read_parquet(dest / "monthly.parquet")
    monthly.index = pd.DatetimeIndex(monthly.index).to_period("M")
    return daily, monthly
