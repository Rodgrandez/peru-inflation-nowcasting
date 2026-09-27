from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

DAILY = {"fx": "PD04638PD", "rate": "PD04692MD", "wti": "PD04705XD", "wheat": "PD31887XD",
         "maize": "PD31888XD", "soyoil": "PD31889XD"}
MONTHLY = {"headline": "PN01271PM", "core": "PN01276PM", "expect": "PD12912AM"}
LEVEL_SERIES = frozenset({"rate"})        # interest rate: changes in percentage points, not logs

DAILY_START, DAILY_END = "2002-01-01", "2026-08-31"
MONTHLY_START, MONTHLY_END = "2002-1", "2026-8"
DESIGN_START = "2003-01"                  # inflation-targeting regime
EVAL_START = "2015-01"

TARGETS = ("headline", "core")
WEEKS = (1, 2, 3, 4)
WEEK_CUTOFF = {1: 7, 2: 14, 3: 21, 4: 31}
ALMON_DEGREE = 2
SEED = 42
COMBO_MODELS = ("U-MIDAS", "MIDAS-Almon", "Ridge", "LASSO", "XGBoost")

# Isolated data glitches in the BCRP daily series (e.g. wheat at 0.46 US$/t on 2005-03-17): drop a value that is
# more than OUTLIER_LOG_MAX (in logs) away from the median of the previous OUTLIER_WINDOW observations.
OUTLIER_SERIES = ("fx", "wheat", "maize", "soyoil")
OUTLIER_WINDOW = 10
OUTLIER_LOG_MAX = 0.4
