import sys

import numpy as np
import pandas as pd

from nowcast import config, plots, report
from nowcast.data import download, load
from nowcast.evaluation import dm_table, end_of_month_summary, error_table, expanding_nowcasts
from nowcast.features import build_design, clean_daily, hf_table


def _nowcasts_path():
    return config.DATA_RAW / "nowcasts.parquet"


def stage_data():
    download()


def stage_nowcast():
    daily, monthly = load(config.DATA_RAW)
    daily = clean_daily(daily)
    frames = []
    for week in config.WEEKS:
        hf = hf_table(daily, week, monthly.index)
        for target in config.TARGETS:
            design = build_design(monthly, hf, target)
            nc = expanding_nowcasts(design, eval_start=config.EVAL_START)
            frames.append(nc.assign(target=target, week=week))
            print(f"   {target} week {week}: {nc['month'].nunique()} months", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out["month"] = out["month"].astype(str)
    out.to_parquet(_nowcasts_path(), index=False)


def stage_report():
    nc = pd.read_parquet(_nowcasts_path())
    nc["month"] = pd.PeriodIndex(nc["month"], freq="M")
    config.TABLES.mkdir(parents=True, exist_ok=True)
    targets, best = {}, {}
    for target in config.TARGETS:
        rows = []
        for week in config.WEEKS:
            sub = nc[(nc["target"] == target) & (nc["week"] == week)]
            t = error_table(sub).merge(dm_table(sub), on="model", how="left").assign(week=week)
            rows.append(t)
        tab = pd.concat(rows, ignore_index=True)
        tab.to_csv(config.TABLES / f"errors_{target}.csv", index=False)
        tab = tab.astype(object).where(pd.notna(tab), None)
        targets[target] = {"by_week": tab[["week", "model", "rmse", "mae", "rel_rmse", "dm_stat", "dm_pvalue"]]
                           .to_dict("records")}
        sub4 = nc[(nc["target"] == target) & (nc["week"] == 4)]
        summary = end_of_month_summary(sub4)
        best[target] = {"week": 4, **{k: None if isinstance(v, float) and np.isnan(v) else v
                                      for k, v in summary.items()}}
        plots.rel_rmse_by_week(pd.DataFrame(targets[target]["by_week"]), target,
                               config.FIGURES / f"rel_rmse_{target}.png")
        models = ["AR"] + ([best[target]["model"]] if best[target]["model"] != "AR" else [])
        plots.nowcast_path(sub4, models, target, config.FIGURES / f"nowcast_{target}.png")
    months = nc["month"].unique()
    results = {"sample": {"eval_start": str(min(months)), "eval_end": str(max(months)),
                          "n_months": len(months)},
               "targets": targets, "best": best}
    report.write_results(results, config.REPORTS / "results.json")
    report.update_readme(config.ROOT / "README.md", results)


STAGES = {"data": [stage_data], "nowcast": [stage_nowcast], "report": [stage_report],
          "all": [stage_data, stage_nowcast, stage_report]}

if __name__ == "__main__":
    for fn in STAGES[sys.argv[1] if len(sys.argv) > 1 else "all"]:
        print(f"== {fn.__name__}", flush=True)
        fn()
