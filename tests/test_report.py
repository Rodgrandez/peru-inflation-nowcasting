import json

import numpy as np
import pandas as pd

from nowcast import plots, report


def _results():
    rows = []
    for w in (1, 2, 3, 4):
        rows += [{"week": w, "model": "AR", "rmse": 0.2, "mae": 0.15, "rel_rmse": 1.0, "dm_stat": None,
                  "dm_pvalue": None},
                 {"week": w, "model": "MIDAS-Almon", "rmse": 0.18, "mae": 0.14, "rel_rmse": 0.9 - 0.02 * w,
                  "dm_stat": -1.5, "dm_pvalue": 0.041}]
    return {"sample": {"eval_start": "2015-01", "eval_end": "2026-08", "n_months": 140},
            "targets": {"headline": {"by_week": rows}, "core": {"by_week": rows}},
            "best": {"headline": {"week": 4, "model": "MIDAS-Almon", "rel_rmse": 0.82, "dm_pvalue": 0.041,
                                  "vs_ar_exp_rel_rmse": 0.97, "vs_ar_exp_dm_pvalue": 0.64, "worst_month": "2026-03",
                                  "worst_share": 0.36, "rel_rmse_ex_worst": 0.91},
                     "core": {"week": 4, "model": "AR", "rel_rmse": 1.0, "dm_pvalue": None,
                              "vs_ar_exp_rel_rmse": None, "vs_ar_exp_dm_pvalue": None, "worst_month": "2026-03",
                              "worst_share": 0.36, "rel_rmse_ex_worst": 1.0}}}


def test_write_results_roundtrip(tmp_path):
    r = dict(_results(), extra=np.float64(2.5))
    assert json.loads(report.write_results(r, tmp_path / "r.json").read_text())["extra"] == 2.5


def test_update_readme_uses_results(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# T\n<!-- RESULTS:START -->\nold 9.99\n<!-- RESULTS:END -->\ntail\n", encoding="utf-8")
    report.update_readme(readme, _results())
    text = readme.read_text(encoding="utf-8")
    assert "old 9.99" not in text and "0.820" in text and "MIDAS-Almon" in text and "140" in text
    assert text.endswith("tail\n")


def test_plots_create_files(tmp_path):
    t = pd.DataFrame(_results()["targets"]["headline"]["by_week"])
    assert plots.rel_rmse_by_week(t, "headline", tmp_path / "a.png").stat().st_size > 0
    idx = pd.period_range("2015-01", periods=10, freq="M")
    nc = pd.concat([pd.DataFrame({"month": idx, "model": m, "y": np.linspace(0, 1, 10),
                                  "yhat": np.linspace(0, 1, 10) + k})
                    for k, m in enumerate(["AR", "MIDAS-Almon"])])
    assert plots.nowcast_path(nc, ["AR", "MIDAS-Almon"], "headline", tmp_path / "b.png").exists()


def test_pipeline_smoke(tmp_path, monkeypatch):
    from conftest import make_daily, make_monthly

    from nowcast import config, pipeline
    raw = tmp_path / "raw"
    raw.mkdir()
    make_daily().to_parquet(raw / "daily.parquet")
    m = make_monthly()
    m.index = m.index.to_timestamp()
    m.to_parquet(raw / "monthly.parquet")
    for name, value in {"DATA_RAW": raw, "ROOT": tmp_path, "REPORTS": tmp_path / "reports",
                        "FIGURES": tmp_path / "reports/figures", "TABLES": tmp_path / "reports/tables",
                        "DESIGN_START": "2010-03", "EVAL_START": "2016-07"}.items():
        monkeypatch.setattr(config, name, value)
    (tmp_path / "README.md").write_text("<!-- RESULTS:START -->\n<!-- RESULTS:END -->\n", encoding="utf-8")
    pipeline.stage_nowcast()
    pipeline.stage_report()
    res = json.loads((tmp_path / "reports/results.json").read_text())
    assert set(res["targets"]) == {"headline", "core"} and res["sample"]["n_months"] == 6
    assert "Best end-of-month model" in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_rel_rmse_plot_leaves_out_random_walk(tmp_path, monkeypatch):
    # RW is 1.5-2x worse than AR and would flatten the differences that matter; it stays in the tables.
    captured = {}
    monkeypatch.setattr(plots, "_save", lambda fig, path: captured.setdefault("labels", [
        t.get_text() for t in fig.axes[0].get_legend().get_texts()]))
    rows = _results()["targets"]["headline"]["by_week"] + [
        {"week": w, "model": "RW", "rmse": 0.3, "mae": 0.2, "rel_rmse": 1.5, "dm_stat": 3.0, "dm_pvalue": 0.0}
        for w in (1, 2, 3, 4)]
    plots.rel_rmse_by_week(pd.DataFrame(rows), "headline", tmp_path / "a.png")
    assert captured["labels"] == ["AR", "MIDAS-Almon"]


def test_findings_compare_with_ar_exp_and_report_worst_month():
    text = report.results_markdown(_results())
    assert "AR+exp" in text and "0.970" in text and "0.640" in text
    assert "2026-03" in text and "36%" in text and "0.910" in text
    assert "No model beats the AR benchmark" in text
    assert "Best end-of-month model: **AR**" not in text
