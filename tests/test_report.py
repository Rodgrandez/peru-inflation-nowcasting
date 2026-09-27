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
            "best": {"headline": {"week": 4, "model": "MIDAS-Almon", "rel_rmse": 0.82, "dm_pvalue": 0.041},
                     "core": {"week": 4, "model": "MIDAS-Almon", "rel_rmse": 0.82, "dm_pvalue": 0.041}}}


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
