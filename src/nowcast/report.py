import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
TITLES = {"headline": "Headline inflation", "core": "Inflation excluding food and energy"}


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def write_results(results: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=_default), encoding="utf-8")
    return path


def _fmt_p(p) -> str:
    return "–" if p is None or (isinstance(p, float) and np.isnan(p)) else f"{p:.3f}"


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def _findings(b: dict) -> str:
    if b["model"] == "AR":
        return "No model beats the AR benchmark at the end of the month (week 4)."
    text = (f"Best end-of-month model: **{b['model']}**, relative RMSE **{b['rel_rmse']:.3f}** "
            f"(Diebold-Mariano p-value vs AR: {_fmt_p(b['dm_pvalue'])}).")
    if not _missing(b["vs_ar_exp_rel_rmse"]):
        text += (f" Against **AR+exp** (the AR plus the lagged expectations survey, which every high-frequency model "
                 f"also uses) its relative RMSE is **{b['vs_ar_exp_rel_rmse']:.3f}** "
                 f"(p-value: {_fmt_p(b['vs_ar_exp_dm_pvalue'])}).")
    text += (f" Robustness: {b['worst_month']} accounts for {b['worst_share']:.0%} of the AR's squared errors; "
             f"without it the relative RMSE is {b['rel_rmse_ex_worst']:.3f}.")
    return text


def results_markdown(r: dict) -> str:
    s = r["sample"]
    out = [(f"Out-of-sample nowcasts, expanding window, {s['eval_start']} to {s['eval_end']} "
            f"({s['n_months']} months). RMSE relative to an AR benchmark (below 1 = better than AR)."), ""]
    for target, block in r["targets"].items():
        t = pd.DataFrame(block["by_week"])
        wide = t.pivot(index="model", columns="week", values="rel_rmse").sort_values(4)
        p4 = t[t["week"] == 4].set_index("model")["dm_pvalue"]
        out += [f"**{TITLES.get(target, target)}**", "",
                "| Model | Week 1 | Week 2 | Week 3 | Week 4 | DM p-value (week 4) |", "|---|---|---|---|---|---|"]
        for model, row in wide.iterrows():
            cells = " | ".join(f"{row[w]:.3f}" for w in (1, 2, 3, 4))
            out.append(f"| {model} | {cells} | {_fmt_p(p4.get(model))} |")
        out += ["", _findings(r["best"][target]), ""]
    return "\n".join(out).rstrip()


def update_readme(readme: Path, results: dict) -> None:
    text = Path(readme).read_text(encoding="utf-8")
    block = f"{START}\n{results_markdown(results)}\n{END}"
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.DOTALL)
    Path(readme).write_text(new, encoding="utf-8")
