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
        b = r["best"][target]
        out += ["", (f"Best end-of-month model: **{b['model']}**, relative RMSE **{b['rel_rmse']:.3f}** "
                     f"(Diebold-Mariano p-value vs AR: {_fmt_p(b['dm_pvalue'])})."), ""]
    return "\n".join(out).rstrip()


def update_readme(readme: Path, results: dict) -> None:
    text = Path(readme).read_text(encoding="utf-8")
    block = f"{START}\n{results_markdown(results)}\n{END}"
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.DOTALL)
    Path(readme).write_text(new, encoding="utf-8")
