from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {"headline": "Headline inflation (m/m, %)", "core": "Inflation ex. food and energy (m/m, %)"}


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def rel_rmse_by_week(table, target: str, path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, g in table.groupby("model"):
        g = g.sort_values("week")
        ax.plot(g["week"], g["rel_rmse"], marker="o", label=model)
    ax.axhline(1.0, c="grey", ls="--", lw=1)
    ax.set(xlabel="Week of the month (information set)", ylabel="RMSE relative to AR", xticks=[1, 2, 3, 4],
           title=LABELS.get(target, target))
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return _save(fig, path)


def nowcast_path(nowcasts, models: list[str], target: str, path) -> Path:
    w = nowcasts.pivot(index="month", columns="model", values="yhat")
    y = nowcasts.drop_duplicates("month").set_index("month")["y"]
    x = w.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(x, y.loc[w.index], c="black", lw=1.6, label="Actual")
    for m in models:
        ax.plot(x, w[m], lw=1, label=m)
    ax.set(ylabel="%", title=f"{LABELS.get(target, target)}: end-of-month nowcast vs actual")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)
