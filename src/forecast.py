"""Rolling-origin backtest of three daily forecasts for every store-item series.

Eight weekly origins cover the last 56 days (25 Apr to 19 Jun 2016). At each origin a
method sees only the days before it and forecasts the next seven, so every score is on
weeks the method has not seen.

  seasonal_naive  same weekday last week
  moving_avg_28   flat mean of the last 28 days
  exp_smoothing   weekday profile from the last 52 weeks, plus an exponentially
                  smoothed level; the smoothing weight is chosen per series from a
                  grid by one-step-ahead error on the training window

Outputs: output/tables/forecasts.parquet (one row per series, origin and day)
         output/tables/forecast_accuracy.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SALES = ROOT / "data" / "processed" / "sales.parquet"
TABLES = ROOT / "output" / "tables"

H = 7
N_ORIGINS = 8
FIT_DAYS = 364
METHODS = ["seasonal_naive", "moving_avg_28", "exp_smoothing"]
ALPHAS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])


def exp_smoothing_forecast(train: np.ndarray) -> tuple[np.ndarray, float]:
    """Seven-day forecast and the standard deviation of one-step-ahead errors.

    The 364-day window is exactly 52 weeks, so position p and forecast day h share a
    weekday when p % 7 == h % 7.
    """
    x = train[-FIT_DAYS:]
    season = x.reshape(-1, 7).mean(0) - x.mean()
    d = x - np.tile(season, FIT_DAYS // 7)
    level = np.full(len(ALPHAS), d[:28].mean())
    sse = np.zeros(len(ALPHAS))
    errs = np.zeros((len(ALPHAS), FIT_DAYS - 28))
    for t in range(28, FIT_DAYS):
        e = d[t] - level
        errs[:, t - 28] = e
        sse += e * e
        level = level + ALPHAS * e
    best = int(np.argmin(sse))
    fc = np.clip(level[best] + season[:H], 0, None)
    return fc, float(np.std(errs[best], ddof=1))


def backtest_series(args) -> list[dict]:
    store, item, dates, y = args
    rows = []
    for k in range(N_ORIGINS, 0, -1):
        o = len(y) - k * H                      # index of the first forecast day
        train, actual = y[:o], y[o:o + H]
        f_es, sigma = exp_smoothing_forecast(train)
        f = {
            "seasonal_naive": train[-7:],
            "moving_avg_28": np.full(H, train[-28:].mean()),
            "exp_smoothing": f_es,
        }
        for h in range(H):
            rows.append({"store": store, "item": item, "origin": dates[o], "date": dates[o + h],
                         "day": h + 1, "actual": actual[h], "sigma": sigma,
                         **{m: float(f[m][h]) for m in METHODS}})
    return rows


def wape(actual: pd.Series, fc: pd.Series) -> float:
    return float((fc - actual).abs().sum() / actual.sum())


def bias(actual: pd.Series, fc: pd.Series) -> float:
    return float((fc - actual).sum() / actual.sum())


def accuracy(fc: pd.DataFrame) -> pd.DataFrame:
    vol = fc.groupby(["store", "item"]).actual.mean()
    band = pd.qcut(vol, 3, labels=["slow", "medium", "fast"]).rename("velocity")
    fc = fc.join(band, on=["store", "item"])
    out = []
    for seg, g in [("all", fc)] + [(v, fc[fc.velocity == v]) for v in ["fast", "medium", "slow"]]:
        for m in METHODS:
            out.append({"segment": seg, "method": m, "wape": wape(g.actual, g[m]),
                        "bias": bias(g.actual, g[m]), "series": g.groupby(["store", "item"]).ngroups})
    return pd.DataFrame(out)


def main() -> pd.DataFrame:
    s = pd.read_parquet(SALES)
    jobs = [(st, it, g.date.to_numpy(), g.units.to_numpy())
            for (it, st), g in s.groupby(["item", "store"], sort=True)]
    rows = [r for job in jobs for r in backtest_series(job)]
    fc = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(TABLES / "forecasts.parquet", index=False)
    acc = accuracy(fc)
    acc.to_csv(TABLES / "forecast_accuracy.csv", index=False, float_format="%.4f")
    print(acc.pivot(index="segment", columns="method", values="wape").round(3))
    return acc


if __name__ == "__main__":
    main()
