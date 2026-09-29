"""Four figures for the README, drawn from the output tables."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from chartstyle import C, apply, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "output" / "tables"
CHARTS = ROOT / "output" / "charts"

LABEL = {"exp_smoothing": "Exponential smoothing", "moving_avg_28": "28-day moving average",
         "seasonal_naive": "Same weekday last week"}
COLOUR = {"exp_smoothing": C["primary"], "moving_avg_28": C["cat"][3], "seasonal_naive": C["ink_sub"]}


def forecast_error():
    acc = pd.read_csv(TABLES / "forecast_accuracy.csv")
    segs = ["all", "fast", "medium", "slow"]
    names = {"all": "All 400 series", "fast": "Fast movers", "medium": "Medium movers", "slow": "Slow movers"}
    methods = ["seasonal_naive", "moving_avg_28", "exp_smoothing"]
    apply(grid="x")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    h = 0.26
    for i, seg in enumerate(segs):
        for j, m in enumerate(methods):
            v = acc[(acc.segment == seg) & (acc.method == m)].wape.iloc[0] * 100
            y = i + (j - 1) * h
            ax.barh(y, v, height=h * 0.92, color=COLOUR[m])
            ax.text(v + 0.8, y, f"{v:.1f}%", va="center", fontsize=8.5, color=C["ink_mut"])
    ax.set_yticks(range(len(segs)), [names[s] for s in segs])
    ax.invert_yaxis()
    ax.set_xlim(0, 95)
    ax.set_xlabel("Forecast error, WAPE (%) — 8 weekly origins, 7-day horizon")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    all_es = acc[(acc.segment == "all") & (acc.method == "exp_smoothing")].wape.iloc[0]
    all_sn = acc[(acc.segment == "all") & (acc.method == "seasonal_naive")].wape.iloc[0]
    ax.set_title(f"Exponential smoothing is the most accurate in every band,\n"
                 f"{(1 - all_es / all_sn) * 100:.0f}% less error than repeating last week")
    for j, m in enumerate(reversed(methods)):
        ax.text(0.99, 0.80 + j * 0.065, LABEL[m], transform=ax.transAxes, ha="right",
                color=COLOUR[m], fontsize=9, fontweight="bold")
    save(fig, CHARTS / "01_forecast_error.png")
    plt.close(fig)


def service_curve(hl):
    c = pd.read_csv(TABLES / "service_curve.csv")
    apply(grid="both")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for t, col, name in [(False, C["ink_sub"], "Weekly top-up only"), (True, C["primary"], "With a mid-week transfer")]:
        g = c[c.transfers == t].sort_values("z")
        ax.plot(g.days_of_cover, g.fill_rate * 100, color=col, marker="o", ms=4.5,
                linestyle="-" if t else "--")
        last = g.iloc[-1]
        ax.text(last.days_of_cover + 0.06, last.fill_rate * 100 + (0.25 if t else -0.45), name,
                color=col, fontsize=9, fontweight="bold", va="center")
    f = hl["fill_rate_with_transfer"] * 100
    x1, x0 = hl["days_of_cover_with_transfer"], hl["days_of_cover_no_transfer_same_fill"]
    ax.annotate("", xy=(x1, f), xytext=(x0, f),
                arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.2))
    ax.annotate(f"{x0 - x1:.2f} days less stock\nfor the same {f:.1f}% fill", xy=((x0 + x1) / 2, f),
                xytext=(x0 + 0.25, f - 2.3), fontsize=8.5, color=C["ink"],
                arrowprops=dict(arrowstyle="-", color=C["ink_sub"], lw=0.8))
    ax.set_xlabel("Average stock held (days of sales)")
    ax.set_ylabel("Fill rate (% of units demanded that were sold)")
    ax.set_xlim(3.4, 7.3)
    ax.set_title(f"A mid-week transfer reaches the same fill rate with "
                 f"{hl['stock_saved_pct_same_fill'] * 100:.0f}% less stock", pad=26)
    ax.text(0, 1.03, "Each point is one safety-stock setting (z from 0 to 2.05). The y-axis starts at 87%.",
            transform=ax.transAxes, fontsize=9, color=C["ink_mut"])
    ax.set_ylim(87, 99.5)
    save(fig, CHARTS / "02_service_curve.png")
    plt.close(fig)


def breakeven(hl):
    apply(grid="y")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    cost = np.linspace(0, 0.7, 141)
    net = hl["lost_units_avoided"] - cost * hl["units_moved"]
    ax.plot(cost * 100, net, color=C["primary"])
    ax.fill_between(cost * 100, net, 0, where=net >= 0, color=C["pos"], alpha=0.12, lw=0)
    ax.fill_between(cost * 100, net, 0, where=net < 0, color=C["neg"], alpha=0.10, lw=0)
    ax.axhline(0, color=C["ink_sub"], lw=0.8)
    be = hl["breakeven_move_cost_vs_lost_sale"] * 100
    ax.plot([be], [0], "o", color=C["ink"], ms=5)
    ax.annotate(f"Break-even at {be:.0f}%", xy=(be, 0), xytext=(be + 4, 500),
                fontsize=9, arrowprops=dict(arrowstyle="-", color=C["ink_sub"], lw=0.8))
    y25 = hl["lost_units_avoided"] - 0.25 * hl["units_moved"]
    ax.plot([25], [y25], "o", color=C["primary"], ms=5)
    ax.annotate("Cost assumed in the model (25%)", xy=(25, y25), xytext=(4, 150), fontsize=8.5,
                color=C["ink_mut"], arrowprops=dict(arrowstyle="-", color=C["ink_sub"], lw=0.8))
    ax.set_xlabel("Cost of moving one unit, as % of the margin lost on a missed sale")
    ax.set_ylabel("Net value over 8 weeks\n(lost-sale equivalents)")
    ax.set_title(f"Only {hl['sales_saved_per_unit_moved'] * 100:.0f}% of moved units saved a sale, "
                 f"so transfers pay only\nwhile a move costs less than {be:.0f}% of a lost sale")
    save(fig, CHARTS / "03_transfer_breakeven.png")
    plt.close(fig)


def weekly_lost():
    w = pd.read_csv(TABLES / "weekly_lost_units.csv", parse_dates=["week_starting"])
    apply(grid="y")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(w))
    ax.bar(x - 0.2, w.lost_no_transfer, 0.38, color=C["ink_sub"], label="Weekly top-up only")
    ax.bar(x + 0.2, w.lost_with_transfer, 0.38, color=C["primary"], label="With a mid-week transfer")
    ax.set_xticks(x, [d.strftime("%d %b") for d in w.week_starting])
    ax.set_xlabel("Week starting (2016)")
    ax.set_ylabel("Units lost to stock-outs")
    ax.grid(axis="x", visible=False)
    peak = int(w.lost_no_transfer.idxmax())
    pk = w.loc[peak]
    peak_share = 1 - pk.lost_with_transfer / pk.lost_no_transfer
    rest = w.drop(index=peak)
    rest_share = 1 - rest.lost_with_transfer.sum() / rest.lost_no_transfer.sum()
    ax.annotate(f"Transfers recovered {peak_share * 100:.0f}% of lost units here,\n"
                f"against {rest_share * 100:.0f}% in the other seven weeks",
                xy=(peak - 0.2, pk.lost_no_transfer), xytext=(peak - 5.3, pk.lost_no_transfer * 0.80),
                fontsize=8.5, arrowprops=dict(arrowstyle="-", color=C["ink_sub"], lw=0.8))
    ax.legend(loc="upper left", bbox_to_anchor=(0.03, 0.70))
    ax.set_title("Memorial Day week: all four stores sold 5-15% above forecast,\nso there was little spare stock to move")
    save(fig, CHARTS / "04_weekly_lost_units.png")
    plt.close(fig)


if __name__ == "__main__":
    CHARTS.mkdir(parents=True, exist_ok=True)
    hl = json.loads((TABLES / "headline.json").read_text())
    forecast_error()
    service_curve(hl)
    breakeven(hl)
    weekly_lost()
    print("charts ->", CHARTS.relative_to(ROOT))
