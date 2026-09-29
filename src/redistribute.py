"""Replay the eight backtest weeks with and without a mid-week stock transfer.

Replenishment: each Monday-equivalent origin, every store orders up to
    S = smoothing forecast for the week + z * sigma * sqrt(7)
where sigma is the standard deviation of the one-step-ahead smoothing errors. Stock left over carries
into the next week. Delivery is assumed to arrive before the first sale.

Transfer: after day 3 each item is re-planned across the four stores with a small linear
program. It moves units from stores holding more than their forecast need for days 4-7
to stores holding less, and only when a lost sale costs more than moving a unit.

Sales are what customers actually bought in M5. They are treated as demand, which they
are not quite: a day with zero sales may have been a day with zero stock.

Outputs: output/tables/service_curve.csv, output/tables/transfer_lanes.csv,
         output/tables/weekly_lost_units.csv
"""
import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "output" / "tables"

REVIEW_DAY = 3                                        # transfer decision after this many days
Z_GRID = [0.0, 0.5, 0.84, 1.28, 1.65, 2.05]           # safety-stock multipliers
Z_BASE = 1.28
# Illustrative economics, not taken from any company: a lost sale costs 1.0 and moving
# one unit between stores costs 0.25 of that (driver time, handling, the van run).
LOST_SALE_COST = 1.0
TRANSFER_COST = 0.25


def plan_transfers(on_hand: np.ndarray, need: np.ndarray,
                   lost_cost: float = LOST_SALE_COST, move_cost: float = TRANSFER_COST) -> np.ndarray:
    """Units to move, as an n x n matrix [from, to]. Whole units, never more than a store holds.

    min  lost_cost * sum(short_i) + move_cost * sum(x_ij)
    s.t. short_i >= need_i - (on_hand_i - out_i + in_i),  out_i <= on_hand_i,  x, short >= 0
    """
    n = len(on_hand)
    arcs = list(permutations(range(n), 2))
    na = len(arcs)
    c = np.r_[np.full(na, move_cost), np.full(n, lost_cost)]
    a_ub, b_ub = [], []
    for i in range(n):
        row = np.zeros(na + n)
        for k, (f, t) in enumerate(arcs):
            if f == i:
                row[k] = -1.0                      # sending lowers what i holds
            if t == i:
                row[k] = 1.0
        row[na + i] = 1.0
        a_ub.append(-row)                          # -(in - out + short) <= on_hand - need
        b_ub.append(on_hand[i] - need[i])
        cap = np.zeros(na + n)
        for k, (f, _) in enumerate(arcs):
            if f == i:
                cap[k] = 1.0
        a_ub.append(cap)
        b_ub.append(on_hand[i])
    res = linprog(c, A_ub=np.array(a_ub), b_ub=np.array(b_ub), bounds=(0, None), method="highs")
    assert res.status == 0, res.message
    moves = np.zeros((n, n))
    for k, (f, t) in enumerate(arcs):
        moves[f, t] = np.floor(res.x[k] + 1e-9)
    return moves


def sell(on_hand: np.ndarray, demand: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One day of sales across stores: returns stock after, units sold, units lost."""
    sold = np.minimum(on_hand, demand)
    return on_hand - sold, sold, demand - sold


def replay(fc: pd.DataFrame, z: float, transfers: bool) -> dict:
    """Run all items through all weeks. fc must be sorted by item, origin, store, day."""
    stores = sorted(fc.store.unique())
    n = len(stores)
    tot = {"demand": 0.0, "sold": 0.0, "lost": 0.0, "stock_days": 0.0, "moved": 0.0, "lanes": 0}
    lanes = np.zeros((n, n))
    weekly = {}
    for _, g_item in fc.groupby("item", sort=True):
        carry = np.zeros(n)
        for origin, g in g_item.groupby("origin", sort=True):
            f = g.pivot(index="day", columns="store", values="exp_smoothing")[stores].to_numpy().T      # n x 7
            d = g.pivot(index="day", columns="store", values="actual")[stores].to_numpy().T
            sig = g.groupby("store").sigma.first()[stores].to_numpy()
            s_level = f.sum(1) + z * sig * np.sqrt(7)
            oh = np.maximum(np.ceil(s_level), carry)
            lost_week = 0.0
            for day in range(7):
                if transfers and day == REVIEW_DAY:
                    rest = 7 - REVIEW_DAY
                    need = f[:, REVIEW_DAY:].sum(1) + z * sig * np.sqrt(rest)
                    m = plan_transfers(oh, need)
                    oh = oh - m.sum(1) + m.sum(0)
                    lanes += m
                    tot["moved"] += m.sum()
                    tot["lanes"] += int((m > 0).sum())
                oh, sold, lost = sell(oh, d[:, day])
                tot["demand"] += d[:, day].sum()
                tot["sold"] += sold.sum()
                tot["lost"] += lost.sum()
                lost_week += lost.sum()
                tot["stock_days"] += oh.sum()          # closing stock, summed over days
            weekly[origin] = weekly.get(origin, 0.0) + lost_week
            carry = oh
    days = fc.groupby(["origin", "day"]).ngroups
    return {"z": z, "transfers": transfers,
            "fill_rate": tot["sold"] / tot["demand"],
            "lost_units": tot["lost"],
            "avg_stock_units": tot["stock_days"] / days,
            "days_of_cover": tot["stock_days"] / tot["demand"],
            "units_moved": tot["moved"], "transfer_lanes_used": tot["lanes"],
            "_lanes": pd.DataFrame(lanes, index=stores, columns=stores),
            "_weekly": weekly}


def main() -> pd.DataFrame:
    fc = pd.read_parquet(TABLES / "forecasts.parquet").sort_values(["item", "origin", "store", "day"])
    runs = [replay(fc, z, t) for z in Z_GRID for t in (False, True)]
    curve = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in runs])
    curve.to_csv(TABLES / "service_curve.csv", index=False, float_format="%.4f")

    base = {r["transfers"]: r for r in runs if r["z"] == Z_BASE}
    lanes = base[True]["_lanes"].stack().rename("units").rename_axis(["from_store", "to_store"]).reset_index()
    lanes[lanes.units > 0].sort_values("units", ascending=False).to_csv(TABLES / "transfer_lanes.csv", index=False)
    weekly = pd.DataFrame({"week_starting": list(base[False]["_weekly"]),
                           "lost_no_transfer": list(base[False]["_weekly"].values()),
                           "lost_with_transfer": list(base[True]["_weekly"].values())})
    weekly.to_csv(TABLES / "weekly_lost_units.csv", index=False)
    headline = summarise(curve, base)
    (TABLES / "headline.json").write_text(json.dumps(headline, indent=2))
    print(curve.round(4).to_string(index=False))
    print(json.dumps(headline, indent=2))
    return curve


def summarise(curve: pd.DataFrame, base: dict) -> dict:
    """Headline numbers at the base safety-stock setting.

    The planner moves a unit whenever it expects that unit to save a sale, and that choice
    is the same for any transfer cost below a lost sale's cost. So the value of transfers
    is linear in the cost of moving a unit, and it breaks even where
        cost per unit moved = lost sales avoided / units moved.
    """
    off, on = base[False], base[True]
    avoided = off["lost_units"] - on["lost_units"]
    no_t = curve[~curve.transfers].sort_values("fill_rate")
    # stock the no-transfer policy needs to match the with-transfer fill rate (linear interpolation)
    cover_needed = float(np.interp(on["fill_rate"], no_t.fill_rate, no_t.days_of_cover))
    return {
        "z": Z_BASE,
        "fill_rate_no_transfer": round(off["fill_rate"], 4),
        "fill_rate_with_transfer": round(on["fill_rate"], 4),
        "lost_units_no_transfer": int(off["lost_units"]),
        "lost_units_with_transfer": int(on["lost_units"]),
        "lost_units_avoided": int(avoided),
        "lost_units_avoided_pct": round(avoided / off["lost_units"], 4),
        "units_moved": int(on["units_moved"]),
        "item_lane_weeks": int(on["transfer_lanes_used"]),
        "avg_units_per_move": round(on["units_moved"] / on["transfer_lanes_used"], 2),
        "sales_saved_per_unit_moved": round(avoided / on["units_moved"], 4),
        "breakeven_move_cost_vs_lost_sale": round(avoided / on["units_moved"], 4),
        "days_of_cover_with_transfer": round(on["days_of_cover"], 3),
        "days_of_cover_no_transfer_same_fill": round(cover_needed, 3),
        "stock_saved_pct_same_fill": round(1 - on["days_of_cover"] / cover_needed, 4),
    }


if __name__ == "__main__":
    main()
