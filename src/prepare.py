"""Cut the M5 panel down to the network this project models.

Four California stores (CA_1 to CA_4) stand in for four dark stores in one city.
The 100 FOODS_3 items with the highest unit sales across those stores over the
final 365 days, kept only if every store has at least two years of history for
the item, make up the assortment.

Output: data/processed/sales.parquet, one row per store, item and day.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "m5.parquet"
OUT = ROOT / "data" / "processed" / "sales.parquet"

STORES = ["CA_1", "CA_2", "CA_3", "CA_4"]
N_ITEMS = 100
HISTORY_DAYS = 730


def main() -> pd.DataFrame:
    t = pq.read_table(RAW, filters=[("state_id", "=", "CA"), ("dept_id", "=", "FOODS_3")]).to_pandas()
    # series are trimmed before their first sale, but every one ends on the same day
    t = t[t.target.map(len) >= HISTORY_DAYS].copy()
    end = pd.Timestamp(t.timestamp.iloc[0][-1])
    assert all(pd.Timestamp(a[-1]) == end for a in t.timestamp), "series do not share an end date"

    t["units"] = t.target.map(lambda a: np.asarray(a[-HISTORY_DAYS:], dtype=float))
    t["last365"] = t.units.map(lambda a: a[-365:].sum())

    stores_per_item = t.groupby("item_id").store_id.nunique()
    full = stores_per_item[stores_per_item == len(STORES)].index
    top = (t[t.item_id.isin(full)].groupby("item_id").last365.sum()
           .sort_values(ascending=False).head(N_ITEMS).index)

    dates = pd.date_range(end=end, periods=HISTORY_DAYS, freq="D")
    rows = t[t.item_id.isin(top)]
    long = pd.DataFrame({
        "store": np.repeat(rows.store_id.values, HISTORY_DAYS),
        "item": np.repeat(("FOODS_3_" + rows.item_id).values, HISTORY_DAYS),
        "date": np.tile(dates, len(rows)),
        "units": np.concatenate(rows.units.values),
    }).sort_values(["item", "store", "date"], ignore_index=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(OUT, index=False)
    print(f"{long['item'].nunique()} items x {long.store.nunique()} stores x {HISTORY_DAYS} days "
          f"({dates[0].date()} to {dates[-1].date()}), {int(long.units.sum()):,} units -> {OUT.relative_to(ROOT)}")
    return long


if __name__ == "__main__":
    main()
