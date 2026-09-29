"""Checks on the mechanics and the outputs. Run: python3 tests/test_pipeline.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from forecast import wape, bias  # noqa: E402
from redistribute import plan_transfers, sell  # noqa: E402

TABLES = ROOT / "output" / "tables"


def test_transfer_moves_surplus_to_shortage():
    on_hand = np.array([10.0, 0.0, 5.0, 5.0])
    need = np.array([2.0, 6.0, 5.0, 5.0])
    m = plan_transfers(on_hand, need)
    after = on_hand - m.sum(1) + m.sum(0)
    assert m.sum() == 6 and m[0, 1] == 6, m
    assert after.sum() == on_hand.sum(), "a transfer created or destroyed stock"
    assert (m.sum(1) <= on_hand).all(), "a store sent more than it held"
    assert (after >= 0).all()


def test_no_transfer_when_moving_costs_more_than_a_lost_sale():
    m = plan_transfers(np.array([10.0, 0.0]), np.array([2.0, 6.0]), lost_cost=1.0, move_cost=1.5)
    assert m.sum() == 0


def test_no_transfer_leaves_the_sender_short():
    m = plan_transfers(np.array([4.0, 0.0]), np.array([4.0, 6.0]))
    assert m.sum() == 0, "sender would have robbed its own forecast need"


def test_sell():
    oh, sold, lost = sell(np.array([5.0, 2.0]), np.array([3.0, 4.0]))
    assert list(oh) == [2, 0] and list(sold) == [3, 2] and list(lost) == [0, 2]


def test_error_metrics():
    a, f = pd.Series([10.0, 0.0, 5.0]), pd.Series([8.0, 1.0, 5.0])
    assert abs(wape(a, f) - 3 / 15) < 1e-12
    assert abs(bias(a, f) - (-1 / 15)) < 1e-12


def test_panel_shape():
    s = pd.read_parquet(ROOT / "data" / "processed" / "sales.parquet")
    assert s.item.nunique() == 100 and s.store.nunique() == 4
    assert len(s) == 100 * 4 * 730
    assert (s.units >= 0).all()


def test_backtest_never_sees_the_future():
    fc = pd.read_parquet(TABLES / "forecasts.parquet")
    assert fc.groupby(["store", "item"]).ngroups == 400
    assert fc.origin.nunique() == 8 and len(fc) == 400 * 8 * 7
    assert (fc.date >= fc.origin).all()
    assert ((fc.date - fc.origin).dt.days == fc.day - 1).all()
    s = pd.read_parquet(ROOT / "data" / "processed" / "sales.parquet")
    merged = fc.merge(s, on=["store", "item", "date"])
    assert np.allclose(merged.actual, merged.units), "backtest actuals do not match the panel"


def test_service_curve():
    c = pd.read_csv(TABLES / "service_curve.csv")
    for t, g in c.groupby("transfers"):
        g = g.sort_values("z")
        assert g.days_of_cover.is_monotonic_increasing, f"more safety stock gave less cover (transfers={t})"
        assert g.fill_rate.is_monotonic_increasing
    assert (c[~c.transfers].units_moved == 0).all()
    assert c.fill_rate.between(0, 1).all()


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
