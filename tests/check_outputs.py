"""Compare freshly rebuilt tables with the committed copies.

Run after the pipeline has rebuilt output/tables into a scratch directory:
    python3 tests/check_outputs.py <committed_dir> <rebuilt_dir>

Counts must match exactly. Rates and averages may differ by 1e-4, the precision the
tables are written at, because floating-point sums can round differently on
another machine.
"""
import json
import sys
from pathlib import Path

import pandas as pd

CSV = ["forecast_accuracy.csv", "service_curve.csv", "transfer_lanes.csv", "weekly_lost_units.csv"]


def main(committed: Path, rebuilt: Path) -> None:
    for name in CSV:
        a, b = pd.read_csv(committed / name), pd.read_csv(rebuilt / name)
        assert list(a.columns) == list(b.columns) and len(a) == len(b), f"{name}: shape changed"
        for col in a.columns:
            if pd.api.types.is_numeric_dtype(a[col]) and not pd.api.types.is_bool_dtype(a[col]):
                diff = (a[col] - b[col]).abs().max()
                assert diff <= 1e-4 + 1e-12, f"{name}.{col}: differs by {diff}"
            else:
                assert (a[col].astype(str) == b[col].astype(str)).all(), f"{name}.{col}: values changed"
        print("ok", name)
    ha = json.loads((committed / "headline.json").read_text())
    hb = json.loads((rebuilt / "headline.json").read_text())
    assert ha.keys() == hb.keys()
    for k in ha:
        assert abs(ha[k] - hb[k]) <= 1e-4 + 1e-12, f"headline {k}: {ha[k]} vs {hb[k]}"
    print("ok headline.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
