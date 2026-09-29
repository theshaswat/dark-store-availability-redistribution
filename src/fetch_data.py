"""Download the M5 sales panel (Walmart, via the Chronos datasets mirror on Hugging Face).

The file is about 78 MB and is not committed to this repository.
"""
import urllib.request
from pathlib import Path

URL = ("https://huggingface.co/datasets/autogluon/chronos_datasets/resolve/main/"
       "m5/train-00000-of-00001.parquet")
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "m5.parquet"

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        print(f"already present: {OUT}")
    else:
        urllib.request.urlretrieve(URL, OUT)
        print(f"saved {OUT.stat().st_size / 1e6:.1f} MB -> {OUT}")
