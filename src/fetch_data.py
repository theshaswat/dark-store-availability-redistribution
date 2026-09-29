"""Download the M5 sales panel (Walmart, via the Chronos datasets mirror on Hugging Face).

The file is about 78 MB and is not committed to this repository. It is pinned by
SHA-256, so a reader can confirm they hold the same file every figure was built from.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = ("https://huggingface.co/datasets/autogluon/chronos_datasets/resolve/main/"
       "m5/train-00000-of-00001.parquet")
SHA256 = "df93472c09b4cd7fcb5b5a2c863fc0827959d7015ba3a2c2b8f5fcf8bce3eff5"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "m5.parquet"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        urllib.request.urlretrieve(URL, OUT)
    got = sha256(OUT)
    if got != SHA256:
        sys.exit(f"checksum mismatch for {OUT}: expected {SHA256}, got {got}")
    print(f"{OUT.stat().st_size / 1e6:.1f} MB, SHA-256 verified -> {OUT}")
