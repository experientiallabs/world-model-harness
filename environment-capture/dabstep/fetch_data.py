"""Download the large DABstep context file(s) from the upstream HuggingFace dataset.

The small context files (manual.md, fees.json, the reference CSVs) are committed under
``datafiles/``; the ~23 MB ``payments.csv`` is gitignored and fetched here so a fresh clone is
runnable without checking a large binary into git. Stdlib-only (urllib) — no extra dependencies.

Usage (from the repo root):
    uv run python environment-capture/dabstep/fetch_data.py            # payments.csv only
    uv run python environment-capture/dabstep/fetch_data.py --all      # every context file
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

_HERE = Path(__file__).parent
_BASE_URL = "https://huggingface.co/datasets/adyen/DABstep/resolve/main/data/context"

# The full context set upstream; only payments.csv is gitignored, the rest are committed.
_ALL_FILES = (
    "payments.csv",
    "acquirer_countries.csv",
    "fees.json",
    "manual.md",
    "merchant_category_codes.csv",
    "merchant_data.json",
    "payments-readme.md",
)
_LARGE_FILES = ("payments.csv",)


def _download(file_id: str, dest: Path) -> None:
    url = f"{_BASE_URL}/{file_id}"
    print(f"fetching {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed https HuggingFace host
    print(f"  wrote {dest.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch every context file (default: only the gitignored large payments.csv)",
    )
    args = parser.parse_args()

    datafiles = _HERE / "datafiles"
    datafiles.mkdir(exist_ok=True)
    wanted = _ALL_FILES if args.all else _LARGE_FILES
    for file_id in wanted:
        _download(file_id, datafiles / file_id)
    print(f"done: {len(wanted)} file(s) in {datafiles}")


if __name__ == "__main__":
    main()
