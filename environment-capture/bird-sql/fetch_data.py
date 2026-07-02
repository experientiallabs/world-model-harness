"""Materialize the real BIRD mini-dev dataset into this benchmark's on-disk shape.

BIRD mini-dev ships as a single zip (databases + questions) on the project's Google Drive; there
is no direct HTTP endpoint for the SQLite databases, so this script does NOT download — it
converts an ALREADY-UNZIPPED MINIDEV directory. Fetch it once (see README) and point
``--minidev-root`` at the unzipped ``.../minidev/MINIDEV`` dir (which holds ``mini_dev_sqlite.json``
and ``dev_databases/<db_id>/<db_id>.sqlite``).

For each selected database the records are seeded-shuffled, capped at ``--per-db``, then split
disjointly into test/train (``--test-frac`` to test, the rest to train) so both splits draw from
every database. Written under this directory:
  - ``data/{train,test}.jsonl`` — agent-visible tasks (question + folded-in evidence hint).
  - ``gold/<task_id>.json`` — ``{"gold_sql": ...}`` sidecars (NEVER staged into the workspace).
  - ``schemas/<db>.sql`` — DDL only, staged as ``schema.sql`` for the agent to read.
  - ``databases/<db>.sqlite`` — a copy of the real db (gitignored; re-materialize with this script).

Usage (from the repo root):
    uv run python environment-capture/bird-sql/fetch_data.py \
        --minidev-root /path/to/minidev/MINIDEV
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sqlite3
from pathlib import Path

_HERE = Path(__file__).parent
_DEFAULT_DATABASES = ("superhero", "toxicology", "student_club", "california_schools")


def _ddl(sqlite_path: Path) -> str:
    """The database's schema (DDL only) — tables, indexes, views, triggers; no data."""
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
        ).fetchall()
    finally:
        con.close()
    return "\n".join(f"{sql};" for (sql,) in rows) + "\n"


def _prompt(record: dict[str, str]) -> str:
    """The question the agent sees, with BIRD's evidence folded in as a hint."""
    prompt = record["question"].strip()
    evidence = record.get("evidence", "").strip()
    if evidence:
        prompt += f"\n\nHint: {evidence}"
    return prompt


def _split_records(
    records: list[dict[str, str]],
    databases: tuple[str, ...],
    *,
    per_db: int,
    test_frac: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Seeded, per-database disjoint test/train partition (every db appears in both)."""
    rng = random.Random(seed)
    test: list[dict[str, str]] = []
    train: list[dict[str, str]] = []
    for db_id in databases:
        db_records = [r for r in records if r["db_id"] == db_id]
        rng.shuffle(db_records)
        db_records = db_records[:per_db]
        n_test = round(len(db_records) * test_frac)
        test.extend(db_records[:n_test])
        train.extend(db_records[n_test:])
    return train, test


def _write_split(
    split: str,
    records: list[dict[str, str]],
    *,
    data_dir: Path,
    gold_dir: Path,
) -> None:
    rows: list[str] = []
    for i, record in enumerate(records):
        task_id = f"bird-{split}-{i}"
        rows.append(
            json.dumps(
                {
                    "task_id": task_id,
                    "prompt": _prompt(record),
                    "data": {"db_name": record["db_id"], "question_id": record["question_id"]},
                }
            )
        )
        (gold_dir / f"{task_id}.json").write_text(
            json.dumps({"gold_sql": record["SQL"]}) + "\n", encoding="utf-8"
        )
    (data_dir / f"{split}.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--minidev-root",
        required=True,
        help="unzipped MINIDEV dir (holds mini_dev_sqlite.json + dev_databases/)",
    )
    parser.add_argument(
        "--databases",
        default=",".join(_DEFAULT_DATABASES),
        help="comma-separated db_ids to include",
    )
    parser.add_argument("--per-db", type=int, default=18, help="max questions per database")
    parser.add_argument("--test-frac", type=float, default=0.3, help="fraction held out as test")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    root = Path(args.minidev_root)
    databases = tuple(d.strip() for d in args.databases.split(",") if d.strip())
    records = json.loads((root / "mini_dev_sqlite.json").read_text(encoding="utf-8"))

    for sub in ("data", "gold", "schemas", "databases"):
        (_HERE / sub).mkdir(exist_ok=True)

    for db_id in databases:
        source_db = root / "dev_databases" / db_id / f"{db_id}.sqlite"
        shutil.copy(source_db, _HERE / "databases" / f"{db_id}.sqlite")
        (_HERE / "schemas" / f"{db_id}.sql").write_text(_ddl(source_db), encoding="utf-8")

    train, test = _split_records(
        records, databases, per_db=args.per_db, test_frac=args.test_frac, seed=args.seed
    )
    _write_split("train", train, data_dir=_HERE / "data", gold_dir=_HERE / "gold")
    _write_split("test", test, data_dir=_HERE / "data", gold_dir=_HERE / "gold")

    print(
        f"materialized {len(databases)} databases, {len(train)} train / {len(test)} test tasks "
        f"-> {_HERE}"
    )


if __name__ == "__main__":
    main()
