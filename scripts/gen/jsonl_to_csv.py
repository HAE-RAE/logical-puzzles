"""Rebuild per-task CSVs from the per-difficulty JSONL files.

The dataset is stored two ways:
  - data/jsonl/{task}_{lang}_{difficulty}.jsonl  (one file per tier)
  - data/csv/{task}_{lang}.csv                   (one file per task+lang, all tiers)

This script reads every JSONL, groups by base name ({task}_{lang}, i.e. the
filename with the _easy/_medium/_hard suffix stripped), concatenates the tiers in
easy -> medium -> hard order, and writes the matching CSV.

CSV format matches the existing files: columns id,question,answer,solution,difficulty
and a UTF-8 BOM (so Excel renders 한글 correctly).

Usage:
    python scripts/gen/jsonl_to_csv.py                       # data/jsonl -> data/csv
    python scripts/gen/jsonl_to_csv.py --jsonl-dir D --csv-dir E
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

COLUMNS = ["id", "question", "answer", "solution", "difficulty"]
TIER_ORDER = {"easy": 0, "medium": 1, "hard": 2}
_SUFFIXES = tuple(f"_{t}" for t in TIER_ORDER)


def _base_and_tier(stem: str) -> tuple[str, str]:
    """('array_formula_en_hard') -> ('array_formula_en', 'hard').

    Files without a recognized tier suffix map to (stem, '') so they still
    produce a CSV of their own.
    """
    for suf in _SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)], suf[1:]
    return stem, ""


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.open(encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{i}: invalid JSON ({e})")
    return rows


def rebuild_csvs(jsonl_dir: Path, csv_dir: Path) -> list[tuple[str, int]]:
    """Write one CSV per base. Returns [(base, row_count), ...] sorted by base."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in jsonl_dir.glob("*.jsonl"):
        base, _ = _base_and_tier(path.stem)
        groups[base].append(path)

    csv_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, int]] = []
    for base in sorted(groups):
        # order the tier files easy -> medium -> hard (unknown tiers last)
        tiers = sorted(
            groups[base],
            key=lambda p: TIER_ORDER.get(_base_and_tier(p.stem)[1], 99),
        )
        rows: list[dict] = []
        for tier_path in tiers:
            rows.extend(_load_jsonl(tier_path))

        out = csv_dir / f"{base}.csv"
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow({c: r.get(c, "") for c in COLUMNS})
        written.append((base, len(rows)))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild data/csv from data/jsonl.")
    parser.add_argument("--jsonl-dir", default=str(_ROOT / "data" / "jsonl"))
    parser.add_argument("--csv-dir", default=str(_ROOT / "data" / "csv"))
    args = parser.parse_args()

    jsonl_dir = Path(args.jsonl_dir)
    csv_dir = Path(args.csv_dir)
    if not jsonl_dir.is_dir():
        raise SystemExit(f"jsonl dir not found: {jsonl_dir}")

    written = rebuild_csvs(jsonl_dir, csv_dir)
    total_rows = sum(n for _, n in written)
    for base, n in written:
        print(f"  {base:28} {n:4d} rows -> {csv_dir / (base + '.csv')}")
    print(f"\nWrote {len(written)} CSV files ({total_rows} rows) to {csv_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
