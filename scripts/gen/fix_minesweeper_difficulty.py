#!/usr/bin/env python3
"""Normalize per-record difficulty in minesweeper JSONL files.

Updates both the top-level ``difficulty`` field and the ``Difficulty:`` line
inside the solution text (STEP 0 meta).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DIFFICULTIES = ("easy", "medium", "hard")
DIFFICULTY_LINE_RE = re.compile(r"(  - Difficulty: )\w+")


def fix_record(record: dict, target: str) -> dict:
    out = dict(record)
    out["difficulty"] = target
    if "solution" in out and isinstance(out["solution"], str):
        out["solution"] = DIFFICULTY_LINE_RE.sub(rf"\g<1>{target}", out["solution"], count=1)
    return out


def fix_jsonl(path: Path, target: str) -> dict:
    rows = []
    before = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            before[row.get("difficulty", "<missing>")] = before.get(row.get("difficulty", "<missing>"), 0) + 1
            rows.append(fix_record(row, target))

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"path": str(path), "rows": len(rows), "before": before, "after": target}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix minesweeper JSONL difficulty labels")
    parser.add_argument("path", type=Path, help="JSONL file to update in place")
    parser.add_argument(
        "--difficulty", required=True, choices=DIFFICULTIES,
        help="Target difficulty written to every record",
    )
    args = parser.parse_args()

    if not args.path.is_file():
        raise SystemExit(f"file not found: {args.path}")

    stats = fix_jsonl(args.path, args.difficulty)
    print(
        f"fixed {stats['rows']} rows in {stats['path']}: "
        f"{stats['before']} -> all '{stats['after']}'"
    )


if __name__ == "__main__":
    main()
