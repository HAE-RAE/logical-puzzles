#!/usr/bin/env python3
"""
합쳐진 JSONL(data/jsonl/foo_en.jsonl 등)을 난이도별 파일로 분리한다.

출력: {stem}_easy.jsonl, {stem}_medium.jsonl, {stem}_hard.jsonl
(stem = 입력 파일명에서 .jsonl 제거)

이미 *_easy.jsonl 처럼 끝나는 파일은 처리하지 않는다.
--delete-source 로 분리 성공 후 입력 합본 파일만 제거한다(CSV 등은 건드리지 않음).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


DIFFS = ("easy", "medium", "hard")
DIFF_SUFFIX_RE = re.compile(r"_(easy|medium|hard)\.jsonl$")


def normalize_difficulty(raw) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip().lower()
    else:
        s = str(raw).strip().lower()
    if s in DIFFS:
        return s
    for d in DIFFS:
        if d in s:
            return d
    return None


def is_already_split(path: Path) -> bool:
    return bool(DIFF_SUFFIX_RE.search(path.name))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_one(
    src: Path,
    out_dir: Path,
    max_per_difficulty: int | None,
    delete_source: bool,
) -> dict[str, int]:
    stem = src.stem
    rows = load_jsonl(src)
    buckets: dict[str, list] = defaultdict(list)
    unknown = 0
    for row in rows:
        d = normalize_difficulty(row.get("difficulty"))
        if d is None:
            unknown += 1
            continue
        buckets[d].append(row)

    counts = {}
    for d in DIFFS:
        items = buckets.get(d, [])
        # Skip tiers absent from this source so a single-tier run doesn't
        # truncate the other tiers' existing files to empty.
        if not items:
            counts[d] = 0
            continue
        if max_per_difficulty is not None and len(items) > max_per_difficulty:
            items = items[:max_per_difficulty]
        out_path = out_dir / f"{stem}_{d}.jsonl"
        save_jsonl(items, out_path)
        counts[d] = len(items)

    if delete_source:
        src.unlink(missing_ok=True)

    return {"source": str(src), **counts, "unknown_difficulty": unknown, "skipped_rows": unknown}


def iter_default_jsonl_dir(jsonl_dir: Path) -> list[Path]:
    out = []
    for p in sorted(jsonl_dir.glob("*.jsonl")):
        if is_already_split(p):
            continue
        out.append(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Split combined JSONL by difficulty")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="JSONL files to split (default: all non-split *.jsonl under --jsonl-dir)",
    )
    parser.add_argument(
        "--jsonl-dir",
        type=Path,
        default=Path("data/jsonl"),
        help="Used when no positional inputs are given",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: same directory as each input file)",
    )
    parser.add_argument(
        "--max-per-difficulty",
        type=int,
        default=None,
        help="If set, cap each difficulty file to this many lines (stable: first N)",
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Remove each input combined .jsonl after writing *_easy|medium|hard.jsonl",
    )
    args = parser.parse_args()

    if args.inputs:
        paths = [p.resolve() for p in args.inputs]
    else:
        base = args.jsonl_dir.resolve()
        if not base.is_dir():
            raise SystemExit(f"Not a directory: {base}")
        paths = iter_default_jsonl_dir(base)

    if not paths:
        print("No JSONL files to split.")
        return

    for src in paths:
        if not src.is_file():
            print(f"[SKIP] missing: {src}")
            continue
        out_dir = args.out_dir.resolve() if args.out_dir else src.parent
        stats = split_one(src, out_dir, args.max_per_difficulty, args.delete_source)
        msg = (
            f"[OK] {src.name} -> "
            f"easy={stats['easy']} medium={stats['medium']} hard={stats['hard']}"
            + (f" unknown_difficulty={stats['unknown_difficulty']}" if stats["unknown_difficulty"] else "")
        )
        if args.delete_source:
            msg += " (source removed)"
        print(msg)


if __name__ == "__main__":
    main()
