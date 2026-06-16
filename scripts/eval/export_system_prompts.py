"""Export every evaluator's system prompt to JSONL + CSV (paper artifact).

For each registry task name, this resolves the system prompt that is actually
used at evaluation time (EN/KO routing via `_get_system_prompt`) and writes:
  - <out>.jsonl : one JSON object per task ({"task", "SYSTEM_PROMPT"})
  - <out>.csv   : spreadsheet-friendly (task, lang, SYSTEM_PROMPT), UTF-8 BOM

The routing-correctness *test* lives in tests/test_evaluator_system_prompts.py
and imports the helpers from this module.

Usage:
    python scripts/eval/export_system_prompts.py            # -> docs/methodology/evaluator_system_prompts.{jsonl,csv}
    python scripts/eval/export_system_prompts.py --out PATH # custom stem (extension ignored)
    python scripts/eval/export_system_prompts.py --check    # also run the EN/KO routing check
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Project root (relative to scripts/eval/)
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.evaluators import get_evaluator, list_tasks  # noqa: E402

# Tasks excluded from bilingual system-prompt checks (separate prompt setup)
_SKIP_TASKS = frozenset({"kinship", "kinship_vision"})

DEFAULT_OUT_STEM = _ROOT / "docs" / "methodology" / "evaluator_system_prompts"
_DUMMY = {"answer": "42", "question": "test", "id": "smoke"}


def _lang_of(task: str) -> str:
    """Language tag inferred from the registry task-name suffix."""
    if task.endswith("_ko"):
        return "ko"
    if task.endswith("_en"):
        return "en"
    return ""


def _resolved_system_prompt(ev, task: str, dummy: dict) -> str:
    """System prompt string used for evaluation for this registry task name."""
    if task == "kinship_vision":
        return getattr(ev, "VISION_SYSTEM_PROMPT", ev.SYSTEM_PROMPT)
    if hasattr(ev, "_get_system_prompt"):
        ev._task_name = task
        return ev._get_system_prompt(dummy)
    return getattr(ev, "SYSTEM_PROMPT", "") or ""


def task_prompt_export_record(task: str, dummy: dict | None = None) -> dict:
    """One JSON object per task: task name and the system prompt actually used."""
    dummy = dummy or _DUMMY
    ev = get_evaluator(task)
    return {
        "task": task,
        "SYSTEM_PROMPT": _resolved_system_prompt(ev, task, dummy),
    }


def dump_evaluator_prompts_jsonl(
    path: Path | None = None,
    *,
    dummy: dict | None = None,
) -> int:
    """Write one JSON object per line (sorted by task). Returns number of records."""
    path = Path(path) if path else DEFAULT_OUT_STEM.with_suffix(".jsonl")
    dummy = dummy or _DUMMY
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks = sorted(list_tasks())
    with path.open("w", encoding="utf-8") as f:
        for task in tasks:
            rec = task_prompt_export_record(task, dummy)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(tasks)


def dump_evaluator_prompts_csv(
    path: Path | None = None,
    *,
    dummy: dict | None = None,
) -> int:
    """Write a spreadsheet-friendly CSV (one row per task). Returns number of rows.

    Columns: task, lang, SYSTEM_PROMPT. Multi-line prompts are kept in a single
    quoted cell (Excel/Sheets render the newlines). UTF-8 BOM so Excel shows 한글.
    """
    path = Path(path) if path else DEFAULT_OUT_STEM.with_suffix(".csv")
    dummy = dummy or _DUMMY
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks = sorted(list_tasks())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "lang", "SYSTEM_PROMPT"])
        for task in tasks:
            rec = task_prompt_export_record(task, dummy)
            writer.writerow([rec["task"], _lang_of(task), rec["SYSTEM_PROMPT"]])
    return len(tasks)


def check_routing(verbose: bool = False) -> list[str]:
    """Return a list of failure messages (empty means pass).

    Bilingual tasks must route _ko -> KOREAN_SYSTEM_PROMPT and _en -> SYSTEM_PROMPT.
    """
    dummy = dict(_DUMMY)
    errors: list[str] = []

    for task in sorted(list_tasks()):
        if task in _SKIP_TASKS:
            continue
        ev = get_evaluator(task)
        if not hasattr(ev, "KOREAN_SYSTEM_PROMPT"):
            if verbose:
                print(f"  (skip) {task} — no bilingual prompts")
            continue
        if not hasattr(ev, "_get_system_prompt"):
            errors.append(f"{task}: missing _get_system_prompt")
            continue

        ev._task_name = task
        got = ev._get_system_prompt(dummy)

        n_err = len(errors)
        if task.endswith("_ko"):
            if got != ev.KOREAN_SYSTEM_PROMPT:
                errors.append(f"{task}: expected KOREAN_SYSTEM_PROMPT for _ko task")
        elif task.endswith("_en"):
            if got != ev.SYSTEM_PROMPT:
                errors.append(f"{task}: expected SYSTEM_PROMPT for _en task")
        else:
            if got != ev.SYSTEM_PROMPT:
                errors.append(
                    f"{task}: no _ko/_en suffix and numeric answer dummy: "
                    f"expected SYSTEM_PROMPT"
                )

        if verbose and len(errors) == n_err:
            label = "KO" if got is ev.KOREAN_SYSTEM_PROMPT else "EN"
            preview = (got[:72] + "…") if len(got) > 72 else got
            print(f"  OK  {task:22} -> {label:2}  {preview!r}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export evaluator system prompts to JSONL + CSV; check routing."
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT_STEM),
        metavar="STEM",
        help=(
            "Output path stem; .jsonl and .csv are written from it "
            f"(default: {DEFAULT_OUT_STEM})"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Also run the EN/KO routing check (off by default).",
    )
    args = parser.parse_args()

    stem = Path(args.out)
    if stem.suffix:  # tolerate a passed extension
        stem = stem.with_suffix("")

    n = dump_evaluator_prompts_jsonl(stem.with_suffix(".jsonl"))
    print(f"Wrote {n} task records to {stem.with_suffix('.jsonl').resolve()}")
    n_csv = dump_evaluator_prompts_csv(stem.with_suffix(".csv"))
    print(f"Wrote {n_csv} task records to {stem.with_suffix('.csv').resolve()}")

    if not args.check:
        return 0

    errors = check_routing(verbose=True)
    if errors:
        print("\nFailures:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nAll bilingual task system-prompt checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
