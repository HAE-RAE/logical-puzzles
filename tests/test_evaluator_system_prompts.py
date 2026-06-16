"""pytest checks for evaluator system prompts.

The export utility (JSONL/CSV dump + CLI) lives in
scripts/eval/export_system_prompts.py; these tests import its helpers so the
routing logic and dump format are covered without duplicating code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Project root (relative to tests/)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.evaluators import list_tasks  # noqa: E402
from scripts.eval.export_system_prompts import (  # noqa: E402
    check_routing,
    dump_evaluator_prompts_csv,
    dump_evaluator_prompts_jsonl,
)


def test_evaluator_system_prompt_branching() -> None:
    """Bilingual tasks must route to the expected (EN/KO) system prompt."""
    errors = check_routing(verbose=False)
    assert not errors, "\n".join(errors)


def test_dump_evaluator_prompts_jsonl_writes_valid_jsonl(tmp_path: Path) -> None:
    """Each registry task produces one valid JSON line: task + SYSTEM_PROMPT only."""
    out = tmp_path / "prompts.jsonl"
    n = dump_evaluator_prompts_jsonl(out)
    assert n == len(list_tasks())
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == n
    for line in lines:
        rec = json.loads(line)
        assert set(rec.keys()) == {"task", "SYSTEM_PROMPT"}
        assert isinstance(rec["task"], str)
        assert isinstance(rec["SYSTEM_PROMPT"], str) and rec["SYSTEM_PROMPT"]


def test_dump_evaluator_prompts_csv_writes_all_rows(tmp_path: Path) -> None:
    """CSV has a header plus one row per task with a non-empty prompt."""
    import csv

    out = tmp_path / "prompts.csv"
    n = dump_evaluator_prompts_csv(out)
    assert n == len(list_tasks())
    with out.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["task", "lang", "SYSTEM_PROMPT"]
    assert len(rows) == n + 1
    for r in rows[1:]:
        assert r[0] and r[2]  # task name and prompt non-empty
        assert r[1] in {"en", "ko", ""}
