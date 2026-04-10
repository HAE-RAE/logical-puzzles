from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Project root (relative to tests/)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.evaluators import get_evaluator, list_tasks  # noqa: E402

# Tasks excluded from bilingual system-prompt checks (separate prompt setup)
_SKIP_TASKS = frozenset({"kinship", "kinship_vision"})

DEFAULT_JSONL_PATH = Path(__file__).resolve().parent / "evaluator_system_prompts.jsonl"


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
    dummy = dummy or {"answer": "42", "question": "test", "id": "smoke"}
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
    path = path or DEFAULT_JSONL_PATH
    dummy = dummy or {"answer": "42", "question": "test", "id": "smoke"}
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks = sorted(list_tasks())
    with path.open("w", encoding="utf-8") as f:
        for task in tasks:
            rec = task_prompt_export_record(task, dummy)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(tasks)


def _check_routing(verbose: bool = False) -> list[str]:
    """Return a list of failure messages (empty means pass)."""
    dummy = {"answer": "42", "question": "test", "id": "smoke"}
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
                errors.append(
                    f"{task}: expected KOREAN_SYSTEM_PROMPT for _ko task"
                )
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


def test_evaluator_system_prompt_branching() -> None:
    """pytest: bilingual tasks must route to the expected system prompt."""
    errors = _check_routing(verbose=False)
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check evaluator system-prompt routing; optionally dump JSONL."
    )
    parser.add_argument(
        "--dump-jsonl",
        nargs="?",
        const=str(DEFAULT_JSONL_PATH),
        default=None,
        metavar="PATH",
        help=(
            "Write all task prompts to JSONL "
            f"(default: {DEFAULT_JSONL_PATH})"
        ),
    )
    args = parser.parse_args()

    exit_code = 0
    if args.dump_jsonl is not None:
        path = Path(args.dump_jsonl)
        n = dump_evaluator_prompts_jsonl(path)
        print(f"Wrote {n} task records to {path.resolve()}")

    errors = _check_routing(verbose=True)
    if errors:
        print("\nFailures:")
        for e in errors:
            print(f"  - {e}")
        exit_code = 1
    else:
        print("\nAll bilingual task system-prompt checks passed.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
