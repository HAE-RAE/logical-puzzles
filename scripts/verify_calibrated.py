#!/usr/bin/env python3
"""
Verify calibrated dataset accuracy with gemini-3-flash-preview.

Usage:
  # English sample (default 20 per cell)
  python scripts/verify_calibrated.py --lang en --sample-size 20

  # Korean sample
  python scripts/verify_calibrated.py --lang ko --sample-size 50

  # Full sweep (100 per cell) — both languages
  python scripts/verify_calibrated.py --full

Targets: easy 65–85% / medium 40–60% / hard 15–35%.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import PROJECT_ROOT, ensure_dotenv, load_jsonl, setup_logger

logger = setup_logger(__name__)

sys.path.insert(0, str(PROJECT_ROOT))
ensure_dotenv(PROJECT_ROOT / ".env" if (PROJECT_ROOT / ".env").exists() else None)

MODEL = "gemini/gemini-3-flash-preview"
GEN_KWARGS = {"temperature": 0.0, "max_tokens": 2000}

TARGETS = {"easy": (0.65, 0.85), "medium": (0.40, 0.60), "hard": (0.15, 0.35)}

LANG_DATA_DIR = {
    "en": PROJECT_ROOT / "data" / "final_calibrated" / "json",
    "ko": PROJECT_ROOT / "data" / "final_calibrated_ko" / "json",
}
LANG_TASKS = {
    "en": ["causal_dag_en", "logic_grid_en", "sat_puzzles_en"],
    "ko": ["causal_dag_ko", "logic_grid_ko", "sat_puzzles_ko"],
}
SHORT_NAME = {
    "causal_dag_en": "Causal DAG", "causal_dag_ko": "Causal DAG",
    "logic_grid_en": "Logic Grid", "logic_grid_ko": "Logic Grid",
    "sat_puzzles_en": "SAT",       "sat_puzzles_ko": "SAT",
}


def load_puzzles(lang: str, task: str, difficulty: str, n: int | None):
    """n=None → load all; else random sample (seed=42)."""
    path = LANG_DATA_DIR[lang] / f"{task}_{difficulty}.jsonl"
    if not path.exists():
        logger.error(f"Missing: {path}")
        return []
    puzzles = load_jsonl(path)
    if n is None:
        return puzzles
    random.seed(42)
    return random.sample(puzzles, min(n, len(puzzles)))


def evaluate_cell(evaluator, puzzles, tag: str, verbose: bool):
    from evaluation.model import create_client
    # client은 caller가 주입; 여기서는 evaluate만.
    return evaluator.evaluate(
        puzzles, evaluate_cell._client,
        verbose=verbose, use_async=True, max_concurrent=20, task_name=tag,
    )


def print_table_simple(lang: str, all_stats: dict, tasks: list, n_label: str):
    """Single-lang ASCII table (verify_calibrated.py style)."""
    print(f"\n{'=' * 90}")
    print(f"  {lang.upper()} VERIFICATION ({n_label})")
    print(f"{'=' * 90}")
    print(f"{'Task + Difficulty':<40} {'N':>4} {'OK':>4} {'Acc':>8} {'Target':>12} {'Status':>8}")
    print(f"{'-' * 90}")
    for task in tasks:
        for diff in ("easy", "medium", "hard"):
            key = f"{task}_{diff}"
            s = all_stats[key]
            lo, hi = TARGETS[diff]
            ok = lo <= s["acc"] <= hi
            bar = "█" * int(s["acc"] * 20) + "░" * (20 - int(s["acc"] * 20))
            status = "✓" if ok else "✗"
            print(f"  {key:<38} {s['total']:>4} {s['correct']:>4} {s['acc']:>7.0%}  "
                  f"({lo:.0%}-{hi:.0%})  {status}  {bar}")
        print()
    print(f"{'=' * 90}")


def print_table_full(results: dict):
    """Both-lang markdown table (verify_calibrated_en_ko_full.py style)."""
    print()
    print("=" * 100)
    print(f"  Final dataset accuracy — gemini-3-flash-preview, temp=0, max_tokens=2000")
    print("=" * 100)
    print()
    for lang_label in ("English", "Korean"):
        print(f"### {lang_label}")
        print()
        print("| Puzzle type   | Easy (65–85%) | Medium (40–60%) | Hard (15–35%) |")
        print("|---------------|---------------:|----------------:|--------------:|")
        lang_key = "en" if lang_label == "English" else "ko"
        for task in LANG_TASKS[lang_key]:
            row = f"| {SHORT_NAME[task]:<13} |"
            for diff in ("easy", "medium", "hard"):
                r = results.get((lang_label, task, diff), {})
                acc = r.get("acc", 0.0)
                c, n = r.get("correct", 0), r.get("n", 0)
                lo, hi = TARGETS[diff]
                mark = "✓" if lo <= acc <= hi else "✗"
                row += f" {acc:>5.0%} ({c}/{n}) {mark} |"
            print(row)
        print()
    print("=" * 100)


def run_sample(lang: str, tasks: list, n: int):
    from evaluation.model import create_client
    from evaluation.evaluators.causal_dag import CausalDAGEvaluator
    from evaluation.evaluators.logic_grid import LogicGridEvaluator
    from evaluation.evaluators.sat_puzzle import SATPuzzleEvaluator

    evaluators = {
        f"causal_dag_{lang}": CausalDAGEvaluator(),
        f"logic_grid_{lang}": LogicGridEvaluator(),
        f"sat_puzzles_{lang}": SATPuzzleEvaluator(),
    }
    evaluate_cell._client = create_client(model=MODEL, timeout=600.0, gen_kwargs=GEN_KWARGS)

    all_stats = {}
    for task in tasks:
        ev = evaluators[task]
        for diff in ("easy", "medium", "hard"):
            key = f"{task}_{diff}"
            logger.info(f"Verifying {key} (N={n})...")
            puzzles = load_puzzles(lang, task, diff, n)
            results = evaluate_cell(ev, puzzles, f"verify_{key}", verbose=True)
            correct = sum(1 for r in results if r.correct)
            acc = correct / len(results) if results else 0.0
            all_stats[key] = {"total": len(results), "correct": correct, "acc": acc}
            logger.info(f"  {key}: {correct}/{len(results)} = {acc:.0%}")
    print_table_simple(lang, all_stats, tasks, n_label=f"N={n} per cell")


def run_full():
    from evaluation.model import create_client
    from evaluation.evaluators.causal_dag import CausalDAGEvaluator
    from evaluation.evaluators.logic_grid import LogicGridEvaluator
    from evaluation.evaluators.sat_puzzle import SATPuzzleEvaluator

    evaluate_cell._client = create_client(model=MODEL, timeout=600.0, gen_kwargs=GEN_KWARGS)
    factories = {"causal_dag": CausalDAGEvaluator, "logic_grid": LogicGridEvaluator, "sat_puzzles": SATPuzzleEvaluator}

    results = {}
    for lang_label, lang in [("English", "en"), ("Korean", "ko")]:
        for task in LANG_TASKS[lang]:
            ev = factories[task.rsplit("_", 1)[0]]()
            for diff in ("easy", "medium", "hard"):
                key = f"{task}_{diff}"
                puzzles = load_puzzles(lang, task, diff, n=None)
                if not puzzles:
                    continue
                n = len(puzzles)
                logger.info(f"{lang_label} {key}: evaluating N={n}...")
                out = evaluate_cell(ev, puzzles, f"full100_{key}", verbose=False)
                correct = sum(1 for r in out if r.correct)
                acc = correct / n if n else 0.0
                results[(lang_label, task, diff)] = {"n": n, "correct": correct, "acc": acc}
                logger.info(f"  -> {correct}/{n} = {acc:.2%}")
    print_table_full(results)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", choices=["en", "ko"], default="en",
                        help="Language for sample mode (ignored when --full)")
    parser.add_argument("--sample-size", type=int, default=20,
                        help="Puzzles per (task, difficulty) cell")
    parser.add_argument("--full", action="store_true",
                        help="Evaluate ALL puzzles for BOTH languages (markdown table output)")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Override task list (e.g. causal_dag_en logic_grid_en)")
    args = parser.parse_args()

    if args.full:
        run_full()
    else:
        tasks = args.tasks or LANG_TASKS[args.lang]
        run_sample(args.lang, tasks, args.sample_size)


if __name__ == "__main__":
    main()
