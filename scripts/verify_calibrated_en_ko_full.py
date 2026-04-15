#!/usr/bin/env python3
"""
최종 캘리브레이션 JSONL(난이도별 100개) 전부에 대해 gemini-3-flash-preview로 정확도 측정.
영어: data/final_calibrated/json/
한국어: data/final_calibrated_ko/json/
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env" if (PROJECT_ROOT / ".env").exists() else None)

MODEL = "gemini/gemini-3-flash-preview"
GEN_KWARGS = {"temperature": 0.0, "max_tokens": 2000}

TARGETS = {"easy": (0.65, 0.85), "medium": (0.40, 0.60), "hard": (0.15, 0.35)}


def load_all(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    from evaluation.model import create_client
    from evaluation.evaluators.causal_dag import CausalDAGEvaluator
    from evaluation.evaluators.logic_grid import LogicGridEvaluator
    from evaluation.evaluators.sat_puzzle import SATPuzzleEvaluator

    client = create_client(model=MODEL, timeout=600.0, gen_kwargs=GEN_KWARGS)

    suites = [
        (
            "English",
            PROJECT_ROOT / "data" / "final_calibrated" / "json",
            [
                ("causal_dag_en", CausalDAGEvaluator()),
                ("logic_grid_en", LogicGridEvaluator()),
                ("sat_puzzles_en", SATPuzzleEvaluator()),
            ],
        ),
        (
            "Korean",
            PROJECT_ROOT / "data" / "final_calibrated_ko" / "json",
            [
                ("causal_dag_ko", CausalDAGEvaluator()),
                ("logic_grid_ko", LogicGridEvaluator()),
                ("sat_puzzles_ko", SATPuzzleEvaluator()),
            ],
        ),
    ]

    difficulties = ["easy", "medium", "hard"]
    results = {}

    for lang_label, data_dir, tasks in suites:
        for task_prefix, ev in tasks:
            for diff in difficulties:
                key = f"{task_prefix}_{diff}"
                path = data_dir / f"{key}.jsonl"
                if not path.exists():
                    logger.error(f"Missing: {path}")
                    continue
                puzzles = load_all(path)
                n = len(puzzles)
                logger.info(f"{lang_label} {key}: evaluating N={n}...")
                tag = f"full100_{key}"
                out = ev.evaluate(
                    puzzles,
                    client,
                    verbose=False,
                    use_async=True,
                    max_concurrent=20,
                    task_name=tag,
                )
                correct = sum(1 for r in out if r.correct)
                acc = correct / n if n else 0.0
                results[(lang_label, task_prefix, diff)] = {
                    "n": n,
                    "correct": correct,
                    "acc": acc,
                }
                logger.info(f"  -> {correct}/{n} = {acc:.2%}")

    # 표 출력
    short = {
        "causal_dag_en": "Causal DAG",
        "logic_grid_en": "Logic Grid",
        "sat_puzzles_en": "SAT",
        "causal_dag_ko": "Causal DAG",
        "logic_grid_ko": "Logic Grid",
        "sat_puzzles_ko": "SAT",
    }

    print()
    print("=" * 100)
    print("  Final dataset accuracy (100 puzzles per cell) — gemini-3-flash-preview, temp=0, max_tokens=2000")
    print("=" * 100)
    print()

    for lang in ["English", "Korean"]:
        print(f"### {lang}")
        print()
        print(
            f"| Puzzle type   | Easy (65–85%) | Medium (40–60%) | Hard (15–35%) |"
        )
        print(f"|---------------|---------------:|----------------:|--------------:|")
        _tasks = suites[0][2] if lang == "English" else suites[1][2]
        for task_prefix, _ev in _tasks:
            row_name = short[task_prefix]
            line = f"| {row_name:<13} |"
            for diff in difficulties:
                r = results.get((lang, task_prefix, diff), {})
                acc = r.get("acc", 0.0)
                c, n = r.get("correct", 0), r.get("n", 0)
                lo, hi = TARGETS[diff]
                ok = lo <= acc <= hi
                mark = "✓" if ok else "✗"
                line += f" {acc:>5.0%} ({c}/{n}) {mark} |"
            print(line)
        print()

    print("=" * 100)


if __name__ == "__main__":
    main()
