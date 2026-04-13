#!/usr/bin/env python3
"""
한국어 캘리브레이션 데이터 정확도 검증 (gemini-3-flash-preview).

설정은 verify_calibrated.py와 동일: temperature=0.0, max_tokens=2000
데이터: data/final_calibrated_ko/json/
"""

import json
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

MODEL = "gemini/gemini-3-flash-preview"
GEN_KWARGS = {"temperature": 0.0, "max_tokens": 2000}
DATA_DIR = PROJECT_ROOT / "data" / "final_calibrated_ko" / "json"


def load_sample(task, difficulty, n):
    path = DATA_DIR / f"{task}_{difficulty}.jsonl"
    with open(path, encoding="utf-8") as f:
        puzzles = [json.loads(line) for line in f if line.strip()]
    random.seed(42)
    return random.sample(puzzles, min(n, len(puzzles)))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["causal_dag_ko", "logic_grid_ko", "sat_puzzles_ko"],
        default=["causal_dag_ko", "logic_grid_ko", "sat_puzzles_ko"],
    )
    args = parser.parse_args()
    n = args.sample_size

    from evaluation.model import create_client
    from evaluation.evaluators.causal_dag import CausalDAGEvaluator
    from evaluation.evaluators.logic_grid import LogicGridEvaluator
    from evaluation.evaluators.sat_puzzle import SATPuzzleEvaluator

    evaluators = {
        "causal_dag_ko": CausalDAGEvaluator(),
        "logic_grid_ko": LogicGridEvaluator(),
        "sat_puzzles_ko": SATPuzzleEvaluator(),
    }
    client = create_client(model=MODEL, timeout=600.0, gen_kwargs=GEN_KWARGS)
    difficulties = ["easy", "medium", "hard"]
    targets = {"easy": (0.65, 0.85), "medium": (0.40, 0.60), "hard": (0.15, 0.35)}

    all_stats = {}
    for task in args.tasks:
        ev = evaluators[task]
        for diff in difficulties:
            key = f"{task}_{diff}"
            logger.info(f"Verifying {key} (N={n})...")
            puzzles = load_sample(task, diff, n)
            results = ev.evaluate(
                puzzles,
                client,
                verbose=True,
                use_async=True,
                max_concurrent=20,
                task_name=f"verify_{key}",
            )
            correct = sum(1 for r in results if r.correct)
            acc = correct / len(results) if results else 0
            all_stats[key] = {"total": len(results), "correct": correct, "acc": acc}
            logger.info(f"  {key}: {correct}/{len(results)} = {acc:.0%}")

    print(f"\n{'=' * 90}")
    print(f"  KOREAN VERIFICATION (N={n} per cell)")
    print(f"{'=' * 90}")
    print(f"{'Task + Difficulty':<40} {'N':>4} {'OK':>4} {'Acc':>8} {'Target':>12} {'Status':>8}")
    print(f"{'-' * 90}")
    for task in args.tasks:
        for diff in difficulties:
            key = f"{task}_{diff}"
            s = all_stats[key]
            lo, hi = targets[diff]
            status = "OK" if lo <= s["acc"] <= hi else "OUT"
            print(
                f"  {key:<38} {s['total']:>4} {s['correct']:>4} {s['acc']:>7.0%}  "
                f"({lo:.0%}-{hi:.0%})  {status}"
            )
        print()
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
