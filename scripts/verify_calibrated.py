#!/usr/bin/env python3
"""
Verify dataset accuracy with gemini-3-flash-preview.
Samples N puzzles per difficulty per task and evaluates.
"""

import sys, os, json, random, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
env_path = PROJECT_ROOT / '.env'
if env_path.exists(): load_dotenv(env_path)
else: load_dotenv()

MODEL = "gemini/gemini-3-flash-preview"
GEN_KWARGS = {"temperature": 0.0, "max_tokens": 2000}
DATA_DIR = PROJECT_ROOT / "data" / "final_calibrated" / "json"


def load_sample(task, difficulty, n):
    path = DATA_DIR / f"{task}_{difficulty}.jsonl"
    with open(path) as f:
        puzzles = [json.loads(l) for l in f]
    random.seed(42)
    return random.sample(puzzles, min(n, len(puzzles)))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=20)
    parser.add_argument('--tasks', nargs='+',
                        choices=['causal_dag_en', 'logic_grid_en', 'sat_puzzles_en'],
                        default=['causal_dag_en', 'logic_grid_en', 'sat_puzzles_en'])
    args = parser.parse_args()
    N = args.sample_size

    from evaluation.model import create_client
    from evaluation.evaluators.causal_dag import CausalDAGEvaluator
    from evaluation.evaluators.logic_grid import LogicGridEvaluator
    from evaluation.evaluators.sat_puzzle import SATPuzzleEvaluator

    evaluators = {
        'causal_dag_en': CausalDAGEvaluator(),
        'logic_grid_en': LogicGridEvaluator(),
        'sat_puzzles_en': SATPuzzleEvaluator(),
    }
    client = create_client(model=MODEL, timeout=600.0, gen_kwargs=GEN_KWARGS)
    difficulties = ['easy', 'medium', 'hard']

    all_stats = {}
    for task in args.tasks:
        ev = evaluators[task]
        for diff in difficulties:
            key = f"{task}_{diff}"
            logger.info(f"Verifying {key} (N={N})...")
            puzzles = load_sample(task, diff, N)
            results = ev.evaluate(puzzles, client, verbose=True, use_async=True,
                                  max_concurrent=20, task_name=f"verify_{key}")
            correct = sum(1 for r in results if r.correct)
            acc = correct / len(results) if results else 0
            all_stats[key] = {'total': len(results), 'correct': correct, 'acc': acc}
            logger.info(f"  {key}: {correct}/{len(results)} = {acc:.0%}")

    # Summary
    targets = {'easy': (0.65, 0.85), 'medium': (0.40, 0.60), 'hard': (0.15, 0.35)}
    print(f"\n{'='*90}")
    print(f"  VERIFICATION (N={N} per cell)")
    print(f"{'='*90}")
    print(f"{'Task + Difficulty':<40} {'N':>4} {'OK':>4} {'Acc':>8} {'Target':>12} {'Status':>8}")
    print(f"{'-'*90}")
    for task in args.tasks:
        for diff in difficulties:
            key = f"{task}_{diff}"
            s = all_stats[key]
            lo, hi = targets[diff]
            bar = '█' * int(s['acc'] * 20) + '░' * (20 - int(s['acc'] * 20))
            status = '✓' if lo <= s['acc'] <= hi else '✗'
            print(f"  {key:<38} {s['total']:>4} {s['correct']:>4} {s['acc']:>7.0%}  "
                  f"({lo:.0%}-{hi:.0%})  {status}  {bar}")
        print()
    print(f"{'='*90}")


if __name__ == '__main__':
    main()
