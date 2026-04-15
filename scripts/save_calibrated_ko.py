#!/usr/bin/env python3
"""
한국어 캘리브레이션 데이터 저장 (영어 save_calibrated.py와 유사 전략).

- Logic Grid / SAT: seed-search 또는 기존 파일 유지로 목표 구간(65-85 / 40-60 / 15-35) 맞춤.
- Causal DAG: 영어와 동일 그래프 파라미터 + 질문에서 엣지 무작위 순서(shuffle_edges)를 써도
  gemini-3-flash-preview 한국어에서 정답률이 구간 상한을 크게 넘는 경우가 많아,
  이 스크립트만으로는 난이도 밴드에 안 들어갈 수 있음(저장은 최선의 시드로 진행).

평가: gemini-3-flash-preview, temperature=0.0, max_tokens=2000
"""

import json
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

from evaluation.evaluators.causal_dag import CausalDAGEvaluator
from evaluation.evaluators.logic_grid import LogicGridEvaluator
from evaluation.evaluators.sat_puzzle import SATPuzzleEvaluator
from evaluation.model import create_client

from scripts.generate_calibrated_ko import (
    PUZZLES_PER_DIFFICULTY,
    gen_causal_dag,
    gen_logic_grid,
    gen_sat,
)

MODEL = "gemini/gemini-3-flash-preview"
GEN_KWARGS = {"temperature": 0.0, "max_tokens": 2000}
OUT_DIR = PROJECT_ROOT / "data" / "final_calibrated_ko" / "json"

TARGETS = {
    "easy": (0.65, 0.85),
    "medium": (0.40, 0.60),
    "hard": (0.15, 0.35),
}

N_SEARCH = 40
N_CONFIRM = 50
MAX_SEEDS = 18


def in_range(acc: float, diff: str) -> bool:
    lo, hi = TARGETS[diff]
    return lo <= acc <= hi


def save_jsonl(puzzles, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in puzzles:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def gen_batch(gen_fn, difficulty: str, n: int, base_seed: int):
    puzzles = []
    attempts = 0
    while len(puzzles) < n and attempts < n * 12:
        seed = base_seed + hash(difficulty) % 10000 + attempts
        p = gen_fn(difficulty, len(puzzles), seed)
        if p:
            puzzles.append(p)
        attempts += 1
    return puzzles


def eval_acc(evaluator, puzzles, n_sample: int, task_tag: str, client) -> float:
    random.seed(42)
    sample = random.sample(puzzles, min(n_sample, len(puzzles)))
    results = evaluator.evaluate(
        sample,
        client,
        verbose=False,
        use_async=True,
        max_concurrent=20,
        task_name=task_tag,
    )
    correct = sum(1 for r in results if r.correct)
    return correct / len(results) if results else 0.0


def seed_search_causal(client, ev: CausalDAGEvaluator, diff: str) -> tuple[list, float, int]:
    for attempt in range(MAX_SEEDS):
        base_seed = 4000 + attempt * 823 + hash(diff) % 500
        puzzles = gen_batch(gen_causal_dag, diff, PUZZLES_PER_DIFFICULTY, base_seed)
        if len(puzzles) < PUZZLES_PER_DIFFICULTY:
            logger.warning(f"  causal {diff} seed={base_seed}: only {len(puzzles)} generated")
            continue
        acc = eval_acc(
            ev,
            puzzles,
            N_SEARCH,
            f"cal_ko_causal_{diff}_{attempt}",
            client,
        )
        logger.info(f"  causal_dag_ko {diff} seed={base_seed}: {acc:.0%}")
        if in_range(acc, diff):
            acc2 = eval_acc(
                ev,
                puzzles,
                N_CONFIRM,
                f"cal_ko_causal_{diff}_confirm",
                client,
            )
            logger.info(f"    confirm N={N_CONFIRM}: {acc2:.0%}")
            return puzzles, acc2, base_seed
    last = gen_batch(gen_causal_dag, diff, PUZZLES_PER_DIFFICULTY, 4000 + (MAX_SEEDS - 1) * 823)
    acc = eval_acc(ev, last, N_CONFIRM, f"cal_ko_causal_{diff}_fallback", client)
    logger.warning(f"  causal_dag_ko {diff}: no seed in range, saving fallback acc={acc:.0%}")
    return last, acc, -1


def seed_search_logic_medium(client, ev: LogicGridEvaluator):
    diff = "medium"
    for attempt in range(MAX_SEEDS):
        base_seed = 7000 + attempt * 823 + hash(diff) % 500
        puzzles = gen_batch(gen_logic_grid, diff, PUZZLES_PER_DIFFICULTY, base_seed)
        if len(puzzles) < PUZZLES_PER_DIFFICULTY:
            logger.warning(f"  logic_grid_ko medium seed={base_seed}: only {len(puzzles)} generated")
            continue
        acc = eval_acc(
            ev,
            puzzles,
            N_SEARCH,
            f"cal_ko_lg_medium_{attempt}",
            client,
        )
        logger.info(f"  logic_grid_ko medium seed={base_seed}: {acc:.0%}")
        if in_range(acc, diff):
            acc2 = eval_acc(
                ev,
                puzzles,
                N_CONFIRM,
                f"cal_ko_lg_medium_confirm",
                client,
            )
            logger.info(f"    confirm N={N_CONFIRM}: {acc2:.0%}")
            return puzzles, acc2, base_seed
    last = gen_batch(gen_logic_grid, diff, PUZZLES_PER_DIFFICULTY, 7000 + (MAX_SEEDS - 1) * 823)
    acc = eval_acc(ev, last, N_CONFIRM, f"cal_ko_lg_medium_fallback", client)
    logger.warning(f"  logic_grid_ko medium: no seed in range, saving fallback acc={acc:.0%}")
    return last, acc, -1


def maybe_keep_or_search(
    client,
    evaluator,
    gen_fn,
    task_prefix: str,
    diff: str,
    path: Path,
) -> None:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            puzzles = [json.loads(line) for line in f if line.strip()]
        if len(puzzles) >= PUZZLES_PER_DIFFICULTY:
            acc = eval_acc(
                evaluator,
                puzzles,
                N_CONFIRM,
                f"cal_ko_chk_{task_prefix}_{diff}",
                client,
            )
            logger.info(f"  existing {path.name}: {acc:.0%} (target {TARGETS[diff]})")
            if in_range(acc, diff):
                logger.info(f"    -> keep existing file")
                return
    for attempt in range(MAX_SEEDS):
        base_seed = 9000 + attempt * 701 + hash(diff) % 400 + hash(task_prefix) % 200
        puzzles = gen_batch(gen_fn, diff, PUZZLES_PER_DIFFICULTY, base_seed)
        if len(puzzles) < PUZZLES_PER_DIFFICULTY:
            continue
        acc = eval_acc(
            evaluator,
            puzzles,
            N_SEARCH,
            f"cal_ko_{task_prefix}_{diff}_{attempt}",
            client,
        )
        logger.info(f"  {task_prefix} {diff} seed={base_seed}: {acc:.0%}")
        if in_range(acc, diff):
            acc2 = eval_acc(
                evaluator,
                puzzles,
                N_CONFIRM,
                f"cal_ko_{task_prefix}_{diff}_cf",
                client,
            )
            logger.info(f"    confirm: {acc2:.0%}")
            save_jsonl(puzzles, path)
            return
    puzzles = gen_batch(gen_fn, diff, PUZZLES_PER_DIFFICULTY, 9000)
    save_jsonl(puzzles, path)
    logger.warning(f"  {task_prefix} {diff}: saved best-effort without range match")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = create_client(model=MODEL, timeout=600.0, gen_kwargs=GEN_KWARGS)
    causal_ev = CausalDAGEvaluator()
    lg_ev = LogicGridEvaluator()
    sat_ev = SATPuzzleEvaluator()

    logger.info("=== Causal DAG (KO) seed search ===")
    for diff in ["easy", "medium", "hard"]:
        puzzles, _, seed = seed_search_causal(client, causal_ev, diff)
        save_jsonl(puzzles, OUT_DIR / f"causal_dag_ko_{diff}.jsonl")
        logger.info(f"  saved causal_dag_ko_{diff}.jsonl (seed={seed})")

    logger.info("=== Logic Grid (KO) medium seed search ===")
    puzzles, _, seed = seed_search_logic_medium(client, lg_ev)
    save_jsonl(puzzles, OUT_DIR / f"logic_grid_ko_medium.jsonl")
    logger.info(f"  saved logic_grid_ko_medium.jsonl (seed={seed})")

    logger.info("=== Logic Grid (KO) easy / hard (keep or search) ===")
    maybe_keep_or_search(client, lg_ev, gen_logic_grid, "lg", "easy", OUT_DIR / "logic_grid_ko_easy.jsonl")
    maybe_keep_or_search(client, lg_ev, gen_logic_grid, "lg", "hard", OUT_DIR / "logic_grid_ko_hard.jsonl")

    logger.info("=== SAT (KO) easy / medium / hard (keep or search) ===")
    for diff in ["easy", "medium", "hard"]:
        maybe_keep_or_search(
            client,
            sat_ev,
            gen_sat,
            "sat",
            diff,
            OUT_DIR / f"sat_puzzles_ko_{diff}.jsonl",
        )

    logger.info(f"Done. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
