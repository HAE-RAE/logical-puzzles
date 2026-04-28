"""
array_formula_en GRPO-vs-SFT 실험용 데이터셋 생성

- 학습용 3000건 (난이도 easy/med/hard 각 1000, problem_type 4종 각 250)
- 평가용 60건 (난이도별 20, problem_type 4종 각 5)
- seed 범위 분리로 학습/평가 disjoint
- question 텍스트 hash 기반 dedup 검증 (학습 내 / 학습-평가 / 학습-기존 102건)
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from generation.array_formula_en import (
    PROBLEM_GENERATORS,
    generate_puzzle,
    puzzle_to_prompt,
)

DIFFICULTIES = ["easy", "medium", "hard"]
PROBLEM_TYPES = list(PROBLEM_GENERATORS.keys())  # 4종

# seed 분리: train [10_000_000, ...), eval [90_000_000, ...)
TRAIN_SEED_BASE = 10_000_000
EVAL_SEED_BASE = 90_000_000


def prompt_hash(puzzle: dict) -> str:
    """학습 instance 단위 dedup용 hash.

    question 텍스트는 generator가 고정된 template 풀에서 뽑기 때문에 동일해도
    실제 학습 prompt(=question+tables)는 tables 데이터에 따라 모두 다르다.
    따라서 contamination은 (question + tables) hash로 판정한다.
    """
    payload = puzzle["question"] + "|" + json.dumps(puzzle["tables"], sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def question_template(puzzle: dict) -> str:
    return puzzle["question"]


def generate_split(num_per_difficulty: int, seed_base: int, id_prefix: str) -> list:
    """난이도×problem_type 균등 생성. seed는 base에서 단조 증가."""
    puzzles = []
    seed = seed_base
    per_ptype = num_per_difficulty // len(PROBLEM_TYPES)
    remainder = num_per_difficulty % len(PROBLEM_TYPES)

    for difficulty in DIFFICULTIES:
        for j, ptype in enumerate(PROBLEM_TYPES):
            count = per_ptype + (1 if j < remainder else 0)
            for _ in range(count):
                puzzle = generate_puzzle(difficulty=difficulty, problem_type=ptype, seed=seed)
                puzzle["id"] = f"{id_prefix}_{len(puzzles):05d}_{difficulty}_{ptype}"
                puzzles.append(puzzle)
                seed += 1
    return puzzles


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dedup_report(train: list, eval_set: list, existing: list) -> dict:
    train_hashes = [prompt_hash(p) for p in train]
    eval_hashes = [prompt_hash(p) for p in eval_set]
    existing_hashes = {prompt_hash(p) for p in existing}

    train_set = set(train_hashes)
    eval_set_h = set(eval_hashes)

    train_qtmpl = {question_template(p) for p in train}
    eval_qtmpl = {question_template(p) for p in eval_set}

    return {
        "_dedup_basis": "prompt = question + tables (instance-level)",
        "train_total": len(train_hashes),
        "train_unique_prompts": len(train_set),
        "train_internal_dups": len(train_hashes) - len(train_set),
        "eval_total": len(eval_hashes),
        "eval_unique_prompts": len(eval_set_h),
        "eval_internal_dups": len(eval_hashes) - len(eval_set_h),
        "train_eval_overlap": len(train_set & eval_set_h),
        "train_existing_overlap": len(train_set & existing_hashes),
        "eval_existing_overlap": len(eval_set_h & existing_hashes),
        "_question_template_stats": {
            "train_unique_question_templates": len(train_qtmpl),
            "eval_unique_question_templates": len(eval_qtmpl),
            "shared_question_templates": len(train_qtmpl & eval_qtmpl),
            "note": "question templates are drawn from a fixed pool by the generator; sharing them is expected and does not imply prompt-instance leakage",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/array_formula_en")
    parser.add_argument("--train-per-difficulty", type=int, default=1000)
    parser.add_argument("--eval-per-difficulty", type=int, default=20)
    parser.add_argument(
        "--existing",
        default="data/json/array_formula_en.jsonl",
        help="기존 데이터(중복 검사용)",
    )
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    print(f"[gen] generating train ({args.train_per_difficulty}/diff × 3 = {args.train_per_difficulty*3}) ...")
    train = generate_split(args.train_per_difficulty, TRAIN_SEED_BASE, "train")
    print(f"[gen] generating eval ({args.eval_per_difficulty}/diff × 3 = {args.eval_per_difficulty*3}) ...")
    eval_set = generate_split(args.eval_per_difficulty, EVAL_SEED_BASE, "eval")

    existing = load_jsonl(PROJECT_ROOT / args.existing)
    rep = dedup_report(train, eval_set, existing)
    print("[dedup]", json.dumps(rep, indent=2))

    write_jsonl(out_dir / "train_3k.jsonl", train)
    write_jsonl(out_dir / "eval_60.jsonl", eval_set)
    (out_dir / "dedup_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

    if rep["train_internal_dups"] or rep["eval_internal_dups"] or rep["train_eval_overlap"]:
        print("[FAIL] 중복 발견 — 위 리포트 확인", file=sys.stderr)
        sys.exit(2)
    print(f"[ok] wrote {out_dir}/train_3k.jsonl, eval_60.jsonl, dedup_report.json")


if __name__ == "__main__":
    main()
