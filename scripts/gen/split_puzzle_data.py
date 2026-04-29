"""
퍼즐 데이터 Train/Test 분할

distillation 실험용 공정 비교를 위해 한 번만 분할하고,
naive/guided 양쪽 모두 동일한 test set을 사용한다.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _lib import PROJECT_ROOT

# 파일명 → task_name 매핑 (evaluator registry와 일치시킴)
FILENAME_TO_TASK = {
    "ferryman_en.jsonl": "ferryman_en",
    "hanoi_en.jsonl": "hanoi_en",
    "array_formula_en.jsonl": "array_formula_en",
    "yacht_dice.jsonl": "yacht_dice_en",  # 파일명에 _en 없음
}


def load_jsonl(path: Path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def stratified_split(items, train_ratio, rng):
    """difficulty별 균등 분할"""
    by_diff = defaultdict(list)
    for item in items:
        by_diff[item.get("difficulty", "unknown")].append(item)

    train, test = [], []
    for diff, group in sorted(by_diff.items()):
        rng.shuffle(group)
        split_idx = int(len(group) * train_ratio)
        train.extend(group[:split_idx])
        test.extend(group[split_idx:])

    return train, test


def save_jsonl(items, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Split puzzle data for distillation experiment")
    parser.add_argument("--data-dir", default="data/json", help="Source data directory")
    parser.add_argument("--output-dir", default="data/distill/split", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tasks", nargs="+",
        default=["ferryman_en", "hanoi_en", "array_formula_en", "yacht_dice_en"],
        help="Task names to split",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir

    # task_name → filename 역매핑
    task_to_file = {v: k for k, v in FILENAME_TO_TASK.items()}

    manifest = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "tasks": {},
    }

    for task_name in args.tasks:
        filename = task_to_file.get(task_name)
        if filename is None:
            print(f"[SKIP] Unknown task: {task_name}")
            continue

        jsonl_path = data_dir / filename
        if not jsonl_path.exists():
            print(f"[SKIP] File not found: {jsonl_path}")
            continue

        items = load_jsonl(jsonl_path)
        train, test = stratified_split(items, args.train_ratio, rng)

        save_jsonl(train, output_dir / "train" / f"{task_name}.jsonl")
        save_jsonl(test, output_dir / "test" / f"{task_name}.jsonl")

        manifest["tasks"][task_name] = {
            "source": str(jsonl_path),
            "total": len(items),
            "train": len(train),
            "test": len(test),
            "train_ids": [it.get("id", "") for it in train],
            "test_ids": [it.get("id", "") for it in test],
        }

        print(f"[OK] {task_name}: {len(items)} total -> {len(train)} train / {len(test)} test")

    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nManifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
