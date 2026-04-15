"""
Distillation 결과 시각화

baseline (학습 전) / naive SFT / guided SFT 세 조건의 accuracy를 모듈별로 비교하는
막대 그래프를 생성한다.

입력 경로 (각 경로 아래 {task}/<timestamp>__<acc>.json 형태):
  - results/qwen3_4b_baseline/{task}/
  - results/qwen3_4b_naive_{task}/{task}/
  - results/qwen3_4b_guided_{task}/{task}/

출력: results/distill_comparison.png, results/distill_comparison.csv
"""

import argparse
import csv
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
TASKS = ["ferryman_en", "array_formula_en"]


def latest_accuracy(result_dir: Path, task: str):
    """result_dir 아래 최신 JSON에서 accuracy 추출 (없으면 None)"""
    patterns = [
        str(result_dir / "**" / task / "*.json"),  # ResultHandler의 {model}/{task}/ 포맷
        str(result_dir / task / "*.json"),
        str(result_dir / "*.json"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    # config_experiment.json 등 메타파일 제외
    files = [f for f in files if not f.split("/")[-1].startswith("config_")]
    if not files:
        return None
    latest = sorted(files)[-1]
    with open(latest) as f:
        d = json.load(f)
    overall = d.get("summary", {}).get("overall", {})
    return overall.get("accuracy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output", default="results/distill_comparison.png")
    parser.add_argument("--csv", default="results/distill_comparison.csv")
    args = parser.parse_args()

    root = PROJECT_ROOT / args.results_root

    rows = []  # (task, condition, accuracy)
    for task in TASKS:
        baseline = latest_accuracy(root / "qwen3_4b_baseline", task)
        naive = latest_accuracy(root / f"qwen3_4b_naive_{task}", task)
        guided = latest_accuracy(root / f"qwen3_4b_guided_{task}", task)
        rows.append((task, "baseline", baseline))
        rows.append((task, "naive_sft", naive))
        rows.append((task, "guided_sft", guided))
        print(f"{task:<22} baseline={baseline}  naive={naive}  guided={guided}")

    # CSV 저장
    csv_path = PROJECT_ROOT / args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "condition", "accuracy"])
        for r in rows:
            w.writerow([r[0], r[1], "" if r[2] is None else f"{r[2]:.4f}"])
    print(f"\nCSV saved: {csv_path}")

    # 플롯 (그룹 막대)
    conditions = ["baseline", "naive_sft", "guided_sft"]
    colors = ["#888888", "#1f77b4", "#ff7f0e"]
    n_tasks = len(TASKS)
    n_cond = len(conditions)

    x = np.arange(n_tasks)
    width = 0.26

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, cond in enumerate(conditions):
        ys = []
        for task in TASKS:
            v = next((r[2] for r in rows if r[0] == task and r[1] == cond), None)
            ys.append(0 if v is None else v)
        bars = ax.bar(x + (i - 1) * width, ys, width, label=cond, color=colors[i])
        for b, v in zip(bars, ys):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS)
    ax.set_ylabel("Accuracy")
    ax.set_title("Qwen3-4B: Pre- vs Post-SFT (Naive / Guided) on test split (21/task)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()

    out = PROJECT_ROOT / args.output
    plt.savefig(out, dpi=150)
    print(f"Plot saved: {out}")


if __name__ == "__main__":
    main()
