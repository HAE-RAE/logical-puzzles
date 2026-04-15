"""
Distillation 비교 리포트 생성

3개 모델(base, naive, guided)의 평가 결과를 비교한다.
results/ 디렉토리에서 JSON 결과를 수집하여 비교 테이블을 출력한다.
"""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_latest_result(results_dir: Path, task_name: str) -> dict | None:
    """주어진 task에 대한 가장 최근 JSON 결과 파일을 찾아 로드.
    ResultHandler는 output_dir/model_safe/task_name/ 구조로 저장하므로
    두 가지 패턴을 모두 탐색한다."""
    # Pattern 1: results_dir/*/task_name/*.json (with model subdir)
    # Pattern 2: results_dir/task_name/*.json (flat)
    json_files = []
    for pattern in [results_dir / "*" / task_name / "*.json",
                    results_dir / task_name / "*.json"]:
        json_files.extend(pattern.parent.parent.glob(f"*/{task_name}/*.json")
                          if "*" in str(pattern.parent) else
                          list(pattern.parent.glob("*.json")))

    # Simpler: just use rglob
    json_files = sorted(results_dir.rglob(f"*/{task_name}/*.json"),
                        key=lambda f: f.stat().st_mtime, reverse=True)
    if not json_files:
        # fallback: flat
        task_dir = results_dir / task_name
        if task_dir.exists():
            json_files = sorted(task_dir.glob("*.json"),
                                key=lambda f: f.stat().st_mtime, reverse=True)

    if not json_files:
        return None

    with open(json_files[0], "r", encoding="utf-8") as f:
        return json.load(f)


def collect_results(model_dirs: dict, tasks: list) -> dict:
    """모든 모델 × 태스크 조합에서 결과 수집"""
    results = {}
    for model_name, results_dir in model_dirs.items():
        results[model_name] = {}
        results_path = Path(results_dir) if Path(results_dir).is_absolute() else PROJECT_ROOT / results_dir
        for task in tasks:
            result = find_latest_result(results_path, task)
            if result:
                results[model_name][task] = result
    return results


def print_comparison_table(results: dict, tasks: list, model_names: list):
    """비교 테이블 출력"""
    # 헤더
    task_width = max(len(t) for t in tasks) + 2
    col_width = 14

    header = f"{'Task':<{task_width}}"
    for name in model_names:
        header += f" | {name:^{col_width}}"
    header += f" | {'Delta(G-N)':^{col_width}}"

    print("=" * len(header))
    print("DISTILLATION COMPARISON REPORT")
    print("=" * len(header))
    print()

    # Overall accuracy
    print("### Overall Accuracy")
    print(header)
    print("-" * len(header))

    for task in tasks:
        row = f"{task:<{task_width}}"
        accuracies = {}
        for name in model_names:
            task_result = results.get(name, {}).get(task)
            if task_result:
                acc = task_result.get("summary", {}).get("overall", {}).get("accuracy", 0)
                accuracies[name] = acc
                row += f" | {acc:^{col_width}.4f}"
            else:
                accuracies[name] = None
                row += f" | {'N/A':^{col_width}}"

        # Delta (guided - naive)
        guided_acc = accuracies.get("guided")
        naive_acc = accuracies.get("naive")
        if guided_acc is not None and naive_acc is not None:
            delta = guided_acc - naive_acc
            sign = "+" if delta >= 0 else ""
            row += f" | {sign}{delta:^{col_width-1}.4f}"
        else:
            row += f" | {'N/A':^{col_width}}"
        print(row)

    # By difficulty
    print()
    print("### Accuracy by Difficulty")
    for difficulty in ["easy", "medium", "hard"]:
        print(f"\n[{difficulty.upper()}]")
        print(header)
        print("-" * len(header))

        for task in tasks:
            row = f"{task:<{task_width}}"
            accuracies = {}
            for name in model_names:
                task_result = results.get(name, {}).get(task)
                if task_result:
                    diff_data = task_result.get("summary", {}).get("by_difficulty", {}).get(difficulty, {})
                    acc = diff_data.get("accuracy", 0)
                    accuracies[name] = acc
                    total = diff_data.get("total", 0)
                    row += f" | {acc:^{col_width}.4f}"
                else:
                    accuracies[name] = None
                    row += f" | {'N/A':^{col_width}}"

            guided_acc = accuracies.get("guided")
            naive_acc = accuracies.get("naive")
            if guided_acc is not None and naive_acc is not None:
                delta = guided_acc - naive_acc
                sign = "+" if delta >= 0 else ""
                row += f" | {sign}{delta:^{col_width-1}.4f}"
            else:
                row += f" | {'N/A':^{col_width}}"
            print(row)

    # 수율 통계 (if available)
    stats_path = PROJECT_ROOT / "data" / "distill" / "generation_stats.json"
    if stats_path.exists():
        print()
        print("### Distillation Yield Statistics")
        with open(stats_path) as f:
            stats = json.load(f)
        print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Compare distillation experiment results")
    parser.add_argument("--base-dir", default="results/Qwen_Qwen3-0.6B",
                        help="Base model results directory")
    parser.add_argument("--naive-dir", default="results/qwen3_0.6b_naive_distill",
                        help="Naive distillation model results directory")
    parser.add_argument("--guided-dir", default="results/qwen3_0.6b_guided_distill",
                        help="Guided distillation model results directory")
    parser.add_argument("--tasks", nargs="+",
                        default=["ferryman_en", "hanoi_en", "array_formula_en", "yacht_dice_en"])
    args = parser.parse_args()

    model_dirs = {
        "base": args.base_dir,
        "naive": args.naive_dir,
        "guided": args.guided_dir,
    }
    model_names = ["base", "naive", "guided"]

    results = collect_results(model_dirs, args.tasks)

    # 결과 있는지 확인
    has_data = any(results.get(m) for m in model_names)
    if not has_data:
        print("No results found. Run evaluations first.")
        print(f"Expected directories: {list(model_dirs.values())}")
        return

    print_comparison_table(results, args.tasks, model_names)

    # JSON 출력도 저장
    output_path = PROJECT_ROOT / "results" / "distillation_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison = {}
    for model_name in model_names:
        comparison[model_name] = {}
        for task in args.tasks:
            task_result = results.get(model_name, {}).get(task)
            if task_result:
                comparison[model_name][task] = task_result.get("summary", {})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\nComparison JSON saved: {output_path}")


if __name__ == "__main__":
    main()
