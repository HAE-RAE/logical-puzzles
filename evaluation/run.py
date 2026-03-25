import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.core import ResultHandler
from evaluation.model import create_client
from evaluation.evaluators import get_evaluator, list_tasks


def parse_gen_kwargs(raw: str) -> dict:
    """'key=val,key=val,...' 문자열을 dict로 파싱한다.

    - reasoning=on/off → enable_thinking=True/False 매핑
    - 숫자는 int/float로 자동 변환
    - True/False 문자열은 bool로 변환
    """
    if not raw:
        return {}
    result = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        k, v = k.strip(), v.strip()

        if k == "reasoning":
            result["enable_thinking"] = v.lower() in ("on", "true", "1")
            continue

        if v.lower() == "true":
            v = True
        elif v.lower() == "false":
            v = False
        else:
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
        result[k] = v
    return result


def load_puzzles(jsonl_path: Path) -> List[Dict]:
    puzzles = []
    if not jsonl_path.exists():
        logger.warning(f"Data file not found: {jsonl_path}")
        return puzzles
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    puzzles.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
    return puzzles


def filter_puzzles(
    puzzles: List[Dict],
    difficulty: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict]:
    filtered = puzzles
    if difficulty:
        difficulty_lower = difficulty.lower()
        filtered = [
            p for p in filtered
            if p.get("difficulty", "").lower() == difficulty_lower
        ]
    if limit is not None and limit > 0:
        filtered = filtered[:limit]
    return filtered


def _normalize_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return project_root / path_str.lstrip("../")


def _calculate_task_summary(results: List) -> Dict[str, float]:
    if not results:
        return {"total": 0, "correct": 0, "accuracy": 0.0}
    correct = sum(1 for r in results if r.correct)
    total = len(results)
    return {"total": total, "correct": correct, "accuracy": correct / total if total > 0 else 0.0}


def main() -> None:
    available_tasks = list_tasks()

    parser = argparse.ArgumentParser(
        description="Unified Puzzle Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # liteLLM mode
  python evaluation/run.py --model gemini/gemini-3-flash-preview \\
      --model_router litellm \\
      --gen-kwargs temperature=1.0,max_tokens=65536,top_p=0.95,top_k=64 \\
      --tasks kinship --async

  # Remote server mode (Colab etc.)
  python evaluation/run.py --model Qwen/Qwen3-0.6B \\
      --model_router remote --remote_url https://xxxx.ngrok-free.app \\
      --gen-kwargs temperature=0.6,max_tokens=32768,top_p=0.95,top_k=20,reasoning=on \\
      --tasks kinship --async --max-concurrent 5

Available tasks: {', '.join(available_tasks)}
        """
    )

    parser.add_argument("--model", required=True, help="Model name (liteLLM format or display name for remote server)")
    parser.add_argument("--model_router", required=True, choices=["litellm", "remote"], help="Backend to use: 'litellm' for liteLLM library, 'remote' for custom API server (Colab etc.)")
    parser.add_argument("--remote_url", default=None, help="Remote server URL (required when --model_router remote)")
    parser.add_argument("--gen-kwargs", default=None, help="Generation params as key=value pairs (e.g. temperature=0.6,max_tokens=32768,reasoning=on)")
    parser.add_argument("--timeout", type=float, default=None, help="Request timeout in seconds (default: 120 for remote, 600 for litellm)")
    parser.add_argument("--tasks", nargs="+", help="List of tasks to evaluate (all if not specified)")
    parser.add_argument("--data-dir", default="data/json", help="Data directory path")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    parser.add_argument("--difficulty", help="Difficulty filter (easy/medium/hard)")
    parser.add_argument("--limit", type=int, help="Maximum number of puzzles to evaluate")
    parser.add_argument("--quiet", action="store_true", help="Minimize progress output")
    parser.add_argument("--async", action="store_true", dest="use_async", help="Run evaluation in async mode")
    parser.add_argument("--max-concurrent", type=int, default=30, help="Maximum concurrent executions in async mode (default: 30)")

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = _normalize_path(args.data_dir, project_root)
    output_dir = _normalize_path(args.output_dir, project_root)

    if args.limit is not None and args.limit < 0:
        logger.warning(f"Invalid limit value: {args.limit}. Using no limit.")
        args.limit = None

    use_remote = args.model_router == "remote"

    if use_remote and not args.remote_url:
        parser.error("--remote_url is required when --model_router is 'remote'")

    gen_kwargs = parse_gen_kwargs(args.gen_kwargs)
    timeout = args.timeout or 600.0

    if use_remote:
        logger.info(f"Initializing LLM client (remote mode): {args.remote_url}")
    else:
        logger.info(f"Initializing LLM client (liteLLM mode): {args.model}")
    if gen_kwargs:
        logger.info(f"gen_kwargs: {gen_kwargs}")

    remote_url = args.remote_url if use_remote else None
    llm_client = create_client(
        model=args.model,
        timeout=timeout,
        remote_url=remote_url,
        gen_kwargs=gen_kwargs,
    )

    result_handler = ResultHandler(str(output_dir))

    tasks = args.tasks or list_tasks()

    logger.info("=" * 100)
    logger.info("Unified Evaluation System")
    logger.info("=" * 100)
    if use_remote:
        logger.info(f"Mode: remote ({args.remote_url})")
    else:
        logger.info(f"Mode: liteLLM")
    logger.info(f"Model: {args.model}")
    logger.info(f"Tasks: {len(tasks)} tasks - {', '.join(tasks)}")
    if args.difficulty:
        logger.info(f"Difficulty filter: {args.difficulty}")
    if args.limit:
        logger.info(f"Limit: {args.limit} puzzles per task")

    all_summaries: Dict[str, Dict[str, float]] = {}
    failed_tasks: Dict[str, str] = {}

    for task_name in tasks:
        print()
        logger.info("=" * 100)
        logger.info(f"Task: {task_name}")
        logger.info("=" * 100)

        data_path = data_dir / f"{task_name}.jsonl"
        puzzles = load_puzzles(data_path)

        if not puzzles:
            logger.warning(f"No puzzles loaded from {data_path}. Skipping...")
            continue

        puzzles = filter_puzzles(puzzles, args.difficulty, args.limit)
        logger.info(f"Loaded {len(puzzles)} puzzles")

        if len(puzzles) == 0:
            if not args.quiet:
                logger.warning("No puzzles after filtering. Skipping...")
            continue

        try:
            evaluator = get_evaluator(task_name)

            evaluate_kwargs = {
                "verbose": not args.quiet,
                "use_async": args.use_async,
                "max_concurrent": args.max_concurrent,
                "task_name": task_name
            }
            results = evaluator.evaluate(
                puzzles,
                llm_client,
                **evaluate_kwargs
            )

            result_handler.save(task_name, results, args.model, puzzles)
            summary = _calculate_task_summary(results)
            logger.info(f"Accuracy: {summary['correct']}/{summary['total']} ({summary['accuracy']:.1%})")
            all_summaries[task_name] = summary

        except Exception as e:
            logger.error(f"Error evaluating {task_name}: {e}")
            import traceback
            if not args.quiet:
                logger.debug(traceback.format_exc())
            failed_tasks[task_name] = str(e)
            continue

    print()
    logger.info("=" * 100)
    logger.info("Overall Summary")
    logger.info("=" * 100)

    if all_summaries:
        for task, stats in all_summaries.items():
            logger.info(f"{task:25s}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

    if failed_tasks:
        logger.warning(f"Failed tasks ({len(failed_tasks)}):")
        for task, error in failed_tasks.items():
            logger.warning(f"  {task:25s}: {error}")

    logger.info("=" * 100)
    total_attempted = len(all_summaries) + len(failed_tasks)
    if all_summaries:
        logger.info(f"Evaluation completed: {len(all_summaries)} succeeded, {len(failed_tasks)} failed (total: {total_attempted})")
    else:
        logger.warning(f"No tasks were successfully evaluated ({len(failed_tasks)} failed)")


if __name__ == "__main__":
    main()
