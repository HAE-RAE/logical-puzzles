import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import yaml

logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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

from evaluation.core import UnifiedLLMClient, ResultHandler
from evaluation.evaluators import get_evaluator, list_tasks


def load_config(config_path: Path) -> Dict:
    if not config_path.exists():
        return {}
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load config.yaml: {e}")
        return {}


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
                    continue
    
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
        return {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0
        }
    
    correct = sum(1 for r in results if r.correct)
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy
    }


def main() -> None:
    from evaluation.evaluators import list_tasks
    available_tasks = list_tasks()
    
    parser = argparse.ArgumentParser(
        description="Unified Puzzle Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Evaluate all tasks
  python evaluation/run.py
  python -m evaluation.run

  # Evaluate specific tasks only
  python evaluation/run.py --tasks kinship cipher

  # Use different model
  python evaluation/run.py --model gpt-4o

  # Filter by difficulty
  python evaluation/run.py --difficulty easy --limit 10

Available tasks: {', '.join(available_tasks)}
        """
    )
    
    parser.add_argument(
        "--model",
        default="gemini/gemini-3-flash-preview",
        help="LLM model (LiteLLM format)"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="List of tasks to evaluate (all if not specified)"
    )
    parser.add_argument(
        "--data-dir",
        default="data/json",
        help="Data directory path (relative to project root)"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results (relative to project root)"
    )
    parser.add_argument(
        "--difficulty",
        help="Difficulty filter (easy/medium/hard)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of puzzles to evaluate"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimize progress output"
    )
    parser.add_argument(
        "--async",
        action="store_true",
        dest="use_async",
        help="Run evaluation in async mode (default from config.yaml)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent executions in async mode (default: 10)"
    )
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    config_path = script_dir / "config.yaml"
    config = load_config(config_path)
    
    llm_config = config.get("llm", {})
    if llm_config.get("model"):
        parser.set_defaults(model=llm_config.get("model"))
    if config.get("data_dir"):
        parser.set_defaults(data_dir=config.get("data_dir"))
    if config.get("output_dir"):
        parser.set_defaults(output_dir=config.get("output_dir"))

    eval_config = config.get("evaluation", {})
    if eval_config.get("use_async") is not None:
        parser.set_defaults(use_async=eval_config.get("use_async"))
    if eval_config.get("max_concurrent") is not None:
        parser.set_defaults(max_concurrent=eval_config.get("max_concurrent"))
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    data_dir = _normalize_path(args.data_dir, project_root)
    output_dir = _normalize_path(args.output_dir, project_root)
    
    if args.limit is not None and args.limit < 0:
        logger.warning(f"Invalid limit value: {args.limit}. Using no limit.")
        args.limit = None
    
    logger.info(f"Initializing LLM client: {args.model}")
    llm_config = config.get("llm", {})
    llm_client = UnifiedLLMClient(
        model=args.model,
        temperature=llm_config.get("temperature", 1.0),
        max_tokens=llm_config.get("max_tokens", 65536),
        timeout=llm_config.get("timeout", 600.0),
        top_p=llm_config.get("top_p"),
        top_k=llm_config.get("top_k"),
        reasoning_effort=llm_config.get("reasoning_effort")
    )
    
    result_handler = ResultHandler(str(output_dir))
    
    tasks = args.tasks or list_tasks()
    
    logger.info("=" * 100)
    logger.info("Unified Evaluation System")
    logger.info("=" * 100)
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
