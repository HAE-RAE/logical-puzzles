import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from collections import defaultdict

from .base import EvaluationResult

logger = logging.getLogger(__name__)


class ResultHandler:
    """Evaluation result storage and aggregation"""
    
    def __init__(self, output_dir: str = "results"):
        """
        Args:
            output_dir: Output directory for results (relative to project root)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        task_name: str,
        results: List[EvaluationResult],
        model: str,
        puzzles: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Path]]:
        """
        Save results to CSV and JSON files (model/task folder structure)
        
        Args:
            task_name: Task name
            results: List of evaluation results
            model: Model name
            puzzles: Original puzzle data (includes question, answer)
            
        Returns:
            Dictionary of saved file paths {"csv": Path, "json": Path}
        """
        if not results:
            logger.warning(f"No results to save for {task_name}")
            return None
        
        model_safe = model.replace("/", "_").replace(":", "_")
        
        model_dir = self.output_dir / model_safe
        task_dir = model_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        summary = self._calculate_summary(task_name, results, model)
        accuracy = summary["accuracy"]
        
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        base_filename = f"{model_safe}_{task_name}_{timestamp}__{accuracy:.2f}"
        
        # CSV
        csv_path = task_dir / f"{base_filename}.csv"
        self._save_csv(csv_path, results, puzzles)
        
        # JSON
        json_path = task_dir / f"{base_filename}.json"
        self._save_json_summary(json_path, task_name, model, summary, timestamp)
        
        logger.info(f"Results saved (CSV): {csv_path.name}")
        logger.info(f"Results saved (JSON): {json_path.name}")
        
        return {"csv": csv_path, "json": json_path}
    
    def _save_csv(
        self,
        filepath: Path,
        results: List[EvaluationResult],
        puzzles: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Save detailed results to CSV file
        
        Args:
            filepath: CSV file path to save
            results: List of evaluation results
            puzzles: Original puzzle data (includes question, answer)
        """
        puzzle_map = {}
        if puzzles:
            for puzzle in puzzles:
                puzzle_map[puzzle["id"]] = puzzle
        
        rows = []
        for result in results:
            puzzle = puzzle_map.get(result.puzzle_id, {})
            
            row = {
                "id": result.puzzle_id,
                "question": puzzle.get("question", ""),
                "answer": str(result.expected),
                "resps": result.raw_response,
                "filtered_resps": str(result.predicted) if result.predicted is not None else "",  # Regex-extracted result
                "exact_match": 1 if result.correct else 0,
                "difficulty": result.difficulty
            }
            rows.append(row)
        
        if rows:
            fieldnames = ["id", "question", "answer", "resps", "filtered_resps", "exact_match", "difficulty"]
            with open(filepath, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    
    def _save_json_summary(
        self,
        filepath: Path,
        task_name: str,
        model: str,
        summary: Dict[str, Any],
        timestamp: str
    ) -> None:
        """
        Save summary by difficulty to JSON file
        
        Args:
            filepath: JSON file path to save
            task_name: Task name
            model: Model name
            summary: Summary statistics dictionary
            timestamp: Timestamp
        """
        output = {
            "metadata": {
                "task": task_name,
                "model": model,
                "timestamp": timestamp,
                "total_puzzles": summary["total_count"]
            },
            "summary": {
                "overall": {
                    "accuracy": summary["accuracy"],
                    "correct_count": summary["correct_count"],
                    "total_count": summary["total_count"],
                    "avg_latency_ms": summary["avg_latency_ms"]
                },
                "by_difficulty": summary["by_difficulty"]
            }
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    
    def _calculate_summary(
        self,
        task_name: str,
        results: List[EvaluationResult],
        model: str
    ) -> Dict[str, Any]:
        """
        Calculate evaluation summary
        
        Args:
            task_name: Task name (currently unused, reserved for future use)
            results: List of evaluation results
            model: Model name (currently unused, reserved for future use)
        
        Returns:
            Summary statistics dictionary
        """
        total = len(results)
        correct = sum(1 for r in results if r.correct)
        avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0
        
        # Statistics by difficulty (normalized to lowercase)
        by_difficulty = {}
        difficulties = set(r.difficulty.lower() if r.difficulty else "unknown" for r in results)
        
        for diff in difficulties:
            # Match original difficulty (case-insensitive)
            diff_results = [r for r in results if (r.difficulty or "").lower() == diff]
            if diff_results:
                diff_correct = sum(1 for r in diff_results if r.correct)
                
                by_difficulty[diff] = {
                    "total": len(diff_results),
                    "correct": diff_correct,
                    "accuracy": diff_correct / len(diff_results)
                }
        
        # Error statistics
        errors = [r.error for r in results if r.error]
        
        return {
            "accuracy": correct / total if total > 0 else 0,
            "correct_count": correct,
            "total_count": total,
            "avg_latency_ms": avg_latency,
            "by_difficulty": by_difficulty,
            "error_count": len(errors),
        }
    
    def load(self, filepath: Path) -> Dict[str, Any]:
        """
        Load saved result file
        
        Args:
            filepath: JSON file path
            
        Returns:
            Result dictionary
        """
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def aggregate_results(self, pattern: str = "*.json") -> Dict[str, List[Dict[str, Any]]]:
        """
        Aggregate multiple result files
        
        Args:
            pattern: File pattern (glob)
            
        Returns:
            Aggregated statistics (list of summaries by task)
        """
        aggregated = defaultdict(list)
        
        for filepath in self.output_dir.glob(pattern):
            data = self.load(filepath)
            task = data["metadata"]["task"]
            aggregated[task].append(data["summary"])
        
        return dict(aggregated)
