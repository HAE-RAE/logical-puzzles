"""
Inequality Evaluator

Evaluates inequality puzzle responses with constraint-based fallback validation.
Answer format: digit sequence (e.g., "3142" or "5 3 12 4 10 1" for size>9)
"""

import logging
import re
import time
from typing import Dict, Any, Tuple, Optional, List, TYPE_CHECKING

from ..core.base import BaseEvaluator, EvaluationResult

if TYPE_CHECKING:
    from ..model.base import BaseLLMClient

logger = logging.getLogger(__name__)


class InequalityEvaluator(BaseEvaluator):
    """
    Inequality puzzle evaluator.

    Supports both concatenated (size<=9) and space-separated (size>9) formats.
    Falls back to constraint-based validation when answer doesn't match pre-computed value.
    """

    SYSTEM_PROMPT = """### Instructions
You are an expert puzzle solver specializing in logical constraint puzzles.

### Rules
Solve the inequality puzzle by filling blanks with numbers.
Each number must be used exactly once.
Inequality symbols (< or >) between positions must be satisfied.

### Output format
Answer: [numbers separated by spaces]"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 부등식 제약 논리 퍼즐 전문가입니다.

### 규칙
빈칸을 숫자로 채워 퍼즐을 완성하세요. 각 숫자는 정확히 한 번씩만 사용합니다.
칸 사이의 부등호(< 또는 >)를 모두 만족해야 합니다.

### 출력 형식
Answer: [공백으로 구분된 숫자들]"""

    def _is_korean(self, puzzle: Optional[Dict] = None) -> bool:
        """Prefer task_name suffix (_ko / _en); else infer from expected answer."""
        task = getattr(self, "_task_name", None) or ""
        if task.endswith("_ko"):
            return True
        if task.endswith("_en"):
            return False
        if puzzle is not None:
            expected = puzzle.get("answer", "")
            return bool(re.search(r"[가-힣]", str(expected)))
        return False

    def _get_system_prompt(self, puzzle: Dict) -> str:
        if self._is_korean(puzzle):
            return self.KOREAN_SYSTEM_PROMPT
        return self.SYSTEM_PROMPT

    def _evaluate_single(
        self,
        puzzle: Dict[str, Any],
        llm_client: "BaseLLMClient",
    ) -> EvaluationResult:
        system_prompt = self._get_system_prompt(puzzle)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": puzzle["question"]},
        ]
        start = time.time()
        try:
            response, usage = llm_client.generate(messages)
            latency = (time.time() - start) * 1000
            return self._process_response(puzzle, response, latency, usage)
        except Exception as e:
            latency = (time.time() - start) * 1000
            return self._process_response(puzzle, "", latency, {"error": str(e)})

    async def _evaluate_async(
        self,
        puzzles: List[Dict[str, Any]],
        llm_client: "BaseLLMClient",
        verbose: bool = True,
        max_concurrent: int = 10,
    ) -> List[EvaluationResult]:
        from ..core.base import logger

        messages_list = []
        for puzzle in puzzles:
            system_prompt = self._get_system_prompt(puzzle)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": puzzle["question"]},
            ]
            messages_list.append(messages)

        total_puzzles = len(puzzles)
        task_name = getattr(self, "_task_name", None)
        task_prefix = f"[{task_name}] " if task_name else ""

        if verbose:
            logger.info(
                f"{task_prefix}Starting async evaluation: {total_puzzles} puzzles, "
                f"max_concurrent={max_concurrent}"
            )

        start_time = time.time()

        def progress_callback(completed, total):
            if verbose:
                percentage = (completed / total) * 100
                if completed % max(1, total // 10) == 0 or completed == total:
                    logger.info(
                        f"{task_prefix}API calls progress: {completed}/{total} ({percentage:.0f}%)"
                    )

        responses = await llm_client.async_batch_generate(
            messages_list,
            max_concurrent=max_concurrent,
            progress_callback=progress_callback if verbose else None,
        )
        total_latency = (time.time() - start_time) * 1000

        if verbose:
            logger.info(
                f"{task_prefix}API calls completed: {total_puzzles}/{total_puzzles} in "
                f"{total_latency:.0f}ms ({total_latency/total_puzzles:.0f}ms per puzzle)"
            )

        results = []
        correct_count = 0
        error_count = 0

        for puzzle, (response, usage) in zip(puzzles, responses):
            latency_ms = usage.get("latency_ms", 0)
            result = self._process_response(puzzle, response, latency_ms, usage)
            if result.correct:
                correct_count += 1
            if result.error:
                error_count += 1
            results.append(result)

        if verbose:
            incorrect_count = total_puzzles - correct_count - error_count
            logger.info(
                f"Processing completed: {correct_count} correct, {incorrect_count} incorrect, "
                f"{error_count} errors"
            )

        return results

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[str]:
        """Extract number sequence from LLM response."""
        expected_answer = puzzle.get("answer", "")
        size = puzzle.get("size", 0)
        if not size and expected_answer:
            # Infer size from answer
            nums = re.findall(r'\d+', expected_answer)
            size = len(nums) if len(nums) > 1 else len(expected_answer)

        # Priority 1: "Answer:" pattern
        patterns = [
            r'Answer:\s*([\d\s]+)',
            r'answer:\s*([\d\s]+)',
            r'solution[:\s]+([\d\s]+)',
            r'sequence[:\s]+([\d\s]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                raw = match.group(1).strip()
                nums = re.findall(r'\d+', raw)
                if len(nums) >= size:
                    nums = nums[:size]
                    if size > 9:
                        return ' '.join(nums)
                    return ''.join(nums)

        # Priority 2: last numeric sequence in last 200 chars
        last_part = response[-200:] if len(response) > 200 else response
        all_nums = re.findall(r'\d+', last_part)
        if len(all_nums) >= size:
            nums = all_nums[-size:]
            if size > 9:
                return ' '.join(nums)
            return ''.join(nums)

        return None

    def _to_int_list(self, s: str, size: int):
        """Convert answer string to list of ints, handling both formats."""
        if not s:
            return []
        nums = re.findall(r'\d+', s)
        # Concatenated single digits (e.g. "641532" for size 6)
        if len(nums) == 1 and len(nums[0]) == size and size <= 9:
            return [int(d) for d in nums[0]]
        return [int(n) for n in nums]

    def _validate_inequality_solution(self, answer_str: str, puzzle: Dict) -> bool:
        """Validate if answer satisfies inequality constraints directly."""
        size = puzzle.get("size", 0)

        # Parse answer to int list
        parsed = self._to_int_list(answer_str, size)
        if len(parsed) != size:
            return False

        # Check permutation of 1..N
        if sorted(parsed) != list(range(1, size + 1)):
            return False

        # Check given numbers
        given_positions = puzzle.get("given_positions", [])
        given_values = puzzle.get("given_values", [])
        for pos, val in zip(given_positions, given_values):
            if pos < len(parsed) and parsed[pos] != val:
                return False

        # Parse inequalities from problem string
        problem_str = puzzle.get("problem", puzzle.get("solution", ""))
        if not problem_str:
            return False

        parts = problem_str.split()
        inequalities = parts[1::2]

        for i, ineq in enumerate(inequalities):
            if i + 1 >= len(parsed):
                break
            if ineq == "?":
                continue
            if ineq == "<" and parsed[i] >= parsed[i + 1]:
                return False
            elif ineq == ">" and parsed[i] <= parsed[i + 1]:
                return False

        return True

    def _check_answer(
        self,
        expected: str,
        predicted: Optional[str]
    ) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0

        # Normalize both to int lists for comparison
        size = len(re.findall(r'\d+', str(expected)))
        if size <= 1:
            size = len(str(expected))

        expected_list = self._to_int_list(str(expected), size)
        predicted_list = self._to_int_list(str(predicted), size)

        correct = expected_list == predicted_list
        return correct, 1.0 if correct else 0.0
