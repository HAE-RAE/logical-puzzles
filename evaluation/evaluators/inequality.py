"""
Inequality Evaluator

Evaluates inequality puzzle responses with constraint-based fallback validation.
Answer format: digit sequence (e.g., "3142" or "5 3 12 4 10 1" for size>9).
"""

import logging
import re
import time
from typing import Dict, Any, Tuple, Optional, List, TYPE_CHECKING

from ..core.base import BaseEvaluator, EvaluationResult
from ..task_names import locale_from_task_name

if TYPE_CHECKING:
    from ..model.base import BaseLLMClient

logger = logging.getLogger(__name__)


class InequalityEvaluator(BaseEvaluator):
    """
    Inequality puzzle evaluator.

    Supports both concatenated (size<=9) and space-separated (size>9) formats.
    Falls back to constraint-based validation when the answer doesn't match
    the pre-computed value but still satisfies all inequality constraints.
    """

    SYSTEM_PROMPT = """### Instructions
You are an expert puzzle solver specializing in logical constraint puzzles.
You will be given an inequality puzzle where you must fill in the blanks with
numbers from 1 to N (where N is the total number of positions in the puzzle).

### Rules
1. Each number from 1 to N must be used exactly one time (it is a permutation of 1..N).
2. Adjacent positions are joined by an inequality symbol that must be satisfied:
   - `<` means the left number is strictly smaller than the right number.
   - `>` means the left number is strictly larger than the right number.
   - `?` means the inequality is hidden/unknown (it may be either `<` or `>`);
     you must still produce a valid assignment consistent with some choice.
3. Some positions may be pre-filled with a number (a hint). These numbers are
   fixed and must appear at that position in your answer.
4. A blank position is shown as an underscore (`_`) in the compact puzzle grid.
5. The compact grid alternates `value symbol value symbol value ...`, for
   example: `_ < 2 > _ < _`.

### Output format (CRITICAL)
Solve the puzzle step by step, then end your response with exactly one line:
Answer: <numbers from left to right>

- For puzzles of size <= 9, concatenate the digits (e.g., `Answer: 53241`).
- For puzzles of size > 9, separate numbers with single spaces
  (e.g., `Answer: 5 3 12 4 10 1`).

Do NOT omit the `Answer:` line. This line is how your solution is graded."""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 부등식 제약 논리 퍼즐 전문가입니다.
부등호 퍼즐이 주어지며, 1부터 N까지의 숫자로 빈칸을 모두 채워야 합니다
(N은 퍼즐의 전체 위치 수입니다).

### 규칙
1. 1부터 N까지의 각 숫자는 정확히 한 번씩만 사용되어야 합니다 (1..N의 순열).
2. 인접한 위치 사이의 부등호를 반드시 만족해야 합니다:
   - `<`: 왼쪽 숫자가 오른쪽 숫자보다 작음
   - `>`: 왼쪽 숫자가 오른쪽 숫자보다 큼
   - `?`: 부등호가 숨겨져 있음 (`<` 또는 `>` 중 어느 쪽이든 가능).
     이 경우에도 어떤 선택과 일관된 유효한 배치를 제시해야 합니다.
3. 일부 위치에는 숫자가 힌트로 미리 주어질 수 있습니다. 힌트 위치의 숫자는
   고정이며 답안에서도 그 위치에 그대로 나타나야 합니다.
4. 빈 위치는 퍼즐 그리드에서 밑줄(`_`)로 표시됩니다.
5. 퍼즐 그리드는 `값 기호 값 기호 값 ...` 순서로 번갈아 나타납니다.
   예: `_ < 2 > _ < _`.

### 출력 형식 (매우 중요)
단계별로 풀이한 뒤, 응답의 마지막 줄은 반드시 다음 형식이어야 합니다:
Answer: <왼쪽에서 오른쪽 순서의 숫자들>

- size <= 9: 숫자를 공백 없이 이어붙임 (예: `Answer: 53241`).
- size > 9: 숫자를 공백 한 칸으로 구분 (예: `Answer: 5 3 12 4 10 1`).

`Answer:` 줄을 절대 생략하지 마세요. 이 줄이 채점의 기준이 됩니다."""

    def _is_korean(self, puzzle: Optional[Dict] = None) -> bool:
        """Prefer task_name (e.g. …_ko_easy); else infer from expected answer."""
        task = getattr(self, "_task_name", None) or ""
        hint = locale_from_task_name(task)
        if hint is not None:
            return hint
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
            nums = re.findall(r'\d+', expected_answer)
            size = len(nums) if len(nums) > 1 else len(expected_answer)

        patterns = [
            r'Answer:\s*([\d\s]+)',
            r'answer:\s*([\d\s]+)',
            r'solution[:\s]+([\d\s]+)',
            r'sequence[:\s]+([\d\s]+)',
        ]

        def _normalize(nums):
            if len(nums) == 1 and size <= 9 and len(nums[0]) == size:
                return ''.join(nums[0])
            if len(nums) >= size:
                nums = nums[:size]
                return ' '.join(nums) if size > 9 else ''.join(nums)
            return None

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                raw = match.group(1).strip()
                nums = re.findall(r'\d+', raw)
                out = _normalize(nums)
                if out is not None:
                    return out

        last_part = response[-200:] if len(response) > 200 else response
        all_nums = re.findall(r'\d+', last_part)
        if size <= 9:
            for cand in reversed(all_nums):
                if len(cand) == size:
                    return cand
        if len(all_nums) >= size:
            return ' '.join(all_nums[-size:]) if size > 9 else ''.join(all_nums[-size:])

        return None

    def _to_int_list(self, s: str, size: int):
        """Convert answer string to list of ints, handling both formats."""
        if not s:
            return []
        nums = re.findall(r'\d+', s)
        if len(nums) == 1 and len(nums[0]) == size and size <= 9:
            return [int(d) for d in nums[0]]
        return [int(n) for n in nums]

    def _validate_inequality_solution(self, answer_str: str, puzzle: Dict) -> bool:
        """Validate an answer against inequality constraints.

        Prefers the schema fields `inequalities` (full list) + `hidden_inequalities`
        (indices that are hidden in display) when present; falls back to parsing
        the compact `problem` string.
        """
        size = puzzle.get("size", 0)

        parsed = self._to_int_list(answer_str, size)
        if len(parsed) != size:
            return False

        if sorted(parsed) != list(range(1, size + 1)):
            return False

        given_positions = puzzle.get("given_positions", [])
        given_values = puzzle.get("given_values", [])
        for pos, val in zip(given_positions, given_values):
            if pos < len(parsed) and parsed[pos] != val:
                return False

        inequalities = puzzle.get("inequalities")

        if inequalities:
            for i, ineq in enumerate(inequalities):
                if i + 1 >= len(parsed):
                    break
                if ineq == "<" and parsed[i] >= parsed[i + 1]:
                    return False
                if ineq == ">" and parsed[i] <= parsed[i + 1]:
                    return False
            return True

        problem_str = puzzle.get("problem", puzzle.get("solution", ""))
        if not problem_str:
            return False

        parts = problem_str.split()
        fallback_ineqs = parts[1::2]
        for i, ineq in enumerate(fallback_ineqs):
            if i + 1 >= len(parsed):
                break
            if ineq == "?":
                continue
            if ineq == "<" and parsed[i] >= parsed[i + 1]:
                return False
            if ineq == ">" and parsed[i] <= parsed[i + 1]:
                return False

        return True

    def _check_answer(
        self,
        expected: str,
        predicted: Optional[str],
        puzzle: Optional[Dict] = None,
    ) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0

        size = len(re.findall(r'\d+', str(expected)))
        if size <= 1:
            size = len(str(expected))

        expected_list = self._to_int_list(str(expected), size)
        predicted_list = self._to_int_list(str(predicted), size)

        correct = expected_list == predicted_list
        if correct:
            return True, 1.0

        # Fallback: accept if answer satisfies visible inequality constraints
        # (handles puzzles with hidden `?` inequalities where multiple visible
        # permutations could be consistent, though by construction visible
        # uniqueness should hold — this keeps the grader robust).
        if puzzle is not None and self._validate_inequality_solution(str(predicted), puzzle):
            return True, 1.0

        return False, 0.0

    def _process_response(
        self,
        puzzle: Dict[str, Any],
        response: str,
        latency_ms: float,
        usage: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """Override base _process_response so _check_answer can see the puzzle
        (needed for constraint-based fallback validation).
        """
        usage = usage or {}

        if "error" in usage:
            return self._create_error_result(
                puzzle,
                response if response else "",
                latency_ms,
                usage["error"],
            )

        try:
            predicted = self._parse_answer(response, puzzle)
            correct, partial_score = self._check_answer(
                puzzle["answer"], predicted, puzzle
            )
            return EvaluationResult(
                puzzle_id=puzzle["id"],
                difficulty=puzzle.get("difficulty", "Unknown"),
                correct=correct,
                partial_score=partial_score,
                expected=puzzle["answer"],
                predicted=predicted,
                raw_response=response,
                latency_ms=latency_ms,
                thinking_content=usage.get("thinking_content", "") if isinstance(usage, dict) else "",
            )
        except Exception as e:
            return self._create_error_result(puzzle, response, latency_ms, str(e))
