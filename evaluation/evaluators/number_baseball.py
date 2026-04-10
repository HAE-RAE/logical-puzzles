"""
Number Baseball Evaluator

Evaluates number baseball (Bulls and Cows) puzzle responses
with constraint-based fallback validation.
Answer format: digit string (e.g., "1234" or "012345")
"""

import logging
import re
import time
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING

from ..core.base import BaseEvaluator, EvaluationResult

if TYPE_CHECKING:
    from ..model.base import BaseLLMClient

logger = logging.getLogger(__name__)


class NumberBaseballEvaluator(BaseEvaluator):
    """
    Number Baseball puzzle evaluator.

    Falls back to constraint-based validation (checking all hints)
    when answer doesn't match pre-computed value.
    """

    SYSTEM_PROMPT = """### Instructions
You are an expert puzzle solver specializing in logical deduction games like Bulls and Cows (Number Baseball).

### Rules
- "Strike" means a digit is correct AND in the correct position
- "Ball" means a digit is correct BUT in the wrong position
- Find the secret number that satisfies ALL hints

### Output format
Answer: [the secret number]"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 숫자 야구(불스 앤 카우즈) 논리 추론 전문가입니다.

### 규칙
- 스트라이크: 숫자와 위치가 모두 맞음
- 볼: 숫자는 맞지만 위치가 틀림
- 모든 힌트를 만족하는 비밀 숫자를 찾으세요

### 출력 형식
Answer: [비밀 숫자]"""

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

    def _extract_hint_numbers(self, puzzle: Dict) -> set:
        """Extract hint guess numbers to use as blacklist for parsing."""
        hint_nums = set()
        hints = puzzle.get("hints", [])
        for h in hints:
            if isinstance(h, dict):
                hint_nums.add(h.get("guess", ""))
            elif isinstance(h, str):
                nums = re.findall(r'\d+', h)
                hint_nums.update(nums)
        return hint_nums

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[str]:
        """Extract digit sequence from LLM response, filtering out hint numbers."""
        num_digits = puzzle.get("num_digits", 3)
        hint_numbers = self._extract_hint_numbers(puzzle)

        # Priority 1: "Answer:" pattern
        patterns = [
            r'Answer:\s*(\d+)',
            r'answer:\s*(\d+)',
            r'secret number[:\s]+(\d+)',
            r'number is[:\s]+(\d+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for m in reversed(matches):
                if len(m) == num_digits and m not in hint_numbers:
                    return m

        # Priority 2: last N-digit number not in hints
        last_part = response[-500:] if len(response) > 500 else response
        numbers = re.findall(rf'\b(\d{{{num_digits}}})\b', last_part)
        for n in reversed(numbers):
            if n not in hint_numbers:
                return n

        # Priority 3: any N-digit number
        if numbers:
            return numbers[-1]

        return None

    def _calculate_strikes_balls(self, secret: str, guess: str) -> Tuple[int, int]:
        """Calculate strikes and balls."""
        strikes = sum(1 for s, g in zip(secret, guess) if s == g)
        balls = sum(1 for i, g in enumerate(guess) if g != secret[i] and g in secret)
        return strikes, balls

    def _validate_solution(self, answer: str, puzzle: Dict) -> bool:
        """Validate if answer satisfies all hints."""
        num_digits = puzzle.get("num_digits", 0)
        if not answer or len(answer) != num_digits:
            return False
        if len(set(answer)) != num_digits:
            return False
        if not all(c.isdigit() for c in answer):
            return False

        hints = puzzle.get("hints", [])
        for hint in hints:
            if isinstance(hint, dict):
                guess = hint["guess"]
                expected_s = hint["strikes"]
                expected_b = hint["balls"]
                s, b = self._calculate_strikes_balls(answer, guess)
                if s != expected_s or b != expected_b:
                    return False

        return True

    def _check_answer(
        self,
        expected: str,
        predicted: Optional[str]
    ) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0

        correct = str(predicted) == str(expected)
        return correct, 1.0 if correct else 0.0
