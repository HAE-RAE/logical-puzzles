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
from ..task_names import locale_from_task_name

if TYPE_CHECKING:
    from ..model.base import BaseLLMClient

logger = logging.getLogger(__name__)


class NumberBaseballEvaluator(BaseEvaluator):
    """
    Number Baseball puzzle evaluator.

    Falls back to constraint-based validation (checking all hints)
    when predicted answer doesn't match expected answer exactly.
    """

    SYSTEM_PROMPT = (
        "You are an expert logical reasoning assistant specialized in solving "
        "Bulls and Cows (Number Baseball) puzzles.\n"
        "\n"
        "Rules of Bulls and Cows:\n"
        "- The secret is a sequence of unique digits (0-9). Leading zeros are "
        "allowed (e.g., '012' is a valid 3-digit secret)\n"
        "- A \"Strike\" (S) means a correct digit in the correct position\n"
        "- A \"Ball\" (B) means a correct digit in the wrong position\n"
        "- You need to find the unique number that satisfies ALL given hints\n"
        "\n"
        "Approach:\n"
        "1. Analyze each hint to understand what digits are in the secret number\n"
        "2. Use the strike and ball counts to determine positions\n"
        "3. Apply logical deduction to eliminate impossible combinations\n"
        "4. Verify your answer against all hints before responding\n"
        "\n"
        "IMPORTANT: Provide your final answer after \"Answer:\" as a single "
        "number with no additional text or formatting."
    )

    KOREAN_SYSTEM_PROMPT = (
        "당신은 숫자 야구(Bulls and Cows) 퍼즐을 푸는 논리적 추론 전문가입니다.\n"
        "\n"
        "숫자 야구 규칙:\n"
        "- 비밀 숫자는 서로 다른 자릿수(0-9)로 이루어진 수열입니다. 앞자리 0이 "
        "허용됩니다 (예: '012'는 유효한 3자리 비밀 숫자)\n"
        "- \"스트라이크\"(S)는 올바른 숫자가 올바른 위치에 있음을 의미합니다\n"
        "- \"볼\"(B)은 올바른 숫자가 잘못된 위치에 있음을 의미합니다\n"
        "- 주어진 모든 힌트를 만족하는 유일한 숫자를 찾아야 합니다\n"
        "\n"
        "풀이 접근법:\n"
        "1. 각 힌트를 분석하여 비밀 숫자에 포함된 숫자를 파악합니다\n"
        "2. 스트라이크와 볼 개수를 이용하여 위치를 결정합니다\n"
        "3. 논리적 추론으로 불가능한 조합을 제거합니다\n"
        "4. 답을 제출하기 전에 모든 힌트에 대해 검증합니다\n"
        "\n"
        "중요: \"Answer:\" 뒤에 추가 텍스트나 서식 없이 숫자만 최종 답으로 "
        "제시하세요."
    )

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
        answer_text = self._extract_final_answer_text(response, allow_boxed_fallback=False) or response

        # Priority 1: "Answer:" pattern
        patterns = [
            r'Answer:\s*(\d+)',
            r'answer:\s*(\d+)',
            r'secret number[:\s]+(\d+)',
            r'number is[:\s]+(\d+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, answer_text, re.IGNORECASE)
            for m in reversed(matches):
                if len(m) == num_digits and m not in hint_numbers:
                    return m

        # Priority 2: last N-digit number not in hints
        last_part = answer_text[-500:] if len(answer_text) > 500 else answer_text
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
        """Validate if answer satisfies all hints (constraint-based fallback)."""
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
