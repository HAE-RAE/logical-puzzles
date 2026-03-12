import logging
import re
import time
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING

from ..core.base import BaseEvaluator, EvaluationResult

if TYPE_CHECKING:
    from ..core.llm_client import UnifiedLLMClient

logger = logging.getLogger(__name__)


class FerrymanEvaluator(BaseEvaluator):
    """
    Ferryman puzzle evaluator (Korean + English)
    
    Korean answer format: X시간 Y분
    English answer format: X hours Y minutes
    """
    
    SYSTEM_PROMPT = """You are an expert at solving boat navigation problems.

### Rules
1. Analyze all given navigation regulations step by step.
2. Apply all speed limits, mandatory rest stops, and cargo regulations in your calculations.
3. After solving the problem, write your final answer in the following format: $\\boxed{X hours Y minutes}$.
4. Inside \\boxed{}, answer ONLY in "X hours Y minutes" format. Do not include other units or explanations.
"""

    KOREAN_SYSTEM_PROMPT = """당신은 뱃사공 운항 문제를 정확히 해결하는 전문가입니다.

### 규칙
1. 주어진 운항 규정을 모두 고려하여 단계별로 분석하세요.
2. 속도 제한, 의무 휴식, 화물 규정을 모두 적용하여 계산하세요.
3. 문제를 푼 후, 최종 답변을 다음과 같은 형식으로 작성하세요: $\\boxed{N시간 M분}$.
4. \\boxed{} 안에는 "X시간 Y분" 형식으로만 답하세요. 다른 단위나 설명은 포함하지 마세요.
"""

    def _is_korean(self, puzzle: Dict) -> bool:
        expected = puzzle.get("answer", "")
        return bool(re.search(r'[가-힣]', expected))

    def _get_system_prompt(self, puzzle: Dict) -> str:
        if self._is_korean(puzzle):
            return self.KOREAN_SYSTEM_PROMPT
        return self.SYSTEM_PROMPT

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[str]:
        """
        Extract time answer from LLM response and normalize.
        Routes to Korean or English parser based on puzzle answer format.
        """
        match = re.search(r'\\boxed\{([^}]+)\}', response)
        text = match.group(1).strip() if match else response

        if self._is_korean(puzzle):
            return self._normalize_time_korean(text)
        return self._normalize_time_english(text)

    def _normalize_time_korean(self, time_str: str) -> Optional[str]:
        match = re.search(r'(\d+)\s*시간\s*(\d+)\s*분', time_str)
        if match:
            return f"{int(match.group(1))}시간 {int(match.group(2))}분"
        numbers = re.findall(r'\d+', time_str)
        if len(numbers) >= 2:
            return f"{numbers[0]}시간 {numbers[1]}분"
        return None

    def _normalize_time_english(self, time_str: str) -> Optional[str]:
        match = re.search(
            r'(\d+)\s*hours?\s*(\d+)\s*minutes?', time_str, re.IGNORECASE)
        if match:
            return f"{int(match.group(1))} hours {int(match.group(2))} minutes"
        match = re.search(
            r'(\d+)\s*hr?s?\s*(\d+)\s*min', time_str, re.IGNORECASE)
        if match:
            return f"{int(match.group(1))} hours {int(match.group(2))} minutes"
        numbers = re.findall(r'\d+', time_str)
        if len(numbers) >= 2:
            return f"{numbers[0]} hours {numbers[1]} minutes"
        return None

    def _parse_time_to_minutes(self, time_str: str) -> Optional[int]:
        match = re.search(r'(\d+)\s*시간\s*(\d+)\s*분', time_str)
        if match:
            return int(match.group(1)) * 60 + int(match.group(2))
        match = re.search(
            r'(\d+)\s*hours?\s*(\d+)\s*minutes?', time_str, re.IGNORECASE)
        if match:
            return int(match.group(1)) * 60 + int(match.group(2))
        return None

    def _check_answer(
        self,
        expected: str,
        predicted: Optional[str]
    ) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0

        expected_minutes = self._parse_time_to_minutes(expected)
        predicted_minutes = self._parse_time_to_minutes(predicted)

        if expected_minutes is None or predicted_minutes is None:
            return False, 0.0

        correct = expected_minutes == predicted_minutes
        return correct, 1.0 if correct else 0.0

    def _evaluate_single(
        self,
        puzzle: Dict[str, Any],
        llm_client: "UnifiedLLMClient"
    ) -> "EvaluationResult":
        system_prompt = self._get_system_prompt(puzzle)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": puzzle["question"]}
        ]

        start = time.time()
        try:
            response, usage = llm_client.generate(messages)
            latency = (time.time() - start) * 1000
            return self._process_response(puzzle, response, latency, usage)
        except Exception as e:
            latency = (time.time() - start) * 1000
            return self._process_response(
                puzzle, "", latency, {"error": str(e)})

    async def _evaluate_async(
        self,
        puzzles: List[Dict[str, Any]],
        llm_client: "UnifiedLLMClient",
        verbose: bool = True,
        max_concurrent: int = 10
    ) -> List["EvaluationResult"]:
        from ..core.base import logger

        messages_list = []
        for puzzle in puzzles:
            system_prompt = self._get_system_prompt(puzzle)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": puzzle["question"]}
            ]
            messages_list.append(messages)

        total_puzzles = len(puzzles)
        task_name = getattr(self, '_task_name', None)
        task_prefix = f"[{task_name}] " if task_name else ""

        if verbose:
            logger.info(
                f"{task_prefix}Starting async evaluation: "
                f"{total_puzzles} puzzles, "
                f"max_concurrent={max_concurrent}")

        start_time = time.time()

        def progress_callback(completed, total):
            if verbose:
                pct = (completed / total) * 100
                if (completed % max(1, total // 10) == 0
                        or completed == total):
                    logger.info(
                        f"{task_prefix}API calls progress: "
                        f"{completed}/{total} ({pct:.0f}%)")

        responses = await llm_client.async_batch_generate(
            messages_list,
            max_concurrent=max_concurrent,
            progress_callback=progress_callback if verbose else None
        )
        total_latency = (time.time() - start_time) * 1000

        if verbose:
            logger.info(
                f"{task_prefix}API calls completed: "
                f"{total_puzzles}/{total_puzzles} in "
                f"{total_latency:.0f}ms "
                f"({total_latency/total_puzzles:.0f}ms per puzzle)")

        results = []
        correct_count = 0
        error_count = 0

        for puzzle, (response, usage) in zip(puzzles, responses):
            latency_ms = usage.get("latency_ms", 0)
            result = self._process_response(
                puzzle, response, latency_ms, usage)
            if result.correct:
                correct_count += 1
            if result.error:
                error_count += 1
            results.append(result)

        if verbose:
            incorrect = total_puzzles - correct_count - error_count
            logger.info(
                f"Processing completed: {correct_count} correct, "
                f"{incorrect} incorrect, {error_count} errors")

        return results
