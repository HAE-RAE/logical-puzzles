"""
Yacht Dice Evaluator

Evaluates Yacht Dice puzzle responses.
Answer format: integer (total score or spotcheck round sum)
"""

import logging
import re
import time
from typing import Dict, Any, Tuple, Optional, List, TYPE_CHECKING

from ..core.base import BaseEvaluator, EvaluationResult

if TYPE_CHECKING:
    from ..model.base import BaseLLMClient

logger = logging.getLogger(__name__)


class YachtDiceEvaluator(BaseEvaluator):
    """
    Yacht Dice puzzle evaluator.

    Parses total score or spotcheck sum from LLM response.
    """

    SYSTEM_PROMPT = """### Instructions
You are an expert at solving Yacht Dice optimization problems.

### Rules
Yacht Dice is a dice game where you roll 5 dice for 12 rounds and assign each round to a scoring category.

Scoring Categories:
- Aces through Sixes: Sum of dice showing that number
- Three-of-a-Kind: Sum of all dice if at least 3 match
- Four-of-a-Kind: Sum of all dice if at least 4 match
- Full House: 25 points for exactly 3 of one number and 2 of another
- Small Straight: 30 points for 4 consecutive numbers
- Large Straight: 40 points for 5 consecutive numbers
- Yacht: 50 points for all 5 dice showing the same number

Upper Section Bonus: If the sum of Aces through Sixes is 63 or more, add 35 bonus points.

Each category can only be used once.

### Output format
CRITICAL: Your very last line MUST be in this exact format:
Answer: [number]"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 요트 다이스(Yacht Dice) 최적화 문제를 푸는 전문가입니다.

### 규칙
요트 다이스는 5개의 주사위를 12라운드 굴리고, 각 라운드를 점수 칸에 한 번씩 배정하는 게임입니다.

점수 칸:
- 에이스~식스: 해당 숫자가 나온 주사위 눈의 합
- 쓰리 카인드 / 포 카인드: 조건을 만족하면 주사위 5개 눈의 합
- 풀하우스: 정확히 3개와 2개 조합이면 25점
- 스몰 스트레이트: 연속 4개면 30점
- 라지 스트레이트: 연속 5개면 40점
- 요트: 5개가 모두 같으면 50점

상단 보너스: 에이스~식스 합이 63 이상이면 35점 보너스.

각 칸은 한 번만 사용할 수 있습니다.

### 출력 형식
반드시 마지막 줄만 아래 형식이어야 합니다:
Answer: [숫자]"""

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

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[int]:
        """Extract integer answer from LLM response."""
        # Remove code blocks
        response = re.sub(r'```[a-z]*\n?', '', response)
        response = re.sub(r'```', '', response)
        response = response.strip()

        # Priority 1: "Answer:" pattern
        answer_matches = re.findall(
            r'(?:Answer|Output|Final\s*Answer)\s*[:\s]*(\d+)',
            response, re.IGNORECASE
        )
        if answer_matches:
            return int(answer_matches[-1])

        # Priority 2: Total/sum patterns
        total_patterns = [
            r'[Tt]otal[:\s]*[=\s]*(\d+)',
            r'[Ss]um[:\s]*[=\s]*(\d+)',
        ]
        for pattern in total_patterns:
            matches = re.findall(pattern, response)
            if matches:
                return int(matches[-1])

        # Priority 3: last number in last 5 lines
        lines = response.strip().split('\n')
        for line in reversed(lines[-5:]):
            nums = re.findall(r'\b(\d+)\b', line.strip())
            if nums:
                # Pick the largest number (likely total score)
                return int(max(nums, key=int))

        # Priority 4: last number anywhere
        all_nums = re.findall(r'\b(\d+)\b', response)
        if all_nums:
            return int(all_nums[-1])

        return None

    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[int]
    ) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0

        try:
            expected_num = int(expected)
        except (ValueError, TypeError):
            return False, 0.0

        correct = predicted == expected_num
        return correct, 1.0 if correct else 0.0
