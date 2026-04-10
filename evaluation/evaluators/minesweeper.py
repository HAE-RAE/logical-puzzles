"""
Minesweeper Evaluator

Evaluates minesweeper puzzle responses using weighted coordinate sum format.
Answer format: single integer (sum of row*C+col for each mine)
"""

import logging
import re
import time
from typing import List, Dict, Any, Tuple, Optional, Set, TYPE_CHECKING

from ..core.base import BaseEvaluator, EvaluationResult

if TYPE_CHECKING:
    from ..model.base import BaseLLMClient

logger = logging.getLogger(__name__)


class MinesweeperEvaluator(BaseEvaluator):
    """
    Minesweeper puzzle evaluator.

    Uses weighted coordinate sum scoring: sum(row * C + col) for each mine.
    """

    SYSTEM_PROMPT = """### Instructions
You are solving a Minesweeper puzzle.

### Rules
Analyze the grid using logical reasoning and deduce the exact location of all mines.

### Output format
Output the sum of linear indices (row * columns + col) for all mine positions as a single integer.

Answer: [number]"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
지뢰찾기 퍼즐을 풀고 있습니다.

### 규칙
격자를 논리적으로 분석하여 모든 지뢰 위치를 추론하세요.

### 출력 형식
각 지뢰 칸에 대해 (행 × 열 개수 + 열) 형태의 선형 인덱스를 모두 더한 값을 하나의 정수로 출력하세요.

답: [숫자]"""

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

    @staticmethod
    def _bitstring_to_coordinates(solution_str: str, R: int, C: int) -> Set[Tuple[int, int]]:
        """Convert solution bitstring to coordinate set."""
        coords = set()
        for i, cell in enumerate(solution_str):
            if cell == '1':
                r, c = divmod(i, C)
                coords.add((r, c))
        return coords

    @staticmethod
    def _compute_total_sum(coords: Set[Tuple[int, int]], C: int) -> int:
        """Compute weighted coordinate sum: sum(row * C + col)."""
        if not coords:
            return 0
        return sum(r * C + c for r, c in coords)

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[int]:
        """Parse LLM output to extract total sum as single integer."""
        # Remove code blocks
        response = re.sub(r'```[a-z]*\n?', '', response)
        response = re.sub(r'```', '', response)
        response = response.strip()

        # Priority 1: "Answer:" or "Output:" line with integer
        answer_matches = re.findall(
            r'(?:Answer|Output|Final\s*Answer|답)\s*[:\s]*(\d+)',
            response, re.IGNORECASE
        )
        if answer_matches:
            return int(answer_matches[-1])

        # Priority 2: last 5 lines for multi-digit integer
        lines = response.strip().split('\n')
        for line in reversed(lines[-5:]):
            match = re.search(r'\b(\d{2,})\b', line.strip())
            if match:
                return int(match.group(1))

        # Priority 3: last multi-digit integer in entire text
        matches = re.findall(r'\b(\d{2,})\b', response)
        if matches:
            return int(matches[-1])

        # Priority 4: single digit
        matches = re.findall(r'\b(\d+)\b', response)
        if matches:
            return int(matches[-1])

        return None

    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[int]
    ) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0

        # expected might be a bitstring or a pre-computed sum
        try:
            expected_num = int(expected)
        except (ValueError, TypeError):
            return False, 0.0

        correct = predicted == expected_num
        return correct, 1.0 if correct else 0.0
