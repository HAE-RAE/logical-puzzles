"""
Hanoi Evaluator

하노이 탑 퍼즐 평가
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


class HanoiEvaluator(BaseEvaluator):
    """
    Hanoi 퍼즐 평가자
    
    답변 형식: (disk, from, to) 튜플
    """
    
    SYSTEM_PROMPT = """### Instructions
You must answer ONLY in the format (disk, from, to).

### Rules
Follow the Hanoi puzzle given in the user message.

### Output format
(disk, from, to) — e.g. (1, 0, 2)"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
반드시 (원반 번호, 출발 기둥, 도착 기둥) 형식으로만 답하세요.

### 규칙
사용자 메시지에 주어진 하노이 퍼즐을 따르세요.

### 출력 형식
예: (1, 0, 2)"""

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

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[Tuple[int, int, int]]:
        """
        LLM 응답에서 (disk, from, to) 튜플 추출
        
        숫자 3개를 찾아서 튜플로 반환
        
        Args:
            response: LLM 응답 텍스트
            puzzle: 퍼즐 데이터 (사용하지 않음)
        """
        # 패턴 1: (숫자, 숫자, 숫자) 형식
        match = re.search(r'\((\d+),\s*(\d+),\s*(\d+)\)', response)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        
        # 패턴 2: 숫자 3개 추출
        nums = re.findall(r'\d+', response)
        if len(nums) >= 3:
            return (int(nums[0]), int(nums[1]), int(nums[2]))
        
        return None
    
    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[Tuple[int, int, int]]
    ) -> Tuple[bool, float]:
        """
        답변 확인
        
        Returns:
            (is_correct, partial_score) 튜플
        """
        if predicted is None:
            return False, 0.0
        
        # expected가 문자열일 수 있으므로 파싱
        if isinstance(expected, str):
            expected = self._parse_answer(expected, {})
        
        correct = predicted == expected
        return correct, 1.0 if correct else 0.0
