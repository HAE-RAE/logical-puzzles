"""
Causal DAG Evaluator

인과관계 DAG 추론 퍼즐 평가 (영문)
"""

import logging
import re
import time
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING

from ..core.base import BaseEvaluator, EvaluationResult

if TYPE_CHECKING:
    from ..model.base import BaseLLMClient

logger = logging.getLogger(__name__)


class CausalDAGEvaluator(BaseEvaluator):
    """
    Causal DAG 퍼즐 평가자
    
    답변 형식: 숫자 (분 단위)
    """
    
    SYSTEM_PROMPT = """### Instructions
You are a logical reasoning expert.

### Rules
Analyze the problem carefully and provide precise numerical answers.

### Output format
Give the numeric answer required by the user message (e.g. minutes)."""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 논리 추론 전문가입니다.

### 규칙
문제를 꼼꼼히 분석하고 분 단위 등 요구되는 정확한 수치 답을 제시하세요.

### 출력 형식
사용자 메시지에서 요구하는 수치 답(예: 분)을 제시하세요."""

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
        """
        LLM 응답에서 숫자 답변 추출 (분 단위)
        
        Args:
            response: LLM 응답 텍스트
            puzzle: 퍼즐 데이터 (사용하지 않음)
        """
        response = response.strip()
        
        # Pattern 1: Just a number
        if response.isdigit():
            return int(response)
        
        # Pattern 2: LaTeX boxed format: \boxed{45} or \\boxed{45}
        match = re.search(r'\\+boxed\{(\d+)\}', response)
        if match:
            return int(match.group(1))
        
        # Pattern 3: "Answer: 45" or "answer: 45"
        match = re.search(r'[Aa]nswer\s*[:：]\s*(\d+)', response)
        if match:
            return int(match.group(1))
        
        # Pattern 4: "event X first occurs at minute 45" or "occurs at minute 45"
        matches = list(re.finditer(r'(?:first\s+)?occurs?\s+at\s+minute\s+(\d+)', response, re.IGNORECASE))
        if matches:
            return int(matches[-1].group(1))
        
        # Pattern 5: "at minute 45" (last occurrence)
        matches = list(re.finditer(r'at\s+minute\s+(\d+)', response, re.IGNORECASE))
        if matches:
            return int(matches[-1].group(1))
        
        # Pattern 6: "minute 45" (last occurrence)
        matches = list(re.finditer(r'[Mm]inute\s+(\d+)', response))
        if matches:
            return int(matches[-1].group(1))
        
        # Pattern 7: "45 minutes"
        match = re.search(r'(\d+)\s+[Mm]inutes?', response)
        if match:
            return int(match.group(1))
        
        # Pattern 8: Last number in response
        numbers = re.findall(r'\b(\d+)\b', response)
        if numbers:
            return int(numbers[-1])
        
        return None
    
    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[int]
    ) -> Tuple[bool, float]:
        """
        답변 확인
        
        Returns:
            (is_correct, partial_score) 튜플
        """
        if predicted is None:
            return False, 0.0
        
        # expected가 문자열일 수 있으므로 정수로 변환
        try:
            expected_num = int(expected)
        except (ValueError, TypeError):
            return False, 0.0
        
        correct = predicted == expected_num
        return correct, 1.0 if correct else 0.0
