"""
Array Formula Evaluator

Excel 배열 수식 퍼즐 평가 (영어/한국어 지원)
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


class ArrayFormulaEvaluator(BaseEvaluator):
    """
    Array Formula 퍼즐 평가자

    숫자 또는 텍스트 답변, 한국어/영어 프롬프트 분기
    """

    SYSTEM_PROMPT = """### Instructions
You are a spreadsheet/Excel expert.
Analyze the given table data and answer the question accurately.

### Rules
1. For numeric results, answer with only the number (no units, commas, or currency symbols)
2. For decimals, truncate unless otherwise specified
3. For text answers, provide the exact value only
4. Briefly explain your reasoning, then end with "Final answer: [answer]"

### Output format
End your response with a line: Final answer: [answer]
"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 스프레드시트/Excel 전문가입니다.
주어진 표 데이터를 분석하고 질문에 정확하게 답하세요.

### 규칙
1. 숫자 결과는 숫자만 답하세요 (단위, 쉼표, 통화 기호 없이)
2. 소수점은 별도 지시가 없으면 버림 처리하세요
3. 텍스트 답변은 정확한 값만 작성하세요
4. 풀이 과정을 간략히 설명한 후, "최종 답: [답]"으로 마무리하세요

### 출력 형식
마지막에 "최종 답: [답]" 형식으로 끝내세요.
"""

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

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[Any]:
        """
        LLM 응답에서 답변 추출

        'Final answer:', 'Answer:', '최종 답:' 패턴을 순서대로 탐색하며,
        매칭 실패 시 마지막 줄을 fallback으로 사용.

        Args:
            response: LLM 응답 텍스트
            puzzle: 퍼즐 데이터 (answer_type 필드로 number/text 분기)

        Returns:
            int/float (number 타입) 또는 str (text 타입), 추출 실패 시 None
        """
        answer_type = puzzle.get("answer_type", "number")

        patterns = [
            r"[Ff]inal\s*[Aa]nswer\s*[:：]\s*(.+?)(?:\n|$)",
            r"[Aa]nswer\s*[:：]\s*(.+?)(?:\n|$)",
            r"최종\s*답\s*[:：]\s*(.+?)(?:\n|$)",
        ]

        answer_text = None
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer_text = match.group(1).strip()
                break

        # Fallback: extract from last line
        if answer_text is None:
            lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
            if lines:
                answer_text = lines[-1]

        if answer_text is None:
            return None

        # Number type processing
        if answer_type == "number":
            number_match = re.search(r"-?[\d,]+\.?\d*", answer_text.replace(",", ""))
            if number_match:
                try:
                    num_str = number_match.group().replace(",", "")
                    if "." in num_str:
                        return float(num_str)
                    return int(num_str)
                except ValueError:
                    pass
            return None

        # Text type
        answer_text = answer_text.strip("'\"")
        return answer_text

    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[Any]
    ) -> Tuple[bool, float]:
        """
        답변 확인

        Returns:
            (is_correct, partial_score) 튜플
        """
        if predicted is None:
            return False, 0.0

        answer_type = "number" if isinstance(predicted, (int, float)) else "text"

        if answer_type == "number":
            try:
                expected_num = float(expected)
                predicted_num = float(predicted)

                # 완전 일치만 정답으로 인정
                exact = abs(expected_num - predicted_num) < 0.001
                return exact, 1.0 if exact else 0.0
            except (ValueError, TypeError):
                return False, 0.0
        else:
            # Text comparison
            expected_str = str(expected).strip().lower()
            predicted_str = str(predicted).strip().lower()
            correct = expected_str == predicted_str
            return correct, 1.0 if correct else 0.0

    def _evaluate_single(
        self,
        puzzle: Dict[str, Any],
        llm_client: "BaseLLMClient"
    ) -> "EvaluationResult":
        """
        단일 퍼즐 평가 (한글/영문에 따라 적절한 SYSTEM_PROMPT 사용)
        """
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
            return self._process_response(puzzle, "", latency, {"error": str(e)})

    async def _evaluate_async(
        self,
        puzzles: List[Dict[str, Any]],
        llm_client: "BaseLLMClient",
        verbose: bool = True,
        max_concurrent: int = 10
    ) -> List["EvaluationResult"]:
        """
        비동기 평가 실행 (한글/영문에 따라 적절한 SYSTEM_PROMPT 사용)
        """
        from ..core.base import logger

        # 모든 메시지 준비 (각 퍼즐에 맞는 SYSTEM_PROMPT 사용)
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
            logger.info(f"{task_prefix}Starting async evaluation: {total_puzzles} puzzles, max_concurrent={max_concurrent}")

        # 비동기 배치 생성
        start_time = time.time()

        def progress_callback(completed, total):
            if verbose:
                percentage = (completed / total) * 100
                if completed % max(1, total // 10) == 0 or completed == total:
                    logger.info(f"{task_prefix}API calls progress: {completed}/{total} ({percentage:.0f}%)")

        responses = await llm_client.async_batch_generate(
            messages_list,
            max_concurrent=max_concurrent,
            progress_callback=progress_callback if verbose else None
        )
        total_latency = (time.time() - start_time) * 1000

        if verbose:
            logger.info(f"{task_prefix}API calls completed: {total_puzzles}/{total_puzzles} in {total_latency:.0f}ms ({total_latency/total_puzzles:.0f}ms per puzzle)")

        # 결과 처리
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
            logger.info(f"Processing completed: {correct_count} correct, {incorrect_count} incorrect, {error_count} errors")

        return results
