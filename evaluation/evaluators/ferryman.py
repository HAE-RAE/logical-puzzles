import logging
import re
import time
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING

from ..core.base import BaseEvaluator, EvaluationResult

if TYPE_CHECKING:
    from ..model.base import BaseLLMClient

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
3. Explain your reasoning clearly, then present your final conclusion in the format below.

### Final answer format
End with $\\boxed{X hours Y minutes}$.
"""

    KOREAN_SYSTEM_PROMPT = """당신은 뱃사공 운항 문제를 정확히 해결하는 전문가입니다.

### 규칙
1. 주어진 운항 규정을 모두 고려하여 단계별로 분석하세요.
2. 속도 제한, 의무 휴식, 화물 규정을 모두 적용하여 계산하세요.
3. 풀이 과정을 명확히 서술한 뒤, 최종 결론을 아래 형식으로 제시하세요.

### 최종 답 형식
마지막에 $\\boxed{N시간 M분}$ 형식으로 정답을 표시하세요.
"""

    def _is_korean(self, puzzle: Optional[Dict] = None) -> bool:
        """Prefer task_name suffix (_ko / _en); else infer from expected answer."""
        task = getattr(self, "_task_name", None) or ""
        if task.endswith("_ko"):
            return True
        if task.endswith("_en"):
            return False
        if puzzle is not None:
            expected = puzzle.get("answer", "")
            return bool(re.search(r"[가-힣]", expected))
        return False

    def _get_system_prompt(self, puzzle: Dict) -> str:
        if self._is_korean(puzzle):
            return self.KOREAN_SYSTEM_PROMPT
        return self.SYSTEM_PROMPT

    @staticmethod
    def _strip_latex(text: str) -> str:
        """Remove LaTeX markup: \\text{}, \\,, $, $$, etc."""
        text = re.sub(r'\\text\s*\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\[,;!]', ' ', text)
        text = text.replace('$', '').replace('\\boxed', '')
        text = re.sub(r'[{}]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[str]:
        """
        Extract time answer from LLM response and normalize.
        Routes to Korean or English parser based on task_name (_ko/_en) or answer format.
        """
        if self._is_korean(puzzle):
            return self._extract_korean(response)
        return self._extract_english(response)

    def _extract_korean(self, response: str) -> Optional[str]:
        # 1) \boxed{X시간 Y분} (with optional \text{})
        m = re.search(r'\\boxed\{((?:[^{}]|\{[^}]*\})*시간[^}]*분[^}]*)\}', response)
        if m:
            clean = self._strip_latex(m.group(1))
            return self._normalize_korean(clean)

        # 2) \boxed{X}시간 ... Y분  or  \boxed{X}$시간 $\boxed{Y}$분
        m = re.search(
            r'\\boxed\{(\d+)\}\s*\$?\s*시간\s*\$?\\boxed\{(\d+)\}\s*\$?\s*분', response)
        if m:
            return f"{int(m.group(1))}시간 {int(m.group(2))}분"

        # 3) \boxed{X}시간 Y분  (only hours boxed)
        m = re.search(r'\\boxed\{(\d+)\}\s*\$?\s*시간\s*(\d+)\s*분', response)
        if m:
            return f"{int(m.group(1))}시간 {int(m.group(2))}분"

        # 4) \boxed{HH:MM} or \boxed{XhYY}
        m = re.search(r'\\boxed\{(\d+)\s*[:hH]\s*(\d+)\s*\}', response)
        if m:
            return f"{int(m.group(1))}시간 {int(m.group(2))}분"

        # 5) Single \boxed{} — strip latex and try to parse
        m = re.search(r'\\boxed\{((?:[^{}]|\{[^}]*\})*)\}', response)
        if m:
            clean = self._strip_latex(m.group(1))
            result = self._normalize_korean(clean)
            if result:
                return result
            result = self._normalize_english(clean)
            if result:
                return result

        # 6) Fallback: last "X시간 Y분" anywhere in the stripped response
        clean = self._strip_latex(response)
        matches = list(re.finditer(r'(\d+)\s*시간\s*(\d+)\s*분', clean))
        if matches:
            m = matches[-1]
            return f"{int(m.group(1))}시간 {int(m.group(2))}분"

        # 7) Fallback: last "X hours Y minutes" (Korean problem with English answer)
        return self._normalize_english(clean)

    def _extract_english(self, response: str) -> Optional[str]:
        # 1) \boxed{X hours Y minutes} (with optional \text{})
        m = re.search(
            r'\\boxed\{((?:[^{}]|\{[^}]*\})*hours?[^}]*minutes?[^}]*)\}',
            response, re.IGNORECASE)
        if m:
            clean = self._strip_latex(m.group(1))
            return self._normalize_english(clean)

        # 2) \boxed{X} ... hours ... \boxed{Y} ... minutes
        m = re.search(
            r'\\boxed\{(\d+)\}\s*[\\,\s]*(?:\\text\{[^}]*\}|hours?)'
            r'\s*[\\,\s]*\\boxed\{(\d+)\}\s*[\\,\s]*(?:\\text\{[^}]*\}|minutes?)?',
            response, re.IGNORECASE)
        if m:
            return f"{int(m.group(1))} hours {int(m.group(2))} minutes"

        # 3) \boxed{X} hours Y minutes  (only hours boxed)
        m = re.search(
            r'\\boxed\{(\d+)\}\s*\$?\s*hours?\s+(\d+)\s*minutes?',
            response, re.IGNORECASE)
        if m:
            return f"{int(m.group(1))} hours {int(m.group(2))} minutes"

        # 4) \boxed{HH:MM} or \boxed{XhYY}
        m = re.search(r'\\boxed\{(\d+)\s*[:hH]\s*(\d+)\s*\}', response)
        if m:
            return f"{int(m.group(1))} hours {int(m.group(2))} minutes"

        # 5) Single \boxed{} — strip latex and try
        m = re.search(r'\\boxed\{((?:[^{}]|\{[^}]*\})*)\}', response)
        if m:
            clean = self._strip_latex(m.group(1))
            result = self._normalize_english(clean)
            if result:
                return result

        # 6) Fallback: last "X hours Y minutes" anywhere
        clean = self._strip_latex(response)
        matches = list(re.finditer(
            r'(\d+)\s*hours?\s+(\d+)\s*minutes?', clean, re.IGNORECASE))
        if matches:
            m = matches[-1]
            return f"{int(m.group(1))} hours {int(m.group(2))} minutes"

        return None

    @staticmethod
    def _normalize_korean(text: str) -> Optional[str]:
        m = re.search(r'(\d+)\s*시간\s*(\d+)\s*분', text)
        if m:
            return f"{int(m.group(1))}시간 {int(m.group(2))}분"
        return None

    @staticmethod
    def _normalize_english(text: str) -> Optional[str]:
        m = re.search(r'(\d+)\s*hours?\s+(\d+)\s*minutes?', text, re.IGNORECASE)
        if m:
            return f"{int(m.group(1))} hours {int(m.group(2))} minutes"
        m = re.search(r'(\d+)\s*hr?s?\s+(\d+)\s*min', text, re.IGNORECASE)
        if m:
            return f"{int(m.group(1))} hours {int(m.group(2))} minutes"
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
        llm_client: "BaseLLMClient"
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
        llm_client: "BaseLLMClient",
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
