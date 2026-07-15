"""
Inequality Evaluator

Evaluates inequality puzzle responses with constraint-based fallback validation.
Answer format: digit sequence (e.g., "3142" or "5 3 12 4 10 1" for size>9).
"""

import logging
import re
from typing import Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class InequalityEvaluator(BaseEvaluator):
    """
    Inequality puzzle evaluator.

    Supports both concatenated (size<=9) and space-separated (size>9) formats.
    Falls back to constraint-based validation when the answer doesn't match
    the pre-computed value but still satisfies all inequality constraints.
    """

    SYSTEM_PROMPT = """### Instructions
You are an expert at inequality-grid (Futoshiki-style) constraint puzzles.

### Rules
1. Fill blanks with a permutation of 1 through N; each of `<`, `>`, and hidden `?` between neighbors must be satisfiable (for `?`, some choice of `<` or `>`).
2. Keep given hints fixed; `_` marks empty cells in the alternating value–symbol row (e.g. `_ < 2 > _ < _`).
3. Explain your reasoning clearly, then present your final conclusion in the format below.

### Output format
Your final line must be:
Answer: <left to right>
(If puzzle size <= 9, use concatenated digits only, e.g. Answer: 53241; if size > 9, separate numbers with single spaces, e.g. Answer: 5 3 12 4 10 1.)
"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 부등호 제약 그리드 퍼즐을 정확히 푸는 전문가입니다.

### 규칙
1. 1부터 N까지 순열로 빈칸을 채우고, 인접 `<`·`>`·`?`(숨은 부등호)를 모두 만족하는 배치를 제시하세요.
2. 주어진 힌트 숫자는 고정이며, `_`는 빈칸(값·기호 교차 나열, 예: `_ < 2 > _ < _`)입니다.
3. 풀이 과정을 명확히 서술한 뒤, 최종 결론을 아래 형식으로 제시하세요.

### 출력 형식
마지막 줄은 반드시 아래 형식으로 작성하세요:
Answer: <왼쪽에서 오른쪽>
(크기 9 이하는 숫자만 이어 쓰기, 예: Answer: 53241; 크기 9 초과는 공백 한 칸으로 구분, 예: Answer: 5 3 12 4 10 1.)
"""

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[str]:
        """Extract number sequence from LLM response."""
        expected_answer = puzzle.get("answer", "")
        size = puzzle.get("size", 0)
        answer_text = self._extract_final_answer_text(response) or response
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
            match = re.search(pattern, answer_text, re.IGNORECASE)
            if match:
                raw = match.group(1).strip()
                nums = re.findall(r'\d+', raw)
                out = _normalize(nums)
                if out is not None:
                    return out

        last_part = answer_text[-200:] if len(answer_text) > 200 else answer_text
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
                finish_reason=usage.get("finish_reason", "") if isinstance(usage, dict) else "",
            )
        except Exception as e:
            return self._create_error_result(
                puzzle, response, latency_ms, str(e),
                finish_reason=usage.get("finish_reason") or "error",
            )
