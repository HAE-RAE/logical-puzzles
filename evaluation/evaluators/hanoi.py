"""
Hanoi Evaluator

하노이 탑 퍼즐 평가
"""

import logging
import re
from typing import Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator

logger = logging.getLogger(__name__)


class HanoiEvaluator(BaseEvaluator):
    """
    Hanoi 퍼즐 평가자

    답변 형식: 문제 유형에 따라 숫자 튜플
    """

    SYSTEM_PROMPT = """### Instructions
You are an expert at solving Tower of Hanoi puzzles.

### Rules
1. Standard rules apply: only one disk can be moved at a time, and a larger disk cannot be placed on top of a smaller disk. Disks are numbered from 1 (smallest) to n (largest).
2. The optimal solution for 'n' disks requires exactly 2^n - 1 moves.
3. Explain your reasoning clearly, then present your final conclusion in the format below.

### Output format
Your final line must be one of the following (matching the question type):
Answer: (H1, H2, H3)
Answer: (H1, H2)
Answer: (sum_sq_peg0, sum_sq_peg1, sum_sq_peg2)
Answer: (empty_dst_count, odd_size_dst_count, c_mult_3, k)
"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 하노이 탑(Tower of Hanoi) 퍼즐을 정확히 해결하는 전문가입니다.

### 규칙
1. 표준 규칙이 적용됩니다: 한 번에 하나의 원판만 이동할 수 있으며, 큰 원판을 작은 원판 위에 놓을 수 없습니다. 원판은 1(가장 작음)부터 n(가장 큼)까지 번호가 매겨져 있습니다.
2. 'n'개의 원판을 옮기는 최적의 해법은 정확히 2^n - 1번의 이동이 필요합니다.
3. 풀이 과정을 명확히 서술한 뒤, 최종 결론을 아래 형식으로 제시하세요.

### 출력 형식
마지막 줄은 반드시 아래 형식 중 하나로 작성하세요 (질문 유형에 맞게):
Answer: (H1, H2, H3)
Answer: (H1, H2)
Answer: (기둥0_제곱합, 기둥1_제곱합, 기둥2_제곱합)
Answer: (빈_목적_횟수, 홀수_크기_목적_횟수, 3의배수_횟수, k)
"""

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[tuple]:
        """LLM 응답에서 숫자 튜플 추출 (2~4개 숫자 지원)."""
        answer_text = self._extract_final_answer_text(response) or response

        # (숫자, 숫자[, 숫자[, 숫자]]) 형식 — 2~4개 원소 튜플 모두 처리
        match = re.search(r'\((\d+(?:,\s*\d+){1,3})\)', answer_text)
        if match:
            nums = re.findall(r'\d+', match.group(0))
            return tuple(int(n) for n in nums)

        return None

    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[tuple]
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
