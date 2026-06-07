"""
Water Jug Evaluator (EN / KO)

물 붓기 최소 동작 수 퍼즐 평가기.
- KO 정답 형식: "N번"   (예: "7번")
- EN 정답 형식: "N moves" (예: "7 moves")
응답에서 정수를 추출해 비교한다(언어 무관).
"""

import re
from typing import Dict, Any, Optional, Tuple

from ..core.base import BaseEvaluator


class WaterJugEvaluator(BaseEvaluator):
    SYSTEM_PROMPT = """You are an expert at solving water jug puzzles.

Rules:
- All jugs start empty.
- One move is one of: FILL a jug to the top, EMPTY a jug completely, or POUR from one jug
  into another until the receiving jug is full or the pouring jug is empty.
- Find the MINIMUM number of moves to get exactly the target amount in some jug.

Reason step by step, then end with exactly this line:
Answer: N moves"""

    KOREAN_SYSTEM_PROMPT = """당신은 물통 퍼즐을 푸는 전문가입니다.

규칙:
- 모든 물통은 비어 있는 상태에서 시작합니다.
- 한 번의 동작 = 채우기(가득) / 비우기(완전히) / 붓기(받는 쪽이 차거나 주는 쪽이 빌 때까지) 중 하나.
- 어느 한 물통에 목표 용량을 정확히 담는 데 필요한 '최소 동작 수'를 구합니다.

단계적으로 상태를 추적하며 추론한 뒤, 마지막 줄에 반드시 다음 형식으로 답하세요:
정답: N번"""

    def _expected_int(self, expected: str) -> Optional[int]:
        m = re.search(r'(-?\d+)', str(expected))
        return int(m.group(1)) if m else None

    def _parse_answer(self, response: str, puzzle: Dict[str, Any]) -> Optional[int]:
        if not response:
            return None

        # 1순위: 정답/Answer 라벨 뒤 숫자 (마지막)
        labeled = re.findall(r'(?:정답|answer|final\s*answer)\s*[:：]?\s*(-?\d+)',
                             response, re.IGNORECASE)
        if labeled:
            return int(labeled[-1])

        # 2순위: 단위(번/moves/times/operations/steps) 동반 숫자 (마지막)
        unit = re.findall(r'(-?\d+)\s*(?:번|moves?|times|operations?|steps?)',
                          response, re.IGNORECASE)
        if unit:
            return int(unit[-1])

        # 3순위: 응답 끝부분의 마지막 정수
        tail = response[-200:]
        nums = re.findall(r'-?\d+', tail)
        if nums:
            return int(nums[-1])

        return None

    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[int]
    ) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0
        exp = self._expected_int(expected)
        if exp is None:
            return False, 0.0
        correct = int(predicted) == exp
        return correct, 1.0 if correct else 0.0
