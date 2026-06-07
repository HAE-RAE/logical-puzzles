"""
Water Jug (KO) Evaluator

물 붓기 최소 동작 수 퍼즐 평가기.
정답 형식: "N번" (예: "7번"). 응답에서 정수를 추출해 비교한다.
"""

import re
from typing import Dict, Any, Optional, Tuple

from ..core.base import BaseEvaluator


class WaterJugKoEvaluator(BaseEvaluator):
    SYSTEM_PROMPT = """당신은 물통 퍼즐을 푸는 전문가입니다.

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
        labeled = re.findall(r'정답\s*[:：]?\s*(-?\d+)', response)
        if labeled:
            return int(labeled[-1])
        beon = re.findall(r'(-?\d+)\s*번', response)
        if beon:
            return int(beon[-1])
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
