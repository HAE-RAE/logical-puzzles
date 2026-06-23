"""
Word Chain Evaluator — 끝말잇기 최장 사슬 (KO)

정답 = 최장 사슬의 단어 개수(정수, 예: "7개"). 응답에서 정수를 추출해 비교한다.
"""

import re
from typing import Dict, Any, Optional, Tuple

from ..core.base import BaseEvaluator


class WordChainEvaluator(BaseEvaluator):
    SYSTEM_PROMPT = """당신은 끝말잇기 최장 사슬을 찾는 전문가입니다.

규칙:
- 앞 단어의 마지막 글자와 다음 단어의 첫 글자가 같아야 이어집니다.
- 각 단어는 최대 한 번만 사용합니다. 주어진 목록의 단어만 씁니다.
- 만들 수 있는 '가장 긴' 사슬의 단어 개수를 구합니다(여러 사슬 중 최대 길이).

여러 사슬을 신중히 탐색해 최댓값을 찾은 뒤, 마지막 줄에 반드시 다음 형식으로 답하세요:
정답: N개"""

    def _expected_int(self, expected: str) -> Optional[int]:
        m = re.search(r'(-?\d+)', str(expected))
        return int(m.group(1)) if m else None

    def _parse_answer(self, response: str, puzzle: Dict[str, Any]) -> Optional[int]:
        if not response:
            return None
        labeled = re.findall(r'정답\s*[:：]?\s*(-?\d+)', response)
        if labeled:
            return int(labeled[-1])
        gae = re.findall(r'(-?\d+)\s*개', response)
        if gae:
            return int(gae[-1])
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
