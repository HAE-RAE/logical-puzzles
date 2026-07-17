"""
Korea Peak Rank Evaluator (KO) — 산 높이 순위 추론(단서+지식)

정답 = 가장 높은 산 이름(보기 중 하나). 응답의 '정답:' 줄(또는 마지막 줄)에서 이름을 추출해
기준 정답명 포함 여부로 채점한다.
"""

import re
from typing import Dict, Any, Optional, Tuple

from ..core.base import BaseEvaluator


def _norm(s):
    return re.sub(r'\s+', '', s or '')


class KoreaPeakRankEvaluator(BaseEvaluator):
    SYSTEM_PROMPT = """당신은 단서 추론과 대한민국 산 지식을 함께 쓰는 전문가입니다.

규칙:
- 'A는 B보다 높다' 단서에서, 다른 산보다 낮다고 나온 산은 가장 높은 산이 될 수 없습니다(후보 제외).
- 그렇게 남은 후보 중 실제로 가장 높은 산을 고릅니다. 후보가 여럿이면 실제 산 높이 지식으로 결정합니다.
- 단서에서 많은 산을 이긴 산이 반드시 최고는 아닙니다(단서는 일부 관계일 뿐).

추론으로 후보를 좁힌 뒤, 마지막 줄에 반드시 다음 형식으로 답하세요:
정답: <산 이름>"""

    def _parse_answer(self, response: str, puzzle: Dict[str, Any]) -> Optional[str]:
        if not response:
            return None
        m = re.findall(r'정답\s*[:：]\s*(.+)', response)
        if m:
            return m[-1].strip()
        for line in reversed(response.splitlines()):
            if line.strip():
                return line.strip()
        return None

    def _check_answer(self, expected: Any, predicted: Optional[str]) -> Tuple[bool, float]:
        if not predicted:
            return False, 0.0
        return (True, 1.0) if _norm(str(expected)) in _norm(predicted) else (False, 0.0)
