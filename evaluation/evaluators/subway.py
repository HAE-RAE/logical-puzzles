"""
Subway Evaluator (EN / KO)

최단 정거장(이동 구간) 수 퍼즐 평가기.
- KO 정답 형식: "N정거장"  (예: "12정거장")
- EN 정답 형식: "N stations" (예: "12 stations")
응답에서 정수를 추출해 비교한다(언어 무관).
"""

import re
from typing import Dict, Any, Optional, Tuple

from ..core.base import BaseEvaluator


class SubwayEvaluator(BaseEvaluator):
    SYSTEM_PROMPT = """You are an expert at reading a line map and counting the shortest route.

Rules:
- Use ONLY the given line map (each line's station order). Do not rely on real-world subway knowledge.
- A station that appears on more than one line (same name) is a transfer station where you may switch lines.
- Moving from one station to an adjacent station counts as 1 stop.
- Find the MINIMUM number of stops from the start station to the destination.

Reason step by step, then end with exactly this line:
Answer: N stations"""

    KOREAN_SYSTEM_PROMPT = """당신은 노선도를 보고 최단 경로의 정거장 수를 계산하는 전문가입니다.

규칙:
- 주어진 노선도(각 호선의 역 순서)만 사용합니다. 실제 지하철 지식이 아니라 제시된 노선도를 기준으로 풉니다.
- 같은 이름의 역은 서로 다른 노선이 만나는 환승역입니다(노선을 갈아탈 수 있음).
- 한 역에서 바로 옆 역으로 이동하는 것을 1정거장으로 셉니다.
- 출발역에서 도착역까지 '최소 정거장 수'를 구합니다.

단계적으로 추론한 뒤, 마지막 줄에 반드시 다음 형식으로 답하세요:
정답: N정거장"""

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

        # 2순위: 단위(정거장/stations/stops) 동반 숫자 (마지막)
        unit = re.findall(r'(-?\d+)\s*(?:정거장|stations?|stops?)', response, re.IGNORECASE)
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
