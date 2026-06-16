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
    SYSTEM_PROMPT = """### Instructions
You are an expert at reading a line map and counting the shortest route.

### Rules
1. Use ONLY the given line map (each line's station order); do not rely on real-world subway knowledge.
2. A station that appears on more than one line (same name) is a transfer station where you may switch lines.
3. Moving from one station to an adjacent station counts as 1 stop; find the MINIMUM number of stops from the start station to the destination.

### Output format
Your final line must be:
Answer: N stations
"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 노선도를 보고 최단 경로의 정거장 수를 계산하는 전문가입니다.

### 규칙
1. 주어진 노선도(각 호선의 역 순서)만 사용하고, 실제 지하철 지식이 아니라 제시된 노선도를 기준으로 풉니다.
2. 같은 이름의 역은 서로 다른 노선이 만나는 환승역입니다(노선을 갈아탈 수 있음).
3. 한 역에서 바로 옆 역으로 이동하는 것을 1정거장으로 세고, 출발역에서 도착역까지 '최소 정거장 수'를 구합니다.

### 출력 형식
마지막 줄은 반드시 아래 형식으로 작성하세요:
Answer: N정거장
"""

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
