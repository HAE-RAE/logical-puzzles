import logging
import re
from typing import List, Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class FerrymanEvaluator(BaseEvaluator):
    """
    Ferryman 퍼즐 평가자
    
    답변 형식: X시간 Y분
    """
    
    SYSTEM_PROMPT = """당신은 뱃사공 운항 문제를 정확히 해결하는 전문가입니다.

### 규칙
1. 주어진 운항 규정을 모두 고려하여 단계별로 분석하세요.
2. 속도 제한, 의무 휴식, 화물 규정을 모두 적용하여 계산하세요.
3. 문제를 푼 후, 최종 답변을 다음과 같은 형식으로 작성하세요: $\\boxed{N시간 M분}$.
4. \\boxed{} 안에는 "X시간 Y분" 형식으로만 답하세요. 다른 단위나 설명은 포함하지 마세요.
"""
    
    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[str]:
        """
        LLM 응답에서 시간 답변 추출 및 정규화
        
        \\boxed{} 패턴에서 추출 후 "X시간 Y분" 형식으로 정규화
        
        Args:
            response: LLM 응답 텍스트
            puzzle: 퍼즐 데이터 (사용하지 않음)
        """
        # 패턴 1: \boxed{} 형식
        match = re.search(r'\\boxed\{([^}]+)\}', response)
        if match:
            time_str = match.group(1).strip()
            return self._normalize_time(time_str)
        
        # 패턴 2: 직접 시간 패턴
        return self._normalize_time(response)
    
    def _normalize_time(self, time_str: str) -> Optional[str]:
        """
        시간 문자열을 "X시간 Y분" 형식으로 정규화
        """
        # 패턴 1: X시간 Y분
        match = re.search(r'(\d+)시간\s*(\d+)분', time_str)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            return f"{hours}시간 {minutes}분"
        
        # 패턴 2: 숫자 2개 추출 (시간, 분)
        numbers = re.findall(r'\d+', time_str)
        if len(numbers) >= 2:
            return f"{numbers[0]}시간 {numbers[1]}분"
        
        return None
    
    def _check_answer(
        self,
        expected: str,
        predicted: Optional[str]
    ) -> Tuple[bool, float]:
        """
        답변 확인
        
        Returns:
            (is_correct, partial_score) 튜플
        """
        if predicted is None:
            return False, 0.0
        
        # 둘 다 정규화
        expected_normalized = self._normalize_time(expected)
        
        correct = expected_normalized == predicted
        return correct, 1.0 if correct else 0.0
