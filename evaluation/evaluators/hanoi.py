"""
Hanoi Evaluator

하노이 탑 퍼즐 평가
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class HanoiEvaluator(BaseEvaluator):
    """
    Hanoi 퍼즐 평가자
    
    답변 형식: (disk, from, to) 튜플
    """
    
    SYSTEM_PROMPT = "You must answer ONLY in the format (disk, from, to)."
    
    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[Tuple[int, int, int]]:
        """
        LLM 응답에서 (disk, from, to) 튜플 추출
        
        숫자 3개를 찾아서 튜플로 반환
        
        Args:
            response: LLM 응답 텍스트
            puzzle: 퍼즐 데이터 (사용하지 않음)
        """
        # 패턴 1: (숫자, 숫자, 숫자) 형식
        match = re.search(r'\((\d+),\s*(\d+),\s*(\d+)\)', response)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        
        # 패턴 2: 숫자 3개 추출
        nums = re.findall(r'\d+', response)
        if len(nums) >= 3:
            return (int(nums[0]), int(nums[1]), int(nums[2]))
        
        return None
    
    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[Tuple[int, int, int]]
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
