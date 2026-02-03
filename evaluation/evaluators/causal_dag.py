"""
Causal DAG Evaluator

인과관계 DAG 추론 퍼즐 평가 (영문)
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class CausalDAGEvaluator(BaseEvaluator):
    """
    Causal DAG 퍼즐 평가자
    
    답변 형식: 숫자 (분 단위)
    """
    
    SYSTEM_PROMPT = """You are a logical reasoning expert. Analyze the problem carefully and provide precise numerical answers."""
    
    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[int]:
        """
        LLM 응답에서 숫자 답변 추출 (분 단위)
        
        Args:
            response: LLM 응답 텍스트
            puzzle: 퍼즐 데이터 (사용하지 않음)
        """
        response = response.strip()
        
        # Pattern 1: Just a number
        if response.isdigit():
            return int(response)
        
        # Pattern 2: LaTeX boxed format: \boxed{45} or \\boxed{45}
        match = re.search(r'\\+boxed\{(\d+)\}', response)
        if match:
            return int(match.group(1))
        
        # Pattern 3: "Answer: 45" or "answer: 45"
        match = re.search(r'[Aa]nswer\s*[:：]\s*(\d+)', response)
        if match:
            return int(match.group(1))
        
        # Pattern 4: "event X first occurs at minute 45" or "occurs at minute 45"
        matches = list(re.finditer(r'(?:first\s+)?occurs?\s+at\s+minute\s+(\d+)', response, re.IGNORECASE))
        if matches:
            return int(matches[-1].group(1))
        
        # Pattern 5: "at minute 45" (last occurrence)
        matches = list(re.finditer(r'at\s+minute\s+(\d+)', response, re.IGNORECASE))
        if matches:
            return int(matches[-1].group(1))
        
        # Pattern 6: "minute 45" (last occurrence)
        matches = list(re.finditer(r'[Mm]inute\s+(\d+)', response))
        if matches:
            return int(matches[-1].group(1))
        
        # Pattern 7: "45 minutes"
        match = re.search(r'(\d+)\s+[Mm]inutes?', response)
        if match:
            return int(match.group(1))
        
        # Pattern 8: Last number in response
        numbers = re.findall(r'\b(\d+)\b', response)
        if numbers:
            return int(numbers[-1])
        
        return None
    
    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[int]
    ) -> Tuple[bool, float]:
        """
        답변 확인
        
        Returns:
            (is_correct, partial_score) 튜플
        """
        if predicted is None:
            return False, 0.0
        
        # expected가 문자열일 수 있으므로 정수로 변환
        try:
            expected_num = int(expected)
        except (ValueError, TypeError):
            return False, 0.0
        
        correct = predicted == expected_num
        return correct, 1.0 if correct else 0.0
