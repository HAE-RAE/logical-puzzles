"""
Cryptarithmetic Evaluator

암호 산술 퍼즐 평가
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class CryptarithmeticEvaluator(BaseEvaluator):
    """
    Cryptarithmetic 퍼즐 평가자
    
    답변 형식: 숫자 (결과 단어의 숫자 값)
    """
    
    SYSTEM_PROMPT = """You are an expert puzzle solver specializing in cryptarithmetic problems."""
    
    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[str]:
        """
        LLM 응답에서 숫자 답변 추출
        
        Args:
            response: LLM 응답 텍스트
            puzzle: 퍼즐 데이터 (사용하지 않음)
        """
        patterns = [
            r'Answer:\s*(\d+)',
            r'answer:\s*(\d+)',
            r'result\s*(?:is|=|equals)\s*(\d+)',
            r'value\s*(?:is|=|equals)\s*(\d+)',
            r'= (\d+)$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1)
        
        # Fallback: Find the last standalone number in the output
        numbers = re.findall(r'\b(\d{2,})\b', response)
        if numbers:
            return numbers[-1]
        
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
        
        correct = str(predicted) == str(expected)
        return correct, 1.0 if correct else 0.0
