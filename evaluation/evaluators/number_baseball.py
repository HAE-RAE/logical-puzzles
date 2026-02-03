"""
Number Baseball Evaluator

숫자 야구 (Bulls and Cows) 퍼즐 평가
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class NumberBaseballEvaluator(BaseEvaluator):
    """
    Number Baseball 퍼즐 평가자
    
    답변 형식: 숫자 시퀀스 (예: "1234")
    """
    
    SYSTEM_PROMPT = """You are an expert puzzle solver specializing in logical deduction games like Bulls and Cows (Number Baseball)."""
    
    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[str]:
        """
        LLM 응답에서 숫자 시퀀스 추출
        
        Args:
            response: LLM 응답 텍스트
            puzzle: 퍼즐 데이터
        """
        num_digits = puzzle.get("num_digits", 3)
        
        # Pattern 1: "Answer: 123" or "Answer: 1234"
        patterns = [
            r'Answer:\s*(\d+)',
            r'answer:\s*(\d+)',
            r'secret number[:\s]+(\d+)',
            r'number is[:\s]+(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                digits = match.group(1)
                if len(digits) == num_digits:
                    return digits
        
        # Fallback: Find standalone N-digit numbers at the end of output
        last_part = response[-500:] if len(response) > 500 else response
        numbers = re.findall(rf'\b(\d{{{num_digits}}})\b', last_part)
        if numbers:
            # Return the last N-digit number found
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
