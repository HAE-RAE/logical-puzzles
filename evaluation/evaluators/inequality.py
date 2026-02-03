"""
Inequality Evaluator

부등호 퍼즐 평가
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class InequalityEvaluator(BaseEvaluator):
    """
    Inequality 퍼즐 평가자
    
    답변 형식: 숫자 시퀀스 (예: "3142")
    """
    
    SYSTEM_PROMPT = """You are an expert puzzle solver specializing in logical constraint puzzles."""
    
    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[str]:
        """
        LLM 응답에서 숫자 시퀀스 추출
        
        Args:
            response: LLM 응답 텍스트
            puzzle: 퍼즐 데이터
        """
        expected_answer = puzzle.get("answer", "")
        expected_length = len(expected_answer) if expected_answer else puzzle.get("size", 0)
        
        # Pattern 1: "Answer: 12345" or "Answer: 1 2 3 4 5"
        patterns = [
            r'Answer:\s*([1-9\s]+)',
            r'answer:\s*([1-9\s]+)',
            r'solution[:\s]+([1-9\s]+)',
            r'sequence[:\s]+([1-9\s]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                # Extract digits only, remove spaces
                digits = re.findall(r'[1-9]', match.group(1))
                if len(digits) >= expected_length:
                    return ''.join(digits[:expected_length])
        
        # Fallback: Find the last sequence of N digits
        digit_sequences = re.findall(r'[1-9][\s,]*[1-9][\s,]*[1-9][\s,]*[1-9]?[\s,]*[1-9]?[\s,]*[1-9]?', response)
        if digit_sequences:
            last_seq = digit_sequences[-1]
            digits = re.findall(r'[1-9]', last_seq)
            if len(digits) >= expected_length:
                return ''.join(digits[:expected_length])
        
        # Last resort: Extract all digits from the end of output
        last_part = response[-200:] if len(response) > 200 else response
        all_digits = re.findall(r'[1-9]', last_part)
        if len(all_digits) >= expected_length:
            return ''.join(all_digits[-expected_length:])
        
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
