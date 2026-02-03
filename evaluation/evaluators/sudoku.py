"""
Sudoku Evaluator

스도쿠 퍼즐 평가
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class SudokuEvaluator(BaseEvaluator):
    """
    Sudoku 퍼즐 평가자
    
    답변 형식: k자리 숫자 (예: "123456")
    """
    
    SYSTEM_PROMPT = """You are a logic puzzle expert."""
    
    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[str]:
        """
        LLM 응답에서 k자리 숫자 추출
        
        Args:
            response: LLM 응답 텍스트
            puzzle: 퍼즐 데이터
        """
        k = puzzle.get("spotcheck", {}).get("k", 0)
        if k == 0:
            # fallback: answer 길이로 추정
            expected_answer = puzzle.get("answer", "")
            k = len(expected_answer) if expected_answer else 0
        
        if k == 0:
            return None
        
        # Find k-digit number after "Answer:"
        patterns = [
            rf'Answer:\s*([1-9]{{{k}}})',
            rf'Answer:\s*\[?([1-9]{{{k}}})\]?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)
        
        # Fallback: Extract all digits after "Answer:"
        answer_match = re.search(r'Answer:\s*(.+)', response)
        if answer_match:
            answer_part = answer_match.group(1)
            digits = re.findall(r'[1-9]', answer_part)
            if len(digits) >= k:
                return ''.join(digits[:k])
        
        # Final fallback: Extract all digits
        digits = re.findall(r'[1-9]', response)
        if len(digits) >= k:
            return ''.join(digits[:k])
        
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
        
        # 완전 일치만 정답으로 인정
        correct = str(predicted) == str(expected)
        return correct, 1.0 if correct else 0.0
