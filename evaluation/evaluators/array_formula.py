"""
Array Formula Evaluator

Excel 배열 수식 퍼즐 평가
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class ArrayFormulaEvaluator(BaseEvaluator):
    """
    Array Formula 퍼즐 평가자
    
    숫자 또는 텍스트 답변, tolerance 지원
    """
    
    SYSTEM_PROMPT = """You are a spreadsheet/Excel expert.
Analyze the given table data and answer the question accurately.

Rules:
1. For numeric results, answer with only the number (no units, commas, or currency symbols)
2. For decimals, truncate unless otherwise specified
3. For text answers, provide the exact value only
4. Briefly explain your reasoning, then end with "Final answer: [answer]"
"""
    
    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[Any]:
        """
        LLM 응답에서 답변 추출
        
        Args:
            response: LLM 응답 텍스트
            puzzle: 퍼즐 데이터
        """
        answer_type = puzzle.get("answer_type", "number")
        
        patterns = [
            r"[Ff]inal\s*[Aa]nswer\s*[:：]\s*(.+?)(?:\n|$)",
            r"[Aa]nswer\s*[:：]\s*(.+?)(?:\n|$)",
        ]
        
        answer_text = None
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer_text = match.group(1).strip()
                break
        
        # Fallback: extract from last line
        if answer_text is None:
            lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
            if lines:
                answer_text = lines[-1]
        
        if answer_text is None:
            return None
        
        # Number type processing
        if answer_type == "number":
            number_match = re.search(r"-?[\d,]+\.?\d*", answer_text.replace(",", ""))
            if number_match:
                try:
                    num_str = number_match.group().replace(",", "")
                    if "." in num_str:
                        return float(num_str)
                    return int(num_str)
                except ValueError:
                    pass
            return None
        
        # Text type
        answer_text = answer_text.strip("'\"")
        return answer_text
    
    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[Any]
    ) -> Tuple[bool, float]:
        """
        답변 확인
        
        Returns:
            (is_correct, partial_score) 튜플
        """
        if predicted is None:
            return False, 0.0
        
        # answer_type과 tolerance는 puzzle에서 가져와야 하지만,
        # _check_answer는 expected만 받으므로 기본값 사용
        # 실제 사용 시에는 _parse_answer에서 puzzle을 받으므로
        # answer_type은 predicted의 타입으로 추정
        answer_type = "number" if isinstance(predicted, (int, float)) else "text"
        tolerance = 0.01
        
        if answer_type == "number":
            try:
                expected_num = float(expected)
                predicted_num = float(predicted)
                
                # 완전 일치만 정답으로 인정
                exact = abs(expected_num - predicted_num) < 0.001
                return exact, 1.0 if exact else 0.0
            except (ValueError, TypeError):
                return False, 0.0
        else:
            # Text comparison
            expected_str = str(expected).strip().lower()
            predicted_str = str(predicted).strip().lower()
            correct = expected_str == predicted_str
            return correct, 1.0 if correct else 0.0
