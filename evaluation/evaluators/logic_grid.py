"""
Logic Grid Evaluator

논리 그리드 퍼즐 평가 (영문)
"""

import logging
import json
import re
from typing import List, Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class LogicGridEvaluator(BaseEvaluator):
    """
    Logic Grid 퍼즐 평가자
    
    답변 형식: JSON (부분 점수 지원)
    """
    
    SYSTEM_PROMPT = """You are an expert at solving logic puzzles. Carefully analyze all constraints and provide accurate answers."""
    
    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[Dict[str, Dict[str, str]]]:
        """
        LLM 응답에서 JSON 답변 추출
        
        Args:
            response: LLM 응답 텍스트
            puzzle: 퍼즐 데이터
        """
        people = puzzle.get("people", [])
        categories = list(puzzle.get("attributes", {}).keys())
        
        try:
            # Try to find JSON in the response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                answer_json = json_match.group(1)
                answer = json.loads(answer_json)
                
                # Validate structure
                if not isinstance(answer, dict):
                    return None
                
                # Check all people are present
                for person in people:
                    if person not in answer:
                        return None
                    if not isinstance(answer[person], dict):
                        return None
                    
                    # Check all categories are present
                    for cat in categories:
                        if cat not in answer[person]:
                            return None
                
                return answer
            
            # Try to find JSON without markdown
            json_match = re.search(r'\{[^{}]*"[^"]+"\s*:\s*\{[^{}]+\}[^{}]*\}', response, re.DOTALL)
            if json_match:
                answer = json.loads(json_match.group(0))
                
                # Validate
                if isinstance(answer, dict):
                    valid = True
                    for person in people:
                        if person not in answer or not isinstance(answer[person], dict):
                            valid = False
                            break
                        for cat in categories:
                            if cat not in answer[person]:
                                valid = False
                                break
                    
                    if valid:
                        return answer
            
            return None
        
        except (json.JSONDecodeError, AttributeError):
            return None
    
    def _check_answer(
        self,
        expected: Dict[str, Dict[str, str]],
        predicted: Optional[Dict[str, Dict[str, str]]]
    ) -> Tuple[bool, float]:
        """
        답변 확인
        
        Returns:
            (is_correct, partial_score) 튜플
        """
        if predicted is None:
            return False, 0.0
        
        # 완전 일치만 정답으로 인정
        for person, attrs in expected.items():
            if person not in predicted:
                return False, 0.0
            
            for cat, val in attrs.items():
                if cat not in predicted[person] or predicted[person][cat] != val:
                    return False, 0.0
        
        return True, 1.0
