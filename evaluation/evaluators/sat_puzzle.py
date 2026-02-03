"""
SAT Puzzle Evaluator

SAT 논리 퍼즐 평가 (영문)
"""

import logging
import json
import re
from typing import List, Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class SATPuzzleEvaluator(BaseEvaluator):
    """
    SAT Puzzle 퍼즐 평가자
    
    답변 형식: JSON (변수: bool 매핑, 부분 점수 지원)
    """
    
    SYSTEM_PROMPT = """You are an expert at solving logic puzzles. Carefully analyze all constraints and provide accurate answers."""
    
    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[Dict[str, bool]]:
        """
        LLM 응답에서 JSON 답변 추출
        
        Args:
            response: LLM 응답 텍스트
            puzzle: 퍼즐 데이터
        """
        variables = puzzle.get("variables", [])
        
        try:
            # Try to find JSON in the response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                answer_json = json_match.group(1)
                # Remove comments
                answer_json = re.sub(r'//.*', '', answer_json)
                answer = json.loads(answer_json)
                
                # Validate structure
                if not isinstance(answer, dict):
                    return None
                
                # Check all variables are present
                for var in variables:
                    if var not in answer:
                        return None
                    if not isinstance(answer[var], bool):
                        return None
                
                return answer
            
            # Try to find JSON without markdown
            json_match = re.search(r'\{[^{}]*"[^"]+"\s*:\s*(true|false)[^{}]*\}', response, re.DOTALL)
            if json_match:
                answer_text = json_match.group(0)
                # Remove comments
                answer_text = re.sub(r'//.*', '', answer_text)
                answer = json.loads(answer_text)
                
                # Validate
                if isinstance(answer, dict):
                    valid = True
                    for var in variables:
                        if var not in answer or not isinstance(answer[var], bool):
                            valid = False
                            break
                    
                    if valid:
                        return answer
            
            # Try to parse line-by-line format
            # "Alice: True" or "Alice: False"
            answer = {}
            for line in response.split('\n'):
                for var in variables:
                    if var in line:
                        if 'true' in line.lower() or ': true' in line.lower():
                            answer[var] = True
                            break
                        elif 'false' in line.lower() or ': false' in line.lower():
                            answer[var] = False
                            break
            
            # Check if we got all variables
            if len(answer) == len(variables) and all(var in answer for var in variables):
                return answer
            
            return None
        
        except (json.JSONDecodeError, AttributeError):
            return None
    
    def _check_answer(
        self,
        expected: Dict[str, bool],
        predicted: Optional[Dict[str, bool]]
    ) -> Tuple[bool, float]:
        """
        답변 확인
        
        Returns:
            (is_correct, partial_score) 튜플
        """
        if predicted is None:
            return False, 0.0
        
        # 완전 일치만 정답으로 인정
        for var in expected:
            if var not in predicted or predicted[var] != expected[var]:
                return False, 0.0
        
        return True, 1.0
