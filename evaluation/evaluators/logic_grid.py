"""
Logic Grid Evaluator

논리 그리드 퍼즐 평가 (영문)
"""

import logging
import json
import re
from typing import Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator

logger = logging.getLogger(__name__)


class LogicGridEvaluator(BaseEvaluator):
    """
    Logic Grid 퍼즐 평가자
    
    답변 형식: JSON (부분 점수 지원)
    """
    
    SYSTEM_PROMPT = """### Instructions
You are an expert at logic-grid (Zebra-style) deduction puzzles.

### Rules
1. Satisfy every clue in the user message and match the required JSON schema for the puzzle.
2. Put a single-line valid JSON object on the final line (double quotes, no trailing commas, no markdown fences, no text after Answer:).
3. Explain your reasoning clearly, then present your final conclusion in the format below.

### Output format
Your final line must be:
Answer: {"Alice":{"Pet":"Cat","Drink":"Tea"},"Bob":{"Pet":"Dog","Drink":"Coffee"}}
"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 논리 그리드(얼룩말 스타일) 추론 퍼즐을 정확히 푸는 전문가입니다.

### 규칙
1. 사용자 메시지의 모든 단서를 만족하고, 요구된 JSON 스키마를 따르세요.
2. 마지막 줄에 한 줄짜리 유효 JSON만(쌍따옴표, trailing comma 금지, 코드펜스 금지, Answer: 뒤 추가 텍스트 금지).
3. 풀이 과정을 명확히 서술한 뒤, 최종 결론을 아래 형식으로 제시하세요.

### 출력 형식
마지막 줄은 반드시 아래 형식으로 작성하세요:
Answer: {"민수":{"애완동물":"고양이","음료":"차"},"지훈":{"애완동물":"강아지","음료":"커피"}}
"""

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[Dict[str, Dict[str, str]]]:
        """
        LLM 응답에서 JSON 답변 추출
        
        Args:
            response: LLM 응답 텍스트
            puzzle: 퍼즐 데이터
        """
        people = puzzle.get("people", [])
        categories = list(puzzle.get("attributes", {}).keys())
        answer_text = self._extract_final_answer_text(response)
        
        try:
            if answer_text and answer_text.strip().startswith("{"):
                answer = json.loads(answer_text.strip())
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
