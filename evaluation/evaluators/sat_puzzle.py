"""
SAT Puzzle Evaluator

SAT 논리 퍼즐 평가 (영문)
"""

import logging
import json
import re
from typing import Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator

logger = logging.getLogger(__name__)


class SATPuzzleEvaluator(BaseEvaluator):
    """
    SAT Puzzle 퍼즐 평가자
    
    답변 형식: JSON (변수: bool 매핑, 부분 점수 지원)
    """
    
    SYSTEM_PROMPT = """### Instructions
You are an expert at propositional logic and SAT-style team puzzles.

### Rules
1. Satisfy every clue in the user message and assign each named variable true or false consistently.
2. Put one-line JSON on the final line with lowercase boolean literals true/false; do not wrap it in markdown code fences.
3. Explain your reasoning clearly, then present your final conclusion in the format below.

### Output format
Your final line must be:
Answer: {"K team": false, "L team": true}
"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 명제 논리·SAT형 팀 추론 퍼즐을 정확히 푸는 전문가입니다.

### 규칙
1. 사용자 메시지의 모든 제약을 만족하도록 각 변수에 참/거짓을 일관되게 부여하세요.
2. 마지막 줄은 한 줄 JSON이며 불리언은 소문자 true/false만 사용하고, 마크다운 코드블록으로 감싸지 마세요.
3. 풀이 과정을 명확히 서술한 뒤, 최종 결론을 아래 형식으로 제시하세요.

### 출력 형식
마지막 줄은 반드시 아래 형식으로 작성하세요:
Answer: {"K팀": false, "L팀": true}
"""

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[Dict[str, bool]]:
        """
        LLM 응답에서 JSON 답변 추출
        
        Args:
            response: LLM 응답 텍스트
            puzzle: 퍼즐 데이터
        """
        variables = puzzle.get("variables", [])
        answer_text = self._extract_final_answer_text(response)

        def _validate(answer_obj: Any) -> Optional[Dict[str, bool]]:
            if not isinstance(answer_obj, dict):
                return None
            for var in variables:
                if var not in answer_obj or not isinstance(answer_obj[var], bool):
                    return None
            return answer_obj

        def _parse_json_like(text: str) -> Optional[Dict[str, bool]]:
            if not text:
                return None
            candidate = text.strip()
            candidate = re.sub(r'//.*', '', candidate)
            # Accept Python-style bools too: True/False -> true/false
            candidate = re.sub(r'\bTrue\b', 'true', candidate)
            candidate = re.sub(r'\bFalse\b', 'false', candidate)
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                return None
            return _validate(parsed)
        
        try:
            if answer_text and answer_text.strip().startswith("{"):
                parsed = _parse_json_like(answer_text)
                if parsed is not None:
                    return parsed

            # Handle multiline style:
            # Answer:
            # ```json
            # { ... }
            # ```
            answer_block = re.search(
                r'(?is)answer\s*[:：]\s*```(?:json)?\s*(\{.*?\})\s*```',
                response,
            )
            if answer_block:
                parsed = _parse_json_like(answer_block.group(1))
                if parsed is not None:
                    return parsed

            # Handle multiline style without fences:
            # Answer:
            # { ... }
            answer_multiline = re.search(
                r'(?is)answer\s*[:：]\s*(\{.*?\})',
                response,
            )
            if answer_multiline:
                parsed = _parse_json_like(answer_multiline.group(1))
                if parsed is not None:
                    return parsed

            # Try to find JSON in the response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                parsed = _parse_json_like(json_match.group(1))
                if parsed is not None:
                    return parsed
            
            # Try to find JSON without markdown
            json_match = re.search(r'\{[^{}]*"[^"]+"\s*:\s*(?:true|false|True|False)[^{}]*\}', response, re.DOTALL)
            if json_match:
                parsed = _parse_json_like(json_match.group(0))
                if parsed is not None:
                    return parsed
            
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
