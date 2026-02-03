import logging
import re
from typing import List, Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class YachtDiceEvaluator(BaseEvaluator):
    """
    Yacht Dice 퍼즐 평가자
    
    답변 형식: 총점 (숫자)
    """
    
    SYSTEM_PROMPT = """You are an expert at solving Yacht Dice (Yahtzee) puzzles. Analyze the dice results and calculate the optimal total score."""
    
    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[int]:
        """
        LLM 응답에서 총점 추출
        
        Args:
            response: LLM 응답 텍스트
            puzzle: 퍼즐 데이터 (사용하지 않음)
        """
        # 최종 정답 섹션 추출
        final_section = self._extract_final_answer_section(response)
        
        # 다양한 패턴으로 총점 찾기
        patterns = [
            r'총점[:\s]*[=\s]*(\d+)',
            r'총[^\d]*점수[:\s]*[=\s]*(\d+)',
            r'합계[:\s]*[=\s]*(\d+)',
            r'total[:\s]*[=\s]*(\d+)',
            r'최종[^\d]*점수[:\s]*[=\s]*(\d+)',
            r'전체[^\d]*점수[:\s]*[=\s]*(\d+)',
        ]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, final_section, re.IGNORECASE))
            if matches:
                return int(matches[-1].group(1))
        
        # 마지막 시도: 응답 끝 부분에서 숫자 찾기
        lines = final_section.strip().split('\n')
        for line in reversed(lines[-5:]):
            if '총' in line or 'total' in line.lower() or '합' in line:
                numbers = re.findall(r'\d+', line)
                if numbers:
                    return int(max(numbers, key=int))
        
        return None
    
    def _extract_final_answer_section(self, response: str) -> str:
        """
        응답에서 최종 정답 섹션만 추출
        
        중간 과정, 시행착오, 고민 과정을 제거하고 최종 정답만 반환
        """
        # 최종 정답을 나타내는 키워드들
        final_keywords = [
            r'최종[^\n]*정답',
            r'최종[^\n]*할당',
            r'최종[^\n]*배정',
            r'정답[:\s]*',
            r'결론[:\s]*',
            r'final[^\n]*answer',
            r'final[^\n]*assignment',
            r'답[:\s]*\n',
        ]
        
        # 키워드 이후 부분을 찾기
        for keyword in final_keywords:
            match = re.search(keyword, response, re.IGNORECASE)
            if match:
                return response[match.start():]
        
        # 키워드가 없으면 "총점" 이전까지의 마지막 부분 추출
        total_patterns = [
            r'총[\s]*점[:\s]*\d+',
            r'total[\s]*[:\s]*\d+',
        ]
        
        for pattern in total_patterns:
            matches = list(re.finditer(pattern, response, re.IGNORECASE))
            if matches:
                last_match = matches[-1]
                start_pos = max(0, last_match.start() - 500)
                return response[start_pos:]
        
        return response
    
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
