"""
Minesweeper Evaluator

지뢰찾기 퍼즐 평가
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class MinesweeperEvaluator(BaseEvaluator):
    """
    Minesweeper 퍼즐 평가자
    
    답변 형식: 좌표 리스트 (예: [(0,1), (0,3), (1,2)])
    """
    
    SYSTEM_PROMPT = """You are solving a Minesweeper puzzle. Analyze the puzzle using logical reasoning and deduce the exact location of all mines."""
    
    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[List[Tuple[int, int]]]:
        """
        LLM 응답에서 좌표 리스트 추출
        
        Args:
            response: LLM 응답 텍스트
            puzzle: 퍼즐 데이터
        """
        num_mines = puzzle.get("mines", 0)
        
        # "Final answer:" 이후 부분 추출
        final_answer_pattern = r'(?:final answer|answer):\s*(.*)'
        final_match = re.search(final_answer_pattern, response, re.IGNORECASE | re.DOTALL)
        
        if final_match:
            search_text = final_match.group(1)
        else:
            # fallback: 마지막 200자
            search_text = response[-200:] if len(response) > 200 else response
        
        # (r,c) 패턴 찾기
        pattern = r'\((\d+),\s*(\d+)\)'
        matches = re.findall(pattern, search_text)
        
        coords = [(int(r), int(c)) for r, c in matches]
        
        # 중복 제거 (순서 유지)
        coords = list(dict.fromkeys(coords))
        
        return coords[:num_mines] if len(coords) >= num_mines else coords
    
    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[List[Tuple[int, int]]]
    ) -> Tuple[bool, float]:
        """
        답변 확인
        
        expected는 solution 그리드일 수 있으므로 좌표 리스트로 변환 필요
        
        Returns:
            (is_correct, partial_score) 튜플
        """
        if predicted is None or len(predicted) == 0:
            return False, 0.0
        
        # expected가 리스트인 경우 (좌표 리스트)
        if isinstance(expected, list):
            expected_set = set(expected)
            predicted_set = set(predicted)
            
            if expected_set == predicted_set:
                return True, 1.0
            
            # 부분 점수: 교집합 비율
            intersection = len(expected_set & predicted_set)
            union = len(expected_set | predicted_set)
            partial_score = intersection / union if union > 0 else 0.0
            return False, partial_score
        
        # expected가 그리드인 경우 (solution lines)
        # 이 경우는 puzzle 데이터에서 solution을 가져와야 함
        # 하지만 _check_answer는 expected만 받으므로, 
        # 정확한 비교를 위해서는 puzzle을 받아야 함
        # 일단 좌표 리스트로 가정
        return False, 0.0
