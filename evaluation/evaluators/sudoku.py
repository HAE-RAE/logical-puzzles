"""
Sudoku Evaluator

Evaluates sudoku puzzle responses.
Answer format: integer (spotcheck sum) or digit code string
"""

import logging
import re
from typing import Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class SudokuEvaluator(BaseEvaluator):
    """
    Sudoku puzzle evaluator.

    Supports spotcheck validation: compares sum of cell values
    at selected positions.
    """

    SYSTEM_PROMPT = """You are a logic puzzle expert specializing in Sudoku.

Solve the Sudoku puzzle following standard rules:
- Each row must contain digits 1-9 exactly once
- Each column must contain digits 1-9 exactly once
- Each 3x3 box must contain digits 1-9 exactly once

Provide your answer in the exact format requested."""

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[int]:
        """Extract integer answer (spotcheck sum) from response."""
        # Remove code blocks
        response = re.sub(r'```[a-z]*\n?', '', response)
        response = re.sub(r'```', '', response)
        response = response.strip()

        # Priority 1: "Answer:" pattern with integer
        answer_matches = re.findall(
            r'(?:Answer|Output|Final\s*Answer)\s*[:\s]*(\d+)',
            response, re.IGNORECASE
        )
        if answer_matches:
            return int(answer_matches[-1])

        # Priority 2: last number in last 5 lines
        lines = response.strip().split('\n')
        for line in reversed(lines[-5:]):
            nums = re.findall(r'\b(\d+)\b', line.strip())
            if nums:
                return int(nums[-1])

        # Priority 3: any number
        all_nums = re.findall(r'\b(\d+)\b', response)
        if all_nums:
            return int(all_nums[-1])

        return None

    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[int]
    ) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0

        try:
            expected_num = int(expected)
        except (ValueError, TypeError):
            # Fall back to string comparison
            correct = str(predicted) == str(expected)
            return correct, 1.0 if correct else 0.0

        correct = predicted == expected_num
        return correct, 1.0 if correct else 0.0
