"""
Cryptarithmetic Evaluator

Evaluates cryptarithmetic puzzle responses.
Answer format: integer (numeric value of result word, or spotcheck sum)
"""

import logging
import re
from typing import Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class CryptarithmeticEvaluator(BaseEvaluator):
    """
    Cryptarithmetic puzzle evaluator.

    Supports spotcheck validation: compares sum of digit values
    for selected letters.
    """

    SYSTEM_PROMPT = """You are an expert puzzle solver specializing in cryptarithmetic problems.

Rules:
- Each letter represents a unique digit (0-9)
- Different letters must map to different digits
- Leading letters cannot be zero
- '*' represents an unknown letter that could be any letter

Solve the puzzle and provide your answer in this exact format:
Answer: [number]"""

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[str]:
        """Extract numeric answer from LLM response."""
        # Remove code blocks
        response = re.sub(r'```[a-z]*\n?', '', response)
        response = re.sub(r'```', '', response)

        # Priority 1: "Answer:" pattern
        answer_matches = re.findall(
            r'(?:Answer|Output|Final\s*Answer)\s*[:\s]*(\d+)',
            response, re.IGNORECASE
        )
        if answer_matches:
            return answer_matches[-1]

        # Priority 2: "result is/equals" pattern
        patterns = [
            r'result\s*(?:is|=|equals)\s*(\d+)',
            r'value\s*(?:is|=|equals)\s*(\d+)',
            r'= (\d+)$',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1)

        # Priority 3: last multi-digit number in last 5 lines
        lines = response.strip().split('\n')
        for line in reversed(lines[-5:]):
            match = re.search(r'\b(\d{2,})\b', line.strip())
            if match:
                return match.group(1)

        # Priority 4: last multi-digit number anywhere
        numbers = re.findall(r'\b(\d{2,})\b', response)
        if numbers:
            return numbers[-1]

        return None

    def _check_answer(
        self,
        expected: str,
        predicted: Optional[str]
    ) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0

        # Check valid_answers if available (multiple valid solutions)
        correct = str(predicted) == str(expected)
        return correct, 1.0 if correct else 0.0
