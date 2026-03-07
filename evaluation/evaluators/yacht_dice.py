"""
Yacht Dice Evaluator

Evaluates Yacht Dice puzzle responses.
Answer format: integer (total score or spotcheck round sum)
"""

import logging
import re
from typing import Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class YachtDiceEvaluator(BaseEvaluator):
    """
    Yacht Dice puzzle evaluator.

    Parses total score or spotcheck sum from LLM response.
    """

    SYSTEM_PROMPT = """You are an expert at solving Yacht Dice optimization problems.

Yacht Dice is a dice game where you roll 5 dice for 12 rounds and assign each round to a scoring category.

Scoring Categories:
- Aces through Sixes: Sum of dice showing that number
- Three-of-a-Kind: Sum of all dice if at least 3 match
- Four-of-a-Kind: Sum of all dice if at least 4 match
- Full House: 25 points for exactly 3 of one number and 2 of another
- Small Straight: 30 points for 4 consecutive numbers
- Large Straight: 40 points for 5 consecutive numbers
- Yacht: 50 points for all 5 dice showing the same number

Upper Section Bonus: If the sum of Aces through Sixes is 63 or more, add 35 bonus points.

Each category can only be used once.

CRITICAL: Your very last line MUST be in this exact format:
Answer: [number]"""

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[int]:
        """Extract integer answer from LLM response."""
        # Remove code blocks
        response = re.sub(r'```[a-z]*\n?', '', response)
        response = re.sub(r'```', '', response)
        response = response.strip()

        # Priority 1: "Answer:" pattern
        answer_matches = re.findall(
            r'(?:Answer|Output|Final\s*Answer)\s*[:\s]*(\d+)',
            response, re.IGNORECASE
        )
        if answer_matches:
            return int(answer_matches[-1])

        # Priority 2: Total/sum patterns
        total_patterns = [
            r'[Tt]otal[:\s]*[=\s]*(\d+)',
            r'[Ss]um[:\s]*[=\s]*(\d+)',
        ]
        for pattern in total_patterns:
            matches = re.findall(pattern, response)
            if matches:
                return int(matches[-1])

        # Priority 3: last number in last 5 lines
        lines = response.strip().split('\n')
        for line in reversed(lines[-5:]):
            nums = re.findall(r'\b(\d+)\b', line.strip())
            if nums:
                # Pick the largest number (likely total score)
                return int(max(nums, key=int))

        # Priority 4: last number anywhere
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
            return False, 0.0

        correct = predicted == expected_num
        return correct, 1.0 if correct else 0.0
