"""
Minesweeper Evaluator

Evaluates minesweeper puzzle responses using weighted coordinate sum format.
Answer format: single integer (sum of row*C+col for each mine)
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional, Set

from ..core.base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class MinesweeperEvaluator(BaseEvaluator):
    """
    Minesweeper puzzle evaluator.

    Uses weighted coordinate sum scoring: sum(row * C + col) for each mine.
    """

    SYSTEM_PROMPT = """You are solving a Minesweeper puzzle. Analyze the grid using logical reasoning and deduce the exact location of all mines.

Output the sum of linear indices (row * columns + col) for all mine positions as a single integer.

Answer: [number]"""

    @staticmethod
    def _bitstring_to_coordinates(solution_str: str, R: int, C: int) -> Set[Tuple[int, int]]:
        """Convert solution bitstring to coordinate set."""
        coords = set()
        for i, cell in enumerate(solution_str):
            if cell == '1':
                r, c = divmod(i, C)
                coords.add((r, c))
        return coords

    @staticmethod
    def _compute_total_sum(coords: Set[Tuple[int, int]], C: int) -> int:
        """Compute weighted coordinate sum: sum(row * C + col)."""
        if not coords:
            return 0
        return sum(r * C + c for r, c in coords)

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[int]:
        """Parse LLM output to extract total sum as single integer."""
        # Remove code blocks
        response = re.sub(r'```[a-z]*\n?', '', response)
        response = re.sub(r'```', '', response)
        response = response.strip()

        # Priority 1: "Answer:" or "Output:" line with integer
        answer_matches = re.findall(
            r'(?:Answer|Output|Final\s*Answer)\s*[:\s]*(\d+)',
            response, re.IGNORECASE
        )
        if answer_matches:
            return int(answer_matches[-1])

        # Priority 2: last 5 lines for multi-digit integer
        lines = response.strip().split('\n')
        for line in reversed(lines[-5:]):
            match = re.search(r'\b(\d{2,})\b', line.strip())
            if match:
                return int(match.group(1))

        # Priority 3: last multi-digit integer in entire text
        matches = re.findall(r'\b(\d{2,})\b', response)
        if matches:
            return int(matches[-1])

        # Priority 4: single digit
        matches = re.findall(r'\b(\d+)\b', response)
        if matches:
            return int(matches[-1])

        return None

    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[int]
    ) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0

        # expected might be a bitstring or a pre-computed sum
        try:
            expected_num = int(expected)
        except (ValueError, TypeError):
            return False, 0.0

        correct = predicted == expected_num
        return correct, 1.0 if correct else 0.0
