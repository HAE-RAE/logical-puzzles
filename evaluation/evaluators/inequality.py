"""
Inequality Evaluator

Evaluates inequality puzzle responses with constraint-based fallback validation.
Answer format: digit sequence (e.g., "3142" or "5 3 12 4 10 1" for size>9)
"""

import logging
import re
from typing import Dict, Any, Tuple, Optional

from ..core.base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class InequalityEvaluator(BaseEvaluator):
    """
    Inequality puzzle evaluator.

    Supports both concatenated (size<=9) and space-separated (size>9) formats.
    Falls back to constraint-based validation when answer doesn't match pre-computed value.
    """

    SYSTEM_PROMPT = """You are an expert puzzle solver specializing in logical constraint puzzles.

Solve the inequality puzzle by filling blanks with numbers.
Each number must be used exactly once.
Inequality symbols (< or >) between positions must be satisfied.

Provide your answer in this exact format:
Answer: [numbers separated by spaces]"""

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[str]:
        """Extract number sequence from LLM response."""
        expected_answer = puzzle.get("answer", "")
        size = puzzle.get("size", 0)
        if not size and expected_answer:
            # Infer size from answer
            nums = re.findall(r'\d+', expected_answer)
            size = len(nums) if len(nums) > 1 else len(expected_answer)

        # Priority 1: "Answer:" pattern
        patterns = [
            r'Answer:\s*([\d\s]+)',
            r'answer:\s*([\d\s]+)',
            r'solution[:\s]+([\d\s]+)',
            r'sequence[:\s]+([\d\s]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                raw = match.group(1).strip()
                nums = re.findall(r'\d+', raw)
                if len(nums) >= size:
                    nums = nums[:size]
                    if size > 9:
                        return ' '.join(nums)
                    return ''.join(nums)

        # Priority 2: last numeric sequence in last 200 chars
        last_part = response[-200:] if len(response) > 200 else response
        all_nums = re.findall(r'\d+', last_part)
        if len(all_nums) >= size:
            nums = all_nums[-size:]
            if size > 9:
                return ' '.join(nums)
            return ''.join(nums)

        return None

    def _to_int_list(self, s: str, size: int):
        """Convert answer string to list of ints, handling both formats."""
        if not s:
            return []
        nums = re.findall(r'\d+', s)
        # Concatenated single digits (e.g. "641532" for size 6)
        if len(nums) == 1 and len(nums[0]) == size and size <= 9:
            return [int(d) for d in nums[0]]
        return [int(n) for n in nums]

    def _validate_inequality_solution(self, answer_str: str, puzzle: Dict) -> bool:
        """Validate if answer satisfies inequality constraints directly."""
        size = puzzle.get("size", 0)

        # Parse answer to int list
        parsed = self._to_int_list(answer_str, size)
        if len(parsed) != size:
            return False

        # Check permutation of 1..N
        if sorted(parsed) != list(range(1, size + 1)):
            return False

        # Check given numbers
        given_positions = puzzle.get("given_positions", [])
        given_values = puzzle.get("given_values", [])
        for pos, val in zip(given_positions, given_values):
            if pos < len(parsed) and parsed[pos] != val:
                return False

        # Parse inequalities from problem string
        problem_str = puzzle.get("problem", puzzle.get("solution", ""))
        if not problem_str:
            return False

        parts = problem_str.split()
        inequalities = parts[1::2]

        for i, ineq in enumerate(inequalities):
            if i + 1 >= len(parsed):
                break
            if ineq == "?":
                continue
            if ineq == "<" and parsed[i] >= parsed[i + 1]:
                return False
            elif ineq == ">" and parsed[i] <= parsed[i + 1]:
                return False

        return True

    def _check_answer(
        self,
        expected: str,
        predicted: Optional[str]
    ) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0

        # Normalize both to int lists for comparison
        size = len(re.findall(r'\d+', str(expected)))
        if size <= 1:
            size = len(str(expected))

        expected_list = self._to_int_list(str(expected), size)
        predicted_list = self._to_int_list(str(predicted), size)

        correct = expected_list == predicted_list
        return correct, 1.0 if correct else 0.0
