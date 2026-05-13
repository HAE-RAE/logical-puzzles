"""
Cryptarithmetic Evaluator

Evaluates cryptarithmetic puzzle responses.
Answer format: integer (numeric value of result word, or spotcheck sum)

Features ported from gemini_eval.py:
- Letter hiding: replaces some letters with '*' for medium/hard
- Spotcheck: asks for sum of K selected letter digit values
- Fallback validation: extracts mapping from response and verifies arithmetic
"""

import logging
import random
import re
import time
from typing import Dict, Any, List, Set, Tuple, Optional, TYPE_CHECKING

from ..core.base import BaseEvaluator, EvaluationResult
from ..task_names import locale_from_task_name

if TYPE_CHECKING:
    from ..model.base import BaseLLMClient

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration constants
# ============================================================================

SPOTCHECK_K = {"easy": 2, "medium": 2, "hard": 3}
# All use simple SUM (sum-of-squares was too harsh combined with letter hiding)
SPOTCHECK_USE_SQUARES = {"easy": False, "medium": False, "hard": False}
# Letter hiding: replace some letter positions with * to remove structural info
HIDE_RATIO = {"easy": 0.0, "medium": 0.1, "hard": 0.25}


class CryptarithmeticEvaluator(BaseEvaluator):
    """
    Cryptarithmetic puzzle evaluator.

    Supports:
    - Letter hiding (medium/hard): some letters replaced with '*'
    - Spotcheck validation: compares sum of digit values for selected letters
    - Fallback validation: extracts full mapping from response and verifies arithmetic
    """

    SYSTEM_PROMPT = """### Instructions
You are an expert at cryptarithmetic (alphametic) puzzles.

### Rules
1. Map each distinct letter to a unique digit 0–9; leading letters of any word cannot be zero.
2. Treat '*' as an unknown letter slot that may stand for any consistent letter; preserve all arithmetic constraints in the puzzle.
3. Explain your reasoning clearly, then present your final conclusion in the format below.

### Output format
Your final line must be:
Answer: [number]
"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 암호산술(숫자 맞추기) 퍼즐을 정확히 푸는 전문가입니다.

### 규칙
1. 서로 다른 글자는 서로 다른 숫자(0–9); 각 단어의 맨 앞 글자는 0이 될 수 없습니다.
2. '*'는 알 수 없는 글자 자리로, 지문과 산술 제약에 맞게 일관되게 해석하세요.
3. 풀이 과정을 명확히 서술한 뒤, 최종 결론을 아래 형식으로 제시하세요.

### 출력 형식
마지막 줄은 반드시 아래 형식으로 작성하세요:
Answer: [숫자]
"""

    def _is_korean(self, puzzle: Optional[Dict] = None) -> bool:
        """Prefer task_name (e.g. …_ko_easy); else infer from expected answer."""
        task = getattr(self, "_task_name", None) or ""
        hint = locale_from_task_name(task)
        if hint is not None:
            return hint
        if puzzle is not None:
            expected = puzzle.get("answer", "")
            return bool(re.search(r"[가-힣]", str(expected)))
        return False

    def _get_system_prompt(self, puzzle: Dict) -> str:
        if self._is_korean(puzzle):
            return self.KOREAN_SYSTEM_PROMPT
        return self.SYSTEM_PROMPT

    # ========================================================================
    # Answer parsing and checking (base interface)
    # ========================================================================

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[str]:
        """Extract numeric answer from LLM response."""
        # Remove code blocks and prioritize unified answer label extraction.
        response = self._strip_code_fences(response)
        answer_text = self._extract_final_answer_text(response) or response

        # Priority 1: "Answer:" pattern
        answer_matches = re.findall(
            r'(?:Answer|Output|Final\s*Answer)\s*[:\s]*(\d+)',
            answer_text, re.IGNORECASE
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
            match = re.search(pattern, answer_text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1)

        # Priority 3: last multi-digit number in last 5 lines
        lines = answer_text.strip().split('\n')
        for line in reversed(lines[-5:]):
            match = re.search(r'\b(\d{2,})\b', line.strip())
            if match:
                return match.group(1)

        # Priority 4: last multi-digit number anywhere
        numbers = re.findall(r'\b(\d{2,})\b', answer_text)
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

        correct = str(predicted) == str(expected)
        return correct, 1.0 if correct else 0.0

    # ========================================================================
    # Letter hiding
    # ========================================================================

    def _hide_letters_in_puzzle(self, puzzle_data: Dict) -> Tuple[str, Set[str]]:
        """Replace some letter positions with '*' to hide structural information.

        Keeps at least one visible occurrence of each unique letter.
        Never hides leading letters (first char of each word).

        Returns (modified_question_text, set_of_visible_letters)
        """
        difficulty = puzzle_data.get("difficulty", "easy")
        hide_ratio = HIDE_RATIO.get(difficulty, 0.0)
        mapping = puzzle_data.get("mapping", {})

        if hide_ratio <= 0 or not mapping:
            return puzzle_data["question"], set(mapping.keys()) if mapping else set()

        puzzle_str = puzzle_data.get("puzzle", "")
        if not puzzle_str:
            return puzzle_data["question"], set(mapping.keys())

        # Parse puzzle string (e.g. "SEND + MORE = MONEY")
        eq_parts = puzzle_str.split("=")
        result_word = eq_parts[-1].strip()
        operand_parts = eq_parts[0].strip().split("+")
        operand_words = [p.strip() for p in operand_parts]
        all_words = operand_words + [result_word]

        # Collect hideable positions (non-leading characters only)
        hideable = []
        for w_idx, word in enumerate(all_words):
            for c_idx in range(1, len(word)):
                hideable.append((w_idx, c_idx, word[c_idx]))

        # Count total occurrences of each letter across all positions
        letter_count: Dict[str, int] = {}
        for word in all_words:
            for ch in word:
                letter_count[ch] = letter_count.get(ch, 0) + 1
        remaining = dict(letter_count)

        # Select positions to hide, keeping at least 1 visible per letter
        num_to_hide = max(1, int(len(hideable) * hide_ratio))
        random.shuffle(hideable)

        hide_set: set = set()
        for w_idx, c_idx, ch in hideable:
            if len(hide_set) >= num_to_hide:
                break
            if remaining[ch] > 1:
                hide_set.add((w_idx, c_idx))
                remaining[ch] -= 1

        # Build modified words
        mod_all = []
        for w_idx, word in enumerate(all_words):
            chars = list(word)
            for c_idx in range(len(chars)):
                if (w_idx, c_idx) in hide_set:
                    chars[c_idx] = '*'
            mod_all.append(''.join(chars))

        mod_operands = mod_all[:-1]
        mod_result = mod_all[-1]

        # Build formatted question
        max_len = max(len(w) for w in mod_all)
        separator = '-' * (max_len + 2)

        op_lines = f"  {mod_operands[0]}\n"
        for w in mod_operands[1:]:
            op_lines += f"+ {w}\n"

        question = (
            f"Solve this cryptarithmetic puzzle where each letter represents a unique digit (0-9). "
            f"Different letters must map to different digits. "
            f"Leading letters cannot be zero. "
            f"'*' represents an unknown letter — it could be any letter, including one already visible in the puzzle. "
            f"Each '*' independently may represent a different letter.\n\n"
            f"{op_lines}"
            f"{separator}\n"
            f"= {mod_result}"
        )

        # Compute visible letters
        visible: Set[str] = set()
        for w_idx, word in enumerate(all_words):
            for c_idx, ch in enumerate(word):
                if (w_idx, c_idx) not in hide_set:
                    visible.add(ch)

        return question, visible

    # ========================================================================
    # Spotcheck generation
    # ========================================================================

    def _generate_spotcheck(self, puzzle: Dict, visible_letters: Set[str]) -> Dict:
        """Generate spotcheck info for a puzzle.

        Selects K visible letters and computes expected sum (or sum of squares).
        Only uses visible letters (not hidden by *).
        """
        difficulty = puzzle.get("difficulty", "easy")
        mapping = puzzle.get("mapping", {})
        if not mapping:
            return {}

        k = SPOTCHECK_K.get(difficulty, 3)
        use_squares = SPOTCHECK_USE_SQUARES.get(difficulty, False)

        available = [l for l in mapping.keys() if l in visible_letters]
        k = min(k, len(available))
        if k <= 0:
            return {}

        selected = random.sample(available, k)

        if use_squares:
            expected_value = sum(mapping[letter] ** 2 for letter in selected)
        else:
            expected_value = sum(mapping[letter] for letter in selected)

        return {
            "letters": selected,
            "expected_value": expected_value,
            "use_squares": use_squares,
        }

    def _build_spotcheck_suffix(self, spotcheck: Dict) -> str:
        """Build prompt suffix with spotcheck instructions."""
        selected_letters = spotcheck.get("letters", [])
        expected_value = spotcheck.get("expected_value")
        use_squares = spotcheck.get("use_squares", False)

        if not selected_letters or expected_value is None:
            return ""

        letters_str = ", ".join(selected_letters)
        if use_squares:
            return (
                f"\n\nAfter solving, for each of these letters: {letters_str}\n"
                f"1. Find the digit assigned to each letter\n"
                f"2. SQUARE each digit (multiply it by itself)\n"
                f"3. Sum all the squared values\n"
                f"Answer: [sum of squares as integer]"
            )
        else:
            return (
                f"\n\nAfter solving, calculate the SUM of the digits assigned to "
                f"these letters: {letters_str}\n"
                f"Answer: [sum as integer]"
            )

    # ========================================================================
    # Fallback validation
    # ========================================================================

    def _extract_mapping_from_response(self, response_text: str) -> Optional[Dict[str, int]]:
        """Extract letter-to-digit mapping from model response.

        Handles patterns like: A=1, B=2 or A: 1, B: 2 or A = 1
        """
        patterns = [
            r'\b([A-Z])\s*=\s*(\d)\b',
            r'\b([A-Z])\s*:\s*(\d)\b',
        ]
        mapping: Dict[str, int] = {}
        for pattern in patterns:
            matches = re.findall(pattern, response_text)
            if matches:
                for letter, digit in matches:
                    mapping[letter] = int(digit)
                break
        return mapping if mapping else None

    def _verify_arithmetic(self, puzzle_str: str, mapping: Dict[str, int]) -> bool:
        """Verify a mapping satisfies the cryptarithmetic puzzle arithmetic.

        puzzle_str format: "ABC + DE = FGH" or "ABC + DE + FG = HIJ"
        """
        # Check all digits are unique
        if len(set(mapping.values())) != len(mapping):
            return False

        # Parse puzzle: split by = to get left and right sides
        parts = puzzle_str.split('=')
        if len(parts) != 2:
            return False

        left_side = parts[0].strip()
        result_word = parts[1].strip()

        operand_words = [w.strip() for w in left_side.split('+')]

        # Check leading letters are not zero
        all_words = operand_words + [result_word]
        for word in all_words:
            if word and word[0] in mapping and mapping[word[0]] == 0:
                return False

        # Convert words to numbers using mapping
        def word_to_num(word: str) -> Optional[int]:
            num_str = ''
            for ch in word:
                if ch not in mapping:
                    return None
                num_str += str(mapping[ch])
            return int(num_str) if num_str else None

        operand_nums = [word_to_num(w) for w in operand_words]
        result_num = word_to_num(result_word)

        if None in operand_nums or result_num is None:
            return False

        return sum(operand_nums) == result_num

    # ========================================================================
    # Puzzle preparation (preprocessing)
    # ========================================================================

    def _prepare_puzzle_for_eval(self, puzzle: Dict) -> Tuple[Dict, str]:
        """Preprocess puzzle: apply letter hiding and generate spotcheck.

        Enriches the puzzle dict with transient _spotcheck_* keys.
        Returns (enriched_puzzle, user_content).
        """
        # Shallow copy to avoid side effects
        puzzle = dict(puzzle)

        # Apply letter hiding
        modified_question, visible_letters = self._hide_letters_in_puzzle(puzzle)

        # Generate spotcheck
        spotcheck = self._generate_spotcheck(puzzle, visible_letters)

        # Build user content
        suffix = self._build_spotcheck_suffix(spotcheck)
        user_content = modified_question + suffix

        # Enrich puzzle dict with transient metadata
        puzzle["_spotcheck_expected"] = spotcheck.get("expected_value")
        puzzle["_spotcheck_letters"] = spotcheck.get("letters", [])
        puzzle["_spotcheck_use_squares"] = spotcheck.get("use_squares", False)

        return puzzle, user_content

    # ========================================================================
    # Overridden evaluation methods
    # ========================================================================

    def _evaluate_single(
        self,
        puzzle: Dict[str, Any],
        llm_client: "BaseLLMClient"
    ) -> EvaluationResult:
        """Evaluate a single puzzle with letter hiding and spotcheck."""
        puzzle, user_content = self._prepare_puzzle_for_eval(puzzle)

        messages = [
            {"role": "system", "content": self._get_system_prompt(puzzle)},
            {"role": "user", "content": user_content}
        ]

        start = time.time()
        try:
            response, usage = llm_client.generate(messages)
            latency = (time.time() - start) * 1000
            return self._process_response(puzzle, response, latency, usage)
        except Exception as e:
            latency = (time.time() - start) * 1000
            return self._process_response(puzzle, "", latency, {"error": str(e)})

    async def _evaluate_async(
        self,
        puzzles: List[Dict[str, Any]],
        llm_client: "BaseLLMClient",
        verbose: bool = True,
        max_concurrent: int = 10
    ) -> List[EvaluationResult]:
        """Async evaluation with letter hiding and spotcheck."""
        # Preprocess all puzzles
        enriched_puzzles = []
        messages_list = []
        for puzzle in puzzles:
            enriched, user_content = self._prepare_puzzle_for_eval(puzzle)
            enriched_puzzles.append(enriched)
            messages_list.append([
                {"role": "system", "content": self._get_system_prompt(enriched)},
                {"role": "user", "content": user_content}
            ])

        total_puzzles = len(enriched_puzzles)
        task_name = getattr(self, '_task_name', None)
        task_prefix = f"[{task_name}] " if task_name else ""

        if verbose:
            logger.info(f"{task_prefix}Starting async evaluation: {total_puzzles} puzzles, max_concurrent={max_concurrent}")

        start_time = time.time()

        def progress_callback(completed, total):
            if verbose:
                percentage = (completed / total) * 100
                if completed % max(1, total // 10) == 0 or completed == total:
                    logger.info(f"{task_prefix}API calls progress: {completed}/{total} ({percentage:.0f}%)")

        responses = await llm_client.async_batch_generate(
            messages_list,
            max_concurrent=max_concurrent,
            progress_callback=progress_callback if verbose else None
        )
        total_latency = (time.time() - start_time) * 1000

        if verbose:
            logger.info(f"{task_prefix}API calls completed: {total_puzzles}/{total_puzzles} in {total_latency:.0f}ms ({total_latency/total_puzzles:.0f}ms per puzzle)")

        # Process results
        results = []
        correct_count = 0
        error_count = 0

        for puzzle, (response, usage) in zip(enriched_puzzles, responses):
            latency_ms = usage.get("latency_ms", 0)
            result = self._process_response(puzzle, response, latency_ms, usage)

            if result.correct:
                correct_count += 1
            if result.error:
                error_count += 1

            results.append(result)

        if verbose:
            incorrect_count = total_puzzles - correct_count - error_count
            logger.info(f"Processing completed: {correct_count} correct, {incorrect_count} incorrect, {error_count} errors")

        return results

    def _process_response(
        self,
        puzzle: Dict[str, Any],
        response: str,
        latency_ms: float,
        usage: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Process response with spotcheck comparison and fallback validation."""
        usage = usage or {}

        if "error" in usage:
            return self._create_error_result(
                puzzle, response if response else "", latency_ms, usage["error"]
            )

        try:
            predicted = self._parse_answer(response, puzzle)
            spotcheck_expected = puzzle.get("_spotcheck_expected")

            if spotcheck_expected is not None:
                # Spotcheck mode: compare parsed answer to expected value
                correct = str(predicted) == str(spotcheck_expected)

                # Fallback: if spotcheck fails, try extracting full mapping
                if not correct and response:
                    extracted = self._extract_mapping_from_response(response)
                    puzzle_str = puzzle.get("puzzle", "")
                    if extracted and puzzle_str and self._verify_arithmetic(puzzle_str, extracted):
                        correct = True

                return EvaluationResult(
                    puzzle_id=puzzle["id"],
                    difficulty=puzzle.get("difficulty", "Unknown"),
                    correct=correct,
                    partial_score=1.0 if correct else 0.0,
                    expected=str(spotcheck_expected),
                    predicted=predicted,
                    raw_response=response,
                    latency_ms=latency_ms,
                )
            else:
                # No spotcheck: fall back to standard check
                correct, score = self._check_answer(puzzle["answer"], predicted)

                # Fallback: verify arithmetic from extracted mapping
                if not correct and response:
                    extracted = self._extract_mapping_from_response(response)
                    puzzle_str = puzzle.get("puzzle", "")
                    if extracted and puzzle_str and self._verify_arithmetic(puzzle_str, extracted):
                        correct = True
                        score = 1.0

                return EvaluationResult(
                    puzzle_id=puzzle["id"],
                    difficulty=puzzle.get("difficulty", "Unknown"),
                    correct=correct,
                    partial_score=score,
                    expected=puzzle["answer"],
                    predicted=predicted,
                    raw_response=response,
                    latency_ms=latency_ms,
                )
        except Exception as e:
            return self._create_error_result(puzzle, response, latency_ms, str(e))
