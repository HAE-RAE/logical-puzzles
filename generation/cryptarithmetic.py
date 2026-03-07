"""Cryptarithmetic Puzzle Generator

Constructive generation: uses reverse arithmetic to guarantee valid puzzles,
with multiple mapping strategies for solution count control.
Supports both 2-operand (A + B = C) and 3-operand (A + B + C = D) puzzles.
"""

import itertools
import random
import string
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field


MAX_SOLUTIONS = 1  # All puzzles require exactly 1 unique solution


@dataclass
class PuzzleCandidate:
    """Represents a puzzle candidate with metadata."""
    word1: str
    word2: str
    result: str
    answer: str  # Primary answer (result value)
    unique_letters: int
    strategy: str  # Mapping strategy used
    word3: str = None  # Optional third operand
    valid_answers: List[str] = None  # All valid result values (for multi-solution puzzles)
    mapping: Dict[str, int] = None  # Letter-to-digit mapping from solution

    @property
    def puzzle_str(self) -> str:
        if self.word3:
            return f"{self.word1} + {self.word2} + {self.word3} = {self.result}"
        return f"{self.word1} + {self.word2} = {self.result}"

    @property
    def operands(self):
        """Return list of operand words."""
        ops = [self.word1, self.word2]
        if self.word3:
            ops.append(self.word3)
        return ops


# ============================================================
# Solver: Column-by-column backtracking
# ============================================================

def find_solutions(puzzle: tuple, max_count: int = 4) -> List[Tuple[str, Dict[str, int]]]:
    """
    Find all solutions up to max_count using column-by-column backtracking.

    puzzle: (word1, word2, result_word) or (word1, word2, word3, result_word)
    Returns list of (result_value, mapping) tuples.
    """
    *operand_words, result_word = puzzle
    all_letters = sorted(set(''.join(operand_words) + result_word))

    if len(all_letters) > 10:
        return []

    # First letters that cannot be 0
    first_letters = set()
    for w in list(operand_words) + [result_word]:
        if len(w) > 1:
            first_letters.add(w[0])

    # Build column constraints (right to left)
    all_words = list(operand_words) + [result_word]
    max_len = max(len(w) for w in all_words)
    reversed_ops = [w[::-1] for w in operand_words]
    wr = result_word[::-1]

    # Determine letter assignment order by columns (right to left)
    # This allows early pruning via arithmetic constraints
    ordered_letters = []
    seen = set()
    for col in range(max_len):
        for w in reversed_ops + [wr]:
            if col < len(w) and w[col] not in seen:
                ordered_letters.append(w[col])
                seen.add(w[col])
    # Add any remaining letters
    for letter in all_letters:
        if letter not in seen:
            ordered_letters.append(letter)
            seen.add(letter)

    solutions = []
    mapping = {}
    used_digits = set()

    def _check_columns(mapping):
        """Check all fully-assigned columns for consistency."""
        carry = 0
        for col in range(max_len):
            op_letters = [rw[col] if col < len(rw) else None for rw in reversed_ops]
            cr = wr[col] if col < len(wr) else None

            for ol in op_letters:
                if ol is not None and ol not in mapping:
                    return True
            if cr is not None and cr not in mapping:
                return True

            total = carry
            for ol in op_letters:
                total += mapping.get(ol, 0) if ol else 0
            dr = mapping.get(cr, 0) if cr else 0

            if total % 10 != dr:
                return False
            carry = total // 10
        return carry == 0

    def backtrack(idx):
        if len(solutions) >= max_count:
            return

        if idx == len(ordered_letters):
            if _check_columns(mapping):
                num_result = int("".join(str(mapping[c]) for c in result_word))
                solutions.append((str(num_result), dict(mapping)))
            return

        letter = ordered_letters[idx]

        for digit in range(10):
            if digit in used_digits:
                continue
            if digit == 0 and letter in first_letters:
                continue

            mapping[letter] = digit
            used_digits.add(digit)

            # Prune: check partial column constraints
            valid = True
            carry = 0
            for col in range(max_len):
                op_letters = [rw[col] if col < len(rw) else None for rw in reversed_ops]
                cr = wr[col] if col < len(wr) else None

                all_assigned = all(ol is None or ol in mapping for ol in op_letters)
                has_cr = cr is None or cr in mapping

                if all_assigned and has_cr:
                    total = carry
                    for ol in op_letters:
                        total += mapping.get(ol, 0) if ol else 0
                    dr = mapping.get(cr, 0) if cr else 0
                    if total % 10 != dr:
                        valid = False
                        break
                    carry = total // 10
                else:
                    break

            if valid:
                backtrack(idx + 1)

            del mapping[letter]
            used_digits.discard(digit)

    backtrack(0)
    return solutions


def count_solutions_fast(puzzle: tuple) -> int:
    """Quick solution count check."""
    return len(find_solutions(puzzle, max_count=MAX_SOLUTIONS + 1))


def has_valid_solutions(puzzle: tuple) -> bool:
    """Check if puzzle has exactly 1 solution."""
    return count_solutions_fast(puzzle) == 1


def get_solution(puzzle: tuple) -> Optional[Tuple[str, Dict[str, int]]]:
    """Get the single solution if exists."""
    solutions = find_solutions(puzzle, max_count=1)
    return solutions[0] if solutions else None


def get_all_result_values(puzzle: tuple, max_count: int = 20) -> List[str]:
    """Get all distinct result values from valid solutions."""
    solutions = find_solutions(puzzle, max_count=max_count)
    return list(set(s[0] for s in solutions))


# ============================================================
# Letter mapping strategies
# ============================================================

def _create_letter_mapping(unique_digits: List[str], strategy: str = 'random') -> Dict[str, str]:
    """
    Create digit-to-letter mapping with different strategies.

    Strategies:
    - 'random': Random letter assignment
    - 'sequential': A, B, C, ... assignment
    - 'reverse': Z, Y, X, ... assignment
    - 'vowel_first': Vowels for common digits
    """
    available_letters = list(string.ascii_uppercase)

    if strategy == 'random':
        random.shuffle(available_letters)
    elif strategy == 'reverse':
        available_letters = available_letters[::-1]
    elif strategy == 'vowel_first':
        vowels = list('AEIOU')
        consonants = [c for c in available_letters if c not in vowels]
        random.shuffle(vowels)
        random.shuffle(consonants)
        available_letters = vowels + consonants
    # 'sequential' uses default order

    return {digit: available_letters[i] for i, digit in enumerate(unique_digits)}


strategy_stats = {
    'random': {'tried': 0, 'success': 0},
    'sequential': {'tried': 0, 'success': 0},
    'reverse': {'tried': 0, 'success': 0},
    'vowel_first': {'tried': 0, 'success': 0},
    'random_fallback': {'tried': 0, 'success': 0}
}


def print_strategy_stats():
    """Print strategy success statistics."""
    print("\nMapping strategy statistics:")
    for name, stats in strategy_stats.items():
        rate = stats['success'] / stats['tried'] * 100 if stats['tried'] > 0 else 0
        print(f"  {name}: {stats['success']}/{stats['tried']} ({rate:.1f}%)")


# ============================================================
# Puzzle generation
# ============================================================

def count_carries(*nums: int) -> int:
    """Count carries in addition of multiple numbers."""
    carries = 0
    carry = 0
    str_nums = [str(n)[::-1] for n in nums]
    max_len = max(len(s) for s in str_nums)

    for i in range(max_len):
        total = carry
        for s in str_nums:
            total += int(s[i]) if i < len(s) else 0
        carry = total // 10
        if carry > 0:
            carries += 1

    return carries


def has_overflow(*nums: int) -> bool:
    """Check if result has more digits than the largest operand."""
    result = sum(nums)
    max_operand_digits = max(len(str(n)) for n in nums)
    result_digits = len(str(result))
    return result_digits > max_operand_digits


def generate_from_arithmetic(
    num1: int,
    num2: int,
    used_patterns: Set[str] = None,
    num3: int = None
) -> Optional[PuzzleCandidate]:
    """
    Constructively generate puzzle from valid arithmetic.

    Requires exactly 1 unique solution for all puzzle types.
    """
    if used_patterns is None:
        used_patterns = set()

    is_3op = num3 is not None
    if is_3op:
        result = num1 + num2 + num3
        str_nums = [str(num1), str(num2), str(num3)]
    else:
        result = num1 + num2
        str_nums = [str(num1), str(num2)]

    str_result = str(result)
    combined = ''.join(str_nums) + str_result

    unique_digits = list(set(combined))

    if len(unique_digits) > 10:
        return None

    max_allowed = MAX_SOLUTIONS

    strategies = ['random', 'sequential', 'reverse', 'vowel_first']
    random.shuffle(strategies)

    for strategy in strategies:
        strategy_stats[strategy]['tried'] += 1
        digit_to_letter = _create_letter_mapping(unique_digits, strategy)

        words = ["".join(digit_to_letter[d] for d in s) for s in str_nums]
        result_word = "".join(digit_to_letter[d] for d in str_result)

        pattern = "+".join(words) + "=" + result_word
        if pattern in used_patterns:
            continue

        puzzle = tuple(words + [result_word])

        solutions = find_solutions(puzzle, max_count=max_allowed + 1)
        solution_count = len(solutions)

        if 1 <= solution_count <= max_allowed:
            used_patterns.add(pattern)
            answer = str(result)
            all_values = list(set(s[0] for s in solutions))
            strategy_stats[strategy]['success'] += 1
            return PuzzleCandidate(
                word1=words[0],
                word2=words[1],
                result=result_word,
                answer=answer,
                unique_letters=len(set(''.join(words) + result_word)),
                strategy=strategy,
                word3=words[2] if len(words) > 2 else None,
                valid_answers=all_values,
                mapping=solutions[0][1]
            )

    # For 2-operand only: try a few more random mappings
    if not is_3op:
        for _ in range(5):
            strategy_stats['random_fallback']['tried'] += 1
            digit_to_letter = _create_letter_mapping(unique_digits, 'random')

            words = ["".join(digit_to_letter[d] for d in s) for s in str_nums]
            result_word = "".join(digit_to_letter[d] for d in str_result)

            pattern = "+".join(words) + "=" + result_word
            if pattern in used_patterns:
                continue

            puzzle = tuple(words + [result_word])

            solutions = find_solutions(puzzle, max_count=max_allowed + 1)
            solution_count = len(solutions)

            if 1 <= solution_count <= max_allowed:
                used_patterns.add(pattern)
                answer = str(result)
                all_values = list(set(s[0] for s in solutions))
                strategy_stats['random_fallback']['success'] += 1
                return PuzzleCandidate(
                    word1=words[0],
                    word2=words[1],
                    result=result_word,
                    answer=answer,
                    unique_letters=len(set(''.join(words) + result_word)),
                    strategy='random_fallback',
                    word3=None,
                    valid_answers=all_values,
                    mapping=solutions[0][1]
                )

    return None


def generate_puzzle_by_difficulty(
    difficulty: str,
    used_patterns: Set[str] = None
) -> Optional[PuzzleCandidate]:
    """Generate a puzzle for specified difficulty level."""
    if used_patterns is None:
        used_patterns = set()

    configs = {
        "easy": {
            "num_operands": 2,
            "num1_range": (100, 999),
            "num2_range": (10, 99),
            "min_carries": None,
            "max_carries": None,
            "require_overflow": None,
            "target_letters": (4, 5),
        },
        "medium": {
            "num_operands": 3,
            "num1_range": (100, 999),
            "num2_range": (10, 99),
            "num3_range": (10, 99),
            "min_carries": None,
            "max_carries": 1,
            "require_overflow": None,
            "target_letters": (4, 5),
        },
        "hard": {
            "num_operands": 3,
            "num1_range": (1000, 9999),
            "num2_range": (100, 999),
            "num3_range": (10, 99),
            "min_carries": 2,
            "max_carries": None,
            "require_overflow": None,
            "target_letters": (7, 9),
        },
    }

    config = configs.get(difficulty, configs["easy"])
    min_letters, max_letters = config["target_letters"]
    min_carries = config["min_carries"]
    max_carries = config["max_carries"]
    require_overflow = config["require_overflow"]
    num_operands = config["num_operands"]

    for _ in range(1000):
        num1 = random.randint(*config["num1_range"])
        num2 = random.randint(*config["num2_range"])
        num3 = None
        if num_operands == 3:
            num3 = random.randint(*config["num3_range"])

        operands = [num1, num2] + ([num3] if num3 is not None else [])
        result = sum(operands)

        if min_carries is not None:
            carries = count_carries(*operands)
            if carries < min_carries:
                continue
            if max_carries is not None and carries > max_carries:
                continue

        if require_overflow is True and not has_overflow(*operands):
            continue
        if require_overflow is False and has_overflow(*operands):
            continue

        combined = ''.join(str(n) for n in operands) + str(result)
        unique_digits = len(set(combined))

        if not (min_letters <= unique_digits <= max_letters):
            continue

        candidate = generate_from_arithmetic(num1, num2, used_patterns, num3=num3)

        if candidate and min_letters <= candidate.unique_letters <= max_letters:
            return candidate

    return None


# ============================================================
# Question formatting
# ============================================================

def create_question(candidate: PuzzleCandidate) -> str:
    """Create question text in English."""
    operand_words = candidate.operands
    max_word_len = max(len(w) for w in operand_words + [candidate.result])
    separator = '-' * (max_word_len + 2)

    operand_lines = f"  {operand_words[0]}\n"
    for w in operand_words[1:]:
        operand_lines += f"+ {w}\n"

    question = (
        f"Solve this cryptarithmetic puzzle where each letter represents a unique digit (0-9). "
        f"Different letters must map to different digits. "
        f"Leading letters cannot be zero. "
        f"Find the numeric value that {candidate.result} represents.\n\n"
        f"{operand_lines}"
        f"{separator}\n"
        f"= {candidate.result}"
    )
    return question


# ============================================================
# Dataset generation
# ============================================================

def generate_dataset(
    puzzles_per_difficulty: int = 3,
    difficulties: List[str] = None,
    verbose: bool = True
) -> List[Dict]:
    """Generate a complete dataset of puzzles."""
    if difficulties is None:
        difficulties = ["easy", "medium", "hard"]

    dataset = []
    used_patterns = set()

    for difficulty in difficulties:
        if verbose:
            print(f"\n=== Generating {difficulty} puzzles ===")

        for i in range(puzzles_per_difficulty):
            candidate = generate_puzzle_by_difficulty(difficulty, used_patterns)

            if candidate:
                question = create_question(candidate)

                puzzle_data = {
                    "question": question,
                    "answer": candidate.answer,
                    "difficulty": difficulty,
                    "puzzle": candidate.puzzle_str,
                    "valid_answers": candidate.valid_answers or [candidate.answer],
                    "mapping": candidate.mapping,
                }

                dataset.append(puzzle_data)

                if verbose:
                    print(f"  [{i+1}/{puzzles_per_difficulty}] {candidate.puzzle_str} -> {candidate.answer}")
            else:
                if verbose:
                    print(f"  [{i+1}/{puzzles_per_difficulty}] Failed to generate")

    if verbose:
        print_strategy_stats()

    return dataset


def generate_and_validate(
    word1: str,
    word2: str,
    result: str,
    word3: str = None
) -> Optional[Dict]:
    """
    Validate a manually specified puzzle and return formatted data if valid.
    """
    words = [word1.upper(), word2.upper()]
    if word3:
        words.append(word3.upper())
    words.append(result.upper())

    puzzle = tuple(words)

    if not has_valid_solutions(puzzle):
        sol_count = count_solutions_fast(puzzle)
        print(f"Invalid: puzzle has {sol_count} solution(s)")
        return None

    solutions = find_solutions(puzzle, max_count=1)
    if solutions:
        answer_val, mapping = solutions[0]

        all_letters = set(''.join(words))
        unique_letters = len(all_letters)

        if unique_letters <= 4:
            difficulty = "easy"
        elif unique_letters <= 6:
            difficulty = "medium"
        else:
            difficulty = "hard"

        candidate = PuzzleCandidate(
            word1=words[0],
            word2=words[1],
            result=words[-1],
            answer=answer_val,
            unique_letters=unique_letters,
            strategy='manual',
            word3=words[2] if len(words) == 4 else None,
            valid_answers=[answer_val],
            mapping=mapping
        )

        return {
            "question": create_question(candidate),
            "answer": answer_val,
            "difficulty": difficulty,
            "puzzle": candidate.puzzle_str,
            "mapping": mapping
        }

    return None


def create_dataset_files(num_questions: int):
    """
    Create dataset files for cryptarithmetic puzzles.

    Args:
        num_questions: Number of questions to generate

    Returns:
        Tuple[pd.DataFrame, List[Dict]]: (dataframe, json list)
    """
    import pandas as pd

    print(f"Generating {num_questions} cryptarithmetic puzzles...")

    difficulties = ["easy", "medium", "hard"]
    puzzles_per_diff = num_questions // len(difficulties)
    remainder = num_questions % len(difficulties)

    all_puzzles = []
    used_patterns = set()

    for i, difficulty in enumerate(difficulties):
        target_count = puzzles_per_diff + (1 if i < remainder else 0)

        if target_count == 0:
            continue

        print(f"\n=== Generating {difficulty} puzzles ({target_count} needed) ===")
        generated = 0
        attempts = 0
        max_total_attempts = 5000

        while generated < target_count and attempts < max_total_attempts:
            attempts += 1
            candidate = generate_puzzle_by_difficulty(
                difficulty,
                used_patterns=used_patterns
            )

            if candidate:
                puzzle_data = {
                    "id": f"cryptarithmetic_{len(all_puzzles)}",
                    "question": create_question(candidate),
                    "answer": candidate.answer,
                    "solution": candidate.puzzle_str,
                    "difficulty": difficulty,
                    "puzzle": candidate.puzzle_str,
                    "valid_answers": candidate.valid_answers or [candidate.answer],
                    "mapping": {k: v for k, v in candidate.mapping.items()} if candidate.mapping else {},
                }
                all_puzzles.append(puzzle_data)
                generated += 1
                print(f"  [{generated}/{target_count}] {candidate.puzzle_str} -> {candidate.answer}")

        if generated < target_count:
            print(f"  Warning: Only generated {generated}/{target_count} {difficulty} puzzles")

    print(f"\nGenerated {len(all_puzzles)} puzzles total")
    print_strategy_stats()

    df = pd.DataFrame(all_puzzles)

    # Save files
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # CSV
    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "cryptarithmetic.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV file created: {csv_path}")

    # JSONL
    json_dir = PROJECT_ROOT / "data" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = json_dir / "cryptarithmetic.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_puzzles:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL file created: {jsonl_path}")

    return df, all_puzzles


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Cryptarithmetic Puzzle Generator")
    parser.add_argument("--num", type=int, default=12, help="Number of questions to generate")

    args = parser.parse_args()

    create_dataset_files(num_questions=args.num)
