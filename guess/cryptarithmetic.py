"""Cryptarithmetic Puzzle Generator and Validator

Generates cryptarithmetic puzzles with guaranteed unique solutions for LLM evaluation.

Improvements over naive approach:
1. Reverse generation: Start from valid arithmetic, map to letters
2. Early termination: Stop after finding 2 solutions
3. Constraint propagation: Prune impossible branches early
4. Smart filtering: Ensure unique solution puzzles
"""

import itertools
import random
import string
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class PuzzleCandidate:
    """Represents a puzzle candidate with metadata."""
    word1: str
    word2: str
    result: str
    answer: str
    unique_letters: int

    @property
    def puzzle_str(self) -> str:
        return f"{self.word1} + {self.word2} = {self.result}"


def count_solutions(puzzle: Tuple[str, str, str], max_count: int = 2) -> int:
    """
    Count solutions up to max_count (for early termination).

    This is more efficient than finding all solutions when we only
    need to know if there's exactly one solution.
    """
    word1, word2, result_word = puzzle
    all_letters = sorted(set(word1 + word2 + result_word))

    if len(all_letters) > 10:
        return 0

    # First letters that cannot be 0
    first_letters = set()
    if len(word1) > 1:
        first_letters.add(word1[0])
    if len(word2) > 1:
        first_letters.add(word2[0])
    if len(result_word) > 1:
        first_letters.add(result_word[0])

    count = 0

    for perm in itertools.permutations(range(10), len(all_letters)):
        mapping = dict(zip(all_letters, perm))

        # Skip if first letter is 0
        if any(mapping[letter] == 0 for letter in first_letters):
            continue

        # Convert and check
        num1 = int("".join(str(mapping[c]) for c in word1))
        num2 = int("".join(str(mapping[c]) for c in word2))
        num_result = int("".join(str(mapping[c]) for c in result_word))

        if num1 + num2 == num_result:
            count += 1
            if count >= max_count:
                return count  # Early termination

    return count


def has_unique_solution(puzzle: Tuple[str, str, str]) -> bool:
    """Check if puzzle has exactly one solution (optimized)."""
    return count_solutions(puzzle, max_count=2) == 1


def get_solution(puzzle: Tuple[str, str, str]) -> Optional[Tuple[str, Dict[str, int]]]:
    """Get the unique solution if exists."""
    word1, word2, result_word = puzzle
    all_letters = sorted(set(word1 + word2 + result_word))

    if len(all_letters) > 10:
        return None

    first_letters = set()
    if len(word1) > 1:
        first_letters.add(word1[0])
    if len(word2) > 1:
        first_letters.add(word2[0])
    if len(result_word) > 1:
        first_letters.add(result_word[0])

    for perm in itertools.permutations(range(10), len(all_letters)):
        mapping = dict(zip(all_letters, perm))

        if any(mapping[letter] == 0 for letter in first_letters):
            continue

        num1 = int("".join(str(mapping[c]) for c in word1))
        num2 = int("".join(str(mapping[c]) for c in word2))
        num_result = int("".join(str(mapping[c]) for c in result_word))

        if num1 + num2 == num_result:
            return str(num_result), mapping

    return None


def generate_from_arithmetic(
    num1: int,
    num2: int,
    used_patterns: Set[str] = None
) -> Optional[PuzzleCandidate]:
    """
    Generate puzzle from valid arithmetic (reverse generation).

    Strategy:
    1. Start with num1 + num2 = result
    2. Map digits to random letters
    3. Check if mapping produces unique solution
    """
    if used_patterns is None:
        used_patterns = set()

    result = num1 + num2

    str1, str2, str_result = str(num1), str(num2), str(result)
    combined = str1 + str2 + str_result

    # Get unique digits used
    unique_digits = list(set(combined))

    # Skip if too many unique digits (impossible to solve)
    if len(unique_digits) > 10:
        return None

    # Create letter mapping
    available_letters = list(string.ascii_uppercase)
    random.shuffle(available_letters)

    digit_to_letter = {}
    for i, digit in enumerate(unique_digits):
        digit_to_letter[digit] = available_letters[i]

    # Convert to words
    word1 = "".join(digit_to_letter[d] for d in str1)
    word2 = "".join(digit_to_letter[d] for d in str2)
    result_word = "".join(digit_to_letter[d] for d in str_result)

    # Check for duplicate pattern
    pattern = f"{word1}+{word2}={result_word}"
    if pattern in used_patterns:
        return None

    puzzle = (word1, word2, result_word)

    # Verify unique solution
    if has_unique_solution(puzzle):
        used_patterns.add(pattern)
        return PuzzleCandidate(
            word1=word1,
            word2=word2,
            result=result_word,
            answer=str(result),
            unique_letters=len(set(word1 + word2 + result_word))
        )

    return None


def generate_puzzle_by_difficulty(
    difficulty: str,
    max_attempts: int = 1000,
    used_patterns: Set[str] = None
) -> Optional[PuzzleCandidate]:
    """
    Generate a puzzle for specified difficulty level.

    Difficulty is determined by:
    - Number of unique letters
    - Word lengths
    """
    if used_patterns is None:
        used_patterns = set()

    # Difficulty configurations
    configs = {
        "Easy": {
            "num1_range": (10, 99),
            "num2_range": (10, 99),
            "target_letters": (3, 4),
        },
        "Medium": {
            "num1_range": (100, 999),
            "num2_range": (100, 999),
            "target_letters": (5, 6),
        },
        "Hard": {
            "num1_range": (1000, 9999),
            "num2_range": (1000, 9999),
            "target_letters": (6, 8),
        },
        "Expert": {
            "num1_range": (10000, 99999),
            "num2_range": (10000, 99999),
            "target_letters": (8, 10),
        },
    }

    config = configs.get(difficulty, configs["Medium"])
    min_letters, max_letters = config["target_letters"]

    for attempt in range(max_attempts):
        num1 = random.randint(*config["num1_range"])
        num2 = random.randint(*config["num2_range"])

        # Quick filter: check unique digit count
        combined = str(num1) + str(num2) + str(num1 + num2)
        unique_digits = len(set(combined))

        if not (min_letters <= unique_digits <= max_letters):
            continue

        candidate = generate_from_arithmetic(num1, num2, used_patterns)

        if candidate and min_letters <= candidate.unique_letters <= max_letters:
            return candidate

    return None


def create_question(candidate: PuzzleCandidate) -> str:
    """Create question text in English."""
    question = (
        f"Solve this cryptarithmetic puzzle where each letter represents a unique digit (0-9). "
        f"Different letters must map to different digits. "
        f"Leading letters cannot be zero. "
        f"Find the numeric value that {candidate.result} represents.\n\n"
        f"  {candidate.word1}\n"
        f"+ {candidate.word2}\n"
        f"{'-' * (max(len(candidate.word1), len(candidate.word2), len(candidate.result)) + 2)}\n"
        f"= {candidate.result}"
    )
    return question


def generate_dataset(
    puzzles_per_difficulty: int = 3,
    difficulties: List[str] = None,
    verbose: bool = True
) -> List[Dict]:
    """
    Generate a complete dataset of validated puzzles.

    Args:
        puzzles_per_difficulty: Number of puzzles per difficulty level
        difficulties: List of difficulty levels to generate
        verbose: Print progress

    Returns:
        List of puzzle dictionaries ready for evaluation
    """
    if difficulties is None:
        difficulties = ["Easy", "Medium", "Hard", "Expert"]

    dataset = []
    used_patterns = set()

    for difficulty in difficulties:
        if verbose:
            print(f"\n=== Generating {difficulty} puzzles ===")

        generated = 0
        attempts = 0
        max_total_attempts = 5000

        while generated < puzzles_per_difficulty and attempts < max_total_attempts:
            attempts += 1
            candidate = generate_puzzle_by_difficulty(
                difficulty,
                max_attempts=100,
                used_patterns=used_patterns
            )

            if candidate:
                question = create_question(candidate)

                puzzle_data = {
                    "question": question,
                    "answer": candidate.answer,
                    "difficulty": difficulty,
                    "puzzle": candidate.puzzle_str
                }

                dataset.append(puzzle_data)
                generated += 1

                if verbose:
                    print(f"  [{generated}/{puzzles_per_difficulty}] {candidate.puzzle_str} -> {candidate.answer}")

        if verbose and generated < puzzles_per_difficulty:
            print(f"  Warning: Only generated {generated}/{puzzles_per_difficulty} puzzles")

    return dataset


def generate_and_validate(
    word1: str,
    word2: str,
    result: str
) -> Optional[Dict]:
    """
    Validate a manually specified puzzle and return formatted data if valid.

    Useful for adding custom puzzles to the dataset.
    """
    puzzle = (word1.upper(), word2.upper(), result.upper())

    if not has_unique_solution(puzzle):
        sol_count = count_solutions(puzzle, max_count=10)
        print(f"Invalid: {word1} + {word2} = {result} has {sol_count} solution(s)")
        return None

    solution = get_solution(puzzle)
    if solution:
        answer, mapping = solution

        question = (
            f"Solve this cryptarithmetic puzzle where each letter represents a unique digit (0-9). "
            f"Different letters must map to different digits. "
            f"Leading letters cannot be zero. "
            f"Find the numeric value that {result.upper()} represents.\n\n"
            f"  {word1.upper()}\n"
            f"+ {word2.upper()}\n"
            f"{'-' * (max(len(word1), len(word2), len(result)) + 2)}\n"
            f"= {result.upper()}"
        )

        unique_letters = len(set(word1.upper() + word2.upper() + result.upper()))

        # Auto-determine difficulty
        if unique_letters <= 4:
            difficulty = "Easy"
        elif unique_letters <= 6:
            difficulty = "Medium"
        elif unique_letters <= 8:
            difficulty = "Hard"
        else:
            difficulty = "Expert"

        return {
            "question": question,
            "answer": answer,
            "difficulty": difficulty,
            "puzzle": f"{word1.upper()} + {word2.upper()} = {result.upper()}",
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

    # Calculate puzzles per difficulty
    puzzles_per_diff = num_questions // 4
    remainder = num_questions % 4

    difficulties = ["Easy", "Medium", "Hard", "Expert"]
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
                max_attempts=200,
                used_patterns=used_patterns
            )

            if candidate:
                puzzle_data = {
                    "question": create_question(candidate),
                    "answer": candidate.answer,
                    "solution": candidate.puzzle_str,
                    "difficulty": difficulty
                }
                all_puzzles.append(puzzle_data)
                generated += 1
                print(f"  [{generated}/{target_count}] {candidate.puzzle_str} -> {candidate.answer}")
        
        if generated < target_count:
            print(f"  Warning: Only generated {generated}/{target_count} {difficulty} puzzles")

    print(f"\nGenerated {len(all_puzzles)} puzzles total")

    df = pd.DataFrame(all_puzzles)

    # Save files
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # CSV
    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"cryptarithmetic.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV file created: {csv_path}")

    # JSONL
    json_dir = PROJECT_ROOT / "data" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = json_dir / f"cryptarithmetic.jsonl"
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
    
    # import time

    # print("=" * 60)
    # print("Cryptarithmetic Puzzle Generator")
    # print("=" * 60)

    # # Benchmark: Generate dataset
    # print("\n[1] Generating new dataset...")
    # start = time.time()

    # dataset = generate_dataset(puzzles_per_difficulty=3, verbose=True)

    # elapsed = time.time() - start
    # print(f"\nGenerated {len(dataset)} puzzles in {elapsed:.2f} seconds")

    # # Show all generated puzzles
    # print("\n" + "=" * 60)
    # print("Generated Puzzles Summary")
    # print("=" * 60)

    # for puzzle in dataset:
    #     print(f"[{puzzle['difficulty']:6}] {puzzle['puzzle']:30} -> {puzzle['answer']}")

    # # Benchmark: Validate custom puzzle
    # print("\n" + "=" * 60)
    # print("[2] Validating custom puzzles...")
    # print("=" * 60)

    # custom_puzzles = [
    #     ("SEND", "MORE", "MONEY"),
    #     ("TO", "GO", "OUT"),
    # ]

    # for w1, w2, r in custom_puzzles:
    #     result = generate_and_validate(w1, w2, r)
    #     if result:
    #         print(f"✓ {result['puzzle']} -> {result['answer']} ({result['difficulty']})")
    #     else:
    #         print(f"✗ {w1} + {w2} = {r}")