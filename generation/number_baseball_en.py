"""Number Baseball (Bulls and Cows) Puzzle Generator and Validator

Constructive generation: builds puzzles by selecting high-information
hints that progressively narrow solutions to 1.
Supports 3-6 digit puzzles via backtracking solver.
"""

import random
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from itertools import permutations
from dataclasses import dataclass
from enum import Enum


@dataclass
class Hint:
    """Represents a guess and its result in Bulls and Cows game"""
    guess: str
    strikes: int
    balls: int

    def __str__(self):
        return f"{self.guess}: {self.strikes}S {self.balls}B"

    def to_dict(self):
        return {
            "guess": self.guess,
            "strikes": self.strikes,
            "balls": self.balls
        }


class Difficulty(Enum):
    """Difficulty levels for problem generation"""
    EASY = 1      # 3 digits, moderate hints
    MEDIUM = 2    # 4 digits, fewer hints
    HARD = 3      # 6 digits, minimal hints


class BullsAndCows:
    """Core game logic for Bulls and Cows (Number Baseball)"""

    def __init__(self, num_digits: int = 3):
        if num_digits not in [3, 4, 5, 6]:
            raise ValueError("Number of digits must be 3, 4, 5, or 6")
        self.num_digits = num_digits

    def generate_number(self) -> str:
        digits = list(range(10))
        random.shuffle(digits)
        return ''.join(str(d) for d in digits[:self.num_digits])

    def calculate_strikes_balls(self, secret: str, guess: str) -> Tuple[int, int]:
        if len(secret) != len(guess):
            raise ValueError("Secret and guess must have the same length")

        strikes = 0
        balls = 0

        for i, digit in enumerate(guess):
            if digit == secret[i]:
                strikes += 1
            elif digit in secret:
                balls += 1

        return strikes, balls

    def check_number_against_hints(self, number: str, hints: List[Hint]) -> bool:
        for hint in hints:
            strikes, balls = self.calculate_strikes_balls(number, hint.guess)
            if strikes != hint.strikes or balls != hint.balls:
                return False
        return True

    def find_all_solutions(self, hints: List[Hint], max_count: int = 0) -> List[str]:
        """Find all possible numbers that satisfy the given hints.

        Args:
            hints: List of hints to satisfy
            max_count: Stop after finding this many solutions (0 = unlimited)
        """
        solutions = []

        for perm in permutations('0123456789', self.num_digits):
            number = ''.join(perm)
            if self.check_number_against_hints(number, hints):
                solutions.append(number)
                if max_count > 0 and len(solutions) >= max_count:
                    break

        return solutions

    def has_unique_solution(self, hints: List[Hint]) -> bool:
        solutions = self.find_all_solutions(hints, max_count=2)
        return len(solutions) == 1

    def generate_hint(self, secret: str, max_attempts: int = 100) -> Optional[Hint]:
        attempts = 0
        while attempts < max_attempts:
            guess = self.generate_number()
            if guess != secret:
                strikes, balls = self.calculate_strikes_balls(secret, guess)
                return Hint(guess, strikes, balls)
            attempts += 1
        return None


MAX_SOLUTIONS = 1  # Only allow exactly 1 solution


class ProblemGenerator:
    """
    Constructive puzzle generator for Bulls and Cows.

    Strategy: Generate hints based on information value, progressively
    adding hints until solution count reaches 1.
    """

    def __init__(self):
        self.game_3digit = BullsAndCows(3)
        self.game_4digit = BullsAndCows(4)
        self.game_5digit = BullsAndCows(5)
        self.game_6digit = BullsAndCows(6)

    def _calculate_hint_info_value(self, hint: Hint, game: BullsAndCows,
                                    current_solutions: List[str]) -> int:
        """
        Calculate information value of a hint.
        Higher value = hint eliminates more invalid solutions.
        """
        if not current_solutions:
            return 0

        eliminated = 0
        for candidate in current_solutions:
            s, b = game.calculate_strikes_balls(hint.guess, candidate)
            if s != hint.strikes or b != hint.balls:
                eliminated += 1

        return eliminated

    def _select_best_hint(self, game: BullsAndCows, secret: str,
                          existing_hints: List[Hint], current_solutions: List[str],
                          difficulty: Difficulty, candidates: List[Hint]) -> Optional[Hint]:
        """Select the best hint that reduces solution count toward 1."""
        best_hint = None
        best_solution_count = float('inf')

        for hint in candidates:
            if self._is_duplicate_hint(hint, existing_hints):
                continue

            if not self._hint_matches_difficulty(hint, difficulty):
                continue

            test_hints = existing_hints + [hint]
            new_solutions = game.find_all_solutions(test_hints, max_count=0)

            if len(new_solutions) == 1:
                return hint

            if len(new_solutions) >= 1 and len(new_solutions) < best_solution_count:
                best_solution_count = len(new_solutions)
                best_hint = hint

        return best_hint

    def _hint_matches_difficulty(self, hint: Hint, difficulty: Difficulty) -> bool:
        """Check if hint matches difficulty constraints."""
        if difficulty == Difficulty.HARD:
            return True
        elif difficulty == Difficulty.MEDIUM:
            return hint.strikes <= 1
        elif difficulty == Difficulty.EASY:
            return hint.strikes <= 1
        return True

    def generate_problem(self, difficulty: Difficulty, max_retries: int = 100) -> Dict:
        """
        Constructively generate a puzzle with exactly 1 solution.

        Process:
        1. Generate secret number
        2. Generate candidate hints with varying information
        3. Select hints that maximize information gain
        4. Stop when solution count reaches 1
        5. Retry with fresh randomisation if needed
        """
        if difficulty == Difficulty.EASY:
            game = self.game_3digit
            num_digits = 3
        elif difficulty == Difficulty.MEDIUM:
            game = self.game_4digit
            num_digits = 4
        else:  # HARD
            game = self.game_6digit
            num_digits = 6

        min_hints = {
            Difficulty.EASY: 4,
            Difficulty.MEDIUM: 3,
            Difficulty.HARD: 2
        }
        max_hints = {
            Difficulty.EASY: 6,
            Difficulty.MEDIUM: 4,
            Difficulty.HARD: 3
        }

        for retry in range(max_retries):
            secret = game.generate_number()

            # Generate pool of candidate hints
            hint_pool = []
            for _ in range(50):
                hint = game.generate_hint(secret)
                if hint and hint not in hint_pool:
                    hint_pool.append(hint)

            # Progressively select hints to narrow solutions
            hints = []
            all_solutions = game.find_all_solutions([], max_count=MAX_SOLUTIONS + 1)

            while len(hints) < max_hints[difficulty]:
                solutions = game.find_all_solutions(hints, max_count=MAX_SOLUTIONS + 1) if hints else all_solutions

                if len(solutions) == 1 and len(hints) >= min_hints[difficulty]:
                    break

                best_hint = self._select_best_hint(game, secret, hints, solutions,
                                                   difficulty, hint_pool)

                if best_hint:
                    hints.append(best_hint)
                    hint_pool.remove(best_hint)
                else:
                    new_hint = game.generate_hint(secret)
                    if new_hint and not self._is_duplicate_hint(new_hint, hints):
                        if self._hint_matches_difficulty(new_hint, difficulty):
                            hints.append(new_hint)
                    else:
                        break

            # Final solution check
            solutions = game.find_all_solutions(hints, max_count=MAX_SOLUTIONS + 1) if hints else [secret]

            if len(solutions) == 1:
                return {
                    "difficulty": difficulty.name.lower(),
                    "num_digits": num_digits,
                    "hints": [hint.to_dict() for hint in hints],
                    "hint_text": self._format_hints(hints),
                    "answer": solutions[0],
                    "problem_text": self._create_problem_text(num_digits, hints)
                }

            # Hard difficulty: allow non-unique solutions (<=5) if we exhausted hints
            if difficulty == Difficulty.HARD and len(solutions) <= 5 and len(hints) >= min_hints[difficulty]:
                return {
                    "difficulty": difficulty.name.lower(),
                    "num_digits": num_digits,
                    "hints": [hint.to_dict() for hint in hints],
                    "hint_text": self._format_hints(hints),
                    "answer": secret,
                    "problem_text": self._create_problem_text(num_digits, hints)
                }

        raise RuntimeError(
            f"Failed to generate {difficulty.name} puzzle with exactly 1 solution "
            f"after {max_retries} retries"
        )

    def _is_duplicate_hint(self, hint: Hint, hints: List[Hint]) -> bool:
        for h in hints:
            if h.guess == hint.guess:
                return True
        return False

    def _format_hints(self, hints: List[Hint]) -> List[str]:
        return [str(hint) for hint in hints]

    def _create_problem_text(self, num_digits: int, hints: List[Hint]) -> str:
        hint_strs = [f"[{hint.guess}: {hint.strikes}S {hint.balls}B]" for hint in hints]
        hints_text = ", ".join(hint_strs)
        return f"Find the {num_digits}-digit number with distinct digits that satisfies all the following hints: {hints_text}"


# ============================================================
# Question formatting
# ============================================================

def create_question(problem: Dict) -> str:
    """Create question text in English."""
    num_digits = problem['num_digits']
    hints = problem['hints']

    hints_text = "\n".join([
        f"  {i+1}. Guess: {h['guess']} → {h['strikes']} Strike(s), {h['balls']} Ball(s)"
        for i, h in enumerate(hints)
    ])

    question = f"""Solve this Number Baseball (Bulls and Cows) puzzle.

Rules:
- The secret number has {num_digits} digits, each digit is unique (0-9)
- "Strike" means a digit is correct AND in the correct position
- "Ball" means a digit is correct BUT in the wrong position
- Your task: Find the secret number that satisfies ALL hints

Hints:
{hints_text}

Think step by step and find the unique {num_digits}-digit secret number.

Provide your answer in this format:
Answer: [the {num_digits}-digit secret number]"""

    return question


def validate_problem(problem: Dict) -> Tuple[bool, str]:
    """Validate a generated problem for correctness."""
    try:
        num_digits = problem['num_digits']
        game = BullsAndCows(num_digits)

        hints = [Hint(h['guess'], h['strikes'], h['balls']) for h in problem['hints']]

        answer = problem['answer']
        if len(answer) != num_digits:
            return False, f"Answer length {len(answer)} doesn't match num_digits {num_digits}"

        if len(set(answer)) != num_digits:
            return False, f"Answer {answer} doesn't have unique digits"

        if not game.check_number_against_hints(answer, hints):
            return False, f"Answer {answer} doesn't satisfy all hints"

        solutions = game.find_all_solutions(hints)
        if len(solutions) == 0:
            return False, "No solution exists for the given hints"
        elif len(solutions) > 1:
            return False, f"Multiple solutions exist: {solutions}"
        elif solutions[0] != answer:
            return False, f"Solution {solutions[0]} doesn't match answer {answer}"

        return True, "Problem is valid with unique solution"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


# ============================================================
# Dataset generation
# ============================================================

def create_dataset_files(num_questions: int):
    """
    Create dataset files for number baseball puzzles.

    Args:
        num_questions: Number of questions to generate

    Returns:
        Tuple[pd.DataFrame, List[Dict]]: (dataframe, json list)
    """
    import pandas as pd

    print(f"Generating {num_questions} number baseball puzzles...")

    generator = ProblemGenerator()

    difficulties = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]
    puzzles_per_diff = num_questions // len(difficulties)
    remainder = num_questions % len(difficulties)

    all_puzzles = []

    for i, difficulty in enumerate(difficulties):
        count = puzzles_per_diff + (1 if i < remainder else 0)
        diff_name = difficulty.name.lower()

        if count == 0:
            continue

        print(f"\n=== Generating {diff_name} puzzles ({count} needed) ===")

        for j in range(count):
            try:
                problem = generator.generate_problem(difficulty)
                is_valid, msg = validate_problem(problem)

                if is_valid or (difficulty == Difficulty.HARD):
                    reordered = {
                        'id': f'number_baseball_en_{len(all_puzzles)}',
                        'question': create_question(problem),
                        'answer': problem['answer'],
                        'difficulty': diff_name,
                        'num_digits': problem['num_digits'],
                        'hints': problem['hints'],
                        'hint_text': problem['hint_text'],
                        'problem_text': problem['problem_text']
                    }
                    all_puzzles.append(reordered)
                    print(f"  [{j+1}/{count}] digits={problem['num_digits']}, "
                          f"hints={len(problem['hints'])}, answer={problem['answer']}")
                else:
                    print(f"  [{j+1}/{count}] Validation failed: {msg}")
            except RuntimeError as e:
                print(f"  [{j+1}/{count}] Failed: {e}")

    print(f"\nGenerated {len(all_puzzles)} puzzles")

    df = pd.DataFrame(all_puzzles)

    # Save files
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # CSV
    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "number_baseball_en.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV file created: {csv_path}")

    # JSONL
    json_dir = PROJECT_ROOT / "data" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = json_dir / "number_baseball_en.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_puzzles:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL file created: {jsonl_path}")

    return df, all_puzzles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Number Baseball Puzzle Generator")
    parser.add_argument("--num", type=int, default=12, help="Number of questions to generate")

    args = parser.parse_args()

    print("=" * 60)
    print("Number Baseball (Bulls and Cows) Puzzle Generator")
    print("=" * 60)

    create_dataset_files(num_questions=args.num)
