"""Number Baseball (Bulls and Cows) Puzzle Generator and Validator

Generates number baseball puzzles with guaranteed unique solutions for LLM evaluation.
The game involves guessing a secret number based on hints (strikes and balls).
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
    EASY = 1      # 3 digits, 3-4 hints, direct hints
    MEDIUM = 2    # 3 digits, 3-5 hints, mixed hints
    HARD = 3      # 4 digits, 4-5 hints, strategic hints
    EXPERT = 4    # 4 digits, minimal hints (3-4), requires deep reasoning


class BullsAndCows:
    """Core game logic for Bulls and Cows (Number Baseball)"""

    def __init__(self, num_digits: int = 3):
        """
        Initialize the game with specified number of digits.

        Args:
            num_digits: Number of digits in the secret number (3 or 4)
        """
        if num_digits not in [3, 4]:
            raise ValueError("Number of digits must be 3 or 4")
        self.num_digits = num_digits

    def generate_number(self) -> str:
        """
        Generate a random number with unique digits.

        Returns:
            A string representing a number with unique digits
        """
        digits = list(range(10))
        # For numbers starting with 0 is allowed in our game
        random.shuffle(digits)
        return ''.join(str(d) for d in digits[:self.num_digits])

    def calculate_strikes_balls(self, secret: str, guess: str) -> Tuple[int, int]:
        """
        Calculate strikes and balls for a guess against the secret number.

        Args:
            secret: The secret number
            guess: The guess to evaluate

        Returns:
            Tuple of (strikes, balls)
        """
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
        """
        Check if a number satisfies all given hints.

        Args:
            number: The number to check
            hints: List of hints to validate against

        Returns:
            True if the number satisfies all hints, False otherwise
        """
        for hint in hints:
            strikes, balls = self.calculate_strikes_balls(number, hint.guess)
            if strikes != hint.strikes or balls != hint.balls:
                return False
        return True

    def find_all_solutions(self, hints: List[Hint]) -> List[str]:
        """
        Find all possible numbers that satisfy the given hints.

        Args:
            hints: List of hints to satisfy

        Returns:
            List of all numbers that satisfy all hints
        """
        solutions = []

        # Generate all possible numbers with unique digits
        for perm in permutations('0123456789', self.num_digits):
            number = ''.join(perm)
            if self.check_number_against_hints(number, hints):
                solutions.append(number)

        return solutions

    def has_unique_solution(self, hints: List[Hint]) -> bool:
        """
        Check if the given hints have exactly one solution.

        Args:
            hints: List of hints to check

        Returns:
            True if there's exactly one solution, False otherwise
        """
        solutions = self.find_all_solutions(hints)
        return len(solutions) == 1

    def generate_hint(self, secret: str, max_attempts: int = 100) -> Optional[Hint]:
        """
        Generate a random hint for the given secret number.

        Args:
            secret: The secret number
            max_attempts: Maximum attempts to find a valid guess

        Returns:
            A Hint object or None if no valid guess found
        """
        attempts = 0
        while attempts < max_attempts:
            guess = self.generate_number()
            if guess != secret:  # Don't use the secret itself as a hint
                strikes, balls = self.calculate_strikes_balls(secret, guess)
                return Hint(guess, strikes, balls)
            attempts += 1
        return None


class ProblemGenerator:
    """Generate Bulls and Cows problems with various difficulty levels"""

    def __init__(self):
        self.game_3digit = BullsAndCows(3)
        self.game_4digit = BullsAndCows(4)

    def generate_problem(self, difficulty: Difficulty, max_attempts: int = 100) -> Optional[Dict]:
        """
        Generate a problem with the specified difficulty level.

        Args:
            difficulty: The difficulty level for the problem
            max_attempts: Maximum attempts to generate a valid problem

        Returns:
            Dictionary containing problem data or None if generation fails
        """
        if difficulty in [Difficulty.EASY, Difficulty.MEDIUM]:
            game = self.game_3digit
            num_digits = 3
        else:
            game = self.game_4digit
            num_digits = 4

        # Generate secret number
        secret = game.generate_number()

        # Generate hints based on difficulty
        hints = self._generate_hints_by_difficulty(game, secret, difficulty, max_attempts)

        if not hints:
            return None

        # Verify unique solution
        solutions = game.find_all_solutions(hints)
        if len(solutions) != 1:
            # Try to refine hints
            refined = self._refine_hints(game, secret, hints, difficulty, max_attempts // 2)
            if refined:
                return refined
            # If refinement fails, return None to regenerate
            return None

        # Create problem dictionary
        problem = {
            "difficulty": difficulty.name,
            "num_digits": num_digits,
            "hints": [hint.to_dict() for hint in hints],
            "hint_text": self._format_hints(hints),
            "answer": secret,
            "problem_text": self._create_problem_text(num_digits, hints)
        }

        return problem

    def _generate_hints_by_difficulty(self, game: BullsAndCows, secret: str,
                                     difficulty: Difficulty, max_attempts: int) -> Optional[List[Hint]]:
        """Generate hints based on difficulty level"""
        hints = []

        if difficulty == Difficulty.EASY:
            # Easy: 3-4 hints with good distribution of strikes and balls
            num_hints = random.randint(3, 4)
            # Try to include at least one hint with high strikes
            for _ in range(max_attempts):
                hint = game.generate_hint(secret)
                if hint and hint.strikes >= 1:
                    hints.append(hint)
                    break

        elif difficulty == Difficulty.MEDIUM:
            # Medium: 3-5 hints with mixed information
            num_hints = random.randint(3, 5)
            # Include varied hints
            for _ in range(max_attempts):
                hint = game.generate_hint(secret)
                if hint:
                    hints.append(hint)
                    if len(hints) >= 2:
                        break

        elif difficulty == Difficulty.HARD:
            # Hard: 4-5 hints, some with no matches
            num_hints = random.randint(4, 5)
            # Include at least one zero match hint
            for _ in range(max_attempts):
                hint = game.generate_hint(secret)
                if hint and hint.strikes == 0 and hint.balls == 0:
                    hints.append(hint)
                    break

        elif difficulty == Difficulty.EXPERT:
            # Expert: 4-6 strategic hints that require deep deduction
            num_hints = random.randint(4, 6)
            # Mix of strategic hint types
            for _ in range(max_attempts):
                hint = game.generate_hint(secret)
                if hint:
                    hints.append(hint)
                    if len(hints) >= 2:
                        break

        # Fill remaining hints
        attempts = 0
        while len(hints) < num_hints and attempts < max_attempts:
            hint = game.generate_hint(secret)
            if hint and not self._is_duplicate_hint(hint, hints):
                hints.append(hint)
            attempts += 1

        return hints if len(hints) >= 3 else None

    def _is_duplicate_hint(self, hint: Hint, hints: List[Hint]) -> bool:
        """Check if a hint already exists in the list"""
        for h in hints:
            if h.guess == hint.guess:
                return True
        return False

    def _refine_hints(self, game: BullsAndCows, secret: str, hints: List[Hint],
                     difficulty: Difficulty, max_attempts: int) -> Optional[Dict]:
        """Refine hints to ensure unique solution"""
        attempts = 0
        max_hints = {
            Difficulty.EASY: 5,
            Difficulty.MEDIUM: 6,
            Difficulty.HARD: 7,
            Difficulty.EXPERT: 8
        }

        while attempts < max_attempts:
            solutions = game.find_all_solutions(hints)

            if len(solutions) == 1:
                # Success! Unique solution found
                return {
                    "difficulty": difficulty.name,
                    "num_digits": game.num_digits,
                    "hints": [hint.to_dict() for hint in hints],
                    "hint_text": self._format_hints(hints),
                    "answer": secret,
                    "problem_text": self._create_problem_text(game.num_digits, hints)
                }

            elif len(solutions) == 0:
                # No solution - hints are contradictory, regenerate
                return None

            else:
                # Multiple solutions - add more hints if within limit
                if len(hints) >= max_hints[difficulty]:
                    # Too many hints needed, regenerate
                    return None

                # Find a hint that eliminates some solutions
                best_hint = None
                best_reduction = 0

                for _ in range(20):
                    candidate = game.generate_hint(secret)
                    if candidate and not self._is_duplicate_hint(candidate, hints):
                        # Check how many solutions this hint eliminates
                        test_hints = hints + [candidate]
                        new_solutions = game.find_all_solutions(test_hints)
                        reduction = len(solutions) - len(new_solutions)

                        if 0 < len(new_solutions) < len(solutions) and reduction > best_reduction:
                            best_hint = candidate
                            best_reduction = reduction
                            # If this hint reduces to unique solution, use it immediately
                            if len(new_solutions) == 1:
                                break

                if best_hint:
                    hints.append(best_hint)
                else:
                    # Can't find good hint, give up
                    return None

            attempts += 1

        return None

    def _format_hints(self, hints: List[Hint]) -> List[str]:
        """Format hints as strings"""
        return [str(hint) for hint in hints]

    def _create_problem_text(self, num_digits: int, hints: List[Hint]) -> str:
        """Create the problem text in English"""
        hint_strs = [f"[{hint.guess}: {hint.strikes}S {hint.balls}B]" for hint in hints]
        hints_text = ", ".join(hint_strs)
        return f"Find the unique {num_digits}-digit number with distinct digits that satisfies all the following hints: {hints_text}"


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
    """
    Validate a generated problem for correctness.

    Args:
        problem: Problem dictionary to validate

    Returns:
        Tuple of (is_valid, validation_message)
    """
    try:
        num_digits = problem['num_digits']
        game = BullsAndCows(num_digits)

        # Convert hints
        hints = [Hint(h['guess'], h['strikes'], h['balls']) for h in problem['hints']]

        # Check answer format
        answer = problem['answer']
        if len(answer) != num_digits:
            return False, f"Answer length {len(answer)} doesn't match num_digits {num_digits}"

        if len(set(answer)) != num_digits:
            return False, f"Answer {answer} doesn't have unique digits"

        # Check if answer satisfies all hints
        if not game.check_number_against_hints(answer, hints):
            return False, f"Answer {answer} doesn't satisfy all hints"

        # Check for unique solution
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


def generate_dataset(problems_per_difficulty: int = 3) -> List[Dict]:
    """
    Generate a complete dataset with problems for all difficulty levels.

    Args:
        problems_per_difficulty: Number of problems to generate per difficulty

    Returns:
        List of generated problems
    """
    generator = ProblemGenerator()
    dataset = []

    # Desired hint limits for each difficulty
    max_hints_per_difficulty = {
        Difficulty.EASY: 5,
        Difficulty.MEDIUM: 6,
        Difficulty.HARD: 7,
        Difficulty.EXPERT: 8
    }

    for difficulty in Difficulty:
        print(f"Generating {difficulty.name} problems...")
        generated = 0
        attempts = 0
        max_attempts_per_problem = 100

        while generated < problems_per_difficulty and attempts < max_attempts_per_problem * 10:
            problem = generator.generate_problem(difficulty)
            if problem:
                # Check hint count is within limits
                if len(problem['hints']) > max_hints_per_difficulty[difficulty]:
                    attempts += 1
                    continue

                # Verify the problem has unique solution
                is_valid, message = validate_problem(problem)

                if is_valid:
                    # Add question in English format
                    problem['question'] = create_question(problem)
                    dataset.append(problem)
                    generated += 1
                    print(f"  Generated {difficulty.name} problem {generated}/{problems_per_difficulty} ({len(problem['hints'])} hints)")

            attempts += 1

        if generated < problems_per_difficulty:
            print(f"  Warning: Only generated {generated}/{problems_per_difficulty} {difficulty.name} problems")

    return dataset


def create_dataset_files(num_questions: int, version: str):
    """
    Create dataset files for number baseball puzzles.

    Args:
        num_questions: Number of questions to generate
        version: Version string for filenames

    Returns:
        Tuple[pd.DataFrame, List[Dict]]: (dataframe, json list)
    """
    import pandas as pd

    print(f"Generating {num_questions} number baseball puzzles...")

    generator = ProblemGenerator()

    # Calculate puzzles per difficulty
    puzzles_per_diff = num_questions // 4
    remainder = num_questions % 4

    difficulties = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD, Difficulty.EXPERT]
    all_puzzles = []

    for i, difficulty in enumerate(difficulties):
        count = puzzles_per_diff + (1 if i < remainder else 0)
        generated = 0
        attempts = 0

        while generated < count and attempts < 1000:
            problem = generator.generate_problem(difficulty)
            if problem:
                is_valid, _ = validate_problem(problem)
                if is_valid:
                    problem['question'] = create_question(problem)
                    all_puzzles.append(problem)
                    generated += 1
            attempts += 1

    print(f"\nGenerated {len(all_puzzles)} puzzles")

    df = pd.DataFrame(all_puzzles)

    # Save files
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # CSV
    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"NUMBER_BASEBALL_{version}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV file created: {csv_path}")

    # JSONL
    json_dir = PROJECT_ROOT / "data" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = json_dir / f"NUMBER_BASEBALL_{version}.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_puzzles:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL file created: {jsonl_path}")

    return df, all_puzzles


if __name__ == "__main__":
    print("=" * 60)
    print("Number Baseball (Bulls and Cows) Puzzle Generator")
    print("=" * 60)

    # Generate dataset
    dataset = generate_dataset(problems_per_difficulty=3)

    print("\n" + "=" * 60)
    print("Generated Puzzles Summary")
    print("=" * 60)

    for problem in dataset:
        print(f"[{problem['difficulty']:6}] Answer: {problem['answer']} ({len(problem['hints'])} hints)")

    # Validate all problems
    print("\n" + "=" * 60)
    print("Validating all problems...")
    print("=" * 60)

    all_valid = True
    for i, problem in enumerate(dataset):
        is_valid, message = validate_problem(problem)
        if is_valid:
            print(f"  Problem {i+1}: ✓ Valid")
        else:
            print(f"  Problem {i+1}: ✗ Invalid - {message}")
            all_valid = False

    if all_valid:
        print("\n✓ All problems validated successfully!")
    else:
        print("\n✗ Some problems have validation issues!")
