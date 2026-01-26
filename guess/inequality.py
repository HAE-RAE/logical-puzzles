"""Inequality Puzzle Generator and Validator

Generates inequality puzzles with varying difficulty levels and validates uniqueness of solutions.
Each puzzle consists of numbers 1 to N arranged in a sequence with inequality operators between them.
"""

import random
import itertools
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class Difficulty(Enum):
    EASY = 1      # 3-4 numbers, many hints
    MEDIUM = 2    # 4-5 numbers, moderate hints
    HARD = 3      # 5-6 numbers, few hints
    EXPERT = 4    # 6-7 numbers, minimal hints


@dataclass
class InequalityPuzzle:
    """Represents an inequality puzzle"""
    size: int
    inequalities: List[str]  # List of inequalities between positions
    given_numbers: Dict[int, int]  # position -> number mapping for hints
    solution: List[int]
    difficulty: Difficulty

    def to_problem_string(self) -> str:
        """Convert puzzle to a readable problem string"""
        problem_parts = []
        for i in range(self.size):
            if i in self.given_numbers:
                problem_parts.append(str(self.given_numbers[i]))
            else:
                problem_parts.append("_")

            if i < len(self.inequalities):
                problem_parts.append(self.inequalities[i])

        return " ".join(problem_parts)

    def get_answer_string(self) -> str:
        """Return answer as a single string of numbers"""
        return "".join(map(str, self.solution))

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "problem": self.to_problem_string(),
            "answer": self.get_answer_string(),
            "difficulty": self.difficulty.name,
            "size": self.size,
            "given_positions": list(self.given_numbers.keys()),
            "given_values": list(self.given_numbers.values())
        }


class InequalityPuzzleGenerator:
    """Generate and validate inequality puzzles"""

    def __init__(self):
        self.difficulty_config = {
            Difficulty.EASY: {
                "size_range": (3, 4),
                "hint_ratio": 0.5,  # 50% of numbers given
                "min_hints": 2
            },
            Difficulty.MEDIUM: {
                "size_range": (4, 5),
                "hint_ratio": 0.4,  # 40% of numbers given
                "min_hints": 2
            },
            Difficulty.HARD: {
                "size_range": (5, 6),
                "hint_ratio": 0.3,  # 30% of numbers given
                "min_hints": 1
            },
            Difficulty.EXPERT: {
                "size_range": (6, 7),
                "hint_ratio": 0.2,  # 20% of numbers given
                "min_hints": 1
            }
        }

    def generate_puzzle(self, difficulty: Difficulty) -> InequalityPuzzle:
        """Generate a puzzle with given difficulty"""
        config = self.difficulty_config[difficulty]
        size = random.randint(*config["size_range"])

        # Generate random permutation as solution
        solution = list(range(1, size + 1))
        random.shuffle(solution)

        # Generate inequalities from solution
        inequalities = []
        for i in range(size - 1):
            if solution[i] < solution[i + 1]:
                inequalities.append("<")
            else:
                inequalities.append(">")

        # Determine number of hints
        num_hints = max(
            config["min_hints"],
            int(size * config["hint_ratio"])
        )

        # Select positions for hints to ensure uniqueness
        given_numbers = self._select_hints(solution, inequalities, num_hints)

        # Validate uniqueness
        if not self._validate_uniqueness(size, inequalities, given_numbers):
            # If not unique, try again
            return self.generate_puzzle(difficulty)

        return InequalityPuzzle(
            size=size,
            inequalities=inequalities,
            given_numbers=given_numbers,
            solution=solution,
            difficulty=difficulty
        )

    def _select_hints(self, solution: List[int], inequalities: List[str],
                     num_hints: int) -> Dict[int, int]:
        """Select hint positions to maximize constraint"""
        given_numbers = {}

        # Always include first or last to anchor the puzzle
        if random.random() < 0.5:
            given_numbers[0] = solution[0]
        else:
            given_numbers[len(solution) - 1] = solution[-1]

        # Add more hints strategically
        remaining_positions = [i for i in range(len(solution)) if i not in given_numbers]

        while len(given_numbers) < num_hints and remaining_positions:
            # Prefer positions that break up long chains
            pos = random.choice(remaining_positions)
            given_numbers[pos] = solution[pos]
            remaining_positions.remove(pos)

        return given_numbers

    def _validate_uniqueness(self, size: int, inequalities: List[str],
                            given_numbers: Dict[int, int]) -> bool:
        """Check if puzzle has unique solution using constraint satisfaction"""
        solutions = []

        # Generate all possible permutations
        for perm in itertools.permutations(range(1, size + 1)):
            if self._check_constraints(perm, inequalities, given_numbers):
                solutions.append(perm)
                if len(solutions) > 1:
                    return False  # Not unique

        return len(solutions) == 1

    def _check_constraints(self, perm: Tuple[int, ...], inequalities: List[str],
                          given_numbers: Dict[int, int]) -> bool:
        """Check if permutation satisfies all constraints"""
        # Check given numbers
        for pos, num in given_numbers.items():
            if perm[pos] != num:
                return False

        # Check inequalities
        for i, ineq in enumerate(inequalities):
            if ineq == "<" and perm[i] >= perm[i + 1]:
                return False
            elif ineq == ">" and perm[i] <= perm[i + 1]:
                return False

        return True

    def solve_puzzle(self, size: int, inequalities: List[str],
                    given_numbers: Dict[int, int]) -> Optional[List[int]]:
        """Solve a puzzle and return the solution if unique"""
        solutions = []

        for perm in itertools.permutations(range(1, size + 1)):
            if self._check_constraints(perm, inequalities, given_numbers):
                solutions.append(list(perm))
                if len(solutions) > 1:
                    return None  # Not unique

        return solutions[0] if solutions else None


def create_question(puzzle: InequalityPuzzle) -> str:
    """Create question text in English."""
    problem_str = puzzle.to_problem_string()

    question = f"""Solve this inequality puzzle. Fill in the blanks with numbers from 1 to {puzzle.size}.

Each number from 1 to {puzzle.size} must be used exactly once.
The inequality symbols (< or >) between positions must be satisfied.

Puzzle: {problem_str}

Rules:
- '_' represents an empty position to fill
- '<' means the left number is smaller than the right number
- '>' means the left number is larger than the right number
- Each number 1 to {puzzle.size} appears exactly once

Provide your answer as a sequence of {puzzle.size} digits (no spaces).

Example format:
Answer: {"".join(str(i) for i in range(1, puzzle.size + 1))}"""

    return question


def generate_dataset(puzzles_per_difficulty: int = 3) -> List[InequalityPuzzle]:
    """Generate complete dataset with all difficulty levels"""
    generator = InequalityPuzzleGenerator()
    dataset = []

    for difficulty in Difficulty:
        print(f"Generating {difficulty.name} puzzles...")
        for i in range(puzzles_per_difficulty):
            puzzle = generator.generate_puzzle(difficulty)
            dataset.append(puzzle)
            print(f"  Generated puzzle {i+1}: {puzzle.to_problem_string()}")
            print(f"    Answer: {puzzle.get_answer_string()}")

    return dataset


def save_dataset(dataset: List[InequalityPuzzle], filename: str = "inequality_dataset.json"):
    """Save dataset to JSON file"""
    data = {
        "problems": [puzzle.to_dict() for puzzle in dataset],
        "metadata": {
            "total_problems": len(dataset),
            "problems_per_difficulty": len(dataset) // len(Difficulty),
            "difficulty_levels": [d.name for d in Difficulty]
        }
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nDataset saved to {filename}")


def validate_dataset(filename: str = "inequality_dataset.json") -> bool:
    """Validate that all problems in dataset have unique solutions"""
    with open(filename, 'r') as f:
        data = json.load(f)

    generator = InequalityPuzzleGenerator()
    all_valid = True

    print("\nValidating dataset...")
    for i, problem in enumerate(data["problems"]):
        # Parse problem
        parts = problem["problem"].split()
        size = problem["size"]
        inequalities = [p for p in parts if p in ["<", ">"]]

        given_numbers = {}
        for pos, val in zip(problem["given_positions"], problem["given_values"]):
            given_numbers[pos] = val

        # Validate uniqueness
        solution = generator.solve_puzzle(size, inequalities, given_numbers)

        if solution is None:
            print(f"  Problem {i+1}: INVALID - No unique solution")
            all_valid = False
        elif "".join(map(str, solution)) != problem["answer"]:
            print(f"  Problem {i+1}: INVALID - Solution mismatch")
            all_valid = False
        else:
            print(f"  Problem {i+1}: VALID")

    return all_valid


def create_dataset_files(num_questions: int):
    """
    Create dataset files for inequality puzzles.

    Args:
        num_questions: Number of questions to generate

    Returns:
        Tuple[pd.DataFrame, List[Dict]]: (dataframe, json list)
    """
    import pandas as pd

    print(f"Generating {num_questions} inequality puzzles...")

    generator = InequalityPuzzleGenerator()

    # Calculate puzzles per difficulty
    puzzles_per_diff = num_questions // 4
    remainder = num_questions % 4

    difficulties = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD, Difficulty.EXPERT]
    all_puzzles = []

    for i, difficulty in enumerate(difficulties):
        count = puzzles_per_diff + (1 if i < remainder else 0)

        for _ in range(count):
            puzzle = generator.generate_puzzle(difficulty)
            puzzle_data = {
                "question": create_question(puzzle),
                "answer": puzzle.get_answer_string(),
                "solution": puzzle.to_problem_string(),
                "difficulty": difficulty.name,
                "size": puzzle.size
            }
            all_puzzles.append(puzzle_data)

    print(f"\nGenerated {len(all_puzzles)} puzzles")

    df = pd.DataFrame(all_puzzles)

    # Save files
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # CSV
    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "inequality.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV file created: {csv_path}")

    # JSONL
    json_dir = PROJECT_ROOT / "data" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = json_dir / "inequality.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_puzzles:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL file created: {jsonl_path}")

    return df, all_puzzles


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inequality Puzzle Generator")
    parser.add_argument("--num", type=int, default=400, help="Number of questions to generate")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Inequality Puzzle Generator")
    print("=" * 50)
    
    create_dataset_files(num_questions=args.num)

    # # Generate dataset
    # print("Generating Inequality Puzzle Dataset")
    # print("=" * 50)

    # dataset = generate_dataset(puzzles_per_difficulty=3)

    # # Save to project data directory
    # PROJECT_ROOT = Path(__file__).resolve().parent.parent
    # json_dir = PROJECT_ROOT / "data" / "json"
    # json_dir.mkdir(parents=True, exist_ok=True)

    # output_path = json_dir / "inequality_dataset.json"
    # save_dataset(dataset, str(output_path))

    # # Validate dataset
    # is_valid = validate_dataset(str(output_path))

    # if is_valid:
    #     print("\n✓ All puzzles validated successfully!")
    # else:
    #     print("\n✗ Some puzzles have validation issues!")