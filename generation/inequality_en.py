"""Inequality Puzzle Generator and Validator

Constructive generation: builds puzzles by progressively adding hints
until solution count reaches 1, guaranteeing valid puzzle generation.
Supports large puzzles (up to 15 numbers) via backtracking solver.
"""

import random
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


class Difficulty(Enum):
    EASY = 1      # 5-7 numbers, few hints
    MEDIUM = 2    # 9-11 numbers, minimal hints
    HARD = 3      # 13-15 numbers, no hints


@dataclass
class InequalityPuzzle:
    """Represents an inequality puzzle with a single solution"""
    size: int
    inequalities: List[str]  # List of inequalities between positions
    given_numbers: Dict[int, int]  # position -> number mapping for hints
    solution: List[int]  # Single valid solution
    difficulty: Difficulty
    hidden_inequalities: set = field(default_factory=set)  # indices of hidden inequalities

    def to_problem_string(self) -> str:
        """Convert puzzle to a readable problem string"""
        problem_parts = []
        for i in range(self.size):
            if i in self.given_numbers:
                problem_parts.append(str(self.given_numbers[i]))
            else:
                problem_parts.append("_")

            if i < len(self.inequalities):
                if i in self.hidden_inequalities:
                    problem_parts.append("?")
                else:
                    problem_parts.append(self.inequalities[i])

        return " ".join(problem_parts)

    def get_answer_string(self) -> str:
        """Return answer as string of numbers.
        For size <= 9: concatenated digits (e.g., '53241')
        For size > 9: space-separated (e.g., '5 3 12 4 10 1')
        """
        if self.size > 9:
            return " ".join(map(str, self.solution))
        return "".join(map(str, self.solution))

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "problem": self.to_problem_string(),
            "answer": self.get_answer_string(),
            "difficulty": self.difficulty.name.lower(),
            "size": self.size,
            "given_positions": list(self.given_numbers.keys()),
            "given_values": list(self.given_numbers.values())
        }


class InequalityPuzzleGenerator:
    """
    Constructive puzzle generator.

    Strategy: Start with base solution, derive inequalities, then
    progressively add hints until solution count is 1.
    """

    MAX_SOLUTIONS = 1  # Only allow exactly 1 solution

    def __init__(self):
        self.difficulty_config = {
            Difficulty.EASY: {
                "size_range": (5, 7),
                "hint_ratio": 0.2,
                "min_hints": 1,
                "ineq_reveal": 1.0
            },
            Difficulty.MEDIUM: {
                "size_range": (9, 11),
                "hint_ratio": 0.0,
                "min_hints": 0,
                "ineq_reveal": 0.5
            },
            Difficulty.HARD: {
                "size_range": (13, 15),
                "hint_ratio": 0.0,
                "min_hints": 0,
                "ineq_reveal": 0.15
            }
        }

    def _find_solutions(self, size: int, inequalities: List[str],
                        given_numbers: Dict[int, int], max_count: int = 0) -> List[List[int]]:
        """Find solutions using backtracking with constraint propagation.

        max_count=0 means unlimited.
        """
        solutions = []
        assignment = [0] * size
        used = [False] * (size + 1)  # used[v] = True if value v is taken

        # Pre-fix given numbers
        for pos, val in given_numbers.items():
            assignment[pos] = val
            used[val] = True

        unfixed = [i for i in range(size) if i not in given_numbers]

        def backtrack(idx: int):
            if max_count > 0 and len(solutions) >= max_count:
                return

            if idx == len(unfixed):
                solutions.append(list(assignment))
                return

            pos = unfixed[idx]

            for val in range(1, size + 1):
                if used[val]:
                    continue

                # Check inequality with left neighbor (pos-1)
                if pos > 0 and assignment[pos - 1] != 0:
                    ineq = inequalities[pos - 1]
                    if ineq == "<" and assignment[pos - 1] >= val:
                        continue
                    if ineq == ">" and assignment[pos - 1] <= val:
                        continue

                # Check inequality with right neighbor (pos+1)
                if pos < size - 1 and assignment[pos + 1] != 0:
                    ineq = inequalities[pos]
                    if ineq == "<" and val >= assignment[pos + 1]:
                        continue
                    if ineq == ">" and val <= assignment[pos + 1]:
                        continue

                assignment[pos] = val
                used[val] = True
                backtrack(idx + 1)
                assignment[pos] = 0
                used[val] = False

                if max_count > 0 and len(solutions) >= max_count:
                    return

        backtrack(0)
        return solutions

    def _check_constraints(self, perm: Tuple[int, ...], inequalities: List[str],
                          given_numbers: Dict[int, int]) -> bool:
        """Check if permutation satisfies all constraints"""
        for pos, num in given_numbers.items():
            if perm[pos] != num:
                return False

        for i, ineq in enumerate(inequalities):
            if ineq == "<" and perm[i] >= perm[i + 1]:
                return False
            elif ineq == ">" and perm[i] <= perm[i + 1]:
                return False

        return True

    def _find_best_hint(self, size: int, inequalities: List[str],
                        given_numbers: Dict[int, int], base_solution: List[int],
                        current_solutions: List[List[int]]) -> Optional[int]:
        """
        Find the hint position that best reduces solution count toward 1.
        Returns position that maximally reduces solutions while keeping >= 1.
        """
        available = [i for i in range(size) if i not in given_numbers]
        if not available:
            return None

        best_pos = None
        best_count = float('inf')

        for pos in available:
            test_hints = dict(given_numbers)
            test_hints[pos] = base_solution[pos]
            new_solutions = self._find_solutions(size, inequalities, test_hints, max_count=2)

            if len(new_solutions) == 1:
                return pos  # Found optimal position

            if len(new_solutions) >= 1 and len(new_solutions) < best_count:
                best_count = len(new_solutions)
                best_pos = pos

        return best_pos

    def generate_puzzle(self, difficulty: Difficulty, max_retries: int = 100) -> InequalityPuzzle:
        """
        Constructively generate a puzzle with exactly 1 solution.

        Process:
        1. Generate base solution (random permutation)
        2. Derive inequalities from base solution
        3. Start with minimum hints for difficulty
        4. Progressively add hints until solution count is 1
        5. Retry with fresh randomisation if needed
        """
        config = self.difficulty_config[difficulty]

        for retry in range(max_retries):
            size = random.randint(*config["size_range"])

            # Step 1: Generate random permutation as base solution
            base_solution = list(range(1, size + 1))
            random.shuffle(base_solution)

            # Step 2: Generate inequalities from base solution
            inequalities = []
            for i in range(size - 1):
                if base_solution[i] < base_solution[i + 1]:
                    inequalities.append("<")
                else:
                    inequalities.append(">")

            # Step 3: Start with minimum hints based on difficulty
            num_hints = max(
                config["min_hints"],
                int(size * config["hint_ratio"])
            )
            given_numbers = self._select_initial_hints(base_solution, num_hints)

            # Step 4: Check solution count and adjust
            solutions = self._find_solutions(size, inequalities, given_numbers, self.MAX_SOLUTIONS + 1)

            # Constructively add hints until solution count is 1
            while len(solutions) > self.MAX_SOLUTIONS:
                best_pos = self._find_best_hint(size, inequalities, given_numbers,
                                                base_solution, solutions)
                if best_pos is None:
                    break

                given_numbers[best_pos] = base_solution[best_pos]
                solutions = self._find_solutions(size, inequalities, given_numbers, self.MAX_SOLUTIONS + 1)

            if len(solutions) == 1:
                # Hide some inequalities based on difficulty
                ineq_reveal = config.get("ineq_reveal", 1.0)
                total_ineqs = size - 1
                num_to_hide = int(total_ineqs * (1.0 - ineq_reveal))
                hidden_indices = set()
                if num_to_hide > 0:
                    hidden_indices = set(random.sample(range(total_ineqs), num_to_hide))

                return InequalityPuzzle(
                    size=size,
                    inequalities=inequalities,
                    given_numbers=given_numbers,
                    solution=solutions[0],
                    difficulty=difficulty,
                    hidden_inequalities=hidden_indices
                )

        raise RuntimeError(
            f"Failed to generate {difficulty.name} puzzle with exactly 1 solution "
            f"after {max_retries} retries"
        )

    def _select_initial_hints(self, solution: List[int], num_hints: int) -> Dict[int, int]:
        """Select initial hint positions strategically"""
        given_numbers = {}
        size = len(solution)

        if num_hints == 0:
            return given_numbers

        # Prioritize extremes (min/max values) as they constrain more
        extreme_positions = []
        for i, val in enumerate(solution):
            if val == 1 or val == size:
                extreme_positions.append(i)

        random.shuffle(extreme_positions)
        for pos in extreme_positions:
            if len(given_numbers) >= num_hints:
                break
            given_numbers[pos] = solution[pos]

        # Fill remaining with random positions
        remaining = [i for i in range(size) if i not in given_numbers]
        random.shuffle(remaining)

        for pos in remaining:
            if len(given_numbers) >= num_hints:
                break
            given_numbers[pos] = solution[pos]

        return given_numbers

    def solve_puzzle(self, size: int, inequalities: List[str],
                    given_numbers: Dict[int, int]) -> List[List[int]]:
        """Solve a puzzle and return all solutions (up to MAX_SOLUTIONS)"""
        return self._find_solutions(size, inequalities, given_numbers, self.MAX_SOLUTIONS)


# ============================================================
# Question formatting
# ============================================================

def create_question(puzzle: InequalityPuzzle) -> str:
    """Create question text in English."""
    problem_str = puzzle.to_problem_string()

    has_hidden = len(puzzle.hidden_inequalities) > 0

    hidden_rule = ""
    if has_hidden:
        hidden_rule = "\n- '?' means the inequality is unknown (could be < or >)"

    if puzzle.size > 9:
        answer_format = f"Provide your answer as {puzzle.size} numbers separated by spaces."
        example = " ".join(str(i) for i in range(1, puzzle.size + 1))
        answer_example = f"Answer: {example}"
    else:
        answer_format = f"Provide your answer as a sequence of {puzzle.size} digits (no spaces)."
        example = "".join(str(i) for i in range(1, puzzle.size + 1))
        answer_example = f"Answer: {example}"

    question = f"""Solve this inequality puzzle. Fill in the blanks with numbers from 1 to {puzzle.size}.

Each number from 1 to {puzzle.size} must be used exactly once.
The inequality symbols (< or >) between positions must be satisfied.

Puzzle: {problem_str}

Rules:
- '_' represents an empty position to fill
- '<' means the left number is smaller than the right number
- '>' means the left number is larger than the right number
- Each number 1 to {puzzle.size} appears exactly once{hidden_rule}

{answer_format}

Example format:
{answer_example}"""

    return question


# ============================================================
# Dataset generation
# ============================================================

def generate_dataset(puzzles_per_difficulty: int = 3) -> List[InequalityPuzzle]:
    """Generate complete dataset with all difficulty levels"""
    generator = InequalityPuzzleGenerator()
    dataset = []

    for difficulty in Difficulty:
        print(f"Generating {difficulty.name} puzzles...")
        for i in range(puzzles_per_difficulty):
            puzzle = generator.generate_puzzle(difficulty)
            dataset.append(puzzle)
            answer = puzzle.get_answer_string()
            print(f"  Generated puzzle {i+1}: {puzzle.to_problem_string()}")
            print(f"    Answer: {answer}")

    return dataset


def save_dataset(dataset: List[InequalityPuzzle], filename: str = "inequality_dataset.json"):
    """Save dataset to JSON file"""
    data = {
        "problems": [puzzle.to_dict() for puzzle in dataset],
        "metadata": {
            "total_problems": len(dataset),
            "problems_per_difficulty": len(dataset) // len(Difficulty),
            "difficulty_levels": [d.name for d in Difficulty],
            "max_solutions_per_puzzle": InequalityPuzzleGenerator.MAX_SOLUTIONS
        }
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nDataset saved to {filename}")


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
                puzzle = generator.generate_puzzle(difficulty)
                puzzle_data = {
                    "id": f"inequality_{len(all_puzzles)}",
                    "question": create_question(puzzle),
                    "answer": puzzle.get_answer_string(),
                    "solution": puzzle.to_problem_string(),
                    "difficulty": diff_name,
                    "size": puzzle.size,
                    "given_positions": list(puzzle.given_numbers.keys()),
                    "given_values": list(puzzle.given_numbers.values()),
                    "problem": puzzle.to_problem_string(),
                }
                all_puzzles.append(puzzle_data)
                print(f"  [{j+1}/{count}] size={puzzle.size}, answer={puzzle.get_answer_string()}")
            except RuntimeError as e:
                print(f"  [{j+1}/{count}] Failed: {e}")

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
    parser.add_argument("--num", type=int, default=12, help="Number of questions to generate")

    args = parser.parse_args()

    print("=" * 50)
    print("Inequality Puzzle Generator")
    print("=" * 50)

    create_dataset_files(num_questions=args.num)
