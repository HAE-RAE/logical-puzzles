"""Yacht Dice Problem Generator and Solver

Generates Yacht Dice optimization problems with difficulty-based dice patterns.
Uses bonus-aware exhaustive search solver (C(12,6) x 720 x 2) instead of
Hungarian algorithm, which cannot handle the upper section bonus correctly.
"""

import random
import json
import itertools
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
from dataclasses import dataclass
from typing import Literal
from datetime import datetime


# ============================================================
# Configuration
# ============================================================

@dataclass
class YachtDiceConfig:
    """Configuration for Yacht Dice scoring rules"""

    bonus_threshold: int = 63
    bonus_points: int = 35

    full_house_points: int = 25
    small_straight_points: int = 30
    large_straight_points: int = 40
    yacht_points: int = 50

    optimization_goal: Literal["maximize", "minimize"] = "maximize"

    def get_system_prompt(self) -> str:
        """Get system prompt in English with Answer: format instruction"""
        return f"""You are an expert at solving Yacht Dice optimization problems.

Yacht Dice is a dice game where you roll 5 dice for 12 rounds and assign each round to a scoring category.

Scoring Categories:
- Aces through Sixes: Sum of dice showing that number
- Three-of-a-Kind: Sum of all dice if at least 3 match
- Four-of-a-Kind: Sum of all dice if at least 4 match
- Full House: {self.full_house_points} points for exactly 3 of one number and 2 of another
- Small Straight: {self.small_straight_points} points for 4 consecutive numbers
- Large Straight: {self.large_straight_points} points for 5 consecutive numbers
- Yacht: {self.yacht_points} points for all 5 dice showing the same number

Upper Section Bonus: If the sum of Aces through Sixes is {self.bonus_threshold} or more, add {self.bonus_points} bonus points.

Your task is to determine the {"maximum" if self.optimization_goal == "maximize" else "minimum"} possible total score by optimally assigning each round to a category.
Each category can only be used once.

CRITICAL: Your very last line MUST be in this exact format:
Answer: [number]

Do NOT omit this line. Follow any specific instructions about what number to provide.
"""

    def get_user_prompt(self, dice_results: List[List[int]]) -> str:
        """Generate user prompt in English"""
        goal = "maximum" if self.optimization_goal == "maximize" else "minimum"
        prompt = f"Given the following 12 rounds of dice results, find the {goal} possible total score:\n\n"
        for i, dice in enumerate(dice_results):
            prompt += f"Round {i+1}: {dice}\n"
        prompt += f"\nCalculate the optimal assignment and provide the {goal} total score."
        return prompt


# ============================================================
# Solver (bonus-aware exhaustive search)
# ============================================================

def get_all_categories() -> List[str]:
    return [
        "Aces", "Twos", "Threes", "Fours", "Fives", "Sixes",
        "Three-Of-A-Kind", "Four-Of-A-Kind", "Full House",
        "Small Straight", "Large Straight", "Yacht"
    ]


def calculate_score(dice: List[int], category: str) -> int:
    """Calculate score for dice in a category (default config values)."""
    counts = Counter(dice)
    sorted_dice = sorted(dice)

    if category == "Aces":
        return dice.count(1) * 1
    elif category == "Twos":
        return dice.count(2) * 2
    elif category == "Threes":
        return dice.count(3) * 3
    elif category == "Fours":
        return dice.count(4) * 4
    elif category == "Fives":
        return dice.count(5) * 5
    elif category == "Sixes":
        return dice.count(6) * 6
    elif category == "Three-Of-A-Kind":
        for num, count in counts.items():
            if count >= 3:
                return sum(dice)
        return 0
    elif category == "Four-Of-A-Kind":
        for num, count in counts.items():
            if count >= 4:
                return sum(dice)
        return 0
    elif category == "Full House":
        counts_values = sorted(counts.values())
        if counts_values == [2, 3]:
            return 25
        return 0
    elif category == "Small Straight":
        unique = set(sorted_dice)
        for straight in [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}]:
            if straight.issubset(unique):
                return 30
        return 0
    elif category == "Large Straight":
        unique = set(sorted_dice)
        if unique == {1, 2, 3, 4, 5} or unique == {2, 3, 4, 5, 6}:
            return 40
        return 0
    elif category == "Yacht":
        if len(counts) == 1:
            return 50
        return 0
    return 0


def calculate_score_with_config(dice: List[int], category: str, config: YachtDiceConfig) -> int:
    """Calculate score with custom config values."""
    counts = Counter(dice)
    sorted_dice = sorted(dice)

    if category == "Aces":
        return dice.count(1) * 1
    elif category == "Twos":
        return dice.count(2) * 2
    elif category == "Threes":
        return dice.count(3) * 3
    elif category == "Fours":
        return dice.count(4) * 4
    elif category == "Fives":
        return dice.count(5) * 5
    elif category == "Sixes":
        return dice.count(6) * 6
    elif category == "Three-Of-A-Kind":
        for num, count in counts.items():
            if count >= 3:
                return sum(dice)
        return 0
    elif category == "Four-Of-A-Kind":
        for num, count in counts.items():
            if count >= 4:
                return sum(dice)
        return 0
    elif category == "Full House":
        counts_values = sorted(counts.values())
        if counts_values == [2, 3]:
            return config.full_house_points
        return 0
    elif category == "Small Straight":
        unique = set(sorted_dice)
        for straight in [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}]:
            if straight.issubset(unique):
                return config.small_straight_points
        return 0
    elif category == "Large Straight":
        unique = set(sorted_dice)
        if unique == {1, 2, 3, 4, 5} or unique == {2, 3, 4, 5, 6}:
            return config.large_straight_points
        return 0
    elif category == "Yacht":
        if len(counts) == 1:
            return config.yacht_points
        return 0
    return 0


def calculate_total_score(assignment: Dict[int, str], dice_results: List[List[int]],
                         config: YachtDiceConfig) -> int:
    """Calculate total score for an assignment (with bonus)."""
    upper_section_score = 0
    total_score = 0
    upper_categories = ["Aces", "Twos", "Threes", "Fours", "Fives", "Sixes"]

    for dice_idx, category in assignment.items():
        dice = dice_results[dice_idx]
        score = calculate_score_with_config(dice, category, config)
        total_score += score
        if category in upper_categories:
            upper_section_score += score

    if upper_section_score >= config.bonus_threshold:
        total_score += config.bonus_points

    return total_score


def solve_yacht_dice(dice_results: List[List[int]], config: YachtDiceConfig) -> Tuple[int, Dict[int, str]]:
    """
    Find optimal solution using bonus-aware exhaustive search.

    Iterates over C(12,6) = 924 subsets for upper section,
    then brute-forces 6! = 720 permutations for each section.
    Total: 924 x 720 x 2 = ~1.3M operations.

    This correctly handles the upper section bonus (35 points)
    which Hungarian algorithm cannot account for.
    """
    categories = get_all_categories()
    upper_cats = categories[:6]
    lower_cats = categories[6:]
    n = len(dice_results)

    # Pre-compute score matrices
    upper_scores = []
    lower_scores = []
    for i in range(n):
        upper_scores.append(
            [calculate_score_with_config(dice_results[i], cat, config) for cat in upper_cats]
        )
        lower_scores.append(
            [calculate_score_with_config(dice_results[i], cat, config) for cat in lower_cats]
        )

    perms_6 = list(itertools.permutations(range(6)))

    is_maximize = config.optimization_goal == "maximize"
    best_total = -1 if is_maximize else float('inf')
    best_assignment = {}

    for upper_rounds in itertools.combinations(range(n), 6):
        lower_rounds = [i for i in range(n) if i not in upper_rounds]
        upper_list = list(upper_rounds)

        # Best upper assignment (6! brute force)
        best_upper_score = -1 if is_maximize else float('inf')
        best_upper_perm = perms_6[0]
        for perm in perms_6:
            s = sum(upper_scores[upper_list[j]][perm[j]] for j in range(6))
            if (is_maximize and s > best_upper_score) or (not is_maximize and s < best_upper_score):
                best_upper_score = s
                best_upper_perm = perm

        # Best lower assignment (6! brute force)
        best_lower_score = -1 if is_maximize else float('inf')
        best_lower_perm = perms_6[0]
        for perm in perms_6:
            s = sum(lower_scores[lower_rounds[j]][perm[j]] for j in range(6))
            if (is_maximize and s > best_lower_score) or (not is_maximize and s < best_lower_score):
                best_lower_score = s
                best_lower_perm = perm

        bonus = config.bonus_points if best_upper_score >= config.bonus_threshold else 0
        total = best_upper_score + best_lower_score + bonus

        if (is_maximize and total > best_total) or (not is_maximize and total < best_total):
            best_total = total
            best_assignment = {}
            for j in range(6):
                best_assignment[upper_list[j]] = upper_cats[best_upper_perm[j]]
                best_assignment[lower_rounds[j]] = lower_cats[best_lower_perm[j]]

    return best_total, best_assignment


def format_solution(dice_results: List[List[int]], assignment: Dict[int, str],
                    config: YachtDiceConfig) -> str:
    """Format solution in readable form."""
    categories = get_all_categories()
    result = []
    upper_section_score = 0
    total_score = 0
    upper_categories = ["Aces", "Twos", "Threes", "Fours", "Fives", "Sixes"]

    for category in categories:
        dice_idx = None
        for idx, cat in assignment.items():
            if cat == category:
                dice_idx = idx
                break

        if dice_idx is not None:
            dice = dice_results[dice_idx]
            score = calculate_score_with_config(dice, category, config)
            result.append(f"{category}: {dice} => {score}")
            total_score += score
            if category in upper_categories:
                upper_section_score += score

    bonus = 0
    if upper_section_score >= config.bonus_threshold:
        bonus = config.bonus_points
        total_score += bonus

    output = "\n".join(result)
    output += f"\n\nUpper section total: {upper_section_score}"
    output += f"\nBonus: {bonus} (threshold: {config.bonus_threshold})"
    output += f"\nTotal: {total_score}"

    return output


# ============================================================
# Dice generation
# ============================================================

def generate_random_dice(num_rounds: int = 12, dice_per_round: int = 5, seed: int = None) -> List[List[int]]:
    """Generate random dice results."""
    if seed is not None:
        random.seed(seed)

    dice_results = []
    for _ in range(num_rounds):
        round_result = [random.randint(1, 6) for _ in range(dice_per_round)]
        round_result.sort()
        dice_results.append(round_result)

    return dice_results


def format_user_prompt(dice_results: List[List[int]]) -> str:
    """Format dice results as user prompt."""
    prompt = "Here are 12 dice results. Assign each result to one of the scoring categories:\n\n"
    for i, dice in enumerate(dice_results, 1):
        prompt += f"{i}. {dice}\n"
    prompt += "\nAssign one result to each category and calculate the score to maximize the total."
    return prompt


# ============================================================
# Difficulty-based problem generator
# ============================================================

class YachtDiceProblemGenerator:
    """Generate Yacht Dice problems with different difficulty levels"""

    def __init__(self):
        self.config = YachtDiceConfig()

    def generate_dice_by_difficulty(self, difficulty: str, seed: int = None) -> List[List[int]]:
        """Generate dice results based on difficulty level."""
        if seed:
            random.seed(seed)

        dice_results = []

        if difficulty == "easy":
            for _ in range(12):
                roll_type = random.choices(
                    ['three_kind', 'pair', 'high_sum', 'random'],
                    weights=[25, 35, 20, 20],
                    k=1
                )[0]
                if roll_type == 'three_kind':
                    num = random.randint(1, 6)
                    dice = [num] * 3
                    dice.extend([random.randint(1, 6) for _ in range(2)])
                    random.shuffle(dice)
                    dice_results.append(dice)
                elif roll_type == 'pair':
                    num = random.randint(1, 6)
                    dice = [num] * 2
                    dice.extend([random.randint(1, 6) for _ in range(3)])
                    random.shuffle(dice)
                    dice_results.append(dice)
                elif roll_type == 'high_sum':
                    dice = [random.choice([4, 5, 6]) for _ in range(5)]
                    dice_results.append(dice)
                else:
                    dice_results.append([random.randint(1, 6) for _ in range(5)])

        elif difficulty == "medium":
            for _ in range(12):
                roll_type = random.choice(['partial_straight', 'pair', 'normal', 'normal', 'normal'])
                if roll_type == 'partial_straight':
                    base = random.sample(range(1, 7), 3)
                    base.extend([random.randint(1, 6) for _ in range(2)])
                    random.shuffle(base)
                    dice_results.append(base)
                elif roll_type == 'pair':
                    num = random.randint(1, 6)
                    dice = [num] * 2
                    dice.extend([random.randint(1, 6) for _ in range(3)])
                    random.shuffle(dice)
                    dice_results.append(dice)
                else:
                    dice_results.append([random.randint(1, 6) for _ in range(5)])

        else:  # hard
            for _ in range(12):
                roll_type = random.choice(['full_house', 'straight', 'three_kind', 'normal'])
                if roll_type == 'full_house':
                    nums = random.sample(range(1, 7), 2)
                    dice = [nums[0]] * 3 + [nums[1]] * 2
                    random.shuffle(dice)
                    dice_results.append(dice)
                elif roll_type == 'straight':
                    if random.random() < 0.5:
                        dice = list(range(1, 6))
                    else:
                        dice = list(range(2, 7))
                    random.shuffle(dice)
                    dice_results.append(dice)
                elif roll_type == 'three_kind':
                    num = random.randint(1, 6)
                    dice = [num] * 3
                    dice.extend([random.randint(1, 6) for _ in range(2)])
                    random.shuffle(dice)
                    dice_results.append(dice)
                else:
                    base = [random.randint(1, 6) for _ in range(5)]
                    if random.random() < 0.4:
                        base[1] = base[0]
                    dice_results.append(base)

        return dice_results

    def generate_problem(self, difficulty: str, problem_id: int = 1) -> Dict:
        """Generate a single Yacht Dice problem with specified difficulty."""
        seed = 1000 + problem_id + hash(difficulty)
        dice_results = self.generate_dice_by_difficulty(difficulty, seed)

        optimal_score, optimal_assignment = solve_yacht_dice(dice_results, self.config)

        problem = {
            "id": problem_id,
            "difficulty": difficulty,
            "dice_results": dice_results,
            "answer": optimal_score,
            "optimal_assignment": {int(k): v for k, v in optimal_assignment.items()},
            "seed": seed
        }

        return problem

    def validate_problem(self, problem: Dict) -> Tuple[bool, str]:
        """Validate that a problem is correctly formed and solvable."""
        try:
            required_fields = ['id', 'difficulty', 'dice_results', 'answer']
            for field in required_fields:
                if field not in problem:
                    return False, f"Missing required field: {field}"

            dice_results = problem['dice_results']
            if len(dice_results) != 12:
                return False, f"Expected 12 rounds, got {len(dice_results)}"

            for round_idx, dice in enumerate(dice_results):
                if len(dice) != 5:
                    return False, f"Round {round_idx+1}: Expected 5 dice, got {len(dice)}"
                for die in dice:
                    if not (1 <= die <= 6):
                        return False, f"Round {round_idx+1}: Invalid die value {die}"

            optimal_score, _ = solve_yacht_dice(dice_results, self.config)
            if optimal_score != problem['answer']:
                return False, f"Answer mismatch: expected {optimal_score}, got {problem['answer']}"

            return True, "Problem is valid"

        except Exception as e:
            return False, f"Validation error: {str(e)}"


# ============================================================
# Dataset generation
# ============================================================

def create_dataset_files(num_questions: int):
    """
    Create dataset files for Yacht Dice puzzles.

    Args:
        num_questions: Number of questions to generate

    Returns:
        Tuple[pd.DataFrame, List[Dict]]: (dataframe, json list)
    """
    import pandas as pd

    print(f"Generating {num_questions} Yacht Dice puzzles...")

    generator = YachtDiceProblemGenerator()
    config = YachtDiceConfig()

    difficulties = ["easy", "medium", "hard"]
    puzzles_per_diff = num_questions // len(difficulties)
    remainder = num_questions % len(difficulties)

    all_puzzles = []
    problem_id = 1

    for i, difficulty in enumerate(difficulties):
        count = puzzles_per_diff + (1 if i < remainder else 0)

        if count == 0:
            continue

        print(f"\n=== Generating {difficulty} puzzles ({count} needed) ===")

        for j in range(count):
            problem = generator.generate_problem(difficulty, problem_id)
            is_valid, message = generator.validate_problem(problem)

            if is_valid:
                dice_results = problem['dice_results']
                optimal_score = problem['answer']
                optimal_assignment = problem.get('optimal_assignment', {})
                solution_str = format_solution(dice_results, optimal_assignment, config)

                question = config.get_user_prompt(dice_results)

                puzzle_data = {
                    "id": f"yacht_dice_{len(all_puzzles)}",
                    "question": question,
                    "answer": str(optimal_score),
                    "solution": solution_str,
                    "difficulty": difficulty,
                    "dice_results": dice_results,
                    "seed": problem['seed']
                }
                all_puzzles.append(puzzle_data)
                print(f"  [{j+1}/{count}] score={optimal_score}")
            else:
                print(f"  [{j+1}/{count}] Invalid: {message}")

            problem_id += 1

    print(f"\nGenerated {len(all_puzzles)} puzzles")

    df = pd.DataFrame(all_puzzles)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # CSV
    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "yacht_dice.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV file created: {csv_path}")

    # JSONL
    json_dir = PROJECT_ROOT / "data" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = json_dir / "yacht_dice.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_puzzles:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL file created: {jsonl_path}")

    return df, all_puzzles


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Yacht Dice Puzzle Generator")
    parser.add_argument("--num", type=int, default=12, help="Number of questions to generate")

    args = parser.parse_args()

    print("=" * 60)
    print("Yacht Dice Puzzle Generator")
    print("=" * 60)

    create_dataset_files(num_questions=args.num)
