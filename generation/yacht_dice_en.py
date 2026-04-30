"""Yacht Dice Problem Generator (EN)

Generates Yacht Dice optimization problems with difficulty-based dice patterns.
Uses bonus-aware exhaustive search solver (C(12,6) x 720 x 2).

Ported from logical-puzzles-me/yacht_dice/generator.py:
- Greedy reference solver for greedy_gap metric
- Per-round top1/top2 margin and decision_complexity metrics
- Delta-band difficulty filtering with co-prime seed retry
- step_metrics exported in puzzle JSONL
"""

import random
import json
import itertools
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
from dataclasses import dataclass
from typing import Literal


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
        goal = "maximum" if self.optimization_goal == "maximize" else "minimum"
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

Your task is to determine the {goal} possible total score by optimally assigning each round to a category.
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
# Scoring primitives
# ============================================================

def get_all_categories() -> List[str]:
    return [
        "Aces", "Twos", "Threes", "Fours", "Fives", "Sixes",
        "Three-Of-A-Kind", "Four-Of-A-Kind", "Full House",
        "Small Straight", "Large Straight", "Yacht"
    ]


UPPER_CATEGORIES = ["Aces", "Twos", "Threes", "Fours", "Fives", "Sixes"]


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
        for _, count in counts.items():
            if count >= 3:
                return sum(dice)
        return 0
    elif category == "Four-Of-A-Kind":
        for _, count in counts.items():
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


def calculate_score(dice: List[int], category: str) -> int:
    """Default-config score calculation."""
    return calculate_score_with_config(dice, category, YachtDiceConfig())


def calculate_total_score(assignment: Dict[int, str], dice_results: List[List[int]],
                          config: YachtDiceConfig) -> int:
    """Calculate total score for an assignment (with upper-section bonus)."""
    upper_section_score = 0
    total_score = 0

    for dice_idx, category in assignment.items():
        dice = dice_results[dice_idx]
        score = calculate_score_with_config(dice, category, config)
        total_score += score
        if category in UPPER_CATEGORIES:
            upper_section_score += score

    if upper_section_score >= config.bonus_threshold:
        total_score += config.bonus_points

    return total_score


# ============================================================
# Solvers
# ============================================================

def solve_yacht_dice(dice_results: List[List[int]], config: YachtDiceConfig) -> Tuple[int, Dict[int, str], bool]:
    """
    Bonus-aware exhaustive optimal solver with optimal-assignment uniqueness check.

    Returns (best_total, best_assignment, is_unique). Collects tied assignments
    during 6! enumeration of upper + lower sections to detect multiple optima.
    """
    categories = get_all_categories()
    upper_cats = categories[:6]
    lower_cats = categories[6:]
    n = len(dice_results)

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
    optimal_assignments: set = set()

    for upper_rounds in itertools.combinations(range(n), 6):
        lower_rounds = [i for i in range(n) if i not in upper_rounds]
        upper_list = list(upper_rounds)

        best_upper_score = -1 if is_maximize else float('inf')
        best_upper_perms: List[Tuple[int, ...]] = []
        for perm in perms_6:
            s = sum(upper_scores[upper_list[j]][perm[j]] for j in range(6))
            if (is_maximize and s > best_upper_score) or (not is_maximize and s < best_upper_score):
                best_upper_score = s
                best_upper_perms = [perm]
            elif s == best_upper_score:
                best_upper_perms.append(perm)

        best_lower_score = -1 if is_maximize else float('inf')
        best_lower_perms: List[Tuple[int, ...]] = []
        for perm in perms_6:
            s = sum(lower_scores[lower_rounds[j]][perm[j]] for j in range(6))
            if (is_maximize and s > best_lower_score) or (not is_maximize and s < best_lower_score):
                best_lower_score = s
                best_lower_perms = [perm]
            elif s == best_lower_score:
                best_lower_perms.append(perm)

        bonus = config.bonus_points if best_upper_score >= config.bonus_threshold else 0
        total = best_upper_score + best_lower_score + bonus

        if (is_maximize and total > best_total) or (not is_maximize and total < best_total):
            best_total = total
            optimal_assignments.clear()

        if total == best_total:
            for up in best_upper_perms:
                for lp in best_lower_perms:
                    key = tuple(
                        (upper_list[j], upper_cats[up[j]]) for j in range(6)
                    ) + tuple(
                        (lower_rounds[j], lower_cats[lp[j]]) for j in range(6)
                    )
                    optimal_assignments.add(key)
                    if len(optimal_assignments) > 1:
                        break
                if len(optimal_assignments) > 1:
                    break

    is_unique = len(optimal_assignments) == 1
    best_assignment: Dict[int, str] = {}
    if optimal_assignments:
        first = next(iter(optimal_assignments))
        best_assignment = {k: v for k, v in first}
    return best_total, best_assignment, is_unique


def solve_yacht_dice_greedy(
    dice_results: List[List[int]], config: YachtDiceConfig
) -> Tuple[int, Dict[int, str]]:
    """Myopic greedy baseline used for greedy_gap metric.

    For each round in order, pick the remaining category that yields the
    highest immediate score. No look-ahead, no bonus planning.
    """
    categories = get_all_categories()
    available = set(categories)
    assignment: Dict[int, str] = {}
    is_maximize = config.optimization_goal == "maximize"

    for idx, dice in enumerate(dice_results):
        best_cat = None
        best_score = -1 if is_maximize else float('inf')
        for cat in available:
            s = calculate_score_with_config(dice, cat, config)
            if (is_maximize and s > best_score) or (not is_maximize and s < best_score):
                best_score = s
                best_cat = cat
        if best_cat is None:
            best_cat = next(iter(available))
        assignment[idx] = best_cat
        available.discard(best_cat)

    total = calculate_total_score(assignment, dice_results, config)
    return total, assignment


def format_solution(dice_results: List[List[int]], assignment: Dict[int, str],
                    config: YachtDiceConfig) -> str:
    """Format solution in readable form."""
    categories = get_all_categories()
    result = []
    upper_section_score = 0
    total_score = 0

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
            if category in UPPER_CATEGORIES:
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
# Difficulty-based dice generator
# ============================================================

DIFFICULTY_CONFIGS: Dict[str, Dict] = {
    # Calibrated to step-count proxy: random_pattern ratio (more random dice =
    # more category-comparison decisions per round). See
    # docs/difficulty_definition.md §2.6 — note the proxy is weakest here
    # because solver is deterministic; categorical complexity (mixed strategy)
    # may dominate over raw step count.
    # v2 recalibration: num_rounds is fixed at 12 (solver core requires it).
    # Differentiation comes from roll_types/weights extremes.
    "easy":   {
        # v7.3: weights [80,20] 가 band [0,15] 와 호환 안 됨 (gap > 15 빈번 → 200 retry fail).
        # v6 weights 회귀 — 빠른 generation. gpt-4o-mini 가 easy 도 못 푸는 것은
        # yacht_dice 의 구조적 한계 (12-round optimization) — paper grade 에서
        # "frontier-resistant" 라벨로 수용.
        "roll_types": ['three_kind', 'pair', 'high_sum', 'normal'],
        "weights":    [60, 20, 5, 15],
    },
    "medium": {
        # v6: gpt-5.4-mini 87% at v3. Push 'normal' fraction up.
        "roll_types": ['partial_straight', 'pair', 'three_kind', 'normal'],
        "weights":    [15, 10, 5, 70],
    },
    "hard":   {
        # v6: gpt-5.4-mini 77% at v3. Pure random — no structured shortcuts.
        "roll_types": ['normal'],
        "weights":    [100],
    },
}


# Extreme-mismatch bands for greedy_gap δ (docs/difficulty_definition.md §2.6).
# The step-count proxy for yacht_dice is weak (solver is deterministic; most
# of the "work" is the 12! assignment search regardless of dice). We only
# reject obvious outliers — e.g., a hard puzzle where greedy already nears
# optimal, or an easy puzzle that traps greedy by a large margin.
_DIFFICULTY_BANDS = {
    # v7.3: weights v6 회귀 (easy 빠른 generation). bands 는 v6 medium/hard
    # 겹침 [24,28] 만 제거 — easy/medium/hard 단조성만 보장.
    'easy': {
        'greedy_gap': {'min': 0, 'max': 15},
        'decision_complexity': {'min': 0.0, 'max': 3.0},
    },
    'medium': {
        'greedy_gap': {'min': 16, 'max': 23},
        'decision_complexity': {'min': 3.2, 'max': 4.7},
    },
    'hard': {
        'greedy_gap': {'min': 24, 'max': None},
        'decision_complexity': {'min': 4.8, 'max': None},
    },
}


class YachtDiceProblemGenerator:
    """Generate Yacht Dice problems with difficulty-calibrated dice patterns."""

    def __init__(self):
        self.config = YachtDiceConfig()

    def generate_dice_by_difficulty(self, difficulty: str, seed: int = None) -> List[List[int]]:
        """Generate 12 rounds of dice results based on difficulty level."""
        rng = random.Random(seed)
        dice_results: List[List[int]] = []

        if difficulty == "easy":
            cfg = DIFFICULTY_CONFIGS["easy"]
            for _ in range(cfg.get("num_rounds", 12)):
                roll_type = rng.choices(cfg["roll_types"], weights=cfg["weights"], k=1)[0]
                if roll_type == 'three_kind':
                    num = rng.randint(1, 6)
                    dice = [num] * 3 + [rng.randint(1, 6) for _ in range(2)]
                    rng.shuffle(dice)
                    dice_results.append(dice)
                elif roll_type == 'pair':
                    num = rng.randint(1, 6)
                    dice = [num] * 2 + [rng.randint(1, 6) for _ in range(3)]
                    rng.shuffle(dice)
                    dice_results.append(dice)
                elif roll_type == 'high_sum':
                    dice_results.append([rng.choice([4, 5, 6]) for _ in range(5)])
                else:
                    dice_results.append([rng.randint(1, 6) for _ in range(5)])

        elif difficulty == "medium":
            cfg = DIFFICULTY_CONFIGS["medium"]
            for _ in range(cfg.get("num_rounds", 12)):
                roll_type = rng.choices(cfg["roll_types"], weights=cfg["weights"], k=1)[0]
                if roll_type == 'partial_straight':
                    base = rng.sample(range(1, 7), 3)
                    base.extend([rng.randint(1, 6) for _ in range(2)])
                    rng.shuffle(base)
                    dice_results.append(base)
                elif roll_type == 'pair':
                    num = rng.randint(1, 6)
                    dice = [num] * 2 + [rng.randint(1, 6) for _ in range(3)]
                    rng.shuffle(dice)
                    dice_results.append(dice)
                elif roll_type == 'three_kind':
                    num = rng.randint(1, 6)
                    dice = [num] * 3 + [rng.randint(1, 6) for _ in range(2)]
                    rng.shuffle(dice)
                    dice_results.append(dice)
                else:
                    dice_results.append([rng.randint(1, 6) for _ in range(5)])

        else:  # hard
            cfg = DIFFICULTY_CONFIGS["hard"]
            for _ in range(cfg.get("num_rounds", 12)):
                roll_type = rng.choices(cfg["roll_types"], weights=cfg["weights"], k=1)[0]
                if roll_type == 'full_house':
                    nums = rng.sample(range(1, 7), 2)
                    dice = [nums[0]] * 3 + [nums[1]] * 2
                    rng.shuffle(dice)
                    dice_results.append(dice)
                elif roll_type == 'three_kind':
                    num = rng.randint(1, 6)
                    dice = [num] * 3 + [rng.randint(1, 6) for _ in range(2)]
                    rng.shuffle(dice)
                    dice_results.append(dice)
                elif roll_type == 'pair':
                    num = rng.randint(1, 6)
                    dice = [num] * 2 + [rng.randint(1, 6) for _ in range(3)]
                    rng.shuffle(dice)
                    dice_results.append(dice)
                else:
                    dice_results.append([rng.randint(1, 6) for _ in range(5)])

        return dice_results

    def generate_problem(self, difficulty: str, problem_id: int = 1, seed: int = None) -> Dict:
        """Generate a single problem; retries seeds until metrics fit difficulty band."""
        if seed is None:
            difficulty_offset = {"easy": 10000, "medium": 20000, "hard": 30000}
            seed = 1000 + problem_id + difficulty_offset.get(difficulty, 0)

        band = _DIFFICULTY_BANDS.get(difficulty, {})
        max_retries = 200
        categories = get_all_categories()
        selected = None
        in_band = False
        trial_seed = seed

        for retry in range(max_retries):
            trial_seed = seed + retry * 997
            dice_results = self.generate_dice_by_difficulty(difficulty, trial_seed)
            optimal_score, optimal_assignment, is_unique = solve_yacht_dice(dice_results, self.config)
            greedy_score, _ = solve_yacht_dice_greedy(dice_results, self.config)
            greedy_gap = optimal_score - greedy_score

            per_round_margin: List[int] = []
            per_round_top1: List[int] = []
            for dice in dice_results:
                scores = sorted(
                    (calculate_score_with_config(dice, c, self.config) for c in categories),
                    reverse=True,
                )
                top1, top2 = scores[0], scores[1]
                per_round_top1.append(top1)
                per_round_margin.append(top1 - top2)
            total_decision_complexity = sum(1.0 / max(m, 1) for m in per_round_margin)
            zero_margin_rounds = sum(1 for m in per_round_margin if m == 0)

            upper_sum = sum(
                calculate_score_with_config(dice_results[idx], cat, self.config)
                for idx, cat in optimal_assignment.items() if cat in UPPER_CATEGORIES
            )
            bonus_applied = upper_sum >= self.config.bonus_threshold

            selected = {
                'dice_results': dice_results,
                'optimal_score': optimal_score,
                'optimal_assignment': optimal_assignment,
                'is_unique_assignment': is_unique,
                'greedy_score': greedy_score,
                'greedy_gap': greedy_gap,
                'per_round_margin': per_round_margin,
                'per_round_top1': per_round_top1,
                'total_decision_complexity': total_decision_complexity,
                'zero_margin_rounds': zero_margin_rounds,
                'bonus_applied': bonus_applied,
            }

            gap_band = band.get('greedy_gap', {})
            complexity_band = band.get('decision_complexity', {})
            ok = True
            if gap_band.get('min') is not None and greedy_gap < gap_band['min']:
                ok = False
            if gap_band.get('max') is not None and greedy_gap > gap_band['max']:
                ok = False
            if complexity_band.get('min') is not None and total_decision_complexity < complexity_band['min']:
                ok = False
            if complexity_band.get('max') is not None and total_decision_complexity > complexity_band['max']:
                ok = False

            if ok:
                in_band = True
                break

        band_violation = not in_band

        problem = {
            "id": problem_id,
            "difficulty": difficulty,
            "dice_results": selected['dice_results'],
            "answer": selected['optimal_score'],
            "optimal_assignment": {int(k): v for k, v in selected['optimal_assignment'].items()},
            "seed": trial_seed,
            "step_metrics": {
                "per_round_margin": selected['per_round_margin'],
                "per_round_top1": selected['per_round_top1'],
                "total_decision_complexity": selected['total_decision_complexity'],
                "zero_margin_rounds": selected['zero_margin_rounds'],
                "greedy_gap": selected['greedy_gap'],
                "bonus_applied": selected['bonus_applied'],
                "is_unique_assignment": selected['is_unique_assignment'],
                "band_violation": band_violation,
            },
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

            optimal_score, _, _ = solve_yacht_dice(dice_results, self.config)
            if optimal_score != problem['answer']:
                return False, f"Answer mismatch: expected {optimal_score}, got {problem['answer']}"

            return True, "Problem is valid"

        except Exception as e:
            return False, f"Validation error: {str(e)}"


# ============================================================
# Dataset generation
# ============================================================

SFT_SOLUTION_RUBRIC_EN = (
    "STEP0=meta · STEP1=given · STEP2=worked solution · "
    "STEP3=answer and verification"
)


def _build_yacht_solution_en(
    dice_results: List[List[int]],
    optimal_assignment: Dict[int, str],
    optimal_score: int,
    config: "YachtDiceConfig",
    difficulty: str,
    step_metrics: Dict,
) -> str:
    """SFT teacher trace: yacht dice with per-round SEG assignments."""
    num_rounds = len(dice_results)
    upper_sum = 0
    bonus = 0
    total = 0

    round_to_cat: Dict[int, str] = dict(optimal_assignment)

    lines: List[str] = [
        SFT_SOLUTION_RUBRIC_EN,
        "[STEP 0] Problem meta",
        f"  - Difficulty: {difficulty}",
        f"  - Rounds: {num_rounds} · 5 dice per round",
        f"  - Upper-section bonus: {config.bonus_points} pts (threshold {config.bonus_threshold})",
        f"  - Greedy-vs-optimal gap: {step_metrics.get('greedy_gap', 0)}",
        "  - Final answer is confirmed in [STEP 3]",
        "[STEP 1] Given",
        "  - Rule: each round's 5 dice must be assigned to one of 12 categories, exactly once.",
        "  - Rule: if the upper section (Aces..Sixes) sum >= threshold, award bonus.",
        "  - Dice per round:",
    ]
    for idx, dice in enumerate(dice_results):
        lines.append(f"    R{idx + 1}: {dice}")

    lines.append("[STEP 2] Worked solution")
    lines.append(
        f"  · Summary: DP over (round, remaining categories) to maximize total score · "
        f"{num_rounds} SEGs"
    )

    for idx in sorted(round_to_cat.keys()):
        dice = dice_results[idx]
        cat = round_to_cat[idx]
        score = calculate_score_with_config(dice, cat, config)
        tag = " (upper)" if cat in UPPER_CATEGORIES else ""
        lines.append(
            f"    [SEG {idx + 1}] R{idx + 1} dice {dice} -> {cat}{tag}: {score} pts"
        )
        if cat in UPPER_CATEGORIES:
            upper_sum += score
        total += score

    if upper_sum >= config.bonus_threshold:
        bonus = config.bonus_points
        total += bonus

    lines.extend([
        "[STEP 3] Answer and verification",
        f"  - Final answer (optimal total): {optimal_score}",
        f"  - Upper section sum: {upper_sum} / threshold {config.bonus_threshold} -> bonus {bonus}",
        f"  - Recomputed total: {total} (must match the final answer)",
        "  - Check that each category is used at most once and the bonus trigger matches the SEG trace.",
    ])
    return "\n".join(lines)


def create_dataset_files(num_questions: int):
    """Create CSV + JSONL dataset files for Yacht Dice puzzles."""
    import pandas as pd

    print(f"Generating {num_questions} Yacht Dice puzzles...")

    generator = YachtDiceProblemGenerator()
    config = YachtDiceConfig()

    difficulties = ["easy", "medium", "hard"]
    puzzles_per_diff = num_questions // len(difficulties)
    remainder = num_questions % len(difficulties)

    all_puzzles: List[Dict] = []
    problem_id = 1
    MAX_RETRIES_PER_PUZZLE = 30

    def _dice_key(problem):
        # per-difficulty dedup key — yacht_dice 는 dice_results 가 입력이므로
        # 동일 sequence 의 puzzle 은 reject (12 라운드 × 6 face value 의 변형이 좁음)
        return tuple(tuple(sorted(r)) for r in problem.get('dice_results', []))

    for i, difficulty in enumerate(difficulties):
        count = puzzles_per_diff + (1 if i < remainder else 0)
        if count == 0:
            continue

        print(f"\n=== Generating {difficulty} puzzles ({count} needed) ===")

        diff_success = 0
        seen = set()
        retries = 0
        max_retries = count * MAX_RETRIES_PER_PUZZLE
        attempt_idx = 0
        while diff_success < count:
            attempt_idx += 1
            problem = generator.generate_problem(difficulty, problem_id)
            is_valid, message = generator.validate_problem(problem)

            if not is_valid:
                print(f"  [attempt {attempt_idx}] Invalid: {message}")
                problem_id += 1
                retries += 1
                if retries > max_retries:
                    print(f"  ⚠️ retry budget exhausted at {diff_success}/{count} for {difficulty}")
                    break
                continue

            key = _dice_key(problem)
            if key in seen:
                retries += 1
                problem_id += 1
                if retries > max_retries:
                    print(f"  ⚠️ dedup retry budget exhausted at {diff_success}/{count} for {difficulty}")
                    break
                continue
            seen.add(key)

            dice_results = problem['dice_results']
            optimal_score = problem['answer']
            optimal_assignment = problem.get('optimal_assignment', {})
            solution_str = _build_yacht_solution_en(
                dice_results=dice_results,
                optimal_assignment=optimal_assignment,
                optimal_score=optimal_score,
                config=config,
                difficulty=difficulty,
                step_metrics=problem.get('step_metrics', {}),
            )

            question = config.get_user_prompt(dice_results)

            puzzle_data = {
                "id": f"yacht_dice_en_{difficulty}_{diff_success:04d}",
                "question": question,
                "answer": str(optimal_score),
                "solution": solution_str,
                "difficulty": difficulty,
            }
            all_puzzles.append(puzzle_data)
            diff_success += 1
            sm = problem['step_metrics']
            print(
                f"  [{diff_success}/{count}] score={optimal_score} "
                f"gap={sm['greedy_gap']} complexity={sm['total_decision_complexity']:.2f} "
                f"band_violation={sm['band_violation']}"
            )
            problem_id += 1

    print(f"\nGenerated {len(all_puzzles)} puzzles")

    df = pd.DataFrame(all_puzzles)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "yacht_dice_en.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV file created: {csv_path}")

    # JSONL
    json_dir = PROJECT_ROOT / "data" / "jsonl"
    json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = json_dir / "yacht_dice_en.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_puzzles:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL file created: {jsonl_path}")

    return df, all_puzzles


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Yacht Dice Puzzle Generator (EN)")
    parser.add_argument("--num", type=int, default=12, help="Number of questions to generate")
    args = parser.parse_args()

    print("=" * 60)
    print("Yacht Dice Puzzle Generator")
    print("=" * 60)
    create_dataset_files(num_questions=args.num)
