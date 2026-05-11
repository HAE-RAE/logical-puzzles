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
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import List, Dict, Tuple, Optional, Literal
from collections import Counter
from dataclasses import dataclass

import numpy as np

from evaluation.yacht_dice_spotcheck import (
    SPOTCHECK_K,
    append_spotcheck_user_suffix,
    deterministic_round_pick_1based,
    spotcheck_grading_total,
)

# Precompute the 720 permutations of (0..5) once (used for every solve call).
_PERMS_6 = np.array(list(itertools.permutations(range(6))), dtype=np.int8)  # (720, 6)
_ROW_IDX_6 = np.arange(6, dtype=np.int8)  # for fancy-indexing


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

    # Score matrix precomputation (numpy-vectorized).
    upper_scores = np.zeros((n, 6), dtype=np.int32)
    lower_scores = np.zeros((n, 6), dtype=np.int32)
    for i in range(n):
        for j, cat in enumerate(upper_cats):
            upper_scores[i, j] = calculate_score_with_config(dice_results[i], cat, config)
        for j, cat in enumerate(lower_cats):
            lower_scores[i, j] = calculate_score_with_config(dice_results[i], cat, config)

    is_maximize = config.optimization_goal == "maximize"
    best_total = -1 if is_maximize else float('inf')
    optimal_assignments: set = set()

    perms = _PERMS_6              # (720, 6)
    perms_list = perms.tolist()   # for fast Python iteration when collecting tied keys

    # C(12, 6) = 924 subset enumeration. Use numpy to evaluate 720 perms at once.
    for upper_rounds in itertools.combinations(range(n), 6):
        lower_rounds = tuple(i for i in range(n) if i not in upper_rounds)
        upper_list = list(upper_rounds)

        # Sub-matrix shape (6, 6) — round_in_subset × category
        sub_upper = upper_scores[upper_list, :]
        sub_lower = lower_scores[list(lower_rounds), :]

        # All perm sums vectorized: sums[k] = sum_j sub[j, perms[k, j]]
        upper_sums = sub_upper[_ROW_IDX_6, perms].sum(axis=1)  # (720,)
        lower_sums = sub_lower[_ROW_IDX_6, perms].sum(axis=1)

        if is_maximize:
            best_upper_score = int(upper_sums.max())
            best_lower_score = int(lower_sums.max())
        else:
            best_upper_score = int(upper_sums.min())
            best_lower_score = int(lower_sums.min())

        bonus = config.bonus_points if best_upper_score >= config.bonus_threshold else 0
        total = best_upper_score + best_lower_score + bonus

        if (is_maximize and total > best_total) or (not is_maximize and total < best_total):
            best_total = total
            optimal_assignments.clear()

        if total == best_total:
            # This subset ties best_total — only enumerate tied permutations.
            upper_tie_mask = upper_sums == best_upper_score
            lower_tie_mask = lower_sums == best_lower_score
            best_upper_perms = [perms_list[i] for i in np.flatnonzero(upper_tie_mask)]
            best_lower_perms = [perms_list[i] for i in np.flatnonzero(lower_tie_mask)]

            stop = False
            for up in best_upper_perms:
                for lp in best_lower_perms:
                    key = tuple(
                        (upper_list[j], upper_cats[up[j]]) for j in range(6)
                    ) + tuple(
                        (lower_rounds[j], lower_cats[lp[j]]) for j in range(6)
                    )
                    optimal_assignments.add(key)
                    if len(optimal_assignments) > 1:
                        stop = True
                        break
                if stop:
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
    # v15: align with ``evaluation.yacht_dice_spotcheck.SPOTCHECK_K`` + acc targets.
    # - easy: even more high-signal rolls (yacht / straights) for reliable 2-round sums.
    # - medium: a bit more ``normal`` chaos vs v14 to pull acc down from ~66%.
    # - hard: same roll machinery as medium; bands separate difficulty.
    "easy": {
        "roll_types": [
            "yacht",
            "large_straight",
            "small_straight",
            "full_house",
            "four_kind_high",
            "three_kind",
            "pair",
        ],
        "weights": [24, 22, 14, 14, 8, 10, 8],
    },
    "medium": {
        "roll_types": ["three_kind", "pair", "normal"],
        "weights": [30, 30, 40],
    },
    "hard": {
        "roll_types": ["three_kind", "pair", "normal"],
        "weights": [18, 22, 60],
    },
}


# Extreme-mismatch bands for greedy_gap δ (docs/difficulty_definition.md §2.6).
# The step-count proxy for yacht_dice is weak (solver is deterministic; most
# of the "work" is the 12! assignment search regardless of dice). We only
# reject obvious outliers — e.g., a hard puzzle where greedy already nears
# optimal, or an easy puzzle that traps greedy by a large margin.
_DIFFICULTY_BANDS = {
    # v14: tighten easy further (push model toward near-greedy, low-tie puzzles).
    # Medium / hard floors slightly relaxed vs v13 so hard dice fix still yields
    # enough accepted puzzles under max_retries.
    "easy": {
        "greedy_gap_abs": {"min": 0, "max": 6},
        "decision_complexity": {"min": 0.0, "max": 2.05},
    },
    "medium": {
        "greedy_gap_abs": {"min": 4, "max": 52},
        "decision_complexity": {"min": 2.5, "max": 12.5},
    },
    "hard": {
        "greedy_gap_abs": {"min": 10, "max": None},
        "decision_complexity": {"min": 4.5, "max": None},
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
                if roll_type == 'yacht':
                    val = rng.randint(1, 6)
                    dice_results.append([val] * 5)
                elif roll_type == 'large_straight':
                    start = rng.choice([1, 2])
                    straight = list(range(start, start + 5))
                    rng.shuffle(straight)
                    dice_results.append(straight)
                elif roll_type == 'small_straight':
                    straight_set = rng.choice(
                        [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}]
                    )
                    extra = rng.randint(1, 6)
                    dice = list(straight_set) + [extra]
                    rng.shuffle(dice)
                    dice_results.append(dice)
                elif roll_type == 'four_kind_high':
                    val = rng.randint(3, 6)
                    other = rng.choice([x for x in range(1, 7) if x != val])
                    dice = [val] * 4 + [other]
                    rng.shuffle(dice)
                    dice_results.append(dice)
                elif roll_type == 'full_house':
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

        elif difficulty == "medium":
            cfg = DIFFICULTY_CONFIGS["medium"]
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
                else:
                    dice_results.append([rng.randint(1, 6) for _ in range(5)])

        else:  # hard
            cfg = DIFFICULTY_CONFIGS["hard"]
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
                else:
                    dice_results.append([rng.randint(1, 6) for _ in range(5)])

        return dice_results

    def generate_problem(self, difficulty: str, problem_id: int = 1, seed: int = None,
                         max_retries: int = 200) -> Dict:
        """Generate a single problem; retries seeds until metrics fit difficulty band."""
        if seed is None:
            difficulty_offset = {"easy": 10000, "medium": 20000, "hard": 30000}
            seed = 1000 + problem_id + difficulty_offset.get(difficulty, 0)

        # use difficulty-specific config if specified
        diff_cfg = DIFFICULTY_CONFIGS.get(difficulty, {})
        config = YachtDiceConfig(
            bonus_threshold=diff_cfg.get("bonus_threshold", self.config.bonus_threshold),
            bonus_points=self.config.bonus_points,
            full_house_points=self.config.full_house_points,
            small_straight_points=self.config.small_straight_points,
            large_straight_points=self.config.large_straight_points,
            yacht_points=self.config.yacht_points,
            optimization_goal=self.config.optimization_goal,
        )

        band = _DIFFICULTY_BANDS.get(difficulty, {})
        categories = get_all_categories()
        selected = None
        in_band = False
        trial_seed = seed

        for retry in range(max_retries):
            trial_seed = seed + retry * 997
            dice_results = self.generate_dice_by_difficulty(difficulty, trial_seed)
            optimal_score, optimal_assignment, is_unique = solve_yacht_dice(dice_results, config)
            greedy_score, _ = solve_yacht_dice_greedy(dice_results, config)
            greedy_gap = optimal_score - greedy_score
            # abs gap unifies maximize (gap>=0) and minimize (gap<=0) modes.
            greedy_gap_abs = abs(greedy_gap)

            per_round_margin: List[int] = []
            per_round_top1: List[int] = []
            for dice in dice_results:
                scores = sorted(
                    (calculate_score_with_config(dice, c, config) for c in categories),
                    reverse=True,
                )
                top1, top2 = scores[0], scores[1]
                per_round_top1.append(top1)
                per_round_margin.append(top1 - top2)
            total_decision_complexity = sum(1.0 / max(m, 1) for m in per_round_margin)
            zero_margin_rounds = sum(1 for m in per_round_margin if m == 0)

            upper_sum = sum(
                calculate_score_with_config(dice_results[idx], cat, config)
                for idx, cat in optimal_assignment.items() if cat in UPPER_CATEGORIES
            )
            bonus_applied = upper_sum >= config.bonus_threshold

            selected = {
                'dice_results': dice_results,
                'optimal_score': optimal_score,
                'optimal_assignment': optimal_assignment,
                'is_unique_assignment': is_unique,
                'greedy_score': greedy_score,
                'greedy_gap': greedy_gap,
                'greedy_gap_abs': greedy_gap_abs,
                'per_round_margin': per_round_margin,
                'per_round_top1': per_round_top1,
                'total_decision_complexity': total_decision_complexity,
                'zero_margin_rounds': zero_margin_rounds,
                'bonus_applied': bonus_applied,
            }

            # New band key: greedy_gap_abs (legacy 'greedy_gap' also accepted).
            gap_band = band.get('greedy_gap_abs') or band.get('greedy_gap', {})
            complexity_band = band.get('decision_complexity', {})
            ok = True
            if gap_band.get('min') is not None and greedy_gap_abs < gap_band['min']:
                ok = False
            if gap_band.get('max') is not None and greedy_gap_abs > gap_band['max']:
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
                "greedy_gap_abs": selected['greedy_gap_abs'],
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

            difficulty = problem.get('difficulty', '')
            diff_cfg = DIFFICULTY_CONFIGS.get(difficulty, {})
            validate_config = YachtDiceConfig(
                bonus_threshold=diff_cfg.get("bonus_threshold", self.config.bonus_threshold),
            )
            optimal_score, _, _ = solve_yacht_dice(dice_results, validate_config)
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
    grading_answer: str,
    spot_rounds_1based: Optional[List[int]] = None,
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

    bonus_note = f"upper section sum {upper_sum} / threshold {config.bonus_threshold} -> bonus {bonus}"
    step3: List[str] = [
        "[STEP 3] Answer and verification",
        f'  - Graded answer (must match dataset "answer" column): {grading_answer}',
    ]
    if spot_rounds_1based:
        rlist = ", ".join(str(r) for r in spot_rounds_1based)
        step3.extend([
            f"  - Spotcheck: sum of optimal-assignment scores for rounds {rlist} only",
            f"  - Full optimal total (reference): {optimal_score}",
        ])
    else:
        step3.append(f"  - Optimal total score: {optimal_score}")
    step3.extend(
        [
            f"  - {bonus_note}",
            f"  - Recomputed total: {total} (must match the optimal total above)",
            "  - Check that each category is used at most once and the bonus trigger matches the SEG trace.",
        ]
    )
    lines.extend(step3)
    return "\n".join(lines)



def create_dataset_files(num_questions: int):
    """Create CSV + JSONL dataset files for Yacht Dice puzzles.

    Calls ``generate_problem`` with a per-difficulty inner retry budget so each
    attempt usually finishes quickly while still preferring in-band dice (see
    ``_DIFFICULTY_BANDS``).  Puzzles that exhaust retries may still be
    out-of-band (``band_violation`` in step_metrics) but remain valid with the
    correct optimal score under the per-difficulty ``YachtDiceConfig``.
    """
    import pandas as pd

    fast_retries_by_diff = {"easy": 200, "medium": 120, "hard": 90}

    print(f"Generating {num_questions} Yacht Dice puzzles (fast mode)...")

    generator = YachtDiceProblemGenerator()
    base_cfg = YachtDiceConfig()

    difficulties = ["easy", "medium", "hard"]
    puzzles_per_diff = num_questions // len(difficulties)
    remainder = num_questions % len(difficulties)

    all_puzzles: List[Dict] = []
    problem_id = 1

    def _dice_key(dice_results):
        return tuple(tuple(sorted(r)) for r in dice_results)

    for di, difficulty in enumerate(difficulties):
        count = puzzles_per_diff + (1 if di < remainder else 0)
        if count == 0:
            continue

        diff_cfg = DIFFICULTY_CONFIGS.get(difficulty, {})
        config = YachtDiceConfig(
            bonus_threshold=int(diff_cfg.get("bonus_threshold", base_cfg.bonus_threshold)),
            bonus_points=base_cfg.bonus_points,
            full_house_points=base_cfg.full_house_points,
            small_straight_points=base_cfg.small_straight_points,
            large_straight_points=base_cfg.large_straight_points,
            yacht_points=base_cfg.yacht_points,
            optimization_goal=base_cfg.optimization_goal,
        )

        print(f"\n=== Generating {difficulty} puzzles ({count} needed) ===")

        seen_keys: set = set()
        diff_success = 0
        retries = 0
        MAX_OUTER_RETRIES = count * 30
        inner_retries = fast_retries_by_diff.get(difficulty, 70)

        while diff_success < count and retries < MAX_OUTER_RETRIES:
            retries += 1
            problem = generator.generate_problem(
                difficulty, problem_id, max_retries=inner_retries
            )
            is_valid, _ = generator.validate_problem(problem)
            if not is_valid:
                problem_id += 1
                continue

            sm = problem.get("step_metrics", {})
            # Easy: only keep in-band puzzles so the full-total task stays on the
            # calibrated "easy" manifold (band_violation rows were ~20% acc drag).
            if difficulty == "easy" and sm.get("band_violation"):
                problem_id += 1
                continue

            key = _dice_key(problem['dice_results'])
            if key in seen_keys:
                problem_id += 1
                continue
            seen_keys.add(key)

            dice_results = problem['dice_results']
            optimal_score = problem['answer']
            optimal_assignment = problem.get('optimal_assignment', {})
            public_id = f"yacht_dice_en_{difficulty}_{diff_success:04d}"

            k_spot = SPOTCHECK_K.get(difficulty, 0)
            spot_sum = spotcheck_grading_total(
                difficulty,
                public_id,
                dice_results,
                optimal_assignment,
                lambda d, c: calculate_score_with_config(d, c, config),
            )
            if spot_sum is not None:
                answer_str = str(spot_sum)
                spot_rounds = deterministic_round_pick_1based(public_id, k_spot)
            else:
                answer_str = str(optimal_score)
                spot_rounds = None

            solution_str = _build_yacht_solution_en(
                dice_results=dice_results,
                optimal_assignment=optimal_assignment,
                optimal_score=optimal_score,
                config=config,
                difficulty=difficulty,
                step_metrics=sm,
                grading_answer=answer_str,
                spot_rounds_1based=spot_rounds,
            )

            base_question = config.get_user_prompt(dice_results)
            question = append_spotcheck_user_suffix(
                base_question, difficulty, public_id, korean=False
            )

            puzzle_data = {
                "id": public_id,
                "question": question,
                "answer": answer_str,
                "solution": solution_str,
                "difficulty": difficulty,
                # Required by YachtDiceEvaluator for spotcheck (medium/hard).
                "dice_results": dice_results,
                "optimal_assignment": {str(k): v for k, v in optimal_assignment.items()},
                "step_metrics": sm,
            }
            all_puzzles.append(puzzle_data)
            diff_success += 1
            print(
                f"  [{diff_success}/{count}] optimal_total={optimal_score} "
                f"answer={answer_str} "
                f"gap={sm.get('greedy_gap', '?')} "
                f"complexity={sm.get('total_decision_complexity', 0):.2f} "
                f"band_violation={sm.get('band_violation', '?')}"
            )
            problem_id += 1

        if diff_success < count:
            print(f"  ⚠️ only generated {diff_success}/{count} for {difficulty}")

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
    parser.add_argument("--workers", type=int, default=0, help="Ignored; reserved for shell compatibility")
    args = parser.parse_args()

    print("=" * 60)
    print("Yacht Dice Puzzle Generator")
    print("=" * 60)
    create_dataset_files(num_questions=args.num)
