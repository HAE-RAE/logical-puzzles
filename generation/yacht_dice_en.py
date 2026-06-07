"""Yacht Dice Problem Generator (EN)

Generates Yacht Dice optimization problems with difficulty-based dice patterns.
Uses bonus-aware exhaustive search solver (C(12,6) x 720 x 2).

Ported from logical-puzzles-me/yacht_dice/generator.py:
- Greedy reference solver for greedy_gap metric
- Per-round top1/top2 margin and decision_complexity metrics
- Delta-band difficulty filtering with co-prime seed retry
- step_metrics used internally during generation (not exported to CSV/JSONL)

Spotcheck helpers (formerly evaluation/yacht_dice_spotcheck.py) are defined
inline below so this file is the single source of truth for generation.
"""

import hashlib
import random
import json
import itertools
import sys
from itertools import combinations
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import List, Dict, Tuple, Optional, Literal, Callable
from collections import Counter
from dataclasses import dataclass

import numpy as np


# ============================================================
# Spotcheck helpers (single source of truth)
# ============================================================

# Rounds in the spotcheck partial sum (0 => answer is full 12-round optimal total).
# v33: easy K=2 + constructive dice -> graded answer is the SUM of two globally
# exclusive fixed-score spot rounds (55/60/65/70/75/80/90/100, ...), not a single
# {25,30,40,50}. Constructive placement keeps both spot scores stable under the
# global optimum. medium/hard unchanged from v32 calibration.
SPOTCHECK_K: Dict[str, int] = {
    "easy": 2,
    "medium": 1,
    "hard": 5,
}


def deterministic_round_pick_1based(public_id: str, k: int) -> List[int]:
    """Return ``k`` distinct round indices in 1..12, stable for a given ``public_id``."""
    if k <= 0:
        return []
    digest = hashlib.sha256(f"yacht_dice_spotcheck_v1|{public_id}".encode()).digest()
    h = int.from_bytes(digest[:16], "big")
    combs = list(combinations(range(1, 13), k))
    chosen = combs[h % len(combs)]
    return sorted(chosen)


def _category_for_round(optimal_assignment: Dict, round_idx0: int) -> str:
    if round_idx0 in optimal_assignment:
        return optimal_assignment[round_idx0]
    return optimal_assignment[str(round_idx0)]


def spotcheck_grading_total(
    difficulty: str,
    public_id: str,
    dice_results: List[List[int]],
    optimal_assignment: Dict,
    score_fn: Callable[[List[int], str], int],
) -> Optional[int]:
    """Partial sum on spotcheck rounds, or ``None`` if ``k == 0``."""
    k = SPOTCHECK_K.get(difficulty, 0)
    if k <= 0:
        return None
    total = 0
    for r1 in deterministic_round_pick_1based(public_id, k):
        i0 = r1 - 1
        cat = _category_for_round(optimal_assignment, i0)
        total += score_fn(dice_results[i0], cat)
    return total


def append_spotcheck_user_suffix(
    base_question: str,
    difficulty: str,
    public_id: str,
) -> str:
    """Append spotcheck instructions when ``SPOTCHECK_K[difficulty] > 0``."""
    k = SPOTCHECK_K.get(difficulty, 0)
    if k <= 0:
        return base_question

    rounds = deterministic_round_pick_1based(public_id, k)
    rlist = ", ".join(str(r) for r in rounds)

    suffix = (
        f"\n\nIMPORTANT — Spotcheck grading (not the full 12-round total):\n"
        f"First compute an optimal assignment that maximizes the usual total score over all 12 rounds "
        f"(same rules and bonus as always). Using that optimal assignment, output ONLY the sum of "
        f"the per-round scores for rounds {rlist}. Your final line must be exactly "
        f"`Answer: <integer>` with that partial sum.\n"
    )

    return base_question + suffix

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
    # v21: easy dice 재구조화. v20 diversification 은 acc 17→20% 미세 개선뿐.
    #   four_kind_high 의 "sum of all dice" 규칙이 모델에게 까다로움 → 제거.
    #   full_house 는 (3a+2b vs 25 비교) 판단이 필요 → 제거.
    #   yacht / LS / SS 만으로 압축 → 카테고리 매핑이 거의 결정적.
    #   K=1 + is_unique 필터와 결합하여 ~30-50% 기대.
    "easy": {
        # v31: fixed-score-dominant pool for K=2 sum answers.
        # Spot rounds must land on Yacht/Straight/Full House (fixed values), so
        # the pool is heavily weighted to those; a little three_kind keeps some
        # non-spot variety without blocking acceptance.
        "roll_types": [
            "yacht", "large_straight", "small_straight", "full_house",
            "three_kind",
        ],
        "weights": [26, 22, 26, 22, 4],
    },
    "medium": {
        "roll_types": ["four_kind_high", "full_house", "three_kind", "pair", "normal"],
        "weights": [12, 13, 25, 25, 25],
    },
    "hard": {
        "roll_types": ["three_kind", "pair", "normal"],
        "weights": [15, 20, 65],
    },
}


# Extreme-mismatch bands for greedy_gap δ (docs/difficulty_definition.md §2.6).
# The step-count proxy for yacht_dice is weak (solver is deterministic; most
# of the "work" is the 12! assignment search regardless of dice). We only
# reject obvious outliers — e.g., a hard puzzle where greedy already nears
# optimal, or an easy puzzle that traps greedy by a large margin.
_DIFFICULTY_BANDS = {
    # v21 calibration (target 75/50/25 vs v20 measured 20/47/31):
    # - easy: band loose (quality 는 dice mix + is_unique 필터로 보장).
    # - medium: 변경 없음 (47% — target 안).
    # - hard: 31% — 1%p 초과. floor 살짝 강화 (gap 18→20, complexity 6.0→6.5).
    "easy": {
        # v24: dice 다양화로 complexity 자연 증가 → band 도 완화.
        # gap_abs 0-3 → 0-8 / complexity 0-1.5 → 0-3.0 로 확장.
        "greedy_gap_abs": {"min": 0, "max": 8},
        "decision_complexity": {"min": 0.0, "max": 3.0},
    },
    "medium": {
        "greedy_gap_abs": {"min": 3, "max": 30},
        "decision_complexity": {"min": 2.0, "max": 7.0},
    },
    "hard": {
        "greedy_gap_abs": {"min": 20, "max": None},
        "decision_complexity": {"min": 6.5, "max": None},
    },
}


# ============================================================
# Easy-tier constructive builders
# ============================================================
# The easy task grades a sum of K spotcheck rounds taken from the GLOBAL optimal
# assignment. Random dice rarely make those spot categories globally exclusive,
# so models place e.g. Yacht on a different round and the spot sum collapses
# below the gold (all errors are negative). To fix this we construct the dice so
# each spot round carries a high-value category that NO other round can match,
# forcing the global optimum to assign it there regardless of the rest.

# Spot categories that are easy to make globally exclusive with a clear margin.
EASY_SPOT_CATEGORIES = ("yacht", "large_straight", "small_straight", "full_house")
EASY_SPOT_SCORE = {
    "yacht": 50,
    "large_straight": 40,
    "small_straight": 30,
    "full_house": 25,
}


def _build_easy_spot_dice(cat: str, rng: random.Random) -> List[int]:
    """Dice for a spot round whose isolated top1 is ``cat`` with a large margin."""
    if cat == "yacht":
        v = rng.randint(1, 6)
        return [v] * 5
    if cat == "large_straight":
        start = rng.choice([1, 2])
        d = list(range(start, start + 5))
        rng.shuffle(d)
        return d
    if cat == "small_straight":
        base = list(rng.choice([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]))
        # Duplicate an existing value so the 5th die cannot extend to a large
        # straight (keeps the round a clean Small Straight = 30).
        d = base + [rng.choice(base)]
        rng.shuffle(d)
        return d
    if cat == "full_house":
        # a=1 keeps the three-of-a-kind sum small (<= 15) so Full House (25)
        # stays the clear top1 with margin >= 10.
        b = rng.choice([2, 3, 4, 5, 6])
        d = [1, 1, 1, b, b]
        rng.shuffle(d)
        return d
    raise ValueError(f"unknown spot category: {cat}")


def _build_easy_noise_dice(rng: random.Random) -> List[int]:
    """Low-value dice that cannot form yacht/straight/full-house/3-4-of-a-kind.

    Uses a 2-1-2 split over three non-consecutive values, so no category that a
    spot round relies on (Yacht/LS/SS/Full House) can be matched elsewhere.
    """
    for _ in range(200):
        vals = rng.sample(range(1, 7), 3)
        a, b, c = vals
        dice = [a, a, b, c, c]  # counts 2-2-1 -> no 3-of-a-kind / full house / yacht
        s = set(dice)
        # reject any 4-in-a-row (small straight) just in case
        if any({x, x + 1, x + 2, x + 3} <= s for x in (1, 2, 3)):
            continue
        rng.shuffle(dice)
        return dice
    # Fallback that is trivially safe.
    return [1, 1, 3, 3, 5]


def _build_easy_constructive_dice(
    spot_rounds_0based: List[int],
    spot_cats: List[str],
    rng: random.Random,
    num_rounds: int = 12,
) -> List[List[int]]:
    """Place each spot category on its spot round; fill the rest with noise."""
    dice: List[Optional[List[int]]] = [None] * num_rounds
    for idx, cat in zip(spot_rounds_0based, spot_cats):
        dice[idx] = _build_easy_spot_dice(cat, rng)
    for i in range(num_rounds):
        if dice[i] is None:
            dice[i] = _build_easy_noise_dice(rng)
    return [list(d) for d in dice]


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
                         max_retries: int = 200, forced_dice: List[List[int]] = None) -> Dict:
        """Generate a single problem; retries seeds until metrics fit difficulty band.

        If ``forced_dice`` is supplied, the band-retry loop is skipped and metrics
        are computed for that exact dice configuration (used by the constructive
        easy-tier generator).
        """
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

        effective_retries = 1 if forced_dice is not None else max_retries
        for retry in range(effective_retries):
            trial_seed = seed + retry * 997
            if forced_dice is not None:
                dice_results = [list(r) for r in forced_dice]
            else:
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

            # Constructive (forced) dice bypass band filtering entirely.
            if forced_dice is not None:
                in_band = True
                break

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


DATASET_COLUMNS = ("id", "question", "answer", "solution", "difficulty")


def create_dataset_files(
    num_questions: int,
    difficulties: Optional[List[str]] = None,
):
    """Create CSV + JSONL dataset files for Yacht Dice puzzles.

    Calls ``generate_problem`` with a per-difficulty inner retry budget so each
    attempt usually finishes quickly while still preferring in-band dice (see
    ``_DIFFICULTY_BANDS``).  Puzzles that exhaust retries may still be
    out-of-band (``band_violation`` in step_metrics) but remain valid with the
    correct optimal score under the per-difficulty ``YachtDiceConfig``.

    If ``difficulties`` is a single tier (e.g. ``["easy"]``), writes only
    ``data/jsonl/yacht_dice_en_easy.jsonl`` and skips the combined CSV/JSONL.
    """
    import pandas as pd

    fast_retries_by_diff = {"easy": 60, "medium": 120, "hard": 90}

    all_tiers = ["easy", "medium", "hard"]
    if difficulties is None:
        difficulties = list(all_tiers)
    else:
        difficulties = [d.lower() for d in difficulties]
        bad = [d for d in difficulties if d not in all_tiers]
        if bad:
            raise ValueError(f"Unknown difficulty tier(s): {bad}; expected {all_tiers}")

    print(
        f"Generating {num_questions} Yacht Dice puzzles "
        f"(tiers: {', '.join(difficulties)})..."
    )

    generator = YachtDiceProblemGenerator()
    base_cfg = YachtDiceConfig()

    puzzles_per_diff = num_questions // len(difficulties)
    remainder = num_questions % len(difficulties)

    all_puzzles: List[Dict] = []
    problem_id = 1

    # Easy-tier acceptance policy (K=2 constructive).
    # Each spot round must be globally exclusive with a clear top1 margin; the
    # graded answer is the SUM of those two fixed-score categories (55..100).
    # No single answer value may exceed this share of the dataset (anti-collapse).
    EASY_MAX_VALUE_SHARE = 0.20

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
        easy_margin_min = 10
        easy_value_counts: Dict[int, int] = {}
        easy_max_per_value = max(2, int(round(EASY_MAX_VALUE_SHARE * count)))

        while diff_success < count and retries < MAX_OUTER_RETRIES:
            retries += 1
            # Lightweight calibration: if acceptance is too low, relax the
            # spot-round margin slightly so generation does not stall.
            if difficulty == "easy" and retries % 40 == 0:
                accepted_rate = diff_success / retries
                if accepted_rate < 0.15:
                    easy_margin_min = max(8, easy_margin_min - 1)
            if difficulty == "easy":
                # Constructive generation: plant globally-exclusive high-value
                # categories on the deterministic spot rounds.
                k_spot = SPOTCHECK_K.get("easy", 0)
                public_id_pre = f"yacht_dice_en_easy_{diff_success:04d}"
                spot_rounds_pre = [
                    r - 1 for r in deterministic_round_pick_1based(public_id_pre, k_spot)
                ]
                crng = random.Random(1000 + problem_id + 10000)
                spot_cats_choice = crng.sample(EASY_SPOT_CATEGORIES, len(spot_rounds_pre))
                # Large + Small Straight cannot coexist: an LS round also scores
                # Small Straight (30), which would break SS exclusivity.
                while ("small_straight" in spot_cats_choice
                       and "large_straight" in spot_cats_choice):
                    spot_cats_choice = crng.sample(EASY_SPOT_CATEGORIES, len(spot_rounds_pre))
                forced = _build_easy_constructive_dice(
                    spot_rounds_pre, spot_cats_choice, crng
                )
                problem = generator.generate_problem(
                    "easy", problem_id, forced_dice=forced
                )
            else:
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
            # v25: v21~v24 의 is_unique_assignment 필터는 generation 을 거의
            # 완전히 막음 (8 dice 타입 다양화에도 pass rate ~0.1% 미만).
            # 진짜 필요한 조건은 spot 라운드의 카테고리만 모든 최적해에서
            # 일치하는 것 (전체 배정 유일성이 아님). K=2 이므로 두 스팟 라운드
            # 대안: spot 라운드가 명백한 dominant 카테고리를 가지면
            # (해당 라운드의 per_round_margin >= 10), 어떤 최적해를 골라도
            # 그 라운드는 항상 같은 카테고리로 배정될 확률이 매우 높음.
            if difficulty == "easy":
                k_spot_check = SPOTCHECK_K.get("easy", 0)
                if k_spot_check >= 1:
                    spot_rounds_0based = [
                        r - 1 for r in deterministic_round_pick_1based(
                            f"yacht_dice_en_easy_{diff_success:04d}",
                            k_spot_check,
                        )
                    ]
                    # v27: ROOT CAUSE FIX. v21~v26 의 per_round_margin 필터는
                    # 잘못된 신호였다 — 고립 평가시 spot 라운드의 best 카테고리를
                    # 봤지만, spotcheck 정답은 *전역 최적 배정이 그 라운드에 실제
                    # 부여한 카테고리*에 달려 있다. 카테고리 경쟁(예: SS 가능한
                    # 라운드 여럿)이 있으면 둘이 달라져, 모델의 자연스러운 greedy
                    # 답(이 라운드 최고점)이 gold(전역 최적의 leftover 배정)와
                    # 불일치 → 47% 천장.
                    # 올바른 조건: spot 라운드가 전역 최적 배정에서 자기 top1
                    # (고립 최고점) 카테고리를 그대로 부여받았는가. 그러면 모델의
                    # greedy 답 = gold. 추가로 margin ≥ 10 으로 그 top1 이 명확
                    # 하도록 한다.
                    opt_assign = problem.get("optimal_assignment", {})
                    top1s = sm.get("per_round_top1", [])
                    margins = sm.get("per_round_margin", [])
                    dice_all = problem["dice_results"]
                    spot_ok = True
                    spot_cats: List[str] = []
                    for idx in spot_rounds_0based:
                        if idx >= len(top1s) or idx >= len(margins):
                            spot_ok = False
                            break
                        cat = opt_assign.get(idx, opt_assign.get(str(idx)))
                        if cat is None:
                            spot_ok = False
                            break
                        assigned_score = calculate_score_with_config(
                            dice_all[idx], cat, config
                        )
                        # (a) The global optimum must assign this round its
                        # isolated top1 category, and that top1 must be clearly
                        # dominant (large margin) on the round itself.
                        if assigned_score != top1s[idx]:
                            spot_ok = False
                            break
                        if margins[idx] < easy_margin_min:
                            spot_ok = False
                            break
                        # (b) GLOBAL EXCLUSIVITY (root-cause fix for sub-target
                        # accuracy): no OTHER round may score as high in this
                        # category. This forces the global optimum to place
                        # `cat` on this exact round, so the graded spot score is
                        # stable no matter how the model solves the other rounds
                        # (previously the model used e.g. Yacht on a different
                        # round, yielding a lower spot sum -> all-negative errors).
                        best_elsewhere = 0
                        for j in range(len(dice_all)):
                            if j == idx:
                                continue
                            s = calculate_score_with_config(dice_all[j], cat, config)
                            if s > best_elsewhere:
                                best_elsewhere = s
                        if best_elsewhere >= assigned_score:
                            spot_ok = False
                            break
                        spot_cats.append(cat)
                    # (c) The two spot rounds must use distinct categories (a
                    # category can only be assigned once across all 12 rounds).
                    if spot_ok and len(set(spot_cats)) != len(spot_cats):
                        spot_ok = False
                    if not spot_ok:
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

            if difficulty == "easy":
                # Distribution control: cap how often any single answer value
                # appears so the dataset does not collapse onto one number.
                ans_num = int(answer_str)
                if easy_value_counts.get(ans_num, 0) >= easy_max_per_value:
                    problem_id += 1
                    continue

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
                base_question, difficulty, public_id
            )

            puzzle_data = {
                "id": public_id,
                "question": question,
                "answer": answer_str,
                "solution": solution_str,
                "difficulty": difficulty,
            }
            all_puzzles.append(puzzle_data)
            if difficulty == "easy":
                easy_value_counts[int(answer_str)] = easy_value_counts.get(int(answer_str), 0) + 1
            diff_success += 1
            print(
                f"  [{diff_success}/{count}] optimal_total={optimal_score} "
                f"answer={answer_str} "
                f"gap={sm.get('greedy_gap', '?')} "
                f"complexity={sm.get('total_decision_complexity', 0):.2f} "
                f"band_violation={sm.get('band_violation', '?')}"
                + (
                    f" easy_margin={easy_margin_min}"
                    if difficulty == "easy" else ""
                )
            )
            problem_id += 1

        if diff_success < count:
            print(f"  ⚠️ only generated {diff_success}/{count} for {difficulty}")
        if difficulty == "easy" and diff_success > 0:
            print(
                "  easy answer distribution: "
                + ", ".join(f"{k}={v}" for k, v in sorted(easy_value_counts.items()))
            )

    print(f"\nGenerated {len(all_puzzles)} puzzles")

    df = pd.DataFrame(all_puzzles, columns=list(DATASET_COLUMNS))
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    json_dir = PROJECT_ROOT / "data" / "jsonl"
    json_dir.mkdir(parents=True, exist_ok=True)

    if len(difficulties) == 1:
        tier = difficulties[0]
        jsonl_path = json_dir / f"yacht_dice_en_{tier}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for item in all_puzzles:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"JSONL file created: {jsonl_path}")
        return df, all_puzzles

    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "yacht_dice_en.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV file created: {csv_path}")

    jsonl_path = json_dir / "yacht_dice_en.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in all_puzzles:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"JSONL file created: {jsonl_path}")

    return df, all_puzzles


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Yacht Dice Puzzle Generator (EN)")
    parser.add_argument(
        "--num",
        type=int,
        default=12,
        help="Question count (per tier if --difficulty is set; else split across easy/medium/hard)",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        nargs="+",
        default=None,
        help="Generate only these tier(s), e.g. --difficulty easy",
    )
    parser.add_argument("--workers", type=int, default=0, help="Ignored; reserved for shell compatibility")
    args = parser.parse_args()

    print("=" * 60)
    print("Yacht Dice Puzzle Generator")
    print("=" * 60)
    create_dataset_files(num_questions=args.num, difficulties=args.difficulty)
