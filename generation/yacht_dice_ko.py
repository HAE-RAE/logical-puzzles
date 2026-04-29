"""요트 다이스 문제 생성기 (한국어 버전)

난이도 기반 주사위 패턴으로 요트 다이스 최적화 문제를 생성합니다.
보너스 인지 완전 탐색 풀이기(C(12,6) x 720 x 2)를 사용합니다.

logical-puzzles-me/yacht_dice/generator.py 에서 이식:
- greedy_gap 지표용 그리디 참조 풀이기
- 라운드별 top1/top2 margin 및 decision_complexity 지표
- 코프라임 시드 재시도를 통한 델타 밴드 난이도 필터링
- step_metrics를 퍼즐 JSONL 에 포함
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
# 설정
# ============================================================

@dataclass
class YachtDiceConfig:
    """요트 다이스 점수 규칙 설정"""

    bonus_threshold: int = 63
    bonus_points: int = 35

    full_house_points: int = 25
    small_straight_points: int = 30
    large_straight_points: int = 40
    yacht_points: int = 50

    optimization_goal: Literal["maximize", "minimize"] = "maximize"

    def get_system_prompt(self) -> str:
        """한국어 시스템 프롬프트 생성 (Answer: 형식 지시 포함)"""
        goal_word = "최대" if self.optimization_goal == "maximize" else "최소"
        return f"""당신은 요트 다이스 최적화 문제를 푸는 전문가입니다.

요트 다이스는 5개의 주사위를 12라운드 동안 굴려 각 라운드를 점수 카테고리에 배정하는 주사위 게임입니다.

점수 카테고리:
- 에이스~식스: 해당 숫자가 나온 주사위의 합
- 쓰리 오브 어 카인드: 3개 이상 같은 숫자면 모든 주사위의 합
- 포 오브 어 카인드: 4개 이상 같은 숫자면 모든 주사위의 합
- 풀 하우스: 한 숫자 3개와 다른 숫자 2개면 {self.full_house_points}점
- 스몰 스트레이트: 연속 4개 숫자면 {self.small_straight_points}점
- 라지 스트레이트: 연속 5개 숫자면 {self.large_straight_points}점
- 요트: 5개 모두 같은 숫자면 {self.yacht_points}점

상단 섹션 보너스: 에이스~식스의 합이 {self.bonus_threshold} 이상이면 {self.bonus_points} 보너스 점수가 추가됩니다.

각 라운드를 카테고리에 최적 배정하여 {goal_word} 총점을 구하세요.
각 카테고리는 한 번만 사용할 수 있습니다.

중요: 반드시 마지막 줄에 다음과 같은 형식으로 답을 작성하세요:
Answer: [숫자]

이 줄을 절대 생략하지 마세요. 구체적인 지시사항에 따라 숫자를 제시하세요.
"""

    def get_user_prompt(self, dice_results: List[List[int]]) -> str:
        """한국어 사용자 프롬프트 생성"""
        goal_word = "최대" if self.optimization_goal == "maximize" else "최소"
        prompt = f"다음 12라운드의 주사위 결과가 주어졌을 때, {goal_word} 가능한 총점을 구하세요:\n\n"
        for i, dice in enumerate(dice_results):
            prompt += f"라운드 {i+1}: {dice}\n"
        prompt += f"\n최적 배정을 계산하고 {goal_word} 총점을 제시하세요."
        return prompt


# ============================================================
# 점수 기본 함수
# ============================================================

def get_all_categories() -> List[str]:
    return [
        "Aces", "Twos", "Threes", "Fours", "Fives", "Sixes",
        "Three-Of-A-Kind", "Four-Of-A-Kind", "Full House",
        "Small Straight", "Large Straight", "Yacht"
    ]


UPPER_CATEGORIES = ["Aces", "Twos", "Threes", "Fours", "Fives", "Sixes"]


# 카테고리 한국어 표시명 매핑
CATEGORY_DISPLAY_NAME = {
    "Aces": "에이스",
    "Twos": "투",
    "Threes": "쓰리",
    "Fours": "포",
    "Fives": "파이브",
    "Sixes": "식스",
    "Three-Of-A-Kind": "쓰리 오브 어 카인드",
    "Four-Of-A-Kind": "포 오브 어 카인드",
    "Full House": "풀 하우스",
    "Small Straight": "스몰 스트레이트",
    "Large Straight": "라지 스트레이트",
    "Yacht": "요트",
}


def calculate_score_with_config(dice: List[int], category: str, config: YachtDiceConfig) -> int:
    """사용자 설정값을 적용한 점수 계산."""
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
    """기본 설정값으로 점수 계산."""
    return calculate_score_with_config(dice, category, YachtDiceConfig())


def calculate_total_score(assignment: Dict[int, str], dice_results: List[List[int]],
                          config: YachtDiceConfig) -> int:
    """배정에 대한 총점 계산 (보너스 포함)."""
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
# 풀이기
# ============================================================

def solve_yacht_dice(dice_results: List[List[int]], config: YachtDiceConfig) -> Tuple[int, Dict[int, str], bool]:
    """보너스 인지 완전 탐색 최적 풀이기 + 다중최적 유일성 판별.

    (best_total, best_assignment, is_unique) 3-튜플 반환. 6! 순열 열거 중 동점
    순열을 수집하여 최적 배정이 유일한지 판별.
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
    """greedy_gap 지표용 근시안 그리디 기준 풀이기.

    각 라운드를 순서대로 순회하며 남은 카테고리 중 즉시 점수가 최대인
    카테고리를 선택합니다. 미래 탐색이나 보너스 플래닝은 하지 않습니다.
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
    """풀이를 읽기 쉬운 형태로 포맷."""
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
            display_name = CATEGORY_DISPLAY_NAME.get(category, category)
            result.append(f"{display_name}: {dice} => {score}")
            total_score += score
            if category in UPPER_CATEGORIES:
                upper_section_score += score

    bonus = 0
    if upper_section_score >= config.bonus_threshold:
        bonus = config.bonus_points
        total_score += bonus

    output = "\n".join(result)
    output += f"\n\n상단 섹션 합계: {upper_section_score}"
    output += f"\n상단 섹션 보너스: {bonus} (기준: {config.bonus_threshold})"
    output += f"\n총점: {total_score}"
    return output


# ============================================================
# 난이도별 주사위 생성기
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
    """난이도에 맞춘 요트 다이스 문제 생성기."""

    def __init__(self):
        self.config = YachtDiceConfig()

    def generate_dice_by_difficulty(self, difficulty: str, seed: int = None) -> List[List[int]]:
        """난이도에 따른 12라운드 주사위 결과 생성."""
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
        """단일 문제 생성; 지표가 난이도 밴드에 들 때까지 시드 재시도."""
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
        """문제 유효성 검증."""
        try:
            required_fields = ['id', 'difficulty', 'dice_results', 'answer']
            for field in required_fields:
                if field not in problem:
                    return False, f"필수 필드 누락: {field}"

            dice_results = problem['dice_results']
            if len(dice_results) != 12:
                return False, f"12라운드가 필요하지만 {len(dice_results)}라운드가 제공됨"

            for round_idx, dice in enumerate(dice_results):
                if len(dice) != 5:
                    return False, f"라운드 {round_idx+1}: 5개의 주사위가 필요하지만 {len(dice)}개가 제공됨"
                for die in dice:
                    if not (1 <= die <= 6):
                        return False, f"라운드 {round_idx+1}: 유효하지 않은 주사위 값 {die}"

            optimal_score, _, _ = solve_yacht_dice(dice_results, self.config)
            if optimal_score != problem['answer']:
                return False, f"정답 불일치: 기대값 {optimal_score}, 실제값 {problem['answer']}"

            return True, "문제가 유효합니다"

        except Exception as e:
            return False, f"검증 오류: {str(e)}"


SFT_SOLUTION_RUBRIC_KO = (
    "STEP0=문제 메타 · STEP1=주어진 조건 · STEP2=풀이 전개 · STEP3=답·검산"
)


def _build_yacht_solution_ko(
    dice_results: List[List[int]],
    optimal_assignment: Dict[int, str],
    optimal_score: int,
    config: YachtDiceConfig,
    difficulty: str,
    step_metrics: Dict,
) -> str:
    """SFT teacher trace: 요트 다이스 라운드별 배정 SEG."""
    categories = get_all_categories()
    num_rounds = len(dice_results)
    upper_sum = 0
    bonus = 0
    total = 0

    round_to_cat: Dict[int, str] = dict(optimal_assignment)

    lines: List[str] = [
        SFT_SOLUTION_RUBRIC_KO,
        "[STEP 0] 문제 메타",
        f"  - 난이도: {difficulty}",
        f"  - 라운드 수: {num_rounds} · 라운드당 5개 주사위",
        f"  - 상단 섹션 보너스: {config.bonus_points}점 (기준 {config.bonus_threshold})",
        f"  - 탐욕/최적 점수 차: {step_metrics.get('greedy_gap', 0)}",
        "  - 최종 답은 [STEP 3]에서 확정",
        "[STEP 1] 주어진 조건",
        "  - 규칙: 각 라운드의 5개 주사위를 12개 카테고리 중 하나에 정확히 한 번 배정.",
        "  - 규칙: 상단 섹션(Aces~Sixes) 합이 기준 이상이면 보너스.",
        "  - 라운드별 주사위:",
    ]
    for idx, dice in enumerate(dice_results):
        lines.append(f"    R{idx + 1}: {dice}")

    lines.append("[STEP 2] 풀이 전개")
    lines.append(
        f"  · 요약: DP로 (라운드, 남은 카테고리) 최대 점수 탐색 → 최적 배정 · "
        f"SEG {num_rounds}개"
    )

    for idx in sorted(round_to_cat.keys()):
        dice = dice_results[idx]
        cat = round_to_cat[idx]
        display_name = CATEGORY_DISPLAY_NAME.get(cat, cat)
        score = calculate_score_with_config(dice, cat, config)
        tag = " (상단)" if cat in UPPER_CATEGORIES else ""
        lines.append(
            f"    [SEG {idx + 1}] R{idx + 1} 주사위 {dice} → {display_name}{tag}: {score}점"
        )
        if cat in UPPER_CATEGORIES:
            upper_sum += score
        total += score

    if upper_sum >= config.bonus_threshold:
        bonus = config.bonus_points
        total += bonus

    lines.extend([
        "[STEP 3] 답·검산",
        f"  - 최종 답(최적 총점): {optimal_score}",
        f"  - 상단 섹션 합계: {upper_sum} / 기준 {config.bonus_threshold} → 보너스 {bonus}",
        f"  - 재계산 총점: {total} (정답과 일치해야 함)",
        "  - 배정 유일성·보너스 조건이 위 SEG와 일관되는지 확인.",
    ])
    return "\n".join(lines)


# ============================================================
# 데이터셋 생성
# ============================================================

def create_dataset_files(num_questions: int):
    """요트 다이스 퍼즐 CSV + JSONL 데이터셋 생성."""
    import pandas as pd

    print(f"{num_questions}개의 요트 다이스 퍼즐을 생성합니다...")

    generator = YachtDiceProblemGenerator()
    config = YachtDiceConfig()

    difficulties = ["easy", "medium", "hard"]
    puzzles_per_diff = num_questions // len(difficulties)
    remainder = num_questions % len(difficulties)

    all_puzzles: List[Dict] = []
    problem_id = 1

    for i, difficulty in enumerate(difficulties):
        count = puzzles_per_diff + (1 if i < remainder else 0)
        if count == 0:
            continue

        print(f"\n=== {difficulty} 난이도 퍼즐 생성 중 ({count}개 필요) ===")

        diff_success = 0
        for j in range(count):
            problem = generator.generate_problem(difficulty, problem_id)
            is_valid, message = generator.validate_problem(problem)

            if is_valid:
                dice_results = problem['dice_results']
                optimal_score = problem['answer']
                optimal_assignment = problem.get('optimal_assignment', {})
                solution_str = _build_yacht_solution_ko(
                    dice_results=dice_results,
                    optimal_assignment=optimal_assignment,
                    optimal_score=optimal_score,
                    config=config,
                    difficulty=difficulty,
                    step_metrics=problem.get('step_metrics', {}),
                )

                question = config.get_user_prompt(dice_results)

                puzzle_data = {
                    "id": f"yacht_dice_ko_{difficulty}_{diff_success:04d}",
                    "question": question,
                    "answer": str(optimal_score),
                    "solution": solution_str,
                    "difficulty": difficulty,
                }
                all_puzzles.append(puzzle_data)
                diff_success += 1
                sm = problem['step_metrics']
                print(
                    f"  [{j+1}/{count}] 점수={optimal_score} "
                    f"gap={sm['greedy_gap']} complexity={sm['total_decision_complexity']:.2f} "
                    f"band_violation={sm['band_violation']}"
                )
            else:
                print(f"  [{j+1}/{count}] 유효하지 않음: {message}")

            problem_id += 1

    print(f"\n총 {len(all_puzzles)}개의 퍼즐이 생성되었습니다")

    df = pd.DataFrame(all_puzzles)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "yacht_dice_ko.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV 파일 생성 완료: {csv_path}")

    # JSONL
    json_dir = PROJECT_ROOT / "data" / "jsonl"
    json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = json_dir / "yacht_dice_ko.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_puzzles:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL 파일 생성 완료: {jsonl_path}")

    return df, all_puzzles


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="요트 다이스 퍼즐 생성기 (한국어)")
    parser.add_argument("--num", type=int, default=12, help="생성할 문제 수")
    args = parser.parse_args()

    print("=" * 60)
    print("요트 다이스 퍼즐 생성기 (한국어)")
    print("=" * 60)
    create_dataset_files(num_questions=args.num)
