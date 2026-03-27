"""요트 다이스 문제 생성기 및 풀이기 (한국어 버전)

난이도 기반 주사위 패턴으로 요트 다이스 최적화 문제를 생성합니다.
헝가리안 알고리즘 대신 보너스 인지 완전 탐색 풀이기(C(12,6) x 720 x 2)를 사용합니다.
헝가리안 알고리즘은 상단 섹션 보너스를 올바르게 처리할 수 없기 때문입니다.
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
# 풀이기 (보너스 인지 완전 탐색)
# ============================================================

def get_all_categories() -> List[str]:
    return [
        "Aces", "Twos", "Threes", "Fours", "Fives", "Sixes",
        "Three-Of-A-Kind", "Four-Of-A-Kind", "Full House",
        "Small Straight", "Large Straight", "Yacht"
    ]


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


def calculate_score(dice: List[int], category: str) -> int:
    """주사위와 카테고리에 대한 점수 계산 (기본 설정값 사용)."""
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
    """배정에 대한 총점 계산 (보너스 포함)."""
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
    보너스 인지 완전 탐색을 사용한 최적 해 탐색.

    상단 섹션에 대해 C(12,6) = 924개의 부분집합을 순회하고,
    각 섹션에 대해 6! = 720개의 순열을 전수 조사합니다.
    총 연산량: 924 x 720 x 2 = 약 130만 회.

    이 방식은 헝가리안 알고리즘으로는 처리할 수 없는
    상단 섹션 보너스(35점)를 올바르게 반영합니다.
    """
    categories = get_all_categories()
    upper_cats = categories[:6]
    lower_cats = categories[6:]
    n = len(dice_results)

    # 점수 행렬 사전 계산
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

        # 최적 상단 배정 (6! 전수 조사)
        best_upper_score = -1 if is_maximize else float('inf')
        best_upper_perm = perms_6[0]
        for perm in perms_6:
            s = sum(upper_scores[upper_list[j]][perm[j]] for j in range(6))
            if (is_maximize and s > best_upper_score) or (not is_maximize and s < best_upper_score):
                best_upper_score = s
                best_upper_perm = perm

        # 최적 하단 배정 (6! 전수 조사)
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
    """풀이를 읽기 쉬운 형태로 포맷."""
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
            display_name = CATEGORY_DISPLAY_NAME.get(category, category)
            result.append(f"{display_name}: {dice} => {score}")
            total_score += score
            if category in upper_categories:
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
# 주사위 생성
# ============================================================

def generate_random_dice(num_rounds: int = 12, dice_per_round: int = 5, seed: int = None) -> List[List[int]]:
    """무작위 주사위 결과 생성."""
    if seed is not None:
        random.seed(seed)

    dice_results = []
    for _ in range(num_rounds):
        round_result = [random.randint(1, 6) for _ in range(dice_per_round)]
        round_result.sort()
        dice_results.append(round_result)

    return dice_results


def format_user_prompt(dice_results: List[List[int]]) -> str:
    """주사위 결과를 사용자 프롬프트로 포맷."""
    prompt = "12라운드의 주사위 결과입니다. 각 결과를 점수 카테고리에 배정하세요:\n\n"
    for i, dice in enumerate(dice_results, 1):
        prompt += f"{i}. {dice}\n"
    prompt += "\n각 카테고리에 하나의 결과를 배정하고 총점을 최대화하도록 점수를 계산하세요."
    return prompt


# ============================================================
# 난이도별 문제 생성기
# ============================================================

class YachtDiceProblemGenerator:
    """난이도별 요트 다이스 문제 생성"""

    def __init__(self):
        self.config = YachtDiceConfig()

    def generate_dice_by_difficulty(self, difficulty: str, seed: int = None) -> List[List[int]]:
        """난이도에 따른 주사위 결과 생성."""
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
        """지정된 난이도의 요트 다이스 문제 하나를 생성."""
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
        """문제가 올바르게 구성되어 있고 풀 수 있는지 검증."""
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

            optimal_score, _ = solve_yacht_dice(dice_results, self.config)
            if optimal_score != problem['answer']:
                return False, f"정답 불일치: 기대값 {optimal_score}, 실제값 {problem['answer']}"

            return True, "문제가 유효합니다"

        except Exception as e:
            return False, f"검증 오류: {str(e)}"


# ============================================================
# 데이터셋 생성
# ============================================================

def create_dataset_files(num_questions: int):
    """
    요트 다이스 퍼즐 데이터셋 파일을 생성합니다.

    Args:
        num_questions: 생성할 문제 수

    Returns:
        Tuple[pd.DataFrame, List[Dict]]: (데이터프레임, JSON 리스트)
    """
    import pandas as pd

    print(f"{num_questions}개의 요트 다이스 퍼즐을 생성합니다...")

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

        print(f"\n=== {difficulty} 난이도 퍼즐 생성 중 ({count}개 필요) ===")

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
                    "id": f"yacht_dice_ko_{len(all_puzzles)}",
                    "question": question,
                    "answer": str(optimal_score),
                    "solution": solution_str,
                    "difficulty": difficulty,
                    "dice_results": dice_results,
                    "seed": problem['seed']
                }
                all_puzzles.append(puzzle_data)
                print(f"  [{j+1}/{count}] 점수={optimal_score}")
            else:
                print(f"  [{j+1}/{count}] 유효하지 않음: {message}")

            problem_id += 1

    print(f"\n총 {len(all_puzzles)}개의 퍼즐이 생성되었습니다")

    df = pd.DataFrame(all_puzzles)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # CSV
    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "yacht_dice_ko.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV 파일 생성 완료: {csv_path}")

    # JSONL
    json_dir = PROJECT_ROOT / "data" / "json"
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
