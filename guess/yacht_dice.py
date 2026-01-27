import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
from dataclasses import dataclass
from typing import Literal


@dataclass
class YachtDiceConfig:
    """Yacht Dice 게임 규칙을 설정하는 클래스"""

    # 상단 항목 보너스 설정
    bonus_threshold: int = 63  # 보너스를 받기 위한 경계 점수
    bonus_points: int = 35  # 보너스 점수

    # 하단 항목 점수 설정
    full_house_points: int = 25
    small_straight_points: int = 30
    large_straight_points: int = 40
    yacht_points: int = 50

    # 최적화 목표
    optimization_goal: Literal["maximize", "minimize"] = "maximize"

    def get_system_prompt(self) -> str:
        """Generate system prompt based on configuration"""

        if self.optimization_goal == "maximize":
            goal_instruction = "Your goal is to find the maximum total score. You must assign dice to categories to achieve the highest possible score."
        else:
            goal_instruction = "Your goal is to find the minimum total score. You must assign dice to categories to achieve the lowest possible score."

        prompt = f"""You are a Yacht Dice scoring expert. You need to assign each of the 12 dice results below to exactly one category and calculate the score. Each category can only be used once.

{goal_instruction}

The scoring rules for each category are as follows:

[Upper Section]
1. Aces: Sum of all dice showing 1
2. Twos: Sum of all dice showing 2
3. Threes: Sum of all dice showing 3
4. Fours: Sum of all dice showing 4
5. Fives: Sum of all dice showing 5
6. Sixes: Sum of all dice showing 6
(If the sum of these 6 upper categories is {self.bonus_threshold} or more, you get a {self.bonus_points} point bonus)

[Lower Section]
7. Three-Of-A-Kind: If 3 or more dice show the same number, score is the sum of all dice
8. Four-Of-A-Kind: If 4 or more dice show the same number, score is the sum of all dice
9. Full House: If dice show 3 of one number and 2 of another (e.g., 3-3-3-5-5), fixed {self.full_house_points} points
10. Small Straight: If 4 consecutive numbers are present (e.g., 1-2-3-4), fixed {self.small_straight_points} points
11. Large Straight: If 5 consecutive numbers are present (1-2-3-4-5 or 2-3-4-5-6), fixed {self.large_straight_points} points
12. Yacht: If all 5 dice show the same number, fixed {self.yacht_points} points

Each dice result must be assigned to exactly one category, and each category must be used exactly once without duplication.

Think step by step, then provide your answer in the following format.

Output format:
[Category]: [d1, d2, d3, d4, d5] => score
Total: XXX
"""

        return prompt


def generate_random_dice(num_rounds: int = 12, dice_per_round: int = 5, seed: int = None) -> List[List[int]]:
    """
    랜덤 주사위 결과를 생성하는 함수

    Args:
        num_rounds: 생성할 주사위 라운드 수 (기본값: 12)
        dice_per_round: 각 라운드당 주사위 개수 (기본값: 5)
        seed: 랜덤 시드 (재현성을 위해 사용)

    Returns:
        List[List[int]]: 주사위 결과 리스트
    """
    if seed is not None:
        random.seed(seed)

    dice_results = []
    for _ in range(num_rounds):
        round_result = [random.randint(1, 6) for _ in range(dice_per_round)]
        round_result.sort()
        dice_results.append(round_result)

    return dice_results


def format_user_prompt(dice_results: List[List[int]]) -> str:
    """
    Format dice results into user prompt

    Args:
        dice_results: List of dice results

    Returns:
        str: Formatted user prompt
    """
    prompt = "Here are 12 dice results. Assign each result to one of the categories above and calculate the score:\n\n"

    for i, dice in enumerate(dice_results, 1):
        prompt += f"{i}. {dice}\n"

    prompt += "\nAssign one dice result to each category and calculate the score to maximize the total."

    return prompt


def calculate_score_with_config(dice: List[int], category: str, config: YachtDiceConfig) -> int:
    """
    설정을 적용하여 점수를 계산

    Args:
        dice: 주사위 결과
        category: 카테고리 이름
        config: Yacht Dice 설정

    Returns:
        int: 해당 카테고리의 점수
    """
    counts = Counter(dice)
    sorted_dice = sorted(dice)

    # 상단 항목 (설정 영향 없음)
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

    # 하단 항목 (설정 적용)
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
        straights = [
            {1, 2, 3, 4},
            {2, 3, 4, 5},
            {3, 4, 5, 6}
        ]
        for straight in straights:
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


def get_all_categories() -> List[str]:
    """모든 카테고리 목록 반환"""
    return [
        "Aces", "Twos", "Threes", "Fours", "Fives", "Sixes",
        "Three-Of-A-Kind", "Four-Of-A-Kind", "Full House",
        "Small Straight", "Large Straight", "Yacht"
    ]


def calculate_total_score(assignment: Dict[int, str], dice_results: List[List[int]],
                         config: YachtDiceConfig) -> int:
    """
    할당에 대한 총점을 계산 (보너스 포함)

    Args:
        assignment: {주사위_인덱스: 카테고리} 매핑
        dice_results: 주사위 결과 리스트
        config: Yacht Dice 설정

    Returns:
        int: 총점 (보너스 포함)
    """
    upper_section_score = 0
    total_score = 0

    upper_categories = ["Aces", "Twos", "Threes", "Fours", "Fives", "Sixes"]

    for dice_idx, category in assignment.items():
        dice = dice_results[dice_idx]
        score = calculate_score_with_config(dice, category, config)
        total_score += score

        if category in upper_categories:
            upper_section_score += score

    # 보너스 계산
    if upper_section_score >= config.bonus_threshold:
        total_score += config.bonus_points

    return total_score


def solve_yacht_dice(dice_results: List[List[int]], config: YachtDiceConfig) -> Tuple[int, Dict[int, str]]:
    """
    Yacht Dice 문제의 최적 해를 구함 (헝가리안 알고리즘 사용)

    Args:
        dice_results: 12개의 주사위 결과 리스트
        config: Yacht Dice 설정

    Returns:
        Tuple[int, Dict[int, str]]: (최적 점수, 최적 할당)
    """
    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np
    except ImportError:
        print("경고: scipy가 설치되지 않았습니다. 빠른 휴리스틱 버전을 사용합니다.")
        return solve_yacht_dice_fast(dice_results, config)

    categories = get_all_categories()
    n = len(dice_results)

    # 각 주사위-카테고리 쌍의 점수를 미리 계산
    score_matrix = np.zeros((n, n))
    for i in range(n):
        for j, category in enumerate(categories):
            score = calculate_score_with_config(dice_results[i], category, config)
            score_matrix[i][j] = score

    # 헝가리안 알고리즘은 최소화 문제를 푸므로, 최대화하려면 음수로 변환
    if config.optimization_goal == "maximize":
        cost_matrix = -score_matrix
    else:
        cost_matrix = score_matrix

    # 최적 할당 찾기
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 할당 딕셔너리 생성
    assignment = {}
    for i, j in zip(row_ind, col_ind):
        assignment[i] = categories[j]

    # 총점 계산
    total_score = calculate_total_score(assignment, dice_results, config)

    return total_score, assignment


def solve_yacht_dice_fast(dice_results: List[List[int]], config: YachtDiceConfig) -> Tuple[int, Dict[int, str]]:
    """
    Yacht Dice 문제의 최적 해를 구함 (휴리스틱 기반 빠른 버전)

    Args:
        dice_results: 12개의 주사위 결과 리스트
        config: Yacht Dice 설정

    Returns:
        Tuple[int, Dict[int, str]]: (점수, 할당)
    """
    categories = get_all_categories()
    n = len(dice_results)

    # 각 주사위-카테고리 쌍의 점수를 계산
    score_matrix = []
    for i in range(n):
        row = []
        for category in categories:
            score = calculate_score_with_config(dice_results[i], category, config)
            row.append(score)
        score_matrix.append(row)

    # 그리디: 각 단계에서 가장 높은 점수를 주는 할당 선택
    used_dice = set()
    used_categories = set()
    assignment = {}

    # 모든 가능한 (주사위, 카테고리) 쌍을 점수 순으로 정렬
    pairs = []
    for i in range(n):
        for j, category in enumerate(categories):
            pairs.append((score_matrix[i][j], i, j, category))

    if config.optimization_goal == "maximize":
        pairs.sort(reverse=True)
    else:
        pairs.sort()

    # 그리디하게 할당
    for score, dice_idx, cat_idx, category in pairs:
        if dice_idx not in used_dice and category not in used_categories:
            used_dice.add(dice_idx)
            used_categories.add(category)
            assignment[dice_idx] = category

            if len(assignment) == n:
                break

    total_score = calculate_total_score(assignment, dice_results, config)
    return total_score, assignment


def format_solution(dice_results: List[List[int]], assignment: Dict[int, str],
                    config: YachtDiceConfig) -> str:
    """
    해를 읽기 쉬운 형식으로 포맷팅

    Args:
        dice_results: 주사위 결과 리스트
        assignment: 할당
        config: Yacht Dice 설정

    Returns:
        str: 포맷팅된 해
    """
    categories = get_all_categories()

    # 카테고리별로 정렬
    result = []
    upper_section_score = 0
    total_score = 0

    upper_categories = ["Aces", "Twos", "Threes", "Fours", "Fives", "Sixes"]

    for category in categories:
        # 이 카테고리에 할당된 주사위 찾기
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

    # 보너스 계산
    bonus = 0
    if upper_section_score >= config.bonus_threshold:
        bonus = config.bonus_points
        total_score += bonus

    output = "\n".join(result)
    output += f"\n\n상단 합계: {upper_section_score}"
    output += f"\n보너스: {bonus} (경계: {config.bonus_threshold})"
    output += f"\n총점: {total_score}"

    return output


def create_dataset_files(num_questions):
    """
    Yacht Dice 문제 데이터셋 파일 생성

    Args:
        num_questions: 생성할 문제 개수
        version: 버전 문자열

    Returns:
        Tuple[pd.DataFrame, List[Dict]]: (데이터프레임, JSON 리스트)
    """
    import pandas as pd
    import json

    print(f"Yacht Dice 문제 {num_questions}개를 생성 중...")

    config = YachtDiceConfig()
    output = []

    for i in range(num_questions):
        # 각 문제마다 다른 seed 사용
        seed = 1000 + i
        dice_results = generate_random_dice(seed=seed)

        # 최적 해 계산
        optimal_score, optimal_assignment = solve_yacht_dice(dice_results, config)
        optimal_solution = format_solution(dice_results, optimal_assignment, config)

        # 질문 생성
        question = format_user_prompt(dice_results)
        answer = f"{optimal_score}"

        output.append([question, answer, optimal_solution])

        if (i + 1) % 10 == 0:
            print(f"  진행: {i+1}/{num_questions}")

    yacht_df = pd.DataFrame(output, columns=['question', 'answer', 'solution'])

    print(f"\n생성 통계:")
    print(f"  생성된 문제 수: {len(yacht_df)}")
    print(f"  고유한 문제 수: {yacht_df['question'].nunique()}")
    print(f"  고유한 정답 수: {yacht_df['answer'].nunique()}")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # CSV
    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    yacht_df.insert(0, 'id', [f"yacht_dice_{i}" for i in range(len(yacht_df))])

    csv_path = csv_dir / "yacht_dice.csv"
    yacht_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV 파일이 생성: {csv_path}")

    # JSONL
    json_dir = PROJECT_ROOT / "data" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    # dice_results도 함께 저장 (정답 재계산을 위해)
    yacht_json = []
    for i in range(len(yacht_df)):
        seed = 1000 + i
        dice_results = generate_random_dice(seed=seed)

        question_data = {
            "id": f"yacht_dice_{i}",
            "question": output[i][0],
            "answer": output[i][1],
            "solution": output[i][2],
            "dice_results": dice_results,
            "seed": seed
        }
        yacht_json.append(question_data)

    jsonl_path = json_dir / "yacht_dice.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in yacht_json:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"JSONL 파일이 생성: {jsonl_path}")

    return yacht_df, yacht_json


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Yacht Dice Puzzle Generator")
    parser.add_argument("--num", type=int, default=100, help="Number of questions to generate")
    
    args = parser.parse_args()
    
    create_dataset_files(num_questions=args.num)

    # 테스트: 샘플 문제 생성
    # yacht_df, yacht_json = create_dataset_files(num_questions=100, version="v1")

    # for i in range(3):
    #     print(f"\n========== 문제{i+1} ==========")
    #     config = YachtDiceConfig()
    #     dice = generate_random_dice(seed=1000+i)
    #     optimal_score, optimal_assignment = solve_yacht_dice(dice, config)

    #     print("- question -")
    #     print(format_user_prompt(dice))
    #     print("\n- answer -")
    #     print(optimal_score)
    #     print("\n- solution -")
    #     print(format_solution(dice, optimal_assignment, config))
    #     print("\n")

