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
        """설정에 따라 시스템 프롬프트를 생성"""

        if self.optimization_goal == "maximize":
            goal_instruction = "너의 목표는 총점의 최댓값을 구하는 것이야. 가능한 한 가장 높은 점수를 얻도록 배정해야 해."
        else:
            goal_instruction = "너의 목표는 총점의 최솟값을 구하는 것이야. 가능한 한 가장 낮은 점수를 얻도록 배정해야 해."

        prompt = f"""너는 Yacht Dice 점수 계산 전문가야. 아래 12개의 주사위 결과를 각각 하나의 항목에 할당하여 점수를 계산해야 해. 각 항목은 정확히 한 번씩만 사용할 수 있어.

{goal_instruction}

각 항목에 해당하는 점수 규칙은 아래와 같아:

[상단 항목]
1. Aces: 주사위에서 1이 나온 눈의 합
2. Twos: 주사위에서 2가 나온 눈의 합
3. Threes: 주사위에서 3이 나온 눈의 합
4. Fours: 주사위에서 4가 나온 눈의 합
5. Fives: 주사위에서 5가 나온 눈의 합
6. Sixes: 주사위에서 6이 나온 눈의 합
(이 상단 6개 항목의 합이 {self.bonus_threshold}점 이상이면 {self.bonus_points}점 보너스를 받음)

[하단 항목]
7. Three-Of-A-Kind: 동일한 눈이 3개 이상일 경우, 주사위 눈의 총합
8. Four-Of-A-Kind: 동일한 눈이 4개 이상일 경우, 주사위 눈의 총합
9. Full House: 동일한 눈이 각각 3개, 2개일 경우 (예: 3-3-3-5-5), 고정 {self.full_house_points}점
10. Small Straight: 4개의 연속된 수가 있을 경우 (예: 1-2-3-4), 고정 {self.small_straight_points}점
11. Large Straight: 5개의 연속된 수가 있을 경우 (1-2-3-4-5 또는 2-3-4-5-6), 고정 {self.large_straight_points}점
12. Yacht: 주사위 5개가 모두 같은 숫자일 경우, 고정 {self.yacht_points}점

각 주사위 결과는 하나의 항목에만 할당되어야 하고, 항목은 중복 없이 정확히 한 번씩만 사용해야 해.

Step by Step으로 생각한 후 출력 형식대로 답변 해줘.

출력 형식은 다음과 같아:
[항목명]: [d1, d2, d3, d4, d5] => 점수
총점: XXX
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
    주사위 결과를 유저 프롬프트 형식으로 포맷팅

    Args:
        dice_results: 주사위 결과 리스트

    Returns:
        str: 포맷팅된 유저 프롬프트
    """
    prompt = "다음은 12개의 주사위 결과야. 이 결과들을 위 항목 중 하나씩에 할당하여 점수를 계산해줘:\n\n"

    for i, dice in enumerate(dice_results, 1):
        prompt += f"{i}. {dice}\n"

    prompt += "\n항목마다 하나씩 할당하고, 총점을 최대화하도록 점수를 계산해줘."

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


def create_dataset_files(num_questions, version):
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

    csv_path = csv_dir / f"YACHT_DICE_{version}.csv"
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
            "question": output[i][0],
            "answer": output[i][1],
            "solution": output[i][2],
            "dice_results": dice_results,  # 추가
            "seed": seed  # 추가
        }
        yacht_json.append(question_data)

    jsonl_path = json_dir / f"YACHT_DICE_{version}.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in yacht_json:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"JSONL 파일이 생성: {jsonl_path}")

    return yacht_df, yacht_json


if __name__ == '__main__':
    # 테스트: 샘플 문제 생성
    yacht_df, yacht_json = create_dataset_files(num_questions=100, version="v1")

    for i in range(3):
        print(f"\n========== 문제{i+1} ==========")
        config = YachtDiceConfig()
        dice = generate_random_dice(seed=1000+i)
        optimal_score, optimal_assignment = solve_yacht_dice(dice, config)

        print("- question -")
        print(format_user_prompt(dice))
        print("\n- answer -")
        print(optimal_score)
        print("\n- solution -")
        print(format_solution(dice, optimal_assignment, config))
        print("\n")
