"""
Array Formula Puzzle Generator
엑셀 배열 수식 기반 논리 퍼즐 생성기

문제 유형:
1. lookup_query: INDEX-MATCH, VLOOKUP 스타일 데이터 조회
2. conditional_aggregation: SUMIF, COUNTIF 스타일 조건부 집계
3. array_computation: SUMPRODUCT 스타일 배열 연산
4. multi_condition: SUMIFS, MAXIFS 스타일 복합 조건

난이도:
- easy: 단일 조건, 작은 테이블 (5행)
- medium: 2개 조건, 중간 테이블 (8행)
- hard: 3개+ 조건, 큰 테이블 (12행), 다중 테이블 참조
"""

import json
import random
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from pathlib import Path


class ProblemType(Enum):
    LOOKUP_QUERY = "lookup_query"
    CONDITIONAL_AGGREGATION = "conditional_aggregation"
    ARRAY_COMPUTATION = "array_computation"
    MULTI_CONDITION = "multi_condition"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class ArrayFormulaConfig:
    """문제 생성 설정"""
    difficulty: str = "medium"
    problem_type: Optional[str] = None  # None이면 랜덤 선택
    seed: Optional[int] = None
    
    # 테이블 크기 설정 (난이도별 자동 조정)
    min_rows: int = 5
    max_rows: int = 12
    
    # 카테고리 및 데이터 다양성
    num_categories: int = 4
    num_regions: int = 3
    
    def __post_init__(self):
        if self.difficulty == "easy":
            self.min_rows, self.max_rows = 5, 6
            self.num_categories = 3
        elif self.difficulty == "medium":
            self.min_rows, self.max_rows = 7, 9
            self.num_categories = 4
        elif self.difficulty == "hard":
            self.min_rows, self.max_rows = 10, 14
            self.num_categories = 5


# ============================================================
# 데이터 생성 유틸리티
# ============================================================

PRODUCT_NAMES = [
    "사과", "배", "포도", "딸기", "바나나", "오렌지", "수박", "참외", "복숭아", "자두",
    "우유", "치즈", "요거트", "버터", "아이스크림", "두부", "계란", "햄", "소시지", "베이컨",
    "빵", "쌀", "라면", "파스타", "시리얼", "과자", "초콜릿", "사탕", "젤리", "껌",
    "콜라", "사이다", "주스", "커피", "녹차", "생수", "맥주", "소주", "와인", "막걸리"
]

CATEGORIES = ["과일", "유제품", "육류", "곡물", "음료", "채소", "수산물", "가공식품"]
REGIONS = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종"]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
DEPARTMENTS = ["영업팀", "마케팅팀", "개발팀", "인사팀", "재무팀", "기획팀"]


def generate_product_table(
    num_rows: int,
    num_categories: int,
    seed: int
) -> List[Dict[str, Any]]:
    """제품 테이블 생성"""
    rng = random.Random(seed)
    
    categories = rng.sample(CATEGORIES, min(num_categories, len(CATEGORIES)))
    products = rng.sample(PRODUCT_NAMES, num_rows)
    
    table = []
    for i, product in enumerate(products):
        row = {
            "id": i + 1,
            "제품명": product,
            "카테고리": rng.choice(categories),
            "단가": rng.randint(5, 50) * 100,  # 500 ~ 5000
            "재고": rng.randint(10, 200),
            "할인율": rng.choice([0, 5, 10, 15, 20]),
        }
        table.append(row)
    
    return table


def generate_sales_table(
    product_table: List[Dict],
    num_orders: int,
    num_regions: int,
    seed: int
) -> List[Dict[str, Any]]:
    """판매 테이블 생성"""
    rng = random.Random(seed + 1000)
    
    regions = rng.sample(REGIONS, min(num_regions, len(REGIONS)))
    products = [p["제품명"] for p in product_table]
    
    table = []
    for i in range(num_orders):
        row = {
            "주문번호": f"ORD-{i+1:03d}",
            "제품명": rng.choice(products),
            "지역": rng.choice(regions),
            "수량": rng.randint(1, 50),
            "분기": rng.choice(QUARTERS),
        }
        table.append(row)
    
    return table


def generate_employee_table(
    num_rows: int,
    seed: int
) -> List[Dict[str, Any]]:
    """직원 테이블 생성 (급여/보너스 계산용)"""
    rng = random.Random(seed + 2000)
    
    last_names = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임"]
    first_names = ["민수", "영희", "철수", "지영", "동현", "수진", "현우", "미나", "준호", "서연"]
    
    departments = rng.sample(DEPARTMENTS, min(4, len(DEPARTMENTS)))
    
    table = []
    for i in range(num_rows):
        base_salary = rng.randint(30, 80) * 100  # 3000 ~ 8000 (만원 단위)
        performance = rng.randint(60, 100)
        
        row = {
            "사번": f"EMP-{i+1:03d}",
            "이름": rng.choice(last_names) + rng.choice(first_names),
            "부서": rng.choice(departments),
            "기본급": base_salary,
            "실적점수": performance,
            "근속연수": rng.randint(1, 15),
        }
        table.append(row)
    
    return table


# ============================================================
# 문제 유형별 생성기
# ============================================================

def generate_lookup_problem(
    config: ArrayFormulaConfig,
    rng: random.Random
) -> Dict[str, Any]:
    """
    LOOKUP 문제 생성
    - VLOOKUP/INDEX-MATCH 스타일의 데이터 조회 문제
    """
    num_rows = rng.randint(config.min_rows, config.max_rows)
    product_table = generate_product_table(num_rows, config.num_categories, rng.randint(1, 10000))
    
    # 타겟 제품 선택
    target = rng.choice(product_table)
    
    if config.difficulty == "easy":
        # Easy: 단순 조회
        question_templates = [
            (f"'{target['제품명']}'의 단가는 얼마인가요?", target["단가"]),
            (f"'{target['제품명']}'의 재고 수량은 몇 개인가요?", target["재고"]),
            (f"'{target['제품명']}'의 할인율은 몇 %인가요?", target["할인율"]),
        ]
    elif config.difficulty == "medium":
        # Medium: 계산 포함 조회
        discounted_price = int(target["단가"] * (100 - target["할인율"]) / 100)
        inventory_value = target["단가"] * target["재고"]
        question_templates = [
            (f"'{target['제품명']}'의 할인 적용 단가는 얼마인가요?", discounted_price),
            (f"'{target['제품명']}'의 재고 가치(단가×재고)는 얼마인가요?", inventory_value),
        ]
    else:
        # Hard: 복합 조건 조회
        category = target["카테고리"]
        same_category = [p for p in product_table if p["카테고리"] == category]
        max_price_product = max(same_category, key=lambda x: x["단가"])
        min_stock_product = min(same_category, key=lambda x: x["재고"])
        
        question_templates = [
            (f"'{category}' 카테고리에서 가장 비싼 제품의 이름은 무엇인가요?", max_price_product["제품명"]),
            (f"'{category}' 카테고리에서 재고가 가장 적은 제품의 단가는 얼마인가요?", min_stock_product["단가"]),
        ]
    
    question, answer = rng.choice(question_templates)
    
    # 엑셀 수식 힌트 생성
    if config.difficulty == "easy":
        formula_hint = "VLOOKUP 또는 INDEX-MATCH 함수를 사용하세요."
    elif config.difficulty == "medium":
        formula_hint = "VLOOKUP과 산술 연산을 조합하세요."
    else:
        formula_hint = "INDEX-MATCH와 MAX/MIN 함수를 조합하세요."
    
    return {
        "type": ProblemType.LOOKUP_QUERY.value,
        "difficulty": config.difficulty,
        "tables": {
            "제품": {
                "columns": ["id", "제품명", "카테고리", "단가", "재고", "할인율"],
                "data": product_table
            }
        },
        "question": question,
        "formula_hint": formula_hint,
        "answer": answer,
        "answer_type": "number" if isinstance(answer, (int, float)) else "text"
    }


def generate_conditional_aggregation_problem(
    config: ArrayFormulaConfig,
    rng: random.Random
) -> Dict[str, Any]:
    """
    조건부 집계 문제 생성
    - SUMIF, COUNTIF, AVERAGEIF 스타일
    """
    num_rows = rng.randint(config.min_rows, config.max_rows)
    product_table = generate_product_table(num_rows, config.num_categories, rng.randint(1, 10000))
    
    # 카테고리별 집계 계산
    categories = list(set(p["카테고리"] for p in product_table))
    target_category = rng.choice(categories)
    
    category_products = [p for p in product_table if p["카테고리"] == target_category]
    
    count_result = len(category_products)
    sum_stock = sum(p["재고"] for p in category_products)
    sum_value = sum(p["단가"] * p["재고"] for p in category_products)
    avg_price = sum(p["단가"] for p in category_products) / len(category_products) if category_products else 0
    
    if config.difficulty == "easy":
        question_templates = [
            (f"'{target_category}' 카테고리의 제품 수는 몇 개인가요?", count_result),
            (f"'{target_category}' 카테고리의 총 재고 수량은 몇 개인가요?", sum_stock),
        ]
        formula_hint = "COUNTIF 또는 SUMIF 함수를 사용하세요."
    elif config.difficulty == "medium":
        question_templates = [
            (f"'{target_category}' 카테고리의 평균 단가는 얼마인가요? (소수점 버림)", int(avg_price)),
            (f"'{target_category}' 카테고리의 총 재고 가치(단가×재고의 합)는 얼마인가요?", sum_value),
        ]
        formula_hint = "AVERAGEIF 또는 SUMPRODUCT와 조건을 조합하세요."
    else:
        # 복합 조건
        threshold_price = rng.choice([1000, 1500, 2000, 2500])
        expensive_products = [p for p in category_products if p["단가"] >= threshold_price]
        count_expensive = len(expensive_products)
        sum_expensive_stock = sum(p["재고"] for p in expensive_products)
        
        question_templates = [
            (f"'{target_category}' 카테고리 중 단가가 {threshold_price}원 이상인 제품 수는?", count_expensive),
            (f"'{target_category}' 카테고리 중 단가가 {threshold_price}원 이상인 제품의 총 재고는?", sum_expensive_stock),
        ]
        formula_hint = "COUNTIFS 또는 SUMIFS 함수를 사용하세요."
    
    question, answer = rng.choice(question_templates)
    
    return {
        "type": ProblemType.CONDITIONAL_AGGREGATION.value,
        "difficulty": config.difficulty,
        "tables": {
            "제품": {
                "columns": ["id", "제품명", "카테고리", "단가", "재고", "할인율"],
                "data": product_table
            }
        },
        "question": question,
        "formula_hint": formula_hint,
        "answer": answer,
        "answer_type": "number"
    }


def generate_array_computation_problem(
    config: ArrayFormulaConfig,
    rng: random.Random
) -> Dict[str, Any]:
    """
    배열 연산 문제 생성
    - SUMPRODUCT 스타일의 복합 계산
    """
    num_rows = rng.randint(config.min_rows, config.max_rows)
    seed = rng.randint(1, 10000)
    product_table = generate_product_table(num_rows, config.num_categories, seed)
    
    if config.difficulty == "easy":
        # 전체 재고 가치
        total_value = sum(p["단가"] * p["재고"] for p in product_table)
        question = "모든 제품의 총 재고 가치(단가×재고의 합)는 얼마인가요?"
        answer = total_value
        formula_hint = "SUMPRODUCT(단가범위, 재고범위) 함수를 사용하세요."
        
    elif config.difficulty == "medium":
        # 할인 적용 총 가치
        discounted_value = sum(
            p["단가"] * (100 - p["할인율"]) / 100 * p["재고"]
            for p in product_table
        )
        question = "모든 제품의 할인 적용 총 재고 가치는 얼마인가요? (소수점 버림)"
        answer = int(discounted_value)
        formula_hint = "SUMPRODUCT와 할인율 계산을 조합하세요."
        
    else:
        # 다중 테이블 참조
        num_orders = rng.randint(8, 15)
        sales_table = generate_sales_table(product_table, num_orders, config.num_regions, seed)
        
        # 총 매출액 계산 (제품 테이블에서 단가 조회)
        product_prices = {p["제품명"]: p["단가"] for p in product_table}
        total_sales = sum(
            product_prices.get(s["제품명"], 0) * s["수량"]
            for s in sales_table
        )
        
        question = "주문 테이블의 총 매출액은 얼마인가요? (제품 테이블에서 단가를 조회하여 계산)"
        answer = total_sales
        formula_hint = "SUMPRODUCT와 INDEX-MATCH 또는 XLOOKUP을 조합하세요."
        
        return {
            "type": ProblemType.ARRAY_COMPUTATION.value,
            "difficulty": config.difficulty,
            "tables": {
                "제품": {
                    "columns": ["id", "제품명", "카테고리", "단가", "재고", "할인율"],
                    "data": product_table
                },
                "주문": {
                    "columns": ["주문번호", "제품명", "지역", "수량", "분기"],
                    "data": sales_table
                }
            },
            "question": question,
            "formula_hint": formula_hint,
            "answer": answer,
            "answer_type": "number"
        }
    
    return {
        "type": ProblemType.ARRAY_COMPUTATION.value,
        "difficulty": config.difficulty,
        "tables": {
            "제품": {
                "columns": ["id", "제품명", "카테고리", "단가", "재고", "할인율"],
                "data": product_table
            }
        },
        "question": question,
        "formula_hint": formula_hint,
        "answer": answer,
        "answer_type": "number"
    }


def generate_multi_condition_problem(
    config: ArrayFormulaConfig,
    rng: random.Random
) -> Dict[str, Any]:
    """
    복합 조건 문제 생성
    - SUMIFS, COUNTIFS, MAXIFS, MINIFS 스타일
    """
    num_rows = rng.randint(config.min_rows, config.max_rows)
    seed = rng.randint(1, 10000)
    product_table = generate_product_table(num_rows, config.num_categories, seed)
    
    categories = list(set(p["카테고리"] for p in product_table))
    target_category = rng.choice(categories)
    
    if config.difficulty == "easy":
        # 2개 조건
        threshold = rng.choice([50, 80, 100])
        filtered = [p for p in product_table 
                   if p["카테고리"] == target_category and p["재고"] >= threshold]
        
        count_result = len(filtered)
        sum_result = sum(p["단가"] for p in filtered)
        
        question_templates = [
            (f"'{target_category}' 카테고리 중 재고가 {threshold}개 이상인 제품 수는?", count_result),
            (f"'{target_category}' 카테고리 중 재고가 {threshold}개 이상인 제품의 단가 합계는?", sum_result),
        ]
        formula_hint = "COUNTIFS 또는 SUMIFS 함수를 사용하세요."
        
    elif config.difficulty == "medium":
        # 2개 조건 + MAX/MIN
        threshold = rng.choice([50, 80, 100])
        filtered = [p for p in product_table 
                   if p["카테고리"] == target_category and p["재고"] >= threshold]
        
        if filtered:
            max_price = max(p["단가"] for p in filtered)
            min_price = min(p["단가"] for p in filtered)
        else:
            max_price = 0
            min_price = 0
        
        question_templates = [
            (f"'{target_category}' 카테고리 중 재고가 {threshold}개 이상인 제품의 최고 단가는?", max_price),
            (f"'{target_category}' 카테고리 중 재고가 {threshold}개 이상인 제품의 최저 단가는?", min_price),
        ]
        formula_hint = "MAXIFS 또는 MINIFS 함수를 사용하세요."
        
    else:
        # 3개 조건 + 다중 테이블
        num_orders = rng.randint(10, 18)
        sales_table = generate_sales_table(product_table, num_orders, config.num_regions, seed)
        
        regions = list(set(s["지역"] for s in sales_table))
        target_region = rng.choice(regions)
        target_quarter = rng.choice(QUARTERS)
        
        # 특정 지역, 특정 분기, 특정 카테고리 제품의 총 주문 수량
        product_categories = {p["제품명"]: p["카테고리"] for p in product_table}
        
        filtered_orders = [
            s for s in sales_table
            if s["지역"] == target_region
            and s["분기"] == target_quarter
            and product_categories.get(s["제품명"]) == target_category
        ]
        
        total_quantity = sum(s["수량"] for s in filtered_orders)
        order_count = len(filtered_orders)
        
        question_templates = [
            (f"'{target_region}' 지역, '{target_quarter}', '{target_category}' 카테고리 제품의 총 주문 수량은?", total_quantity),
            (f"'{target_region}' 지역, '{target_quarter}', '{target_category}' 카테고리 제품의 주문 건수는?", order_count),
        ]
        formula_hint = "SUMIFS/COUNTIFS와 다중 테이블 조회를 조합하세요."
        
        question, answer = rng.choice(question_templates)
        
        return {
            "type": ProblemType.MULTI_CONDITION.value,
            "difficulty": config.difficulty,
            "tables": {
                "제품": {
                    "columns": ["id", "제품명", "카테고리", "단가", "재고", "할인율"],
                    "data": product_table
                },
                "주문": {
                    "columns": ["주문번호", "제품명", "지역", "수량", "분기"],
                    "data": sales_table
                }
            },
            "question": question,
            "formula_hint": formula_hint,
            "answer": answer,
            "answer_type": "number"
        }
    
    question, answer = rng.choice(question_templates)
    
    return {
        "type": ProblemType.MULTI_CONDITION.value,
        "difficulty": config.difficulty,
        "tables": {
            "제품": {
                "columns": ["id", "제품명", "카테고리", "단가", "재고", "할인율"],
                "data": product_table
            }
        },
        "question": question,
        "formula_hint": formula_hint,
        "answer": answer,
        "answer_type": "number"
    }


# ============================================================
# 메인 생성 함수
# ============================================================

PROBLEM_GENERATORS = {
    ProblemType.LOOKUP_QUERY.value: generate_lookup_problem,
    ProblemType.CONDITIONAL_AGGREGATION.value: generate_conditional_aggregation_problem,
    ProblemType.ARRAY_COMPUTATION.value: generate_array_computation_problem,
    ProblemType.MULTI_CONDITION.value: generate_multi_condition_problem,
}


def generate_puzzle(
    difficulty: str = "medium",
    problem_type: Optional[str] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    단일 퍼즐 생성
    
    Args:
        difficulty: 난이도 ("easy", "medium", "hard")
        problem_type: 문제 유형 (None이면 랜덤)
        seed: 랜덤 시드
    
    Returns:
        퍼즐 딕셔너리
    """
    if seed is None:
        seed = random.randint(1, 1000000)
    
    rng = random.Random(seed)
    config = ArrayFormulaConfig(difficulty=difficulty, seed=seed)
    
    if problem_type is None:
        problem_type = rng.choice(list(PROBLEM_GENERATORS.keys()))
    
    generator = PROBLEM_GENERATORS[problem_type]
    puzzle = generator(config, rng)
    
    # ID 생성
    puzzle_hash = hashlib.md5(json.dumps(puzzle, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]
    puzzle["id"] = f"af_{difficulty}_{problem_type}_{puzzle_hash}"
    puzzle["seed"] = seed
    
    return puzzle


def generate_dataset(
    num_puzzles_per_config: int = 10,
    seed: int = 2025
) -> List[Dict[str, Any]]:
    """
    난이도 × 문제유형별 데이터셋 생성
    
    Args:
        num_puzzles_per_config: 설정당 문제 수
        seed: 기본 시드
    
    Returns:
        퍼즐 리스트
    """
    puzzles = []
    difficulties = ["easy", "medium", "hard"]
    problem_types = list(PROBLEM_GENERATORS.keys())
    
    puzzle_seed = seed
    for difficulty in difficulties:
        for ptype in problem_types:
            for i in range(num_puzzles_per_config):
                puzzle = generate_puzzle(
                    difficulty=difficulty,
                    problem_type=ptype,
                    seed=puzzle_seed
                )
                puzzles.append(puzzle)
                puzzle_seed += 1
    
    return puzzles


def format_table_for_prompt(table_name: str, table_data: Dict) -> str:
    """테이블을 프롬프트용 문자열로 포맷팅"""
    columns = table_data["columns"]
    data = table_data["data"]
    
    # 헤더
    lines = [f"[{table_name} 테이블]"]
    header = " | ".join(str(col) for col in columns)
    lines.append(header)
    lines.append("-" * len(header))
    
    # 데이터 행
    for row in data:
        row_str = " | ".join(str(row.get(col, "")) for col in columns)
        lines.append(row_str)
    
    return "\n".join(lines)


def puzzle_to_prompt(puzzle: Dict[str, Any], include_hint: bool = True) -> str:
    """퍼즐을 LLM 프롬프트로 변환"""
    prompt_parts = []
    
    prompt_parts.append("다음은 엑셀/스프레드시트 데이터입니다.\n")
    
    # 테이블 출력
    for table_name, table_data in puzzle["tables"].items():
        prompt_parts.append(format_table_for_prompt(table_name, table_data))
        prompt_parts.append("")
    
    # 질문
    prompt_parts.append(f"질문: {puzzle['question']}")
    
    # 힌트 (옵션)
    if include_hint and "formula_hint" in puzzle:
        prompt_parts.append(f"힌트: {puzzle['formula_hint']}")
    
    # 응답 형식
    if puzzle.get("answer_type") == "number":
        prompt_parts.append("\n숫자로만 답하세요. (단위 없이)")
    else:
        prompt_parts.append("\n정확한 값으로 답하세요.")
    
    return "\n".join(prompt_parts)


def save_dataset(
    puzzles: List[Dict],
    output_dir: str = "../evaluation_data/array_formula",
    filename: str = "array_formula_dataset.json"
):
    """데이터셋 저장"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(puzzles, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(puzzles)} puzzles to {filepath}")
    
    # 난이도별 통계
    stats = {}
    for puzzle in puzzles:
        key = f"{puzzle['difficulty']}_{puzzle['type']}"
        stats[key] = stats.get(key, 0) + 1
    
    print("\n데이터셋 통계:")
    for key, count in sorted(stats.items()):
        print(f"  {key}: {count}개")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Array Formula 퍼즐 생성기")
    parser.add_argument("--num", type=int, default=5, help="설정당 문제 수")
    parser.add_argument("--seed", type=int, default=2025, help="랜덤 시드")
    parser.add_argument("--output", type=str, default="../evaluation_data/array_formula", help="출력 디렉토리")
    parser.add_argument("--demo", action="store_true", help="데모 문제 출력")
    
    args = parser.parse_args()
    
    if args.demo:
        # 데모: 각 유형별 샘플 출력
        print("=" * 60)
        print("Array Formula Puzzle 데모")
        print("=" * 60)
        
        for ptype in PROBLEM_GENERATORS.keys():
            for difficulty in ["easy", "medium", "hard"]:
                puzzle = generate_puzzle(
                    difficulty=difficulty,
                    problem_type=ptype,
                    seed=42
                )
                print(f"\n[{ptype} - {difficulty}]")
                print("-" * 40)
                print(puzzle_to_prompt(puzzle))
                print(f"\n정답: {puzzle['answer']}")
                print("=" * 60)
                break  # 각 유형에서 하나만
    else:
        # 데이터셋 생성
        puzzles = generate_dataset(
            num_puzzles_per_config=args.num,
            seed=args.seed
        )
        save_dataset(puzzles, args.output)
