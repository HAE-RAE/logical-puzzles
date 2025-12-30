"""
Array Formula Puzzle Generator
Excel array formula-based logical puzzle generator

Problem Types:
1. lookup_query: INDEX-MATCH, VLOOKUP style data lookup
2. conditional_aggregation: SUMIF, COUNTIF style conditional aggregation
3. array_computation: SUMPRODUCT style array computation
4. multi_condition: SUMIFS, MAXIFS style multi-condition problems

Difficulty Levels:
- easy: Single condition, small table (5 rows)
- medium: 2 conditions, medium table (8 rows)
- hard: 3+ conditions, large table (12 rows), multi-table reference
"""

import json
import random
import hashlib
import csv
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
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
    """Problem generation configuration"""
    difficulty: str = "medium"
    problem_type: Optional[str] = None
    seed: Optional[int] = None

    min_rows: int = 5
    max_rows: int = 12

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
# Data Generation Utilities
# ============================================================

PRODUCT_NAMES = [
    "Apple", "Pear", "Grape", "Strawberry", "Banana", "Orange", "Watermelon", "Melon", "Peach", "Plum",
    "Milk", "Cheese", "Yogurt", "Butter", "IceCream", "Tofu", "Egg", "Ham", "Sausage", "Bacon",
    "Bread", "Rice", "Ramen", "Pasta", "Cereal", "Cookie", "Chocolate", "Candy", "Jelly", "Gum",
    "Cola", "Sprite", "Juice", "Coffee", "GreenTea", "Water", "Beer", "Soju", "Wine", "Makgeolli"
]

CATEGORIES = ["Fruit", "Dairy", "Meat", "Grain", "Beverage", "Vegetable", "Seafood", "Processed"]
REGIONS = ["Seoul", "Busan", "Daegu", "Incheon", "Gwangju", "Daejeon", "Ulsan", "Sejong"]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
DEPARTMENTS = ["Sales", "Marketing", "Development", "HR", "Finance", "Planning"]


def generate_product_table(
    num_rows: int,
    num_categories: int,
    seed: int
) -> List[Dict[str, Any]]:
    """Generate product table"""
    rng = random.Random(seed)

    categories = rng.sample(CATEGORIES, min(num_categories, len(CATEGORIES)))
    products = rng.sample(PRODUCT_NAMES, num_rows)

    table = []
    for i, product in enumerate(products):
        row = {
            "id": i + 1,
            "product": product,
            "category": rng.choice(categories),
            "price": rng.randint(5, 50) * 100,
            "stock": rng.randint(10, 200),
            "discount": rng.choice([0, 5, 10, 15, 20]),
        }
        table.append(row)

    return table


def generate_sales_table(
    product_table: List[Dict],
    num_orders: int,
    num_regions: int,
    seed: int
) -> List[Dict[str, Any]]:
    """Generate sales table"""
    rng = random.Random(seed + 1000)

    regions = rng.sample(REGIONS, min(num_regions, len(REGIONS)))
    products = [p["product"] for p in product_table]

    table = []
    for i in range(num_orders):
        row = {
            "order_id": f"ORD-{i+1:03d}",
            "product": rng.choice(products),
            "region": rng.choice(regions),
            "quantity": rng.randint(1, 50),
            "quarter": rng.choice(QUARTERS),
        }
        table.append(row)

    return table


def generate_employee_table(
    num_rows: int,
    seed: int
) -> List[Dict[str, Any]]:
    """Generate employee table for salary/bonus calculations"""
    rng = random.Random(seed + 2000)

    last_names = ["Kim", "Lee", "Park", "Choi", "Jung", "Kang", "Cho", "Yoon", "Jang", "Lim"]
    first_names = ["Minsu", "Younghee", "Cheolsu", "Jiyoung", "Donghyun", "Sujin", "Hyunwoo", "Mina", "Junho", "Seoyeon"]

    departments = rng.sample(DEPARTMENTS, min(4, len(DEPARTMENTS)))

    table = []
    for i in range(num_rows):
        base_salary = rng.randint(30, 80) * 100
        performance = rng.randint(60, 100)

        row = {
            "emp_id": f"EMP-{i+1:03d}",
            "name": rng.choice(last_names) + rng.choice(first_names),
            "department": rng.choice(departments),
            "base_salary": base_salary,
            "performance": performance,
            "years": rng.randint(1, 15),
        }
        table.append(row)

    return table


# ============================================================
# Problem Type Generators
# ============================================================

def generate_lookup_problem(
    config: ArrayFormulaConfig,
    rng: random.Random
) -> Dict[str, Any]:
    """
    Generate LOOKUP problem
    - VLOOKUP/INDEX-MATCH style data lookup
    """
    num_rows = rng.randint(config.min_rows, config.max_rows)
    product_table = generate_product_table(num_rows, config.num_categories, rng.randint(1, 10000))

    target = rng.choice(product_table)

    if config.difficulty == "easy":
        question_templates = [
            (f"What is the price of '{target['product']}'?", target["price"]),
            (f"What is the stock quantity of '{target['product']}'?", target["stock"]),
            (f"What is the discount rate of '{target['product']}' in %?", target["discount"]),
        ]
    elif config.difficulty == "medium":
        discounted_price = int(target["price"] * (100 - target["discount"]) / 100)
        inventory_value = target["price"] * target["stock"]
        question_templates = [
            (f"What is the discounted price of '{target['product']}'?", discounted_price),
            (f"What is the inventory value (price x stock) of '{target['product']}'?", inventory_value),
        ]
    else:
        category = target["category"]
        same_category = [p for p in product_table if p["category"] == category]
        max_price_product = max(same_category, key=lambda x: x["price"])
        min_stock_product = min(same_category, key=lambda x: x["stock"])

        question_templates = [
            (f"What is the name of the most expensive product in '{category}' category?", max_price_product["product"]),
            (f"What is the price of the product with least stock in '{category}' category?", min_stock_product["price"]),
        ]

    question, answer = rng.choice(question_templates)

    if config.difficulty == "easy":
        formula_hint = "Use VLOOKUP or INDEX-MATCH function."
    elif config.difficulty == "medium":
        formula_hint = "Combine VLOOKUP with arithmetic operations."
    else:
        formula_hint = "Combine INDEX-MATCH with MAX/MIN functions."

    return {
        "type": ProblemType.LOOKUP_QUERY.value,
        "difficulty": config.difficulty,
        "tables": {
            "Products": {
                "columns": ["id", "product", "category", "price", "stock", "discount"],
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
    Generate conditional aggregation problem
    - SUMIF, COUNTIF, AVERAGEIF style
    """
    num_rows = rng.randint(config.min_rows, config.max_rows)
    product_table = generate_product_table(num_rows, config.num_categories, rng.randint(1, 10000))

    categories = list(set(p["category"] for p in product_table))
    target_category = rng.choice(categories)

    category_products = [p for p in product_table if p["category"] == target_category]

    count_result = len(category_products)
    sum_stock = sum(p["stock"] for p in category_products)
    sum_value = sum(p["price"] * p["stock"] for p in category_products)
    avg_price = sum(p["price"] for p in category_products) / len(category_products) if category_products else 0

    if config.difficulty == "easy":
        question_templates = [
            (f"How many products are in the '{target_category}' category?", count_result),
            (f"What is the total stock quantity in the '{target_category}' category?", sum_stock),
        ]
        formula_hint = "Use COUNTIF or SUMIF function."
    elif config.difficulty == "medium":
        question_templates = [
            (f"What is the average price in the '{target_category}' category? (truncate decimals)", int(avg_price)),
            (f"What is the total inventory value (sum of price x stock) in the '{target_category}' category?", sum_value),
        ]
        formula_hint = "Use AVERAGEIF or combine SUMPRODUCT with conditions."
    else:
        threshold_price = rng.choice([1000, 1500, 2000, 2500])
        expensive_products = [p for p in category_products if p["price"] >= threshold_price]
        count_expensive = len(expensive_products)
        sum_expensive_stock = sum(p["stock"] for p in expensive_products)

        question_templates = [
            (f"How many products in '{target_category}' category have price >= {threshold_price}?", count_expensive),
            (f"What is the total stock of products in '{target_category}' category with price >= {threshold_price}?", sum_expensive_stock),
        ]
        formula_hint = "Use COUNTIFS or SUMIFS function."

    question, answer = rng.choice(question_templates)

    return {
        "type": ProblemType.CONDITIONAL_AGGREGATION.value,
        "difficulty": config.difficulty,
        "tables": {
            "Products": {
                "columns": ["id", "product", "category", "price", "stock", "discount"],
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
    Generate array computation problem
    - SUMPRODUCT style complex calculations
    """
    num_rows = rng.randint(config.min_rows, config.max_rows)
    seed = rng.randint(1, 10000)
    product_table = generate_product_table(num_rows, config.num_categories, seed)

    if config.difficulty == "easy":
        total_value = sum(p["price"] * p["stock"] for p in product_table)
        question = "What is the total inventory value (sum of price x stock) for all products?"
        answer = total_value
        formula_hint = "Use SUMPRODUCT(price_range, stock_range)."

    elif config.difficulty == "medium":
        discounted_value = sum(
            p["price"] * (100 - p["discount"]) / 100 * p["stock"]
            for p in product_table
        )
        question = "What is the total discounted inventory value? (truncate decimals)"
        answer = int(discounted_value)
        formula_hint = "Combine SUMPRODUCT with discount calculation."

    else:
        num_orders = rng.randint(8, 15)
        sales_table = generate_sales_table(product_table, num_orders, config.num_regions, seed)

        product_prices = {p["product"]: p["price"] for p in product_table}
        total_sales = sum(
            product_prices.get(s["product"], 0) * s["quantity"]
            for s in sales_table
        )

        question = "What is the total sales amount? (lookup price from Products table)"
        answer = total_sales
        formula_hint = "Combine SUMPRODUCT with INDEX-MATCH or XLOOKUP."

        return {
            "type": ProblemType.ARRAY_COMPUTATION.value,
            "difficulty": config.difficulty,
            "tables": {
                "Products": {
                    "columns": ["id", "product", "category", "price", "stock", "discount"],
                    "data": product_table
                },
                "Orders": {
                    "columns": ["order_id", "product", "region", "quantity", "quarter"],
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
            "Products": {
                "columns": ["id", "product", "category", "price", "stock", "discount"],
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
    Generate multi-condition problem
    - SUMIFS, COUNTIFS, MAXIFS, MINIFS style
    """
    num_rows = rng.randint(config.min_rows, config.max_rows)
    seed = rng.randint(1, 10000)
    product_table = generate_product_table(num_rows, config.num_categories, seed)

    categories = list(set(p["category"] for p in product_table))
    target_category = rng.choice(categories)

    if config.difficulty == "easy":
        threshold = rng.choice([50, 80, 100])
        filtered = [p for p in product_table
                   if p["category"] == target_category and p["stock"] >= threshold]

        count_result = len(filtered)
        sum_result = sum(p["price"] for p in filtered)

        question_templates = [
            (f"How many products in '{target_category}' category have stock >= {threshold}?", count_result),
            (f"What is the sum of prices for products in '{target_category}' category with stock >= {threshold}?", sum_result),
        ]
        formula_hint = "Use COUNTIFS or SUMIFS function."

    elif config.difficulty == "medium":
        threshold = rng.choice([50, 80, 100])
        filtered = [p for p in product_table
                   if p["category"] == target_category and p["stock"] >= threshold]

        if filtered:
            max_price = max(p["price"] for p in filtered)
            min_price = min(p["price"] for p in filtered)
        else:
            max_price = 0
            min_price = 0

        question_templates = [
            (f"What is the highest price among products in '{target_category}' category with stock >= {threshold}?", max_price),
            (f"What is the lowest price among products in '{target_category}' category with stock >= {threshold}?", min_price),
        ]
        formula_hint = "Use MAXIFS or MINIFS function."

    else:
        num_orders = rng.randint(10, 18)
        sales_table = generate_sales_table(product_table, num_orders, config.num_regions, seed)

        regions = list(set(s["region"] for s in sales_table))
        target_region = rng.choice(regions)
        target_quarter = rng.choice(QUARTERS)

        product_categories = {p["product"]: p["category"] for p in product_table}

        filtered_orders = [
            s for s in sales_table
            if s["region"] == target_region
            and s["quarter"] == target_quarter
            and product_categories.get(s["product"]) == target_category
        ]

        total_quantity = sum(s["quantity"] for s in filtered_orders)
        order_count = len(filtered_orders)

        question_templates = [
            (f"What is the total order quantity for '{target_category}' products in '{target_region}' during {target_quarter}?", total_quantity),
            (f"How many orders for '{target_category}' products in '{target_region}' during {target_quarter}?", order_count),
        ]
        formula_hint = "Combine SUMIFS/COUNTIFS with multi-table lookup."

        question, answer = rng.choice(question_templates)

        return {
            "type": ProblemType.MULTI_CONDITION.value,
            "difficulty": config.difficulty,
            "tables": {
                "Products": {
                    "columns": ["id", "product", "category", "price", "stock", "discount"],
                    "data": product_table
                },
                "Orders": {
                    "columns": ["order_id", "product", "region", "quantity", "quarter"],
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
            "Products": {
                "columns": ["id", "product", "category", "price", "stock", "discount"],
                "data": product_table
            }
        },
        "question": question,
        "formula_hint": formula_hint,
        "answer": answer,
        "answer_type": "number"
    }


# ============================================================
# Main Generation Functions
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
    Generate a single puzzle

    Args:
        difficulty: Difficulty level ("easy", "medium", "hard")
        problem_type: Problem type (None for random)
        seed: Random seed

    Returns:
        Puzzle dictionary
    """
    if seed is None:
        seed = random.randint(1, 1000000)

    rng = random.Random(seed)
    config = ArrayFormulaConfig(difficulty=difficulty, seed=seed)

    if problem_type is None:
        problem_type = rng.choice(list(PROBLEM_GENERATORS.keys()))

    generator = PROBLEM_GENERATORS[problem_type]
    puzzle = generator(config, rng)

    # Generate ID
    puzzle_hash = hashlib.md5(json.dumps(puzzle, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]
    puzzle["id"] = f"af_{difficulty}_{problem_type}_{puzzle_hash}"
    puzzle["seed"] = seed

    return puzzle


def generate_dataset(
    num_puzzles_per_config: int = 10,
    seed: int = 2025
) -> List[Dict[str, Any]]:
    """
    Generate dataset by difficulty x problem_type

    Args:
        num_puzzles_per_config: Number of puzzles per configuration
        seed: Base seed

    Returns:
        List of puzzles
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
    """Format table as prompt string"""
    columns = table_data["columns"]
    data = table_data["data"]

    lines = [f"[{table_name} Table]"]
    header = " | ".join(str(col) for col in columns)
    lines.append(header)
    lines.append("-" * len(header))

    for row in data:
        row_str = " | ".join(str(row.get(col, "")) for col in columns)
        lines.append(row_str)

    return "\n".join(lines)


def puzzle_to_prompt(puzzle: Dict[str, Any], include_hint: bool = True) -> str:
    """Convert puzzle to LLM prompt"""
    prompt_parts = []

    prompt_parts.append("The following is spreadsheet data.\n")

    for table_name, table_data in puzzle["tables"].items():
        prompt_parts.append(format_table_for_prompt(table_name, table_data))
        prompt_parts.append("")

    prompt_parts.append(f"Question: {puzzle['question']}")

    if include_hint and "formula_hint" in puzzle:
        prompt_parts.append(f"Hint: {puzzle['formula_hint']}")

    if puzzle.get("answer_type") == "number":
        prompt_parts.append("\nAnswer with only a number. (no units)")
    else:
        prompt_parts.append("\nAnswer with the exact value.")

    return "\n".join(prompt_parts)


def save_dataset(
    puzzles: List[Dict],
    base_dir: str = "../data"
):
    """
    Save dataset as CSV and JSONL

    Output paths:
    - data/csv/array_formula.csv
    - data/json/array_formula.jsonl
    """
    base_path = Path(base_dir)
    csv_dir = base_path / "csv"
    json_dir = base_path / "json"

    csv_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    csv_path = csv_dir / "array_formula.csv"
    jsonl_path = json_dir / "array_formula.jsonl"

    # Add question prompt to each puzzle
    processed_puzzles = []
    for puzzle in puzzles:
        question = puzzle_to_prompt(puzzle, include_hint=False)

        processed = {
            "id": puzzle["id"],
            "difficulty": puzzle["difficulty"],
            "type": puzzle["type"],
            "question": question,
            "answer": puzzle["answer"],
            "answer_type": puzzle.get("answer_type", "number"),
            "formula_hint": puzzle.get("formula_hint", ""),
            "tables": puzzle["tables"],
            "seed": puzzle.get("seed"),
        }
        processed_puzzles.append(processed)

    # Save JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for puzzle in processed_puzzles:
            f.write(json.dumps(puzzle, ensure_ascii=False) + "\n")

    print(f"Saved {len(processed_puzzles)} puzzles to {jsonl_path}")

    # Save CSV
    csv_columns = ["id", "difficulty", "type", "question", "answer", "answer_type", "formula_hint", "tables", "seed"]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

        for puzzle in processed_puzzles:
            row = {
                "id": puzzle["id"],
                "difficulty": puzzle["difficulty"],
                "type": puzzle["type"],
                "question": puzzle["question"],
                "answer": puzzle["answer"],
                "answer_type": puzzle["answer_type"],
                "formula_hint": puzzle["formula_hint"],
                "tables": json.dumps(puzzle["tables"], ensure_ascii=False),
                "seed": puzzle["seed"],
            }
            writer.writerow(row)

    print(f"Saved {len(processed_puzzles)} puzzles to {csv_path}")

    # Print statistics
    stats = {}
    for puzzle in processed_puzzles:
        key = f"{puzzle['difficulty']}_{puzzle['type']}"
        stats[key] = stats.get(key, 0) + 1

    print("\nDataset Statistics:")
    for key, count in sorted(stats.items()):
        print(f"  {key}: {count}")

    return csv_path, jsonl_path


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Array Formula Puzzle Generator")
    parser.add_argument("--num", type=int, default=5, help="Number of puzzles per config")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--output", type=str, default="../data", help="Output base directory")
    parser.add_argument("--demo", action="store_true", help="Print demo puzzles")

    args = parser.parse_args()

    if args.demo:
        print("=" * 60)
        print("Array Formula Puzzle Demo")
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
                print(f"\nAnswer: {puzzle['answer']}")
                print("=" * 60)
                break
    else:
        puzzles = generate_dataset(
            num_puzzles_per_config=args.num,
            seed=args.seed
        )
        save_dataset(puzzles, args.output)
