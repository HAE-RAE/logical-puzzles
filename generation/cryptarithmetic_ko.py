"""복면산(Cryptarithmetic) 퍼즐 생성기 - 한국어 버전

구성적 생성: 역산술을 사용하여 유효한 퍼즐을 보장하며,
다양한 매핑 전략으로 해의 개수를 제어합니다.

logical-puzzles-me/cryptarithmetic/generator.py 기반 이식:
- find_solutions 내부 _stats 를 통한 solver_steps 계측
- min_solver_steps(역추적 노드 수) 기반 난이도 게이팅
- 퍼즐 JSONL 에 step_metrics 필드 포함
"""

import random
import string
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass


MAX_SOLUTIONS = 1


@dataclass
class PuzzleCandidate:
    word1: str
    word2: str
    result: str
    answer: str
    unique_letters: int
    strategy: str
    word3: str = None
    valid_answers: List[str] = None
    mapping: Dict[str, int] = None
    solver_steps: int = 0

    @property
    def puzzle_str(self) -> str:
        if self.word3:
            return f"{self.word1} + {self.word2} + {self.word3} = {self.result}"
        return f"{self.word1} + {self.word2} = {self.result}"

    @property
    def operands(self):
        ops = [self.word1, self.word2]
        if self.word3:
            ops.append(self.word3)
        return ops


def find_solutions(
    puzzle: tuple,
    max_count: int = 4,
    _stats: Optional[Dict] = None,
) -> List[Tuple[str, Dict[str, int]]]:
    *operand_words, result_word = puzzle
    all_letters = sorted(set(''.join(operand_words) + result_word))

    if len(all_letters) > 10:
        return []

    first_letters = set()
    for w in list(operand_words) + [result_word]:
        first_letters.add(w[0])

    all_words = list(operand_words) + [result_word]
    max_len = max(len(w) for w in all_words)
    reversed_ops = [w[::-1] for w in operand_words]
    wr = result_word[::-1]

    ordered_letters = []
    seen = set()
    for col in range(max_len):
        for w in reversed_ops + [wr]:
            if col < len(w) and w[col] not in seen:
                ordered_letters.append(w[col])
                seen.add(w[col])
    for letter in all_letters:
        if letter not in seen:
            ordered_letters.append(letter)
            seen.add(letter)

    solutions = []
    mapping = {}
    used_digits = set()

    def _check_columns(mapping):
        carry = 0
        for col in range(max_len):
            op_letters = [rw[col] if col < len(rw) else None for rw in reversed_ops]
            cr = wr[col] if col < len(wr) else None

            for ol in op_letters:
                if ol is not None and ol not in mapping:
                    return True
            if cr is not None and cr not in mapping:
                return True

            total = carry
            for ol in op_letters:
                total += mapping.get(ol, 0) if ol else 0
            dr = mapping.get(cr, 0) if cr else 0

            if total % 10 != dr:
                return False
            carry = total // 10
        return carry == 0

    def backtrack(idx):
        if _stats is not None:
            _stats['nodes'] = _stats.get('nodes', 0) + 1
        if len(solutions) >= max_count:
            return

        if idx == len(ordered_letters):
            if _check_columns(mapping):
                num_result = int("".join(str(mapping[c]) for c in result_word))
                solutions.append((str(num_result), dict(mapping)))
            return

        letter = ordered_letters[idx]

        for digit in range(10):
            if digit in used_digits:
                continue
            if digit == 0 and letter in first_letters:
                continue

            mapping[letter] = digit
            used_digits.add(digit)

            valid = True
            carry = 0
            for col in range(max_len):
                op_letters = [rw[col] if col < len(rw) else None for rw in reversed_ops]
                cr = wr[col] if col < len(wr) else None

                all_assigned = all(ol is None or ol in mapping for ol in op_letters)
                has_cr = cr is None or cr in mapping

                if all_assigned and has_cr:
                    total = carry
                    for ol in op_letters:
                        total += mapping.get(ol, 0) if ol else 0
                    dr = mapping.get(cr, 0) if cr else 0
                    if total % 10 != dr:
                        valid = False
                        break
                    carry = total // 10
                else:
                    break

            if valid:
                backtrack(idx + 1)

            del mapping[letter]
            used_digits.discard(digit)

    backtrack(0)
    return solutions


def count_solutions_fast(puzzle: tuple) -> int:
    return len(find_solutions(puzzle, max_count=MAX_SOLUTIONS + 1))


def has_valid_solutions(puzzle: tuple) -> bool:
    return count_solutions_fast(puzzle) == 1


def _create_letter_mapping(unique_digits: List[str], strategy: str = 'random') -> Dict[str, str]:
    available_letters = list(string.ascii_uppercase)

    if strategy == 'random':
        random.shuffle(available_letters)
    elif strategy == 'reverse':
        available_letters = available_letters[::-1]
    elif strategy == 'vowel_first':
        vowels = list('AEIOU')
        consonants = [c for c in available_letters if c not in vowels]
        random.shuffle(vowels)
        random.shuffle(consonants)
        available_letters = vowels + consonants

    return {digit: available_letters[i] for i, digit in enumerate(unique_digits)}


strategy_stats = {
    'random': {'tried': 0, 'success': 0},
    'sequential': {'tried': 0, 'success': 0},
    'reverse': {'tried': 0, 'success': 0},
    'vowel_first': {'tried': 0, 'success': 0},
    'random_fallback': {'tried': 0, 'success': 0}
}


def print_strategy_stats():
    print("\n매핑 전략 통계:")
    for name, stats in strategy_stats.items():
        rate = stats['success'] / stats['tried'] * 100 if stats['tried'] > 0 else 0
        print(f"  {name}: {stats['success']}/{stats['tried']} ({rate:.1f}%)")


def count_carries(*nums: int) -> int:
    carries = 0
    carry = 0
    str_nums = [str(n)[::-1] for n in nums]
    max_len = max(len(s) for s in str_nums)

    for i in range(max_len):
        total = carry
        for s in str_nums:
            total += int(s[i]) if i < len(s) else 0
        carry = total // 10
        if carry > 0:
            carries += 1

    return carries


def has_overflow(*nums: int) -> bool:
    result = sum(nums)
    max_operand_digits = max(len(str(n)) for n in nums)
    result_digits = len(str(result))
    return result_digits > max_operand_digits


def generate_from_arithmetic(
    num1: int,
    num2: int,
    used_patterns: Set[str] = None,
    num3: int = None,
) -> Optional[PuzzleCandidate]:
    if used_patterns is None:
        used_patterns = set()

    is_3op = num3 is not None
    if is_3op:
        result = num1 + num2 + num3
        str_nums = [str(num1), str(num2), str(num3)]
    else:
        result = num1 + num2
        str_nums = [str(num1), str(num2)]

    str_result = str(result)
    combined = ''.join(str_nums) + str_result

    unique_digits = sorted(set(combined))

    if len(unique_digits) > 10:
        return None

    max_allowed = MAX_SOLUTIONS

    strategies = ['random', 'sequential', 'reverse', 'vowel_first']
    random.shuffle(strategies)

    for strategy in strategies:
        strategy_stats[strategy]['tried'] += 1
        digit_to_letter = _create_letter_mapping(unique_digits, strategy)

        words = ["".join(digit_to_letter[d] for d in s) for s in str_nums]
        result_word = "".join(digit_to_letter[d] for d in str_result)

        pattern = "+".join(words) + "=" + result_word
        if pattern in used_patterns:
            continue

        puzzle = tuple(words + [result_word])

        stats = {'nodes': 0}
        solutions = find_solutions(puzzle, max_count=max_allowed + 1, _stats=stats)
        solution_count = len(solutions)

        if solution_count == 1:
            used_patterns.add(pattern)
            answer = str(result)
            strategy_stats[strategy]['success'] += 1
            return PuzzleCandidate(
                word1=words[0],
                word2=words[1],
                result=result_word,
                answer=answer,
                unique_letters=len(set(''.join(words) + result_word)),
                strategy=strategy,
                word3=words[2] if len(words) > 2 else None,
                valid_answers=[answer],
                mapping=solutions[0][1],
                solver_steps=stats['nodes'],
            )

    if not is_3op:
        for _ in range(5):
            strategy_stats['random_fallback']['tried'] += 1
            digit_to_letter = _create_letter_mapping(unique_digits, 'random')

            words = ["".join(digit_to_letter[d] for d in s) for s in str_nums]
            result_word = "".join(digit_to_letter[d] for d in str_result)

            pattern = "+".join(words) + "=" + result_word
            if pattern in used_patterns:
                continue

            puzzle = tuple(words + [result_word])

            stats = {'nodes': 0}
            solutions = find_solutions(puzzle, max_count=max_allowed + 1, _stats=stats)
            solution_count = len(solutions)

            if solution_count == 1:
                used_patterns.add(pattern)
                answer = str(result)
                strategy_stats['random_fallback']['success'] += 1
                return PuzzleCandidate(
                    word1=words[0],
                    word2=words[1],
                    result=result_word,
                    answer=answer,
                    unique_letters=len(set(''.join(words) + result_word)),
                    strategy='random_fallback',
                    word3=None,
                    valid_answers=[answer],
                    mapping=solutions[0][1],
                    solver_steps=stats['nodes'],
                )

    return None


DIFFICULTY_CONFIGS: Dict[str, Dict] = {
    "easy": {
        # v2 recalibration: 3-digit + 2-digit, carries 0-1, few letters — LLM-solvable in single pass.
        "num_operands": 2,
        "num1_range": (100, 999),
        "num2_range": (10, 99),
        "min_carries": 0,
        "max_carries": 1,
        "require_overflow": None,
        "target_letters": (4, 6),
        "min_solver_steps": 40,
        "max_attempts": 4000,
    },
    "medium": {
        "num_operands": 2,
        "num1_range": (1000, 9999),
        "num2_range": (1000, 9999),
        "min_carries": 2,
        "max_carries": 3,
        "require_overflow": None,
        "target_letters": (7, 8),
        "min_solver_steps": 650,
        "max_attempts": 5000,
    },
    "hard": {
        # v2 recalibration: frontier-grade stress with longer carry chains and more letters.
        "num_operands": 2,
        "num1_range": (10000, 99999),
        "num2_range": (10000, 99999),
        "min_carries": 4,
        "max_carries": None,
        "require_overflow": None,
        "target_letters": (9, 10),
        "min_solver_steps": 2500,
        "max_attempts": 6000,
    },
}


def generate_puzzle_by_difficulty(
    difficulty: str,
    used_patterns: Set[str] = None,
    **overrides,
) -> Optional[PuzzleCandidate]:
    if used_patterns is None:
        used_patterns = set()

    base_config = dict(DIFFICULTY_CONFIGS.get(difficulty, DIFFICULTY_CONFIGS["easy"]))
    base_config.update(overrides)

    relaxations = [dict(base_config)]
    if difficulty == "easy":
        relaxations.append({
            **base_config,
            "target_letters": (base_config["target_letters"][0], base_config["target_letters"][1] + 1),
            "min_solver_steps": max(100, base_config.get("min_solver_steps", 0) - 20),
            "max_attempts": base_config.get("max_attempts", 3000) // 2,
        })
    elif difficulty == "medium":
        relaxations.append({
            **base_config,
            "min_solver_steps": max(500, base_config.get("min_solver_steps", 0) - 80),
            "max_attempts": base_config.get("max_attempts", 3000) // 2,
        })

    for config in relaxations:
        min_letters, max_letters = config["target_letters"]
        min_carries = config["min_carries"]
        max_carries = config["max_carries"]
        require_overflow = config["require_overflow"]
        num_operands = config["num_operands"]
        max_attempts = config.get("max_attempts", 3000)

        for _ in range(max_attempts):
            num1 = random.randint(*config["num1_range"])
            num2 = random.randint(*config["num2_range"])
            num3 = None
            if num_operands == 3:
                num3 = random.randint(*config["num3_range"])

            operands = [num1, num2] + ([num3] if num3 is not None else [])
            result = sum(operands)

            if min_carries is not None or max_carries is not None:
                carries = count_carries(*operands)
                if min_carries is not None and carries < min_carries:
                    continue
                if max_carries is not None and carries > max_carries:
                    continue

            if require_overflow is True and not has_overflow(*operands):
                continue
            if require_overflow is False and has_overflow(*operands):
                continue

            combined = ''.join(str(n) for n in operands) + str(result)
            unique_digits = len(set(combined))
            if not (min_letters <= unique_digits <= max_letters):
                continue

            candidate = generate_from_arithmetic(num1, num2, used_patterns, num3=num3)
            if (
                candidate
                and min_letters <= candidate.unique_letters <= max_letters
                and candidate.solver_steps >= config.get("min_solver_steps", 0)
            ):
                return candidate

    return None


def create_question(candidate: PuzzleCandidate) -> str:
    operand_words = candidate.operands
    max_word_len = max(len(w) for w in operand_words + [candidate.result])
    separator = '-' * (max_word_len + 2)

    operand_lines = f"  {operand_words[0]}\n"
    for w in operand_words[1:]:
        operand_lines += f"+ {w}\n"

    question = (
        f"각 글자가 고유한 숫자(0-9)를 나타내는 복면산 퍼즐을 풀어주세요. "
        f"서로 다른 글자는 서로 다른 숫자에 대응해야 합니다. "
        f"첫 글자는 0이 될 수 없습니다. "
        f"{candidate.result}이(가) 나타내는 숫자 값을 구하세요.\n\n"
        f"{operand_lines}"
        f"{separator}\n"
        f"= {candidate.result}"
    )
    return question


def create_dataset_files(num_questions: int):
    import pandas as pd

    print(f"{num_questions}개의 복면산 퍼즐을 생성합니다...")

    difficulties = ["easy", "medium", "hard"]
    puzzles_per_diff = num_questions // len(difficulties)
    remainder = num_questions % len(difficulties)

    all_puzzles = []
    used_patterns = set()

    for i, difficulty in enumerate(difficulties):
        target_count = puzzles_per_diff + (1 if i < remainder else 0)

        if target_count == 0:
            continue

        print(f"\n=== {difficulty} 퍼즐 생성 중 ({target_count}개 필요) ===")
        generated = 0
        attempts = 0
        max_total_attempts = 5000

        while generated < target_count and attempts < max_total_attempts:
            attempts += 1
            candidate = generate_puzzle_by_difficulty(
                difficulty,
                used_patterns=used_patterns,
            )

            if candidate:
                operand_digits = [
                    int(''.join(str(candidate.mapping[c]) for c in w))
                    for w in candidate.operands
                ]
                carries = count_carries(*operand_digits)

                puzzle_data = {
                    "id": f"cryptarithmetic_ko_{len(all_puzzles)}",
                    "question": create_question(candidate),
                    "answer": candidate.answer,
                    "solution": candidate.puzzle_str,
                    "difficulty": difficulty,
                    "puzzle": candidate.puzzle_str,
                    "valid_answers": candidate.valid_answers or [candidate.answer],
                    "mapping": {k: v for k, v in candidate.mapping.items()} if candidate.mapping else {},
                    "step_metrics": {
                        "solver_steps": candidate.solver_steps,
                        "unique_letters": candidate.unique_letters,
                        "num_operands": len(candidate.operands),
                        "carries": carries,
                    },
                }
                all_puzzles.append(puzzle_data)
                generated += 1
                print(f"  [{generated}/{target_count}] {candidate.puzzle_str} -> {candidate.answer} (steps={candidate.solver_steps})")

        if generated < target_count:
            print(f"  경고: {difficulty} 퍼즐을 {target_count}개 중 {generated}개만 생성했습니다")

    print(f"\n총 {len(all_puzzles)}개의 퍼즐을 생성했습니다")
    print_strategy_stats()

    df = pd.DataFrame(all_puzzles)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "cryptarithmetic_ko.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV 파일 생성 완료: {csv_path}")

    # JSONL
    json_dir = PROJECT_ROOT / "data" / "jsonl"
    json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = json_dir / "cryptarithmetic_ko.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_puzzles:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL 파일 생성 완료: {jsonl_path}")

    return df, all_puzzles


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="복면산(Cryptarithmetic) 퍼즐 생성기 - 한국어")
    parser.add_argument("--num", type=int, default=12, help="생성할 문제 수")

    args = parser.parse_args()

    create_dataset_files(num_questions=args.num)
