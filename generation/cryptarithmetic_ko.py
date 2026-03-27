"""복면산(Cryptarithmetic) 퍼즐 생성기 - 한국어 버전

구성적 생성: 역산술을 사용하여 유효한 퍼즐을 보장하며,
다양한 매핑 전략으로 해의 개수를 제어합니다.
2항 연산(A + B = C) 및 3항 연산(A + B + C = D) 퍼즐을 모두 지원합니다.
"""

import itertools
import random
import string
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field


MAX_SOLUTIONS = 1  # 모든 퍼즐은 정확히 1개의 고유한 해를 요구


@dataclass
class PuzzleCandidate:
    """퍼즐 후보와 메타데이터를 나타냅니다."""
    word1: str
    word2: str
    result: str
    answer: str  # 주요 정답 (결과 값)
    unique_letters: int
    strategy: str  # 사용된 매핑 전략
    word3: str = None  # 선택적 세 번째 피연산자
    valid_answers: List[str] = None  # 모든 유효한 결과 값 (다중 해 퍼즐용)
    mapping: Dict[str, int] = None  # 해에서 얻은 글자-숫자 매핑

    @property
    def puzzle_str(self) -> str:
        if self.word3:
            return f"{self.word1} + {self.word2} + {self.word3} = {self.result}"
        return f"{self.word1} + {self.word2} = {self.result}"

    @property
    def operands(self):
        """피연산자 단어 목록을 반환합니다."""
        ops = [self.word1, self.word2]
        if self.word3:
            ops.append(self.word3)
        return ops


# ============================================================
# 풀이기: 열 단위 역추적(backtracking)
# ============================================================

def find_solutions(puzzle: tuple, max_count: int = 4) -> List[Tuple[str, Dict[str, int]]]:
    """
    열 단위 역추적을 사용하여 max_count까지의 모든 해를 찾습니다.

    puzzle: (word1, word2, result_word) 또는 (word1, word2, word3, result_word)
    (결과값, 매핑) 튜플의 리스트를 반환합니다.
    """
    *operand_words, result_word = puzzle
    all_letters = sorted(set(''.join(operand_words) + result_word))

    if len(all_letters) > 10:
        return []

    # 0이 될 수 없는 첫 글자
    first_letters = set()
    for w in list(operand_words) + [result_word]:
        if len(w) > 1:
            first_letters.add(w[0])

    # 열 제약 조건 구축 (오른쪽에서 왼쪽으로)
    all_words = list(operand_words) + [result_word]
    max_len = max(len(w) for w in all_words)
    reversed_ops = [w[::-1] for w in operand_words]
    wr = result_word[::-1]

    # 열 순서에 따른 글자 할당 순서 결정 (오른쪽에서 왼쪽으로)
    # 이를 통해 산술 제약 조건을 활용한 조기 가지치기가 가능합니다
    ordered_letters = []
    seen = set()
    for col in range(max_len):
        for w in reversed_ops + [wr]:
            if col < len(w) and w[col] not in seen:
                ordered_letters.append(w[col])
                seen.add(w[col])
    # 나머지 글자 추가
    for letter in all_letters:
        if letter not in seen:
            ordered_letters.append(letter)
            seen.add(letter)

    solutions = []
    mapping = {}
    used_digits = set()

    def _check_columns(mapping):
        """완전히 할당된 모든 열의 일관성을 검사합니다."""
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

            # 가지치기: 부분 열 제약 조건 검사
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
    """빠른 해 개수 검사."""
    return len(find_solutions(puzzle, max_count=MAX_SOLUTIONS + 1))


def has_valid_solutions(puzzle: tuple) -> bool:
    """퍼즐이 정확히 1개의 해를 가지는지 확인합니다."""
    return count_solutions_fast(puzzle) == 1


def get_solution(puzzle: tuple) -> Optional[Tuple[str, Dict[str, int]]]:
    """유일한 해가 존재하면 반환합니다."""
    solutions = find_solutions(puzzle, max_count=1)
    return solutions[0] if solutions else None


def get_all_result_values(puzzle: tuple, max_count: int = 20) -> List[str]:
    """유효한 해에서 모든 고유한 결과 값을 가져옵니다."""
    solutions = find_solutions(puzzle, max_count=max_count)
    return list(set(s[0] for s in solutions))


# ============================================================
# 글자 매핑 전략
# ============================================================

def _create_letter_mapping(unique_digits: List[str], strategy: str = 'random') -> Dict[str, str]:
    """
    다양한 전략으로 숫자-글자 매핑을 생성합니다.

    전략:
    - 'random': 무작위 글자 할당
    - 'sequential': A, B, C, ... 순차 할당
    - 'reverse': Z, Y, X, ... 역순 할당
    - 'vowel_first': 자주 사용되는 숫자에 모음 우선 할당
    """
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
    # 'sequential'은 기본 순서 사용

    return {digit: available_letters[i] for i, digit in enumerate(unique_digits)}


strategy_stats = {
    'random': {'tried': 0, 'success': 0},
    'sequential': {'tried': 0, 'success': 0},
    'reverse': {'tried': 0, 'success': 0},
    'vowel_first': {'tried': 0, 'success': 0},
    'random_fallback': {'tried': 0, 'success': 0}
}


def print_strategy_stats():
    """매핑 전략 성공 통계를 출력합니다."""
    print("\n매핑 전략 통계:")
    for name, stats in strategy_stats.items():
        rate = stats['success'] / stats['tried'] * 100 if stats['tried'] > 0 else 0
        print(f"  {name}: {stats['success']}/{stats['tried']} ({rate:.1f}%)")


# ============================================================
# 퍼즐 생성
# ============================================================

def count_carries(*nums: int) -> int:
    """여러 수의 덧셈에서 올림(carry) 횟수를 셉니다."""
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
    """결과가 가장 큰 피연산자보다 더 많은 자릿수를 가지는지 확인합니다."""
    result = sum(nums)
    max_operand_digits = max(len(str(n)) for n in nums)
    result_digits = len(str(result))
    return result_digits > max_operand_digits


def generate_from_arithmetic(
    num1: int,
    num2: int,
    used_patterns: Set[str] = None,
    num3: int = None
) -> Optional[PuzzleCandidate]:
    """
    유효한 산술로부터 퍼즐을 구성적으로 생성합니다.

    모든 퍼즐 유형에 대해 정확히 1개의 고유한 해를 요구합니다.
    """
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

    unique_digits = list(set(combined))

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

        solutions = find_solutions(puzzle, max_count=max_allowed + 1)
        solution_count = len(solutions)

        if 1 <= solution_count <= max_allowed:
            used_patterns.add(pattern)
            answer = str(result)
            all_values = list(set(s[0] for s in solutions))
            strategy_stats[strategy]['success'] += 1
            return PuzzleCandidate(
                word1=words[0],
                word2=words[1],
                result=result_word,
                answer=answer,
                unique_letters=len(set(''.join(words) + result_word)),
                strategy=strategy,
                word3=words[2] if len(words) > 2 else None,
                valid_answers=all_values,
                mapping=solutions[0][1]
            )

    # 2항 연산에 대해서만: 몇 번 더 무작위 매핑 시도
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

            solutions = find_solutions(puzzle, max_count=max_allowed + 1)
            solution_count = len(solutions)

            if 1 <= solution_count <= max_allowed:
                used_patterns.add(pattern)
                answer = str(result)
                all_values = list(set(s[0] for s in solutions))
                strategy_stats['random_fallback']['success'] += 1
                return PuzzleCandidate(
                    word1=words[0],
                    word2=words[1],
                    result=result_word,
                    answer=answer,
                    unique_letters=len(set(''.join(words) + result_word)),
                    strategy='random_fallback',
                    word3=None,
                    valid_answers=all_values,
                    mapping=solutions[0][1]
                )

    return None


def generate_puzzle_by_difficulty(
    difficulty: str,
    used_patterns: Set[str] = None
) -> Optional[PuzzleCandidate]:
    """지정된 난이도에 맞는 퍼즐을 생성합니다."""
    if used_patterns is None:
        used_patterns = set()

    configs = {
        "easy": {
            "num_operands": 2,
            "num1_range": (100, 999),
            "num2_range": (10, 99),
            "min_carries": None,
            "max_carries": None,
            "require_overflow": None,
            "target_letters": (4, 5),
        },
        "medium": {
            "num_operands": 3,
            "num1_range": (100, 999),
            "num2_range": (10, 99),
            "num3_range": (10, 99),
            "min_carries": None,
            "max_carries": 1,
            "require_overflow": None,
            "target_letters": (4, 5),
        },
        "hard": {
            "num_operands": 3,
            "num1_range": (1000, 9999),
            "num2_range": (100, 999),
            "num3_range": (10, 99),
            "min_carries": 2,
            "max_carries": None,
            "require_overflow": None,
            "target_letters": (7, 9),
        },
    }

    config = configs.get(difficulty, configs["easy"])
    min_letters, max_letters = config["target_letters"]
    min_carries = config["min_carries"]
    max_carries = config["max_carries"]
    require_overflow = config["require_overflow"]
    num_operands = config["num_operands"]

    for _ in range(1000):
        num1 = random.randint(*config["num1_range"])
        num2 = random.randint(*config["num2_range"])
        num3 = None
        if num_operands == 3:
            num3 = random.randint(*config["num3_range"])

        operands = [num1, num2] + ([num3] if num3 is not None else [])
        result = sum(operands)

        if min_carries is not None:
            carries = count_carries(*operands)
            if carries < min_carries:
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

        if candidate and min_letters <= candidate.unique_letters <= max_letters:
            return candidate

    return None


# ============================================================
# 문제 텍스트 생성
# ============================================================

def create_question(candidate: PuzzleCandidate) -> str:
    """한국어로 문제 텍스트를 생성합니다."""
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


# ============================================================
# 데이터셋 생성
# ============================================================

def generate_dataset(
    puzzles_per_difficulty: int = 3,
    difficulties: List[str] = None,
    verbose: bool = True
) -> List[Dict]:
    """퍼즐의 전체 데이터셋을 생성합니다."""
    if difficulties is None:
        difficulties = ["easy", "medium", "hard"]

    dataset = []
    used_patterns = set()

    for difficulty in difficulties:
        if verbose:
            print(f"\n=== {difficulty} 퍼즐 생성 중 ===")

        for i in range(puzzles_per_difficulty):
            candidate = generate_puzzle_by_difficulty(difficulty, used_patterns)

            if candidate:
                question = create_question(candidate)

                puzzle_data = {
                    "question": question,
                    "answer": candidate.answer,
                    "difficulty": difficulty,
                    "puzzle": candidate.puzzle_str,
                    "valid_answers": candidate.valid_answers or [candidate.answer],
                    "mapping": candidate.mapping,
                }

                dataset.append(puzzle_data)

                if verbose:
                    print(f"  [{i+1}/{puzzles_per_difficulty}] {candidate.puzzle_str} -> {candidate.answer}")
            else:
                if verbose:
                    print(f"  [{i+1}/{puzzles_per_difficulty}] 생성 실패")

    if verbose:
        print_strategy_stats()

    return dataset


def generate_and_validate(
    word1: str,
    word2: str,
    result: str,
    word3: str = None
) -> Optional[Dict]:
    """
    수동으로 지정된 퍼즐을 검증하고, 유효하면 형식화된 데이터를 반환합니다.
    """
    words = [word1.upper(), word2.upper()]
    if word3:
        words.append(word3.upper())
    words.append(result.upper())

    puzzle = tuple(words)

    if not has_valid_solutions(puzzle):
        sol_count = count_solutions_fast(puzzle)
        print(f"유효하지 않음: 퍼즐의 해가 {sol_count}개입니다")
        return None

    solutions = find_solutions(puzzle, max_count=1)
    if solutions:
        answer_val, mapping = solutions[0]

        all_letters = set(''.join(words))
        unique_letters = len(all_letters)

        if unique_letters <= 4:
            difficulty = "easy"
        elif unique_letters <= 6:
            difficulty = "medium"
        else:
            difficulty = "hard"

        candidate = PuzzleCandidate(
            word1=words[0],
            word2=words[1],
            result=words[-1],
            answer=answer_val,
            unique_letters=unique_letters,
            strategy='manual',
            word3=words[2] if len(words) == 4 else None,
            valid_answers=[answer_val],
            mapping=mapping
        )

        return {
            "question": create_question(candidate),
            "answer": answer_val,
            "difficulty": difficulty,
            "puzzle": candidate.puzzle_str,
            "mapping": mapping
        }

    return None


def create_dataset_files(num_questions: int):
    """
    복면산 퍼즐 데이터셋 파일을 생성합니다.

    Args:
        num_questions: 생성할 문제 수

    Returns:
        Tuple[pd.DataFrame, List[Dict]]: (데이터프레임, JSON 리스트)
    """
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
                used_patterns=used_patterns
            )

            if candidate:
                puzzle_data = {
                    "id": f"cryptarithmetic_ko_{len(all_puzzles)}",
                    "question": create_question(candidate),
                    "answer": candidate.answer,
                    "solution": candidate.puzzle_str,
                    "difficulty": difficulty,
                    "puzzle": candidate.puzzle_str,
                    "valid_answers": candidate.valid_answers or [candidate.answer],
                    "mapping": {k: v for k, v in candidate.mapping.items()} if candidate.mapping else {},
                }
                all_puzzles.append(puzzle_data)
                generated += 1
                print(f"  [{generated}/{target_count}] {candidate.puzzle_str} -> {candidate.answer}")

        if generated < target_count:
            print(f"  경고: {difficulty} 퍼즐을 {target_count}개 중 {generated}개만 생성했습니다")

    print(f"\n총 {len(all_puzzles)}개의 퍼즐을 생성했습니다")
    print_strategy_stats()

    df = pd.DataFrame(all_puzzles)

    # 파일 저장
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # CSV
    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "cryptarithmetic_ko.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV 파일 생성 완료: {csv_path}")

    # JSONL
    json_dir = PROJECT_ROOT / "data" / "json"
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
