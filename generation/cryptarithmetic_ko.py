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

    # 열별 letter 구조 (operands + result) 를 한 번만 미리 계산.
    # 매 노드 prune 마다 list-comp 재구축을 회피.
    cols_op_letters: List[Tuple[str, ...]] = []
    cols_cr: List[Optional[str]] = []
    for col in range(max_len):
        cols_op_letters.append(tuple(
            rw[col] for rw in reversed_ops if col < len(rw)
        ))
        cols_cr.append(wr[col] if col < len(wr) else None)

    def _check_columns(mapping):
        carry = 0
        for col in range(max_len):
            ops = cols_op_letters[col]
            cr = cols_cr[col]

            for ol in ops:
                if ol not in mapping:
                    return True
            if cr is not None and cr not in mapping:
                return True

            total = carry
            for ol in ops:
                total += mapping[ol]
            dr = mapping[cr] if cr is not None else 0

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
                ops = cols_op_letters[col]
                cr = cols_cr[col]

                all_assigned = True
                for ol in ops:
                    if ol not in mapping:
                        all_assigned = False
                        break
                has_cr = cr is None or cr in mapping

                if all_assigned and has_cr:
                    total = carry
                    for ol in ops:
                        total += mapping[ol]
                    dr = mapping[cr] if cr is not None else 0
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
    # Calibrated to step-count proxy: letters × num_operands × (1 + carries).
    # See docs/difficulty_definition.md §2.4. Prior sweep data invalid (infra
    # bug); these values are set from algorithmic argument and will be
    # fine-tuned after the next diagnostic.
    "easy": {
        # v6: gemini 90 / gpt-5.4-mini 100 — 충분한 gradient. v7 시도 (3-4 letters)
        # 는 사용자 판정상 불필요 → v6 유지.
        "num_operands": 2,
        "num1_range": (100, 999),
        "num2_range": (100, 999),
        "min_carries": 1,
        "max_carries": 2,
        "require_overflow": None,
        "target_letters": (5, 7),
        "min_solver_steps": 200,
        "max_attempts": 5000,
    },
    "medium": {
        # v8 shift: medium 슬롯이 v7 hard config 를 채택. 기존 v7 medium (7-8L, 1500s)
        # 에서 9-10L, 5000s 로 step proxy 약 3× 증가. v7 medium 데이터는 *_v7.jsonl 백업.
        "num_operands": 2,
        "num1_range": (10000, 99999),
        "num2_range": (10000, 99999),
        "min_carries": 4,
        "max_carries": None,
        "require_overflow": None,
        "target_letters": (9, 10),
        "min_solver_steps": 5000,
        "max_attempts": 10000,
    },
    "hard": {
        # v8.1 candidate: letters 10-11, steps 9800 — easy(200) → medium(5000) gap=4800,
        # medium(5000) → hard(9800) gap=4800 → 균일 step proxy gap. v8 (11-12L, 15000s)
        # 는 0/5 (100 retries) fail — 9800/10-11L 로 완화.
        "num_operands": 2,
        "num1_range": (10000, 99999),
        "num2_range": (10000, 99999),
        "min_carries": 4,
        "max_carries": None,
        "require_overflow": None,
        "target_letters": (10, 11),
        "min_solver_steps": 9800,
        "max_attempts": 12000,
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


SFT_SOLUTION_RUBRIC_KO = (
    "STEP0=문제 메타 · STEP1=주어진 조건 · STEP2=풀이 전개 · STEP3=답·검산"
)


def _build_cryptarithmetic_solution_ko(
    candidate: PuzzleCandidate,
    difficulty: str,
    carries: int,
) -> str:
    """SFT teacher trace: 복면산 역산 + 열별 SEG."""
    operand_words = candidate.operands
    result_word = candidate.result
    mapping = candidate.mapping or {}
    answer = candidate.answer

    mapping_items = sorted(mapping.items(), key=lambda kv: kv[0])
    mapping_text = ", ".join(f"{k}={v}" for k, v in mapping_items)

    def digits_of(word: str) -> str:
        return "".join(str(mapping.get(c, "?")) for c in word)

    operand_digit_strs = [digits_of(w) for w in operand_words]
    result_digit_str = digits_of(result_word)

    op_lines_words = " + ".join(operand_words) + f" = {result_word}"
    op_lines_digits = " + ".join(operand_digit_strs) + f" = {result_digit_str}"

    lines: List[str] = [
        SFT_SOLUTION_RUBRIC_KO,
        "[STEP 0] 문제 메타",
        f"  - 난이도: {difficulty}",
        f"  - 퍼즐: {op_lines_words}",
        f"  - 고유 글자 수: {candidate.unique_letters}",
        f"  - 받아올림(carry) 수: {carries}",
        "  - 최종 답은 [STEP 3]에서 확정",
        "[STEP 1] 주어진 조건",
        "  - 규칙: 서로 다른 글자 = 서로 다른 숫자(0–9).",
        "  - 규칙: 각 단어의 **첫 글자는 0이 될 수 없음**.",
        f"  - 구하려는 값: {result_word}의 숫자값.",
    ]

    max_len = max(len(w) for w in operand_words + [result_word])
    padded_operands = [w.rjust(max_len) for w in operand_words]
    padded_result = result_word.rjust(max_len)

    carry = 0
    seg_lines: List[str] = []
    for pos in range(max_len):
        col_idx = max_len - 1 - pos
        col_digit_letters = [w[col_idx] for w in padded_operands if w[col_idx] != ' ']
        col_digits = [mapping[c] for c in col_digit_letters if c in mapping]
        s = sum(col_digits) + carry
        new_carry = s // 10
        unit = s % 10
        res_letter = padded_result[col_idx] if padded_result[col_idx] != ' ' else ''
        res_digit = mapping.get(res_letter, None) if res_letter else None
        terms = " + ".join(f"{c}={mapping[c]}" for c in col_digit_letters if c in mapping)
        if carry:
            terms += f" + carry {carry}"
        verdict = ""
        if res_digit is not None:
            verdict = f" → 결과 자리 {res_letter}={res_digit} (받아올림 {new_carry})"
        else:
            verdict = f" → 받아올림 {new_carry}"
        seg_lines.append(
            f"    [SEG {pos + 1}] 자리 {pos + 1}(우→좌): {terms} = {s}{verdict}"
        )
        carry = new_carry

    lines.append("[STEP 2] 풀이 전개")
    lines.append(
        f"  · 요약: 우→좌 자리별 합 · 글자→숫자 매핑({mapping_text}) · "
        f"받아올림 총 {carries}회 · SEG {len(seg_lines)}개"
    )
    lines.append(f"  · 매핑 검증: {op_lines_digits}")
    lines.extend(seg_lines)

    lines.extend([
        "[STEP 3] 답·검산",
        f"  - 최종 답: {result_word} = {answer}",
        f"  - 매핑: {mapping_text}",
        "  - 각 자리 합과 받아올림이 일치하는지 [SEG] 전개로 재확인.",
        "  - 모든 단어의 첫 글자 숫자가 0이 아닌지 확인.",
    ])
    return "\n".join(lines)


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
                    "id": f"cryptarithmetic_ko_{difficulty}_{generated:04d}",
                    "question": create_question(candidate),
                    "answer": candidate.answer,
                    "solution": _build_cryptarithmetic_solution_ko(
                        candidate, difficulty, carries
                    ),
                    "difficulty": difficulty,
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
