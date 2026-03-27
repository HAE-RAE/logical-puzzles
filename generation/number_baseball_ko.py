"""숫자 야구 퍼즐 생성기 및 검증기

구성적 생성 방식: 정보 가치가 높은 힌트를 선택하여
해를 점진적으로 1개로 좁혀가는 퍼즐을 구축합니다.
백트래킹 솔버를 통해 3~6자리 퍼즐을 지원합니다.
"""

import random
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from itertools import permutations
from dataclasses import dataclass
from enum import Enum


@dataclass
class Hint:
    """숫자 야구 게임에서 추측과 그 결과를 나타내는 클래스"""
    guess: str
    strikes: int
    balls: int

    def __str__(self):
        return f"{self.guess}: {self.strikes}S {self.balls}B"

    def to_dict(self):
        return {
            "guess": self.guess,
            "strikes": self.strikes,
            "balls": self.balls
        }


class Difficulty(Enum):
    """문제 생성 난이도"""
    EASY = 1      # 3자리, 보통 수준의 힌트
    MEDIUM = 2    # 4자리, 적은 힌트
    HARD = 3      # 6자리, 최소한의 힌트


class BullsAndCows:
    """숫자 야구의 핵심 게임 로직"""

    def __init__(self, num_digits: int = 3):
        if num_digits not in [3, 4, 5, 6]:
            raise ValueError("자릿수는 3, 4, 5, 6 중 하나여야 합니다")
        self.num_digits = num_digits

    def generate_number(self) -> str:
        digits = list(range(10))
        random.shuffle(digits)
        return ''.join(str(d) for d in digits[:self.num_digits])

    def calculate_strikes_balls(self, secret: str, guess: str) -> Tuple[int, int]:
        if len(secret) != len(guess):
            raise ValueError("비밀 숫자와 추측의 자릿수가 같아야 합니다")

        strikes = 0
        balls = 0

        for i, digit in enumerate(guess):
            if digit == secret[i]:
                strikes += 1
            elif digit in secret:
                balls += 1

        return strikes, balls

    def check_number_against_hints(self, number: str, hints: List[Hint]) -> bool:
        for hint in hints:
            strikes, balls = self.calculate_strikes_balls(number, hint.guess)
            if strikes != hint.strikes or balls != hint.balls:
                return False
        return True

    def find_all_solutions(self, hints: List[Hint], max_count: int = 0) -> List[str]:
        """주어진 힌트를 모두 만족하는 가능한 숫자를 모두 찾습니다.

        Args:
            hints: 만족해야 하는 힌트 목록
            max_count: 이 수만큼 해를 찾으면 중단 (0 = 무제한)
        """
        solutions = []

        for perm in permutations('0123456789', self.num_digits):
            number = ''.join(perm)
            if self.check_number_against_hints(number, hints):
                solutions.append(number)
                if max_count > 0 and len(solutions) >= max_count:
                    break

        return solutions

    def has_unique_solution(self, hints: List[Hint]) -> bool:
        solutions = self.find_all_solutions(hints, max_count=2)
        return len(solutions) == 1

    def generate_hint(self, secret: str, max_attempts: int = 100) -> Optional[Hint]:
        attempts = 0
        while attempts < max_attempts:
            guess = self.generate_number()
            if guess != secret:
                strikes, balls = self.calculate_strikes_balls(secret, guess)
                return Hint(guess, strikes, balls)
            attempts += 1
        return None


MAX_SOLUTIONS = 1  # 정확히 1개의 해만 허용


class ProblemGenerator:
    """
    숫자 야구의 구성적 퍼즐 생성기.

    전략: 정보 가치에 기반하여 힌트를 생성하고,
    해의 수가 1이 될 때까지 점진적으로 힌트를 추가합니다.
    """

    def __init__(self):
        self.game_3digit = BullsAndCows(3)
        self.game_4digit = BullsAndCows(4)
        self.game_5digit = BullsAndCows(5)
        self.game_6digit = BullsAndCows(6)

    def _calculate_hint_info_value(self, hint: Hint, game: BullsAndCows,
                                    current_solutions: List[str]) -> int:
        """
        힌트의 정보 가치를 계산합니다.
        값이 높을수록 = 더 많은 잘못된 후보를 제거하는 힌트입니다.
        """
        if not current_solutions:
            return 0

        eliminated = 0
        for candidate in current_solutions:
            s, b = game.calculate_strikes_balls(hint.guess, candidate)
            if s != hint.strikes or b != hint.balls:
                eliminated += 1

        return eliminated

    def _select_best_hint(self, game: BullsAndCows, secret: str,
                          existing_hints: List[Hint], current_solutions: List[str],
                          difficulty: Difficulty, candidates: List[Hint]) -> Optional[Hint]:
        """해의 수를 1로 줄이는 데 가장 효과적인 힌트를 선택합니다."""
        best_hint = None
        best_solution_count = float('inf')

        for hint in candidates:
            if self._is_duplicate_hint(hint, existing_hints):
                continue

            if not self._hint_matches_difficulty(hint, difficulty):
                continue

            test_hints = existing_hints + [hint]
            new_solutions = game.find_all_solutions(test_hints, max_count=0)

            if len(new_solutions) == 1:
                return hint

            if len(new_solutions) >= 1 and len(new_solutions) < best_solution_count:
                best_solution_count = len(new_solutions)
                best_hint = hint

        return best_hint

    def _hint_matches_difficulty(self, hint: Hint, difficulty: Difficulty) -> bool:
        """힌트가 난이도 제약 조건에 맞는지 확인합니다."""
        if difficulty == Difficulty.HARD:
            return True
        elif difficulty == Difficulty.MEDIUM:
            return hint.strikes <= 1
        elif difficulty == Difficulty.EASY:
            return hint.strikes <= 1
        return True

    def generate_problem(self, difficulty: Difficulty, max_retries: int = 100) -> Dict:
        """
        정확히 1개의 해를 가진 퍼즐을 구성적으로 생성합니다.

        과정:
        1. 비밀 숫자 생성
        2. 다양한 정보량을 가진 후보 힌트 생성
        3. 정보 이득을 최대화하는 힌트 선택
        4. 해의 수가 1이 되면 중단
        5. 필요 시 새로운 무작위화로 재시도
        """
        if difficulty == Difficulty.EASY:
            game = self.game_3digit
            num_digits = 3
        elif difficulty == Difficulty.MEDIUM:
            game = self.game_4digit
            num_digits = 4
        else:  # HARD
            game = self.game_6digit
            num_digits = 6

        min_hints = {
            Difficulty.EASY: 4,
            Difficulty.MEDIUM: 3,
            Difficulty.HARD: 2
        }
        max_hints = {
            Difficulty.EASY: 6,
            Difficulty.MEDIUM: 4,
            Difficulty.HARD: 3
        }

        for retry in range(max_retries):
            secret = game.generate_number()

            # 후보 힌트 풀 생성
            hint_pool = []
            for _ in range(50):
                hint = game.generate_hint(secret)
                if hint and hint not in hint_pool:
                    hint_pool.append(hint)

            # 해를 좁히기 위해 점진적으로 힌트 선택
            hints = []
            all_solutions = game.find_all_solutions([], max_count=MAX_SOLUTIONS + 1)

            while len(hints) < max_hints[difficulty]:
                solutions = game.find_all_solutions(hints, max_count=MAX_SOLUTIONS + 1) if hints else all_solutions

                if len(solutions) == 1 and len(hints) >= min_hints[difficulty]:
                    break

                best_hint = self._select_best_hint(game, secret, hints, solutions,
                                                   difficulty, hint_pool)

                if best_hint:
                    hints.append(best_hint)
                    hint_pool.remove(best_hint)
                else:
                    new_hint = game.generate_hint(secret)
                    if new_hint and not self._is_duplicate_hint(new_hint, hints):
                        if self._hint_matches_difficulty(new_hint, difficulty):
                            hints.append(new_hint)
                    else:
                        break

            # 최종 해 확인
            solutions = game.find_all_solutions(hints, max_count=MAX_SOLUTIONS + 1) if hints else [secret]

            if len(solutions) == 1:
                return {
                    "difficulty": difficulty.name.lower(),
                    "num_digits": num_digits,
                    "hints": [hint.to_dict() for hint in hints],
                    "hint_text": self._format_hints(hints),
                    "answer": solutions[0],
                    "problem_text": self._create_problem_text(num_digits, hints)
                }

            # 고난도: 힌트를 모두 사용했을 때 해가 5개 이하면 허용
            if difficulty == Difficulty.HARD and len(solutions) <= 5 and len(hints) >= min_hints[difficulty]:
                return {
                    "difficulty": difficulty.name.lower(),
                    "num_digits": num_digits,
                    "hints": [hint.to_dict() for hint in hints],
                    "hint_text": self._format_hints(hints),
                    "answer": secret,
                    "problem_text": self._create_problem_text(num_digits, hints)
                }

        raise RuntimeError(
            f"{max_retries}번 재시도 후에도 정확히 1개의 해를 가진 "
            f"{difficulty.name} 난이도 퍼즐을 생성하지 못했습니다"
        )

    def _is_duplicate_hint(self, hint: Hint, hints: List[Hint]) -> bool:
        for h in hints:
            if h.guess == hint.guess:
                return True
        return False

    def _format_hints(self, hints: List[Hint]) -> List[str]:
        return [str(hint) for hint in hints]

    def _create_problem_text(self, num_digits: int, hints: List[Hint]) -> str:
        hint_strs = [f"[{hint.guess}: {hint.strikes}S {hint.balls}B]" for hint in hints]
        hints_text = ", ".join(hint_strs)
        return f"다음 모든 힌트를 만족하는, 각 자릿수가 서로 다른 {num_digits}자리 숫자를 찾으세요: {hints_text}"


# ============================================================
# 질문 포맷팅
# ============================================================

def create_question(problem: Dict) -> str:
    """한국어로 질문 텍스트를 생성합니다."""
    num_digits = problem['num_digits']
    hints = problem['hints']

    hints_text = "\n".join([
        f"  {i+1}. 추측: {h['guess']} → {h['strikes']} 스트라이크(S), {h['balls']} 볼(B)"
        for i, h in enumerate(hints)
    ])

    question = f"""다음 숫자 야구 퍼즐을 풀어보세요.

규칙:
- 비밀 숫자는 {num_digits}자리이며, 각 자릿수는 서로 다릅니다 (0-9)
- "스트라이크(S)"는 숫자가 맞고 위치도 맞음을 의미합니다
- "볼(B)"은 숫자는 맞지만 위치가 틀림을 의미합니다
- 모든 힌트를 만족하는 비밀 숫자를 찾으세요

힌트:
{hints_text}

단계별로 생각하며 유일한 {num_digits}자리 비밀 숫자를 찾으세요.

다음 형식으로 답을 제시하세요:
Answer: [{num_digits}자리 비밀 숫자]"""

    return question


def validate_problem(problem: Dict) -> Tuple[bool, str]:
    """생성된 문제의 정확성을 검증합니다."""
    try:
        num_digits = problem['num_digits']
        game = BullsAndCows(num_digits)

        hints = [Hint(h['guess'], h['strikes'], h['balls']) for h in problem['hints']]

        answer = problem['answer']
        if len(answer) != num_digits:
            return False, f"정답 길이 {len(answer)}가 자릿수 {num_digits}와 일치하지 않습니다"

        if len(set(answer)) != num_digits:
            return False, f"정답 {answer}에 중복된 숫자가 있습니다"

        if not game.check_number_against_hints(answer, hints):
            return False, f"정답 {answer}이 모든 힌트를 만족하지 않습니다"

        solutions = game.find_all_solutions(hints)
        if len(solutions) == 0:
            return False, "주어진 힌트를 만족하는 해가 존재하지 않습니다"
        elif len(solutions) > 1:
            return False, f"여러 개의 해가 존재합니다: {solutions}"
        elif solutions[0] != answer:
            return False, f"해 {solutions[0]}가 정답 {answer}과 일치하지 않습니다"

        return True, "유일한 해를 가진 유효한 문제입니다"

    except Exception as e:
        return False, f"검증 오류: {str(e)}"


# ============================================================
# 데이터셋 생성
# ============================================================

def create_dataset_files(num_questions: int):
    """
    숫자 야구 퍼즐 데이터셋 파일을 생성합니다.

    Args:
        num_questions: 생성할 문제 수

    Returns:
        Tuple[pd.DataFrame, List[Dict]]: (데이터프레임, JSON 리스트)
    """
    import pandas as pd

    print(f"{num_questions}개의 숫자 야구 퍼즐을 생성합니다...")

    generator = ProblemGenerator()

    difficulties = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]
    puzzles_per_diff = num_questions // len(difficulties)
    remainder = num_questions % len(difficulties)

    all_puzzles = []

    for i, difficulty in enumerate(difficulties):
        count = puzzles_per_diff + (1 if i < remainder else 0)
        diff_name = difficulty.name.lower()

        if count == 0:
            continue

        print(f"\n=== {diff_name} 난이도 퍼즐 생성 중 ({count}개 필요) ===")

        for j in range(count):
            try:
                problem = generator.generate_problem(difficulty)
                is_valid, msg = validate_problem(problem)

                if is_valid or (difficulty == Difficulty.HARD):
                    reordered = {
                        'id': f'number_baseball_ko_{len(all_puzzles)}',
                        'question': create_question(problem),
                        'answer': problem['answer'],
                        'difficulty': diff_name,
                        'num_digits': problem['num_digits'],
                        'hints': problem['hints'],
                        'hint_text': problem['hint_text'],
                        'problem_text': problem['problem_text']
                    }
                    all_puzzles.append(reordered)
                    print(f"  [{j+1}/{count}] 자릿수={problem['num_digits']}, "
                          f"힌트={len(problem['hints'])}개, 정답={problem['answer']}")
                else:
                    print(f"  [{j+1}/{count}] 검증 실패: {msg}")
            except RuntimeError as e:
                print(f"  [{j+1}/{count}] 실패: {e}")

    print(f"\n총 {len(all_puzzles)}개의 퍼즐이 생성되었습니다")

    df = pd.DataFrame(all_puzzles)

    # 파일 저장
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # CSV
    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "number_baseball_ko.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV 파일 생성 완료: {csv_path}")

    # JSONL
    json_dir = PROJECT_ROOT / "data" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = json_dir / "number_baseball_ko.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_puzzles:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL 파일 생성 완료: {jsonl_path}")

    return df, all_puzzles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="숫자 야구 퍼즐 생성기")
    parser.add_argument("--num", type=int, default=12, help="생성할 문제 수")

    args = parser.parse_args()

    print("=" * 60)
    print("숫자 야구 퍼즐 생성기")
    print("=" * 60)

    create_dataset_files(num_questions=args.num)
