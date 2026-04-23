"""숫자 야구(Bulls and Cows) 퍼즐 생성기 - 한국어 버전

구성적 생성 방식: 정보 가치가 높은 힌트를 선택하여
해를 점진적으로 정확히 1개로 좁혀가는 퍼즐을 구축합니다.

logical-puzzles-me/number_baseball/generator.py 기반 이식:
- 비밀 숫자의 순열을 포함한 후보 힌트 풀 (볼 중심 힌트 생성)
- 중/상 난이도를 위한 2단계 전방 탐색(2-step lookahead) 스코어링
- 상 난이도 전용 볼 중심 체인 전략
- 모든 난이도에서 엄격한 유일 해(MAX_SOLUTIONS = 1) 보장
- 퍼즐 JSONL 에 step_metrics 필드 포함
"""

import itertools
import math
import random
import statistics
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from itertools import permutations
from dataclasses import dataclass
from enum import Enum


MAX_SOLUTIONS = 1  # 모든 난이도에서 정확히 1개의 해만 허용


@dataclass
class Hint:
    guess: str
    strikes: int
    balls: int

    def __str__(self):
        return f"{self.guess}: {self.strikes}S {self.balls}B"

    def to_dict(self):
        return {"guess": self.guess, "strikes": self.strikes, "balls": self.balls}


class Difficulty(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3


DIFFICULTY_CONFIGS: Dict[str, Dict] = {
    "easy": {
        # v2 recalibration: 3-digit candidate space is small (720), so 4-5 hints
        # typically narrow to 1 in 1-2 deductions.
        "num_digits": 3,
        "min_hints": 4,
        "max_hints": 5,
        "preferred_strikes": (0, 2),
        "preferred_balls": (2, 3),
        "target_residual": (1, 12),
        "min_ball_heavy_ratio": 0.10,
    },
    "medium": {
        # v3 recalibration: fewer hints (2-3) + stronger ball-heavy to break gemini's
        # 100% saturation at v2 (3-4 hints). 4-digit candidate space = 5040, so
        # 2-3 hints requires careful cross-hint deduction.
        "num_digits": 4,
        "min_hints": 2,
        "max_hints": 3,
        "preferred_strikes": (0, 1),
        "preferred_balls": (2, 4),
        "target_residual": (2, 24),
        "min_ball_heavy_ratio": 0.55,
    },
    "hard": {
        # v2 recalibration: fewer hints + heavier ball-bias forces cross-hint intersection.
        "num_digits": 6,
        "min_hints": 3,
        "max_hints": 4,
        "preferred_strikes": (0, 1),
        "preferred_balls": (2, 5),
        "target_residual": (4, 45),
        "min_ball_heavy_ratio": 0.70,
    },
}


class BullsAndCows:
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
            s, b = self.calculate_strikes_balls(number, hint.guess)
            if s != hint.strikes or b != hint.balls:
                return False
        return True

    def find_all_solutions(self, hints: List[Hint], max_count: int = 0) -> List[str]:
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
                s, b = self.calculate_strikes_balls(secret, guess)
                return Hint(guess, s, b)
            attempts += 1
        return None


class ProblemGenerator:
    """숫자 야구의 구성적 퍼즐 생성기."""

    def __init__(self):
        self.game_3digit = BullsAndCows(3)
        self.game_4digit = BullsAndCows(4)
        self.game_5digit = BullsAndCows(5)
        self.game_6digit = BullsAndCows(6)

    def _is_duplicate_hint(self, hint: Hint, hints: List[Hint]) -> bool:
        for h in hints:
            if h.guess == hint.guess and h.strikes == hint.strikes and h.balls == hint.balls:
                return True
        return False

    def _hint_matches_difficulty(self, hint: Hint, difficulty: Difficulty) -> bool:
        cfg = DIFFICULTY_CONFIGS[difficulty.name.lower()]
        slo, shi = cfg["preferred_strikes"]
        blo, bhi = cfg["preferred_balls"]
        if not (slo <= hint.strikes <= shi):
            return False
        if not (blo <= hint.balls <= bhi):
            return False
        return True

    def _build_candidate_pool(
        self,
        game: BullsAndCows,
        secret: str,
        difficulty: Difficulty,
        target_size: int = 80,
    ) -> List[Hint]:
        """비밀 숫자 순열을 활용하여 후보 힌트 풀을 구성합니다
        (낮은 스트라이크/높은 볼 힌트 확보에 유리)."""
        pool: Dict[tuple, Hint] = {}

        if difficulty != Difficulty.EASY:
            perms = list(itertools.permutations(secret))
            random.shuffle(perms)
            for perm in perms:
                guess = ''.join(perm)
                if guess == secret:
                    continue
                s, b = game.calculate_strikes_balls(secret, guess)
                hint = Hint(guess, s, b)
                if self._hint_matches_difficulty(hint, difficulty):
                    pool[(hint.guess, hint.strikes, hint.balls)] = hint
                if len(pool) >= target_size:
                    break

        attempts = 0
        while len(pool) < target_size and attempts < target_size * 20:
            attempts += 1
            hint = game.generate_hint(secret)
            if hint and self._hint_matches_difficulty(hint, difficulty):
                pool[(hint.guess, hint.strikes, hint.balls)] = hint

        hints = list(pool.values())
        if difficulty == Difficulty.HARD:
            hints.sort(key=lambda h: (h.balls, -h.strikes), reverse=True)
        elif difficulty == Difficulty.MEDIUM:
            hints.sort(key=lambda h: (h.balls, -h.strikes), reverse=True)
        else:
            hints.sort(key=lambda h: (h.strikes, -h.balls), reverse=True)
        return hints[:target_size]

    def _project_two_step_residual(
        self,
        game: BullsAndCows,
        existing_hints: List[Hint],
        current_solutions: List[str],
        candidate: Hint,
        difficulty: Difficulty,
        candidates: List[Hint],
        max_followups: int = 12,
    ) -> int:
        """2단계 전방 탐색: `candidate` 적용 후 다음 힌트가 도달 가능한
        최적 잔여 후보 수를 추정합니다. 중/상 난이도에서 사용."""
        best = None
        base_hints = existing_hints + [candidate]
        base_candidates = [
            s for s in current_solutions
            if game.calculate_strikes_balls(s, candidate.guess) == (candidate.strikes, candidate.balls)
        ]
        followups = candidates[:max_followups]
        for nxt in followups:
            if nxt.guess == candidate.guess:
                continue
            if self._is_duplicate_hint(nxt, base_hints):
                continue
            if not self._hint_matches_difficulty(nxt, difficulty):
                continue
            residual = sum(
                1 for s in base_candidates
                if game.calculate_strikes_balls(s, nxt.guess) == (nxt.strikes, nxt.balls)
            )
            if residual < 1:
                continue
            if best is None or residual < best:
                best = residual
                if residual == 1:
                    break
        return best if best is not None else 10**9

    def _select_best_hint(
        self,
        game: BullsAndCows,
        secret: str,
        existing_hints: List[Hint],
        current_solutions: List[str],
        difficulty: Difficulty,
        candidates: List[Hint],
        cfg: Dict[str, int],
    ) -> Optional[Hint]:
        """난이도 프로파일에 따른 최적 다음 힌트 선택."""
        best_hint = None
        best_score = None
        next_index = len(existing_hints) + 1
        min_hints = cfg["min_hints"]
        max_hints = cfg["max_hints"]
        target_lo, target_hi = cfg["target_residual"]

        for hint in candidates:
            if self._is_duplicate_hint(hint, existing_hints):
                continue
            if not self._hint_matches_difficulty(hint, difficulty):
                continue

            residual_candidates = [
                s for s in current_solutions
                if game.calculate_strikes_balls(s, hint.guess) == (hint.strikes, hint.balls)
            ]
            residual = len(residual_candidates)
            if residual < 1:
                continue

            if next_index >= min_hints and residual == 1:
                return hint

            lookahead = self._project_two_step_residual(
                game,
                existing_hints,
                current_solutions,
                hint,
                difficulty,
                candidates,
                max_followups=8 if difficulty == Difficulty.MEDIUM else 10,
            ) if difficulty != Difficulty.EASY else residual

            if difficulty == Difficulty.EASY:
                score = (
                    residual == 1,
                    -residual,
                    hint.strikes,
                    -hint.balls,
                )
            elif difficulty == Difficulty.MEDIUM:
                in_band = target_lo <= residual <= target_hi
                score = (
                    in_band,
                    -(residual == 1),
                    -abs(residual - (target_lo + target_hi) / 2),
                    -(lookahead == 10**9),
                    -abs(lookahead - max(1, target_lo // 2)),
                    hint.balls,
                    -hint.strikes,
                )
            else:
                if next_index < max_hints and residual == 1:
                    continue
                in_band = target_lo <= residual <= target_hi
                ball_heavy = hint.balls >= 2
                low_strike = hint.strikes <= 1
                score = (
                    low_strike,
                    ball_heavy,
                    in_band,
                    -(residual == 1),
                    -abs(residual - (target_lo + target_hi) / 2),
                    -(lookahead == 10**9),
                    -abs(lookahead - max(1, target_lo // 2)),
                    hint.balls,
                    -hint.strikes,
                )

            if best_score is None or score > best_score:
                best_score = score
                best_hint = hint

        return best_hint

    def _select_hard_hint_sequence(
        self,
        game: BullsAndCows,
        secret: str,
        cfg: Dict[str, int],
        candidate_space: List[str],
        hint_pool: List[Hint],
    ) -> Optional[Tuple[List[Hint], List[int]]]:
        """상 난이도 전략: 초반에는 볼 중심 체인을 유지하고, 마지막에 유일 해로 수렴."""
        if not hint_pool:
            return None

        hints: List[Hint] = []
        residuals: List[int] = []
        current = list(candidate_space)
        max_hints = cfg["max_hints"]
        min_hints = cfg["min_hints"]
        target_lo, target_hi = cfg["target_residual"]

        ranked_pool = sorted(
            hint_pool,
            key=lambda h: (h.balls, -h.strikes),
            reverse=True,
        )

        for step in range(max_hints):
            best = None
            best_filtered = None
            best_score = None
            for hint in ranked_pool:
                if self._is_duplicate_hint(hint, hints):
                    continue
                filtered = [
                    s for s in current
                    if game.calculate_strikes_balls(s, hint.guess) == (hint.strikes, hint.balls)
                ]
                residual = len(filtered)
                if residual < 1:
                    continue
                if step + 1 < min_hints and residual == 1:
                    continue
                if step + 1 < max_hints and residual == 1:
                    continue

                ball_heavy = hint.balls >= 2 and hint.strikes <= 1
                in_band = target_lo <= residual <= target_hi
                closeness = -abs(residual - (target_lo + target_hi) / 2)
                score = (ball_heavy, in_band, closeness, hint.balls, -hint.strikes)
                if best_score is None or score > best_score:
                    best = hint
                    best_filtered = filtered
                    best_score = score

            if best is None:
                break

            hints.append(best)
            current = best_filtered
            residuals.append(len(current))

            if len(current) == 1 and len(hints) >= min_hints:
                return hints, residuals

        if len(hints) < min_hints:
            return None

        for hint in ranked_pool:
            if self._is_duplicate_hint(hint, hints):
                continue
            filtered = [
                s for s in current
                if game.calculate_strikes_balls(s, hint.guess) == (hint.strikes, hint.balls)
            ]
            if len(filtered) == 1:
                hints.append(hint)
                residuals.append(1)
                return hints, residuals

        return None

    def generate_problem(self, difficulty: Difficulty, max_retries: int = 4000) -> Dict:
        """정확히 1개의 해를 가진 퍼즐을 구성적으로 생성합니다."""
        cfg = DIFFICULTY_CONFIGS[difficulty.name.lower()]
        num_digits = cfg["num_digits"]
        if num_digits == 3:
            game = self.game_3digit
        elif num_digits == 4:
            game = self.game_4digit
        elif num_digits == 5:
            game = self.game_5digit
        else:
            game = self.game_6digit

        min_hints = {difficulty: cfg["min_hints"]}
        max_hints = {difficulty: cfg["max_hints"]}

        for retry in range(max_retries):
            secret = game.generate_number()

            hint_pool = self._build_candidate_pool(
                game,
                secret,
                difficulty,
                target_size=36 if difficulty == Difficulty.EASY else 28,
            )

            candidate_space = [''.join(p) for p in itertools.permutations('0123456789', num_digits)]

            if difficulty == Difficulty.HARD:
                structured = self._select_hard_hint_sequence(
                    game, secret, cfg, candidate_space, hint_pool
                )
                if structured is None:
                    continue
                hints, residuals = structured
                solutions = [secret]
            else:
                hints = []
                solutions = candidate_space
                while len(hints) < max_hints[difficulty]:
                    if len(solutions) == 1 and len(hints) >= min_hints[difficulty]:
                        break

                    best_hint = self._select_best_hint(
                        game, secret, hints, solutions,
                        difficulty, hint_pool, cfg,
                    )

                    if best_hint:
                        hints.append(best_hint)
                        hint_pool.remove(best_hint)
                        solutions = [
                            s for s in solutions
                            if game.calculate_strikes_balls(s, best_hint.guess) == (best_hint.strikes, best_hint.balls)
                        ]
                    else:
                        replenished = [
                            h for h in self._build_candidate_pool(game, secret, difficulty, target_size=24)
                            if not self._is_duplicate_hint(h, hints)
                        ]
                        if replenished:
                            hint_pool.extend(replenished)
                        else:
                            break

                solutions = solutions if hints else [secret]
                residuals = None

            if len(solutions) == 1 and len(hints) >= min_hints[difficulty]:
                candidates = [''.join(p) for p in itertools.permutations('0123456789', num_digits)]
                initial_candidates = len(candidates)
                if residuals is None:
                    residuals = []
                    for h in hints:
                        candidates = [
                            c for c in candidates
                            if game.calculate_strikes_balls(c, h.guess) == (h.strikes, h.balls)
                        ]
                        residuals.append(len(candidates))

                prev = initial_candidates
                per_hint_bits = []
                for r in residuals:
                    if r <= 0 or prev <= 0:
                        per_hint_bits.append(0.0)
                    else:
                        per_hint_bits.append(math.log2(prev / r))
                    prev = r
                total_deduction_bits = math.log2(initial_candidates) if initial_candidates > 0 else 0.0
                min_per_hint_bits = min(per_hint_bits) if per_hint_bits else 0.0
                ball_heavy_ratio = (
                    sum(1 for h in hints if h.balls >= 2 and h.strikes <= 1) / len(hints)
                    if hints else 0.0
                )
                late_resolution_index = next(
                    (i + 1 for i, r in enumerate(residuals) if r == 1),
                    len(residuals),
                )
                residual_drop_variance = statistics.pvariance(per_hint_bits) if len(per_hint_bits) > 1 else 0.0

                if ball_heavy_ratio < cfg.get("min_ball_heavy_ratio", 0.0):
                    continue

                return {
                    "difficulty": difficulty.name.lower(),
                    "num_digits": num_digits,
                    "hints": [hint.to_dict() for hint in hints],
                    "hint_text": self._format_hints(hints),
                    "answer": solutions[0],
                    "problem_text": self._create_problem_text(num_digits, hints),
                    "step_metrics": {
                        "initial_candidates": initial_candidates,
                        "residuals": residuals,
                        "per_hint_bits": per_hint_bits,
                        "total_deduction_bits": total_deduction_bits,
                        "min_per_hint_bits": min_per_hint_bits,
                        "hint_count": len(hints),
                        "ball_heavy_ratio": ball_heavy_ratio,
                        "late_resolution_index": late_resolution_index,
                        "residual_drop_variance": residual_drop_variance,
                    },
                }

        raise RuntimeError(
            f"{max_retries}번 재시도 후에도 정확히 1개의 해를 가진 "
            f"{difficulty.name} 난이도 퍼즐을 생성하지 못했습니다"
        )

    def _format_hints(self, hints: List[Hint]) -> List[str]:
        return [str(hint) for hint in hints]

    def _create_problem_text(self, num_digits: int, hints: List[Hint]) -> str:
        hint_strs = [f"[{hint.guess}: {hint.strikes}S {hint.balls}B]" for hint in hints]
        hints_text = ", ".join(hint_strs)
        return (
            f"다음 모든 힌트를 만족하는, 각 자릿수가 서로 다른 "
            f"{num_digits}자리 숫자를 찾으세요: {hints_text}"
        )


SFT_SOLUTION_RUBRIC_KO = (
    "STEP0=문제 메타 · STEP1=주어진 조건 · STEP2=풀이 전개 · STEP3=답·검산"
)


def _build_baseball_solution_ko(problem: Dict) -> str:
    """SFT teacher trace: 숫자 야구 · 힌트별 후보 축소 SEG."""
    num_digits = problem['num_digits']
    hints = problem['hints']
    answer = problem['answer']
    metrics = problem.get('step_metrics', {})
    initial = metrics.get('initial_candidates', 0)
    residuals = metrics.get('residuals', [])
    per_bits = metrics.get('per_hint_bits', [])

    lines: List[str] = [
        SFT_SOLUTION_RUBRIC_KO,
        "[STEP 0] 문제 메타",
        f"  - 난이도: {problem.get('difficulty', '')}",
        f"  - 자릿수: {num_digits} (서로 다른 숫자)",
        f"  - 힌트 수: {len(hints)} · 초기 후보: {initial}",
        "  - 최종 답은 [STEP 3]에서 확정",
        "[STEP 1] 주어진 조건",
        "  - 규칙: 각 자리 숫자는 서로 다름(0–9).",
        "  - S(스트라이크) = 숫자·위치 모두 일치, B(볼) = 숫자만 일치.",
    ]
    for i, h in enumerate(hints, 1):
        lines.append(
            f"  {i}. 추측 {h['guess']} → {h['strikes']}S {h['balls']}B"
        )

    lines.append("[STEP 2] 풀이 전개")
    lines.append(
        f"  · 요약: 각 힌트(S/B)로 후보 공간 축소 · 초기 {initial} → "
        f"최종 1 · SEG {len(hints)}개"
    )
    prev = initial
    for i, h in enumerate(hints, 1):
        resid = residuals[i - 1] if i - 1 < len(residuals) else None
        bits = per_bits[i - 1] if i - 1 < len(per_bits) else None
        info_parts = []
        if resid is not None:
            info_parts.append(f"후보 {prev}→{resid}")
            prev = resid
        if bits is not None:
            info_parts.append(f"정보량 {bits:.2f} bits")
        info_text = " · ".join(info_parts) if info_parts else ""
        lines.append(
            f"    [SEG {i}] 힌트 {i} 반영: {h['guess']} → {h['strikes']}S {h['balls']}B · "
            f"{info_text}"
        )

    lines.extend([
        "[STEP 3] 답·검산",
        f"  - 최종 답: {answer}",
        "  - 각 힌트에 대해 정답과 S/B를 재계산하여 모두 일치하는지 확인.",
    ])
    return "\n".join(lines)


# ============================================================
# 질문 포맷팅
# ============================================================

def create_question(problem: Dict) -> str:
    num_digits = problem['num_digits']
    hints = problem['hints']

    hints_text = "\n".join([
        f"  {i+1}. 추측: {h['guess']} -> {h['strikes']} 스트라이크(S), {h['balls']} 볼(B)"
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

        solutions = game.find_all_solutions(hints, max_count=2)
        if len(solutions) == 0:
            return False, "주어진 힌트를 만족하는 해가 존재하지 않습니다"
        elif len(solutions) > 1:
            return False, "여러 개의 해가 존재합니다"
        elif solutions[0] != answer:
            return False, f"해 {solutions[0]}가 정답 {answer}과 일치하지 않습니다"

        return True, "유일한 해를 가진 유효한 문제입니다"

    except Exception as e:
        return False, f"검증 오류: {str(e)}"


# ============================================================
# 데이터셋 생성
# ============================================================

def create_dataset_files(num_questions: int):
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

        diff_success = 0
        for j in range(count):
            try:
                problem = generator.generate_problem(difficulty)
                is_valid, msg = validate_problem(problem)

                if is_valid:
                    puzzle_data = {
                        'id': f'number_baseball_ko_{diff_name}_{diff_success:04d}',
                        'question': create_question(problem),
                        'answer': problem['answer'],
                        'solution': _build_baseball_solution_ko(problem),
                        'difficulty': diff_name,
                        'num_digits': problem['num_digits'],
                        'hints': problem['hints'],
                        'hint_text': problem['hint_text'],
                        'problem_text': problem['problem_text'],
                        'step_metrics': problem.get('step_metrics', {}),
                    }
                    all_puzzles.append(puzzle_data)
                    diff_success += 1
                    print(f"  [{j+1}/{count}] 자릿수={problem['num_digits']}, "
                          f"힌트={len(problem['hints'])}개, 정답={problem['answer']}")
                else:
                    print(f"  [{j+1}/{count}] 검증 실패: {msg}")
            except RuntimeError as e:
                print(f"  [{j+1}/{count}] 실패: {e}")

    print(f"\n총 {len(all_puzzles)}개의 퍼즐이 생성되었습니다")

    df = pd.DataFrame(all_puzzles)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "number_baseball_ko.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV 파일 생성 완료: {csv_path}")

    # JSONL
    json_dir = PROJECT_ROOT / "data" / "jsonl"
    json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = json_dir / "number_baseball_ko.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_puzzles:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL 파일 생성 완료: {jsonl_path}")

    return df, all_puzzles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="숫자 야구 퍼즐 생성기 (한국어)")
    parser.add_argument("--num", type=int, default=12, help="생성할 문제 수")

    args = parser.parse_args()

    print("=" * 60)
    print("숫자 야구 퍼즐 생성기")
    print("=" * 60)

    create_dataset_files(num_questions=args.num)
