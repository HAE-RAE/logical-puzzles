"""Number Baseball (Bulls and Cows) Puzzle Generator (EN)

Constructive generation: builds puzzles by selecting high-information
hints that progressively narrow solutions to exactly 1.

Ported from logical-puzzles-me/number_baseball/generator.py:
- Permutation-based candidate pool for ball-heavy hints
- 2-step lookahead scoring for medium/hard
- Hard-specific ball-heavy chain strategy
- Strict uniqueness (MAX_SOLUTIONS = 1) for all difficulties
- step_metrics exported in puzzle JSONL
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


MAX_SOLUTIONS = 1  # Only allow exactly 1 solution for ALL difficulties


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
            raise ValueError("Number of digits must be 3, 4, 5, or 6")
        self.num_digits = num_digits

    def generate_number(self) -> str:
        digits = list(range(10))
        random.shuffle(digits)
        return ''.join(str(d) for d in digits[:self.num_digits])

    def calculate_strikes_balls(self, secret: str, guess: str) -> Tuple[int, int]:
        if len(secret) != len(guess):
            raise ValueError("Secret and guess must have the same length")
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
    """Constructive puzzle generator for Bulls and Cows."""

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
        """Build candidate hint pool enriched with permutations of the secret
        digits (yields more low-strike/high-ball hints)."""
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
        """2-step lookahead: after applying `candidate`, find the best possible
        residual from a next hint. Used for medium/hard."""
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
        """Select best next hint given difficulty profile."""
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
        """Hard strategy: ball-heavy chain early, resolve uniqueness at end."""
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
        """Constructively generate a puzzle with exactly 1 solution."""
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
            f"Failed to generate {difficulty.name} puzzle with exactly 1 solution "
            f"after {max_retries} retries"
        )

    def _format_hints(self, hints: List[Hint]) -> List[str]:
        return [str(hint) for hint in hints]

    def _create_problem_text(self, num_digits: int, hints: List[Hint]) -> str:
        hint_strs = [f"[{hint.guess}: {hint.strikes}S {hint.balls}B]" for hint in hints]
        hints_text = ", ".join(hint_strs)
        return (
            f"Find the {num_digits}-digit number with distinct digits that satisfies "
            f"all the following hints: {hints_text}"
        )


# ============================================================
# Question formatting
# ============================================================

def create_question(problem: Dict) -> str:
    num_digits = problem['num_digits']
    hints = problem['hints']

    hints_text = "\n".join([
        f"  {i+1}. Guess: {h['guess']} -> {h['strikes']} Strike(s), {h['balls']} Ball(s)"
        for i, h in enumerate(hints)
    ])

    question = f"""Solve this Number Baseball (Bulls and Cows) puzzle.

Rules:
- The secret number has {num_digits} digits, each digit is unique (0-9)
- "Strike" means a digit is correct AND in the correct position
- "Ball" means a digit is correct BUT in the wrong position
- Your task: Find the secret number that satisfies ALL hints

Hints:
{hints_text}

Think step by step and find the unique {num_digits}-digit secret number.

Provide your answer in this format:
Answer: [the {num_digits}-digit secret number]"""

    return question


def validate_problem(problem: Dict) -> Tuple[bool, str]:
    try:
        num_digits = problem['num_digits']
        game = BullsAndCows(num_digits)

        hints = [Hint(h['guess'], h['strikes'], h['balls']) for h in problem['hints']]

        answer = problem['answer']
        if len(answer) != num_digits:
            return False, f"Answer length {len(answer)} doesn't match num_digits {num_digits}"

        if len(set(answer)) != num_digits:
            return False, f"Answer {answer} doesn't have unique digits"

        if not game.check_number_against_hints(answer, hints):
            return False, f"Answer {answer} doesn't satisfy all hints"

        solutions = game.find_all_solutions(hints, max_count=2)
        if len(solutions) == 0:
            return False, "No solution exists for the given hints"
        elif len(solutions) > 1:
            return False, f"Multiple solutions exist"
        elif solutions[0] != answer:
            return False, f"Solution {solutions[0]} doesn't match answer {answer}"

        return True, "Problem is valid with unique solution"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


# ============================================================
# Dataset generation
# ============================================================

def create_dataset_files(num_questions: int):
    import pandas as pd

    print(f"Generating {num_questions} number baseball puzzles...")

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

        print(f"\n=== Generating {diff_name} puzzles ({count} needed) ===")

        for j in range(count):
            try:
                problem = generator.generate_problem(difficulty)
                is_valid, msg = validate_problem(problem)

                if is_valid:
                    puzzle_data = {
                        'id': f'number_baseball_en_{len(all_puzzles)}',
                        'question': create_question(problem),
                        'answer': problem['answer'],
                        'solution': problem['problem_text'],
                        'difficulty': diff_name,
                        'num_digits': problem['num_digits'],
                        'hints': problem['hints'],
                        'hint_text': problem['hint_text'],
                        'problem_text': problem['problem_text'],
                        'step_metrics': problem.get('step_metrics', {}),
                    }
                    all_puzzles.append(puzzle_data)
                    print(f"  [{j+1}/{count}] digits={problem['num_digits']}, "
                          f"hints={len(problem['hints'])}, answer={problem['answer']}")
                else:
                    print(f"  [{j+1}/{count}] Validation failed: {msg}")
            except RuntimeError as e:
                print(f"  [{j+1}/{count}] Failed: {e}")

    print(f"\nGenerated {len(all_puzzles)} puzzles")

    df = pd.DataFrame(all_puzzles)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "number_baseball_en.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV file created: {csv_path}")

    json_dir = PROJECT_ROOT / "data" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = json_dir / "number_baseball_en.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_puzzles:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL file created: {jsonl_path}")

    return df, all_puzzles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Number Baseball Puzzle Generator (EN)")
    parser.add_argument("--num", type=int, default=12, help="Number of questions to generate")

    args = parser.parse_args()

    print("=" * 60)
    print("Number Baseball (Bulls and Cows) Puzzle Generator")
    print("=" * 60)

    create_dataset_files(num_questions=args.num)
