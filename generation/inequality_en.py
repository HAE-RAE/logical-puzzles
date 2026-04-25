"""Inequality Puzzle Generator (EN)

Constructive generation: builds puzzles by progressively adding hints until
solution count reaches 1, then greedily hides inequalities while preserving
visible uniqueness. Ported from logical-puzzles-me/inequality/generator.py:
- _stats instrumentation for solver-step counting
- domain-minimization variable ordering in backtracking
- greedy hidden-inequality selection maximizing visible_solver_steps
- dual-track uniqueness (full + visible constraints)
- difficulty configs with min_visible_solver_steps
- step_metrics exported in puzzle JSONL
"""

import random
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


class Difficulty(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3


@dataclass
class InequalityPuzzle:
    size: int
    inequalities: List[str]
    given_numbers: Dict[int, int]
    solution: List[int]
    difficulty: Difficulty
    hidden_inequalities: set = field(default_factory=set)
    step_metrics: dict = field(default_factory=dict)

    def to_problem_string(self) -> str:
        parts = []
        for i in range(self.size):
            if i in self.given_numbers:
                parts.append(str(self.given_numbers[i]))
            else:
                parts.append("_")
            if i < len(self.inequalities):
                if i in self.hidden_inequalities:
                    parts.append("?")
                else:
                    parts.append(self.inequalities[i])
        return " ".join(parts)

    def get_answer_string(self) -> str:
        """size<=9: concatenated digits; size>9: space-separated."""
        if self.size > 9:
            return " ".join(map(str, self.solution))
        return "".join(map(str, self.solution))


DIFFICULTY_CONFIGS: Dict[str, Dict] = {
    "easy": {
        # v2 recalibration: very small permutations + mostly-visible inequalities.
        "size_range": (5, 6),
        "hint_ratio": 0.0,
        "min_hints": 1,
        "ineq_reveal": 0.70,
        "min_visible_solver_steps": 4,
        "max_retries": 800,
    },
    "medium": {
        # v3 recalibration: size bumped to (12,14) to harden for frontier models.
        # reveal 0.35 retained (0.30 exhausted retries in pretest).
        "size_range": (12, 14),
        "hint_ratio": 0.0,
        "min_hints": 1,
        "ineq_reveal": 0.35,
        "min_visible_solver_steps": 30,
        "max_retries": 1000,
    },
    "hard": {
        # v3 recalibration: reveal 0.30 caused 4000-retry exhaustion in
        # production (tail-risk seed regions). Reverted to 0.34; stiffened
        # only via min_visible_solver_steps (55 → 60).
        "size_range": (13, 15),
        "hint_ratio": 0.0,
        "min_hints": 1,
        "ineq_reveal": 0.34,
        "min_visible_solver_steps": 60,
        "max_retries": 4000,
    },
}


class InequalityPuzzleGenerator:
    MAX_SOLUTIONS = 1

    def __init__(self):
        self.difficulty_config = {
            Difficulty.EASY: DIFFICULTY_CONFIGS["easy"],
            Difficulty.MEDIUM: DIFFICULTY_CONFIGS["medium"],
            Difficulty.HARD: DIFFICULTY_CONFIGS["hard"],
        }

    def _find_solutions(
        self,
        size: int,
        inequalities: List[str],
        given_numbers: Dict[int, int],
        max_count: int = 0,
        _stats: Optional[Dict] = None,
    ) -> List[List[int]]:
        """Backtracking solver with domain-minimization variable ordering.

        If _stats dict is provided, _stats['nodes'] is incremented per backtrack
        call (step-count proxy).
        """
        solutions: List[List[int]] = []
        assignment = [0] * size
        used = [False] * (size + 1)

        for pos, val in given_numbers.items():
            assignment[pos] = val
            used[val] = True

        unfixed = [i for i in range(size) if i not in given_numbers]

        def domain_values(pos: int):
            values = []
            for val in range(1, size + 1):
                if used[val]:
                    continue
                if pos > 0 and assignment[pos - 1] != 0:
                    ineq = inequalities[pos - 1]
                    if ineq == "<" and assignment[pos - 1] >= val:
                        continue
                    if ineq == ">" and assignment[pos - 1] <= val:
                        continue
                if pos < size - 1 and assignment[pos + 1] != 0:
                    ineq = inequalities[pos]
                    if ineq == "<" and val >= assignment[pos + 1]:
                        continue
                    if ineq == ">" and val <= assignment[pos + 1]:
                        continue
                values.append(val)
            return values

        def choose_next_pos():
            best_pos = None
            best_domain = None
            best_constraint_score = None
            for pos in unfixed:
                if assignment[pos] != 0:
                    continue
                values = domain_values(pos)
                if not values:
                    return pos, []
                constraint_score = 0
                if pos > 0 and assignment[pos - 1] != 0:
                    constraint_score += 1
                if pos < size - 1 and assignment[pos + 1] != 0:
                    constraint_score += 1
                if (
                    best_pos is None
                    or len(values) < len(best_domain)
                    or (len(values) == len(best_domain) and constraint_score > best_constraint_score)
                ):
                    best_pos = pos
                    best_domain = values
                    best_constraint_score = constraint_score
            return best_pos, best_domain

        def backtrack(idx: int):
            if _stats is not None:
                _stats['nodes'] = _stats.get('nodes', 0) + 1
            if max_count > 0 and len(solutions) >= max_count:
                return
            if idx == len(unfixed):
                solutions.append(list(assignment))
                return
            pos, values = choose_next_pos()
            if pos is None:
                solutions.append(list(assignment))
                return
            if not values:
                return
            for val in values:
                assignment[pos] = val
                used[val] = True
                backtrack(idx + 1)
                assignment[pos] = 0
                used[val] = False
                if max_count > 0 and len(solutions) >= max_count:
                    return

        backtrack(0)
        return solutions

    def _find_best_hint(
        self,
        size: int,
        inequalities: List[str],
        given_numbers: Dict[int, int],
        base_solution: List[int],
        current_solutions: List[List[int]],
    ) -> Optional[int]:
        available = [i for i in range(size) if i not in given_numbers]
        if not available:
            return None
        best_pos = None
        best_count = float('inf')
        for pos in available:
            test_hints = dict(given_numbers)
            test_hints[pos] = base_solution[pos]
            new_solutions = self._find_solutions(size, inequalities, test_hints, max_count=2)
            if len(new_solutions) == 1:
                return pos
            if len(new_solutions) >= 1 and len(new_solutions) < best_count:
                best_count = len(new_solutions)
                best_pos = pos
        return best_pos

    def _minimize_hints_for_visible_uniqueness(
        self,
        size: int,
        visible_inequalities: List[str],
        given_numbers: Dict[int, int],
        min_hints: int,
    ) -> Dict[int, int]:
        """Greedily drop redundant hints while keeping visible uniqueness."""
        if len(given_numbers) <= min_hints:
            return dict(given_numbers)
        hints = dict(given_numbers)
        positions = list(hints.keys())
        random.shuffle(positions)
        for pos in positions:
            if len(hints) <= min_hints:
                break
            test_hints = dict(hints)
            del test_hints[pos]
            visible_solutions = self._find_solutions(
                size, visible_inequalities, test_hints, max_count=2
            )
            if len(visible_solutions) == 1:
                hints = test_hints
        return hints

    def _select_hidden_inequalities(
        self,
        size: int,
        inequalities: List[str],
        given_numbers: Dict[int, int],
        num_to_hide: int,
    ) -> Optional[set]:
        """Greedy hiding: pick the index that maximizes visible solver steps
        while preserving visible uniqueness. Returns None if no feasible choice.
        """
        hidden = set()
        total_ineqs = len(inequalities)
        for _ in range(num_to_hide):
            best_idx = None
            best_nodes = -1
            candidates = list(range(total_ineqs))
            random.shuffle(candidates)
            for idx in candidates:
                if idx in hidden:
                    continue
                trial_hidden = hidden | {idx}
                visible_ineqs = [
                    '?' if i in trial_hidden else ineq
                    for i, ineq in enumerate(inequalities)
                ]
                stats = {'nodes': 0}
                visible_solutions = self._find_solutions(
                    size, visible_ineqs, given_numbers, max_count=2, _stats=stats
                )
                if len(visible_solutions) == 1 and stats['nodes'] > best_nodes:
                    best_idx = idx
                    best_nodes = stats['nodes']
            if best_idx is None:
                return None
            hidden.add(best_idx)
        return hidden

    def generate_puzzle(self, difficulty: Difficulty, max_retries: int = 800) -> InequalityPuzzle:
        config = self.difficulty_config[difficulty]
        effective_max_retries = config.get("max_retries", max_retries)

        for retry in range(effective_max_retries):
            size = random.randint(*config["size_range"])

            base_solution = list(range(1, size + 1))
            random.shuffle(base_solution)

            inequalities = []
            for i in range(size - 1):
                if base_solution[i] < base_solution[i + 1]:
                    inequalities.append("<")
                else:
                    inequalities.append(">")

            num_hints = max(config["min_hints"], int(size * config["hint_ratio"]))
            given_numbers = self._select_initial_hints(base_solution, num_hints)

            solutions = self._find_solutions(size, inequalities, given_numbers, self.MAX_SOLUTIONS + 1)

            while len(solutions) > self.MAX_SOLUTIONS:
                best_pos = self._find_best_hint(size, inequalities, given_numbers,
                                                base_solution, solutions)
                if best_pos is None:
                    break
                given_numbers[best_pos] = base_solution[best_pos]
                solutions = self._find_solutions(size, inequalities, given_numbers, self.MAX_SOLUTIONS + 1)

            if len(solutions) != 1:
                continue

            ineq_reveal = config.get("ineq_reveal", 1.0)
            total_ineqs = size - 1
            num_to_hide = int(total_ineqs * (1.0 - ineq_reveal))

            hidden_indices = set()
            if num_to_hide > 0:
                selected = self._select_hidden_inequalities(
                    size, inequalities, given_numbers, num_to_hide
                )
                if selected is None:
                    continue
                hidden_indices = selected

            visible_ineqs_final = [
                '?' if i in hidden_indices else ineq
                for i, ineq in enumerate(inequalities)
            ]

            minimized_hints = self._minimize_hints_for_visible_uniqueness(
                size,
                visible_ineqs_final,
                given_numbers,
                min_hints=config["min_hints"],
            )
            visible_solutions = self._find_solutions(
                size, visible_ineqs_final, minimized_hints, max_count=2
            )
            if len(visible_solutions) != 1:
                continue

            full_stats = {'nodes': 0}
            self._find_solutions(size, inequalities, minimized_hints,
                                 max_count=2, _stats=full_stats)
            visible_stats = {'nodes': 0}
            self._find_solutions(size, visible_ineqs_final, minimized_hints,
                                 max_count=2, _stats=visible_stats)
            if visible_stats['nodes'] < config.get("min_visible_solver_steps", 0):
                continue

            return InequalityPuzzle(
                size=size,
                inequalities=inequalities,
                given_numbers=minimized_hints,
                solution=solutions[0],
                difficulty=difficulty,
                hidden_inequalities=hidden_indices,
                step_metrics={
                    'solver_steps': full_stats['nodes'],
                    'visible_solver_steps': visible_stats['nodes'],
                    'size': size,
                    'hidden_count': len(hidden_indices),
                    'hint_count': len(minimized_hints),
                },
            )

        raise RuntimeError(
            f"Failed to generate {difficulty.name} puzzle with exactly 1 solution "
            f"and configured ineq_reveal after {effective_max_retries} retries"
        )

    def _select_initial_hints(self, solution: List[int], num_hints: int) -> Dict[int, int]:
        given_numbers: Dict[int, int] = {}
        size = len(solution)
        if num_hints == 0:
            return given_numbers
        extreme_positions = [i for i, val in enumerate(solution) if val == 1 or val == size]
        random.shuffle(extreme_positions)
        for pos in extreme_positions:
            if len(given_numbers) >= num_hints:
                break
            given_numbers[pos] = solution[pos]
        remaining = [i for i in range(size) if i not in given_numbers]
        random.shuffle(remaining)
        for pos in remaining:
            if len(given_numbers) >= num_hints:
                break
            given_numbers[pos] = solution[pos]
        return given_numbers

    def solve_puzzle(self, size: int, inequalities: List[str],
                     given_numbers: Dict[int, int], max_count: int = 0) -> List[List[int]]:
        return self._find_solutions(size, inequalities, given_numbers, max_count)


def create_question(puzzle: InequalityPuzzle) -> str:
    """Create question text in English."""
    problem_str = puzzle.to_problem_string()
    has_hidden = len(puzzle.hidden_inequalities) > 0

    hidden_rule = ""
    if has_hidden:
        hidden_rule = "\n- '?' means the inequality is unknown (could be < or >)"

    if puzzle.size > 9:
        answer_format = f"Provide your answer as {puzzle.size} numbers separated by spaces."
        example = " ".join(str(i) for i in range(1, puzzle.size + 1))
        answer_example = f"Answer: {example}"
    else:
        answer_format = f"Provide your answer as a sequence of {puzzle.size} digits (no spaces)."
        example = "".join(str(i) for i in range(1, puzzle.size + 1))
        answer_example = f"Answer: {example}"

    question = f"""Solve this inequality puzzle. Fill in the blanks with numbers from 1 to {puzzle.size}.

Each number from 1 to {puzzle.size} must be used exactly once.
The inequality symbols (< or >) between positions must be satisfied.

Puzzle: {problem_str}

Rules:
- '_' represents an empty position to fill
- '<' means the left number is smaller than the right number
- '>' means the left number is larger than the right number
- Each number 1 to {puzzle.size} appears exactly once{hidden_rule}

{answer_format}

Example format:
{answer_example}"""
    return question


SFT_SOLUTION_RUBRIC_EN = (
    "STEP0=meta · STEP1=given · STEP2=worked solution · "
    "STEP3=answer and verification"
)


def _build_inequality_solution_en(puzzle: InequalityPuzzle) -> str:
    """SFT teacher trace: inequality puzzle with SEG-per-constraint."""
    size = puzzle.size
    solution = puzzle.solution
    ineqs = puzzle.inequalities
    givens = puzzle.given_numbers
    hidden = puzzle.hidden_inequalities
    problem_str = puzzle.to_problem_string()
    ans_str = puzzle.get_answer_string()
    visible_cnt = len(ineqs) - len(hidden)

    lines: List[str] = [
        SFT_SOLUTION_RUBRIC_EN,
        "[STEP 0] Problem meta",
        f"  - Difficulty: {puzzle.difficulty.name.lower()}",
        f"  - Grid size: {size} (permutation of 1..{size})",
        f"  - Givens: {len(givens)} · visible inequalities: {visible_cnt} · hidden: {len(hidden)}",
        "  - Final answer is confirmed in [STEP 3]",
        "[STEP 1] Given",
        f"  - Puzzle: {problem_str}",
        f"  - Givens (pos:value): {', '.join(f'{p}:{v}' for p, v in sorted(givens.items())) or '(none)'}",
    ]

    lines.append("[STEP 2] Worked solution")
    lines.append(
        f"  · Summary: permutation 1..{size} + hint/inequality propagation -> unique model · "
        f"{len(ineqs)} inequalities (visible {visible_cnt} / hidden {len(hidden)}) · "
        f"{len(ineqs)} SEGs"
    )
    lines.append(f"  · Solution vector: [{', '.join(str(v) for v in solution)}]")
    for i, op in enumerate(ineqs):
        left, right = solution[i], solution[i + 1]
        hidden_flag = "hidden" if i in hidden else "visible"
        if op == "<":
            ok = left < right
        elif op == ">":
            ok = left > right
        else:
            ok = None
        status = "holds" if ok else ("fails" if ok is False else "check")
        lines.append(
            f"    [SEG {i + 1}] positions {i}<->{i + 1} ({hidden_flag}): {left} {op} {right} -> {status}"
        )

    lines.extend([
        "[STEP 3] Answer and verification",
        f"  - Final answer: {ans_str}",
        f"  - Each of 1..{size} used exactly once: "
        f"{'OK' if sorted(solution) == list(range(1, size + 1)) else 'FAIL'}",
        "  - Givens match and every inequality (visible + hidden) holds as shown in the [SEG] trace.",
    ])
    return "\n".join(lines)


def create_dataset_files(num_questions: int):
    """Create inequality puzzle dataset files (CSV + JSONL)."""
    import pandas as pd

    print(f"Generating {num_questions} inequality puzzles...")

    generator = InequalityPuzzleGenerator()

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
                puzzle = generator.generate_puzzle(difficulty)
                puzzle_data = {
                    "id": f"inequality_en_{diff_name}_{j:04d}",
                    "question": create_question(puzzle),
                    "answer": puzzle.get_answer_string(),
                    "solution": _build_inequality_solution_en(puzzle),
                    "difficulty": diff_name,
                }
                all_puzzles.append(puzzle_data)
                print(f"  [{j+1}/{count}] size={puzzle.size}, answer={puzzle.get_answer_string()}, "
                      f"steps={puzzle.step_metrics.get('visible_solver_steps', 0)}")
            except RuntimeError as e:
                print(f"  [{j+1}/{count}] Failed: {e}")

    print(f"\nGenerated {len(all_puzzles)} puzzles")

    df = pd.DataFrame(all_puzzles)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "inequality_en.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV file created: {csv_path}")

    # JSONL
    json_dir = PROJECT_ROOT / "data" / "jsonl"
    json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = json_dir / "inequality_en.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_puzzles:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL file created: {jsonl_path}")

    return df, all_puzzles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inequality Puzzle Generator")
    parser.add_argument("--num", type=int, default=12, help="Number of questions to generate")

    args = parser.parse_args()

    print("=" * 50)
    print("Inequality Puzzle Generator")
    print("=" * 50)

    create_dataset_files(num_questions=args.num)
