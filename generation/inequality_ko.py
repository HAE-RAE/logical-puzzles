"""부등호 퍼즐 생성기 (KO)

구성적 생성: 해의 개수가 1이 될 때까지 힌트를 점진적으로 추가한 뒤,
visible 제약의 유일성을 유지하며 부등호를 탐욕적으로 숨깁니다.
logical-puzzles-me/inequality/generator.py에서 포팅:
- 솔버 단계 수 측정을 위한 _stats 계측
- 백트래킹의 domain-minimization 변수 선택
- visible_solver_steps를 최대화하는 탐욕적 숨김 부등호 선택
- dual-track 유일성(전체 제약 + visible 제약)
- min_visible_solver_steps 기반 난이도 설정
- step_metrics를 퍼즐 JSONL에 내보냄
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
        """size<=9: 숫자를 이어붙임; size>9: 공백으로 구분."""
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
        """domain-minimization 변수 선택을 사용하는 백트래킹 솔버.

        _stats dict이 주어지면 각 backtrack 호출마다 _stats['nodes']를 증가시킨다.
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
        """visible 유일성을 유지하면서 불필요한 힌트를 탐욕적으로 제거."""
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
        """탐욕적 숨김: visible 유일성을 유지하면서 visible 탐색 노드 수를 최대화하는
        인덱스를 선택한다. 선택 불가 시 None 반환.
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
            f"{effective_max_retries}번 재시도 후에도 정확히 1개의 해를 가진 "
            f"{difficulty.name} 퍼즐 생성에 실패했습니다"
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
    """한국어로 질문 텍스트를 생성합니다."""
    problem_str = puzzle.to_problem_string()
    has_hidden = len(puzzle.hidden_inequalities) > 0

    hidden_rule = ""
    if has_hidden:
        hidden_rule = "\n- '?'는 부등호가 알려지지 않았음을 의미합니다 (< 또는 > 가능)"

    if puzzle.size > 9:
        answer_format = f"답을 {puzzle.size}개의 숫자로 공백으로 구분하여 제출하세요."
        example = " ".join(str(i) for i in range(1, puzzle.size + 1))
        answer_example = f"Answer: {example}"
    else:
        answer_format = f"답을 {puzzle.size}자리 숫자열로 제출하세요 (공백 없이)."
        example = "".join(str(i) for i in range(1, puzzle.size + 1))
        answer_example = f"Answer: {example}"

    question = f"""다음 부등호 퍼즐을 풀어주세요. 빈칸을 채우세요. 1부터 {puzzle.size}까지의 숫자를 사용합니다.

1부터 {puzzle.size}까지의 각 숫자는 정확히 한 번만 사용해야 합니다.
위치 사이의 부등호 기호 (< 또는 >)를 만족해야 합니다.

퍼즐: {problem_str}

규칙:
- '_'는 채워야 할 빈 위치를 나타냅니다
- '<'는 왼쪽 숫자가 오른쪽 숫자보다 작음을 의미합니다
- '>'는 왼쪽 숫자가 오른쪽 숫자보다 큼을 의미합니다
- 1부터 {puzzle.size}까지의 각 숫자는 정확히 한 번만 사용됩니다{hidden_rule}

{answer_format}

답 형식 예시:
{answer_example}"""
    return question


def create_dataset_files(num_questions: int):
    """부등호 퍼즐 데이터셋 파일(CSV + JSONL)을 생성합니다."""
    import pandas as pd

    print(f"부등호 퍼즐 {num_questions}개 생성 중...")

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

        print(f"\n=== {diff_name} 퍼즐 생성 중 ({count}개 필요) ===")

        for j in range(count):
            try:
                puzzle = generator.generate_puzzle(difficulty)
                puzzle_data = {
                    "id": f"inequality_ko_{len(all_puzzles)}",
                    "question": create_question(puzzle),
                    "answer": puzzle.get_answer_string(),
                    "solution": puzzle.to_problem_string(),
                    "difficulty": diff_name,
                    "size": puzzle.size,
                    "given_positions": list(puzzle.given_numbers.keys()),
                    "given_values": list(puzzle.given_numbers.values()),
                    "problem": puzzle.to_problem_string(),
                    "inequalities": list(puzzle.inequalities),
                    "hidden_inequalities": sorted(puzzle.hidden_inequalities),
                    "step_metrics": dict(puzzle.step_metrics),
                }
                all_puzzles.append(puzzle_data)
                print(f"  [{j+1}/{count}] size={puzzle.size}, 정답={puzzle.get_answer_string()}, "
                      f"steps={puzzle.step_metrics.get('visible_solver_steps', 0)}")
            except RuntimeError as e:
                print(f"  [{j+1}/{count}] 실패: {e}")

    print(f"\n총 {len(all_puzzles)}개 퍼즐 생성 완료")

    df = pd.DataFrame(all_puzzles)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "inequality_ko.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV 파일 생성 완료: {csv_path}")

    json_dir = PROJECT_ROOT / "data" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = json_dir / "inequality_ko.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_puzzles:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL 파일 생성 완료: {jsonl_path}")

    return df, all_puzzles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="부등호 퍼즐 생성기")
    parser.add_argument("--num", type=int, default=12, help="생성할 질문의 수")

    args = parser.parse_args()

    print("=" * 50)
    print("부등호 퍼즐 생성기")
    print("=" * 50)

    create_dataset_files(num_questions=args.num)
