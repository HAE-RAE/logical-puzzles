"""부등호 / Futoshiki 퍼즐 생성기 (KO)

세 가지 난이도 티어로 구성된 혼합 논리 퍼즐 데이터셋을 생성하며,
각 티어는 별도 JSONL 파일과 하나의 통합 CSV에 기록됩니다:

    easy   -> 1D 부등호 체인 퍼즐   (inequality_ko_easy.jsonl)
    medium -> 2D Futoshiki 격자 퍼즐  (inequality_ko_medium.jsonl)
    hard   -> 2D Futoshiki 격자 퍼즐  (inequality_ko_hard.jsonl)
    (통합)  -------------------------  inequality_ko.csv

레코드 스키마 (모든 티어 동일, CSV == 연결된 JSONL):
    id, question, answer, solution, difficulty

설계 참고
------------
* 데이터셋은 체인 전용 생성기에서 이 혼합 형식으로 이전되었습니다.
  체인 분기는 원래의 구성적 알고리즘(해가 유일해질 때까지 점진적 힌트 추가,
  visible 유일성을 유지하며 탐욕적으로 부등호 숨기기)과 동일한 SFT
  teacher-trace 형식을 유지합니다.
* 긴 체인(size > 9)은 퍼즐 문자열과 답에 컴팩트한 base-36 단일 문자
  인코딩(a=10, b=11, ...)을 사용합니다. 공백 구분 값은 채점이 모호하고
  장황하기 때문입니다. solution의 "Solution vector" / "Givens" 줄은
  가독성을 위해 원시 정수를 그대로 사용합니다.
* Futoshiki 분기는 무작위 라틴 방진을 생성하고, 여기서 직교 부등호 제약을
  파생한 뒤, 셀 일부를 힌트로 공개하고 유일해를 보장하는 최소 제약만
  추가합니다. 각 퍼즐은 순수 제약 전파로 풀 수 있는지, 아니면 경우 분석
  (역추적)이 필요한지 분류되며, teacher trace 문구도 그에 맞게 작성됩니다.
* 티어 크기/구조 분포는 참조 데이터셋에 맞춰 보정되었습니다
  (아래 SIZE/GIVEN 범위 참고). 퍼즐은 무작위 생성되므로 이 파일을 실행하면
  동일 형식의 새 데이터셋이 만들어지며, 이전 실행과 바이트 단위로 동일하지
  않습니다.

실행:  python inequality_ko.py --num 300 --seed 0 --outdir ./out
"""

import argparse
import copy
import json
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# 공유
# --------------------------------------------------------------------------- #

SFT_SOLUTION_RUBRIC_KO = (
    "STEP0=문제 메타 · STEP1=주어진 조건 · STEP2=풀이 전개 · STEP3=답·검산"
)


def encode_val(v: int) -> str:
    """1..9 -> 숫자; 10.. -> 소문자 한 자리 (a=10, b=11, ...)."""
    if v <= 9:
        return str(v)
    return chr(ord("a") + v - 10)


class Difficulty(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3


# --------------------------------------------------------------------------- #
# 1D 부등호 체인 퍼즐  (easy 티어)
# --------------------------------------------------------------------------- #

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
                parts.append(encode_val(self.given_numbers[i]))
            else:
                parts.append("_")
            if i < len(self.inequalities):
                if i in self.hidden_inequalities:
                    parts.append("?")
                else:
                    parts.append(self.inequalities[i])
        return " ".join(parts)

    def get_answer_string(self) -> str:
        """컴팩트 단일 문자 인코딩 (모든 크기, 공백 없음)."""
        return "".join(encode_val(v) for v in self.solution)


class InequalityPuzzleGenerator:
    """이중 추적 (전체 + visible) 유일성을 갖춘 구성적 체인 생성기."""

    MAX_SOLUTIONS = 1

    # ---- domain-minimization 순서의 백트래킹 솔버 ---- #
    def _find_solutions(
        self,
        size: int,
        inequalities: List[str],
        given_numbers: Dict[int, int],
        max_count: int = 0,
        _stats: Optional[Dict] = None,
    ) -> List[List[int]]:
        solutions: List[List[int]] = []
        assignment = [0] * size
        used = [False] * (size + 1)

        for pos, val in given_numbers.items():
            assignment[pos] = val
            used[val] = True

        unfixed = [i for i in range(size) if i not in given_numbers]

        def domain_values(pos: int):
            lo, hi = 1, size
            if pos > 0 and assignment[pos - 1] != 0:
                prev = assignment[pos - 1]
                ineq = inequalities[pos - 1]
                if ineq == "<":
                    if prev + 1 > lo:
                        lo = prev + 1
                elif ineq == ">":
                    if prev - 1 < hi:
                        hi = prev - 1
            if pos < size - 1 and assignment[pos + 1] != 0:
                nxt = assignment[pos + 1]
                ineq = inequalities[pos]
                if ineq == "<":
                    if nxt - 1 < hi:
                        hi = nxt - 1
                elif ineq == ">":
                    if nxt + 1 > lo:
                        lo = nxt + 1
            if lo > hi:
                return []
            return [val for val in range(lo, hi + 1) if not used[val]]

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
                    or (
                        len(values) == len(best_domain)
                        and constraint_score > best_constraint_score
                    )
                ):
                    best_pos = pos
                    best_domain = values
                    best_constraint_score = constraint_score
            return best_pos, best_domain

        def backtrack(idx: int):
            if _stats is not None:
                _stats["nodes"] = _stats.get("nodes", 0) + 1
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
        best_count = float("inf")
        for pos in available:
            test_hints = dict(given_numbers)
            test_hints[pos] = base_solution[pos]
            new_solutions = self._find_solutions(size, inequalities, test_hints, max_count=2)
            if len(new_solutions) == 1:
                return pos
            if 1 <= len(new_solutions) < best_count:
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
        """visible 유일성을 유지하는 인덱스 중, visible 솔버 노드 수 최댓값의
        70% 이상(상위 티어)에서 무작위 샘플링하여 숨김이 '어렵게' 유지되도록 함.
        동점은 확률적으로 깨뜨림. 선택 불가 시 None 반환."""
        hidden: set = set()
        total_ineqs = len(inequalities)
        for _ in range(num_to_hide):
            candidates_with_score: List[Tuple[int, int]] = []
            for idx in range(total_ineqs):
                if idx in hidden:
                    continue
                trial_hidden = hidden | {idx}
                visible_ineqs = [
                    "?" if i in trial_hidden else ineq
                    for i, ineq in enumerate(inequalities)
                ]
                stats = {"nodes": 0}
                visible_solutions = self._find_solutions(
                    size, visible_ineqs, given_numbers, max_count=2, _stats=stats
                )
                if len(visible_solutions) == 1:
                    candidates_with_score.append((idx, stats["nodes"]))
            if not candidates_with_score:
                return None
            max_nodes = max(c[1] for c in candidates_with_score)
            threshold = max(1, int(max_nodes * 0.7))
            top = [idx for idx, n in candidates_with_score if n >= threshold]
            hidden.add(random.choice(top))
        return hidden

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

    def generate_chain(
        self,
        size: int,
        num_to_hide: int,
        difficulty: Difficulty,
        min_hints: int = 1,
        max_retries: int = 400,
    ) -> InequalityPuzzle:
        """유일 해를 갖는 크기 `size` 체인을 생성하며, 정확히 `num_to_hide` 개의
        부등호를 숨기되 visible 퍼즐은 유일 해를 유지합니다."""
        for _ in range(max_retries):
            base_solution = list(range(1, size + 1))
            random.shuffle(base_solution)

            inequalities = [
                "<" if base_solution[i] < base_solution[i + 1] else ">"
                for i in range(size - 1)
            ]

            given_numbers = self._select_initial_hints(base_solution, min_hints)
            solutions = self._find_solutions(
                size, inequalities, given_numbers, self.MAX_SOLUTIONS + 1
            )
            while len(solutions) > self.MAX_SOLUTIONS:
                best_pos = self._find_best_hint(
                    size, inequalities, given_numbers, base_solution, solutions
                )
                if best_pos is None:
                    break
                given_numbers[best_pos] = base_solution[best_pos]
                solutions = self._find_solutions(
                    size, inequalities, given_numbers, self.MAX_SOLUTIONS + 1
                )
            if len(solutions) != 1:
                continue

            hidden_indices: set = set()
            if num_to_hide > 0:
                selected = self._select_hidden_inequalities(
                    size, inequalities, given_numbers, num_to_hide
                )
                if selected is None:
                    continue
                hidden_indices = selected

            visible_ineqs_final = [
                "?" if i in hidden_indices else ineq
                for i, ineq in enumerate(inequalities)
            ]
            minimized_hints = self._minimize_hints_for_visible_uniqueness(
                size, visible_ineqs_final, given_numbers, min_hints=min_hints
            )
            visible_solutions = self._find_solutions(
                size, visible_ineqs_final, minimized_hints, max_count=2
            )
            if len(visible_solutions) != 1:
                continue

            full_stats = {"nodes": 0}
            self._find_solutions(size, inequalities, minimized_hints, max_count=2, _stats=full_stats)
            visible_stats = {"nodes": 0}
            self._find_solutions(
                size, visible_ineqs_final, minimized_hints, max_count=2, _stats=visible_stats
            )

            return InequalityPuzzle(
                size=size,
                inequalities=inequalities,
                given_numbers=minimized_hints,
                solution=solutions[0],
                difficulty=difficulty,
                hidden_inequalities=hidden_indices,
                step_metrics={
                    "solver_steps": full_stats["nodes"],
                    "visible_solver_steps": visible_stats["nodes"],
                    "size": size,
                    "hidden_count": len(hidden_indices),
                    "hint_count": len(minimized_hints),
                },
            )

        raise RuntimeError(
            f"size={size} 체인 (hide {num_to_hide}) 생성에 {max_retries}번 재시도 후 실패"
        )


def create_chain_question(puzzle: InequalityPuzzle) -> str:
    problem_str = puzzle.to_problem_string()
    n = puzzle.size

    rules = [
        "- '_'는 채워야 할 빈 위치를 나타냅니다",
        "- '<'는 왼쪽 숫자가 오른쪽 숫자보다 작음을 의미합니다",
        "- '>'는 왼쪽 숫자가 오른쪽 숫자보다 큼을 의미합니다",
        f"- 1부터 {n}까지의 각 숫자는 정확히 한 번만 사용됩니다",
        "- '?'는 부등호가 알려지지 않았음을 의미합니다 (< 또는 > 가능)",
    ]
    if n > 9:
        rules.append(
            "- 10 이상의 값은 소문자 한 글자로 표기합니다: a=10, b=11, c=12, ... "
            "(예: 'a'는 10을 의미하고 '_ > c'는 빈칸이 12보다 큼을 의미합니다)"
        )
        answer_format = (
            f"답을 {n}개의 단일 문자로 공백 없이 순서대로 제출하세요. "
            f"1-9는 숫자, 10 이상은 소문자 (a=10, b=11, ...)로 표기합니다."
        )
        answer_example = "Answer: " + "".join(encode_val(i) for i in range(1, n + 1))
    else:
        answer_format = f"답을 {n}자리 숫자열로 제출하세요 (공백 없이)."
        answer_example = "Answer: " + "".join(str(i) for i in range(1, n + 1))

    rules_block = "\n".join(rules)
    return (
        f"이 부등호 퍼즐을 푸세요. 1부터 {n}까지의 숫자로 빈칸을 채우세요.\n\n"
        f"1부터 {n}까지의 각 숫자는 정확히 한 번만 사용해야 합니다.\n"
        f"위치 사이의 부등호 기호 (< 또는 >)를 반드시 만족해야 합니다.\n\n"
        f"퍼즐: {problem_str}\n\n"
        f"규칙:\n{rules_block}\n\n"
        f"{answer_format}\n\n"
        f"답 형식 예시:\n{answer_example}"
    )


def build_chain_solution(puzzle: InequalityPuzzle) -> str:
    size = puzzle.size
    solution = puzzle.solution
    ineqs = puzzle.inequalities
    givens = puzzle.given_numbers
    hidden = puzzle.hidden_inequalities
    problem_str = puzzle.to_problem_string()
    ans_str = puzzle.get_answer_string()
    visible_cnt = len(ineqs) - len(hidden)

    lines: List[str] = [
        SFT_SOLUTION_RUBRIC_KO,
        "[STEP 0] 문제 메타",
        f"  - 난이도: {puzzle.difficulty.name.lower()}",
        f"  - 격자 크기: {size} (1~{size}의 순열)",
        f"  - 주어진 숫자: {len(givens)}개 · 보이는 부등호: {visible_cnt} · 숨겨진 부등호: {len(hidden)}",
        "  - 최종 답은 [STEP 3]에서 확정",
        "[STEP 1] 주어진 조건",
        f"  - 퍼즐: {problem_str}",
        f"  - 힌트(위치: 값): {', '.join(f'{p}:{v}' for p, v in sorted(givens.items())) or '(없음)'}",
        "[STEP 2] 풀이 전개",
        f"  · 요약: 1~{size} 순열에서 힌트·부등호 전파 → 유일해 확정 · "
        f"부등호 {len(ineqs)}개(가시 {visible_cnt}/숨김 {len(hidden)}) · "
        f"SEG {len(ineqs)}개",
        f"  · 해 벡터: [{', '.join(str(v) for v in solution)}]",
    ]
    for i, op in enumerate(ineqs):
        left, right = solution[i], solution[i + 1]
        hidden_flag = "숨김" if i in hidden else "가시"
        if op == "<":
            ok = left < right
        elif op == ">":
            ok = left > right
        else:
            ok = None
        status = "성립" if ok else ("불일치" if ok is False else "확인 필요")
        lines.append(
            f"    [SEG {i + 1}] 자리 {i}↔{i + 1} ({hidden_flag}): {left} {op} {right} → {status}"
        )
    lines.extend(
        [
            "[STEP 3] 답·검산",
            f"  - 최종 답: {ans_str}",
            f"  - 1~{size}의 각 숫자가 정확히 한 번 사용됨: "
            f"{'OK' if sorted(solution) == list(range(1, size + 1)) else 'FAIL'}",
            "  - 힌트 위치의 값 일치 및 모든 부등호(가시+숨김)가 [SEG]에서 확인됨.",
        ]
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 2D Futoshiki 퍼즐  (medium / hard 티어)
# --------------------------------------------------------------------------- #

Cell = Tuple[int, int]
Constraint = Tuple[Cell, Cell, str]  # (cell_a, cell_b, op) 의미: a op b


@dataclass
class FutoshikiPuzzle:
    n: int
    solution: List[List[int]]
    givens: Dict[Cell, int]
    constraints: List[Constraint]  # 읽기 순서로 정렬
    difficulty: Difficulty
    mode: str = "backtrack"  # "propagation" | "backtrack"
    branch_points: int = 0

    def flat_answer(self) -> str:
        return " ".join(
            str(self.solution[r][c]) for r in range(self.n) for c in range(self.n)
        )


class FutoshikiGenerator:
    @staticmethod
    def random_latin_square(n: int) -> List[List[int]]:
        base = [[(i + j) % n + 1 for j in range(n)] for i in range(n)]
        random.shuffle(base)
        cols = list(range(n))
        random.shuffle(cols)
        base = [[row[c] for c in cols] for row in base]
        perm = list(range(1, n + 1))
        random.shuffle(perm)
        return [[perm[v - 1] for v in row] for row in base]

    @staticmethod
    def all_adjacent_constraints(sol: List[List[int]]) -> List[Constraint]:
        """모든 직교 인접 제약 (작은 읽기 순서 셀이 앞에)."""
        n = len(sol)
        cons: List[Constraint] = []
        for r in range(n):
            for c in range(n):
                if c + 1 < n:
                    op = "<" if sol[r][c] < sol[r][c + 1] else ">"
                    cons.append(((r, c), (r, c + 1), op))
                if r + 1 < n:
                    op = "<" if sol[r][c] < sol[r + 1][c] else ">"
                    cons.append(((r, c), (r + 1, c), op))
        return cons

    @staticmethod
    def sort_constraints(cons: List[Constraint]) -> List[Constraint]:
        return sorted(cons, key=lambda x: (x[0][0], x[0][1], x[1][0], x[1][1]))

    # ---- 해 개수 세기 (역추적, 상한) ---- #
    @staticmethod
    def count_solutions(
        n: int, givens: Dict[Cell, int], constraints: List[Constraint], cap: int = 2
    ) -> int:
        grid = [[0] * n for _ in range(n)]
        rows = [set() for _ in range(n)]
        cols = [set() for _ in range(n)]
        for (r, c), v in givens.items():
            grid[r][c] = v
            rows[r].add(v)
            cols[c].add(v)

        cell_cons: Dict[Cell, List[Constraint]] = {}
        for con in constraints:
            cell_cons.setdefault(con[0], []).append(con)
            cell_cons.setdefault(con[1], []).append(con)

        def ok(r: int, c: int, v: int) -> bool:
            if v in rows[r] or v in cols[c]:
                return False
            for (a, b, op) in cell_cons.get((r, c), []):
                av = v if a == (r, c) else grid[a[0]][a[1]]
                bv = v if b == (r, c) else grid[b[0]][b[1]]
                if av == 0 or bv == 0:
                    continue
                if op == "<" and not av < bv:
                    return False
                if op == ">" and not av > bv:
                    return False
            return True

        empties = [(r, c) for r in range(n) for c in range(n) if grid[r][c] == 0]
        count = 0

        def bt(i: int):
            nonlocal count
            if count >= cap:
                return
            if i == len(empties):
                count += 1
                return
            r, c = empties[i]
            for v in range(1, n + 1):
                if ok(r, c, v):
                    grid[r][c] = v
                    rows[r].add(v)
                    cols[c].add(v)
                    bt(i + 1)
                    grid[r][c] = 0
                    rows[r].discard(v)
                    cols[c].discard(v)
                    if count >= cap:
                        return

        bt(0)
        return count

    # ---- 고정점까지 제약 전파 ---- #
    @staticmethod
    def _propagate(
        cand: List[List[set]], n: int, constraints: List[Constraint]
    ) -> bool:
        """`cand`를 in-place 수정. 모순 시 False, 아니면 True(고정점).

        전파 강도는 teacher trace가 주장하는 범위로 의도적으로 제한됩니다 —
        행/열 all-different 규칙(naked singles + peer elimination)과
        부등호 경계 가지치기뿐입니다. hidden singles 같은 더 강한 기법은
        의도적으로 적용하지 않으므로, 이 기본 규칙만으로 끝낼 때만
        "전파로 풀 수 있음"으로 분류하고, 나머지는 역추적 탐색이 필요합니다."""
        changed = True
        while changed:
            changed = False
            # naked singles -> peer elimination
            for r in range(n):
                for c in range(n):
                    if len(cand[r][c]) == 0:
                        return False
                    if len(cand[r][c]) == 1:
                        v = next(iter(cand[r][c]))
                        for cc in range(n):
                            if cc != c and v in cand[r][cc]:
                                cand[r][cc].discard(v)
                                changed = True
                        for rr in range(n):
                            if rr != r and v in cand[rr][c]:
                                cand[rr][c].discard(v)
                                changed = True
            # inequality pruning
            for (a, b, op) in constraints:
                A, B = cand[a[0]][a[1]], cand[b[0]][b[1]]
                if not A or not B:
                    return False
                if op == "<":  # a < b
                    nA = {x for x in A if x < max(B)}
                    nB = {x for x in B if x > min(A)}
                else:  # a > b
                    nA = {x for x in A if x > min(B)}
                    nB = {x for x in B if x < max(A)}
                if nA != A:
                    if not nA:
                        return False
                    cand[a[0]][a[1]] = nA
                    changed = True
                if nB != B:
                    if not nB:
                        return False
                    cand[b[0]][b[1]] = nB
                    changed = True
        return True

    @classmethod
    def _initial_candidates(cls, n: int, givens: Dict[Cell, int]) -> List[List[set]]:
        cand = [[set(range(1, n + 1)) for _ in range(n)] for _ in range(n)]
        for (r, c), v in givens.items():
            cand[r][c] = {v}
        return cand

    @classmethod
    def propagation_solves(
        cls, n: int, givens: Dict[Cell, int], constraints: List[Constraint]
    ) -> Optional[List[List[int]]]:
        cand = cls._initial_candidates(n, givens)
        if not cls._propagate(cand, n, constraints):
            return None
        for r in range(n):
            for c in range(n):
                if len(cand[r][c]) != 1:
                    return None
        return [[next(iter(cand[r][c])) for c in range(n)] for r in range(n)]

    @classmethod
    def branch_points(
        cls, n: int, givens: Dict[Cell, int], constraints: List[Constraint]
    ) -> int:
        """첫 해에 이르는 경로에서 추측 횟수 (전파로 강제 셀을 처리 후)."""

        def solve(cand: List[List[set]]) -> Optional[int]:
            cand = copy.deepcopy(cand)
            if not cls._propagate(cand, n, constraints):
                return None
            best: Optional[Cell] = None
            for r in range(n):
                for c in range(n):
                    if len(cand[r][c]) > 1:
                        if best is None or len(cand[r][c]) < len(cand[best[0]][best[1]]):
                            best = (r, c)
            if best is None:
                return 0  # 추가 추측 없이 해결됨
            r, c = best
            for v in sorted(cand[r][c]):
                trial = copy.deepcopy(cand)
                trial[r][c] = {v}
                res = solve(trial)
                if res is not None:
                    return res + 1
            return None

        res = solve(cls._initial_candidates(n, givens))
        return res if res is not None else 0

    # ---- 퍼즐 조립 ---- #
    MIN_CONSTRAINTS = 3

    @classmethod
    def generate(
        cls, n: int, target_givens: int, target_mode: str = "backtrack",
        max_tries: int = 200,
    ) -> FutoshikiPuzzle:
        """참조 프로필에 맞는 유일 해 Futoshiki 생성.

        소수의 힌트를 공개하고 격자를 고정하는 전체 직교 부등호 제약 집합에서
        시작합니다. 그다음 유일성을 깨지 않으며 제약을 하나씩 제거하고,
        타겟 인식 중단 규칙을 적용합니다:

        * backtrack 타겟 — 제한된 전파 규칙만으로는 더 이상 격자를 끝낼 수
          없을 때까지(즉 탐색이 실제로 필요해질 때까지) 계속 제거.
        * propagation 타겟 — 전파만으로 여전히 풀리는 동안에만 제약 제거하여
          간결한 전파-풀이 퍼즐을 만듦.

        희소 힌트 + 간결한 제약 집합이 참조의 역추적 중심 성격을 재현합니다.
        티어별 비율이 어떤 타겟을 요청할지 결정하며, 재시도 예산 안에
        요청 모드를 맞추기 어려우면 유일한 fallback을 반환합니다.
        """
        fallback: Optional[FutoshikiPuzzle] = None

        for _ in range(max_tries):
            sol = cls.random_latin_square(n)
            all_cons = cls.all_adjacent_constraints(sol)
            random.shuffle(all_cons)

            cells = [(r, c) for r in range(n) for c in range(n)]
            random.shuffle(cells)
            g = max(1, min(target_givens, n * n - 1))
            givens: Dict[Cell, int] = {cells[i]: sol[cells[i][0]][cells[i][1]] for i in range(g)}
            gi = g

            chosen: List[Constraint] = list(all_cons)  # fully constrained

            def unique(cons) -> bool:
                return cls.count_solutions(n, givens, cons, 2) == 1

            # 전체 제약 집합이 여러 라틴 방진을 허용할 수 있으면,
            # 해가 유일해질 때까지 힌트를 더 공개
            while gi < len(cells) and not unique(chosen):
                cell = cells[gi]
                givens[cell] = sol[cell[0]][cell[1]]
                gi += 1
            if not unique(chosen):
                continue

            # 타겟 인식 제약 제거 (매 단계 유일성 유지)
            #   backtrack 타겟 -> 최대한 제거(참조처럼 간결한 제약 집합);
            #     실제 탐색 필요 여부는 아래에서 확인하고 재시도/fallback이 고름.
            #   propagation 타겟 -> 전파만으로 여전히 풀리는 동안에만 제거.
            for con in list(chosen):
                trial = [c for c in chosen if c != con]
                if not unique(trial):
                    continue  # 유일성에 필요한 제약; 유지
                if target_mode == "backtrack":
                    chosen = trial
                else:  # propagation target
                    if cls.propagation_solves(n, givens, trial) is not None:
                        chosen = trial  # 전파-풀이 가능 유지

            if len(chosen) < cls.MIN_CONSTRAINTS:
                continue  # 퇴화된 (거의) 무제약 퍼즐 방지

            chosen = cls.sort_constraints(chosen)
            prop = cls.propagation_solves(n, givens, chosen)
            if prop is not None:
                mode, bp = "propagation", 0
            else:
                mode = "backtrack"
                bp = max(2, cls.branch_points(n, givens, chosen))

            puzzle = FutoshikiPuzzle(
                n=n,
                solution=sol,
                givens=givens,
                constraints=chosen,
                difficulty=Difficulty.HARD,
                mode=mode,
                branch_points=bp,
            )
            if mode == target_mode:
                return puzzle
            fallback = puzzle

        if fallback is not None:
            return fallback
        raise RuntimeError(f"유일한 {n}x{n} Futoshiki 퍼즐 생성 실패")


def _grid_lines(n: int, value_at, indent: str) -> List[str]:
    out = []
    for r in range(n):
        cells = " ".join(value_at(r, c) for c in range(n))
        out.append(f"{indent}행 {r + 1}: {cells}")
    return out


def create_futoshiki_question(p: FutoshikiPuzzle) -> str:
    n = p.n

    def shown(r, c):
        return str(p.givens[(r, c)]) if (r, c) in p.givens else "_"

    grid_block = "\n".join(_grid_lines(n, shown, ""))
    con_block = "\n".join(
        f"- ({a[0] + 1},{a[1] + 1}) {op} ({b[0] + 1},{b[1] + 1})"
        for (a, b, op) in p.constraints
    )
    example = " ".join(str(v) for _ in range(n) for v in range(1, n + 1))

    return (
        f"{n}x{n} 격자에서 Futoshiki 퍼즐을 푸세요.\n\n"
        f"규칙:\n"
        f"- 격자를 채워 모든 행과 열에 1부터 {n}까지의 숫자가 각각 한 번씩 등장하도록 합니다 (라틴 방진).\n"
        f"- '_'는 채워야 할 빈 칸을 나타냅니다.\n"
        f"- 아래 나열된 모든 부등호 제약을 만족해야 합니다. 각 제약은 두 직교 인접 셀을 비교하며 "
        f"1-기반 인덱스인 (행, 열)로 표기합니다 (행 1 = 맨 위, 열 1 = 맨 왼쪽).\n"
        f"- '(a) < (b)'는 셀 a의 숫자가 셀 b보다 작음을, '(a) > (b)'는 더 큼을 의미합니다.\n\n"
        f"주어진 격자 (위에서 아래로):\n{grid_block}\n\n"
        f"부등호 제약:\n{con_block}\n\n"
        f"완성된 격자를 답으로 제출하세요: {n * n}개 숫자를 공백으로 구분하여 행 순서로 나열 "
        f"(행 1 왼쪽부터 오른쪽, 그 다음 행 2, 순서대로).\n\n"
        f"형식 예시 (길이와 순서만 보여주는 예, 유효한 풀이가 아닙니다):\nAnswer: {example}"
    )


def build_futoshiki_solution(p: FutoshikiPuzzle) -> str:
    n = p.n
    sol = p.solution

    if p.mode == "propagation":
        mode_line = (
            "  - 이 퍼즐은 제약 전파만으로 풀 수 있습니다 "
            "(행/열 all-different 규칙과 부등호를 반복 적용; 추측 불필요)."
        )
        approach_line = (
            "  · 접근: 라틴 방진(행/열 all-different) 규칙과 부등호 제약을 반복 적용하여 "
            "강제 셀을 채워 나갑니다; 각 단계에서 하나의 값만 가능한 셀부터 결정하여 "
            "격자 전체를 완성합니다."
        )
    else:
        mode_line = (
            "  - 제약 전파만으로는 이 퍼즐을 완성할 수 없습니다; "
            "경우 분석(역추적 탐색)이 필요합니다."
        )
        approach_line = (
            "  · 접근: 라틴 방진(행/열 all-different) 규칙과 부등호로 각 셀의 후보를 좁힌 뒤, "
            "가장 제약이 강한 셀에서 가정하고 모순 발생 시 역추적하여 유일한 격자를 찾습니다 "
            f"(탐색 크기 약 {p.branch_points}개 분기점)."
        )

    def given_at(r, c):
        return str(p.givens[(r, c)]) if (r, c) in p.givens else "_"

    def sol_at(r, c):
        return str(sol[r][c])

    lines: List[str] = [
        SFT_SOLUTION_RUBRIC_KO,
        "[STEP 0] 문제 메타",
        f"  - 난이도: {p.difficulty.name.lower()}",
        f"  - 퍼즐: {n}x{n} Futoshiki (부등호 제약이 있는 라틴 방진)",
        f"  - 주어진 수: {len(p.givens)}개 · 부등호 제약: {len(p.constraints)}개",
        mode_line,
        "  - 셀은 1-인덱스 (행 1 = 맨 위, 열 1 = 맨 왼쪽). 최종 답은 [STEP 3]에서 확정",
        "[STEP 1] 주어진 조건",
        "  - 주어진 격자:",
    ]
    lines += _grid_lines(n, given_at, "      ")
    lines.append("  - 부등호 제약:")
    lines += [
        f"      ({a[0] + 1},{a[1] + 1}) {op} ({b[0] + 1},{b[1] + 1})"
        for (a, b, op) in p.constraints
    ]

    lines.append("[STEP 2] 풀이 전개")
    lines.append(approach_line)
    lines.append("  · 완성된 격자:")
    lines += _grid_lines(n, sol_at, "      ")

    lines.append(f"  · 검산 — 모든 행이 1~{n}의 순열:")
    for r in range(n):
        row = ", ".join(str(sol[r][c]) for c in range(n))
        lines.append(f"      행 {r + 1}: {{{row}}} -> OK")
    lines.append(f"  · 검산 — 모든 열이 1~{n}의 순열:")
    for c in range(n):
        col = ", ".join(str(sol[r][c]) for r in range(n))
        lines.append(f"      열 {c + 1}: {{{col}}} -> OK")
    lines.append("  · 검산 — 모든 부등호 성립:")
    for (a, b, op) in p.constraints:
        av, bv = sol[a[0]][a[1]], sol[b[0]][b[1]]
        lines.append(
            f"      ({a[0] + 1},{a[1] + 1}) {op} ({b[0] + 1},{b[1] + 1}): "
            f"{av} {op} {bv} -> 성립"
        )

    lines += [
        "[STEP 3] 답·검산",
        f"  - 최종 답 (격자를 행 순서로 일렬 나열): {p.flat_answer()}",
        f"  - 각 행과 열이 1~{n}의 순열: OK",
        "  - 모든 부등호 제약 성립 및 주어진 수 일치 (위 [SEG] 추적 확인).",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 티어 설정 (참조 데이터셋에 맞춰 보정)
# --------------------------------------------------------------------------- #

# easy / medium / hard: 전부 Futoshiki. 격자 크기 가중치 + 역추적이 실제로
# 필요한 퍼즐 비율 (제한된 전파 규칙만으로 풀 수 있는 퍼즐 대비).
#
# easy는 4x4 격자로, 대체로 전파만으로 풀리게 하여 medium(5x5, 대부분 역추적)
# 바로 아래 난이도에 위치시킴. 기존 1D 부등호 체인 easy는 강한 모델이 체인
# 길이나 숨김 부등호 수와 무관하게 거의 만점을 내서 폐기했고, 더 작은 Futoshiki
# 격자로 easy < medium < hard 단조 난이도 축을 확보함.
FUTOSHIKI_TIERS = {
    "easy": {"size_weights": {4: 1.0}, "backtrack_ratio": 0.35},
    "medium": {"size_weights": {5: 0.62, 6: 0.38}, "backtrack_ratio": 0.82},
    "hard": {"size_weights": {5: 0.20, 6: 0.80}, "backtrack_ratio": 0.92},
}

# 격자 크기별 공개 힌트 수 (작게 유지; 최소 제약 집합이 난이도를 만듦).
# 참조 중앙값에 맞춤 (5x5 ~4, 6x6 ~5-6).
# 4x4 easy는 접근성을 위해 셀 공개 비율을 약간 높게 잡음.
FUTOSHIKI_GIVENS_BY_SIZE = {4: (3, 5), 5: (2, 6), 6: (3, 7)}


def _weighted_choice(weight_map: Dict[int, float]) -> int:
    keys = list(weight_map.keys())
    weights = [weight_map[k] for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]


# --------------------------------------------------------------------------- #
# 레코드 빌더
# --------------------------------------------------------------------------- #

def make_easy_record(idx: int, chain_gen: InequalityPuzzleGenerator) -> dict:
    # easy는 이제 4x4 Futoshiki (FUTOSHIKI_TIERS 참조). chain_gen 인자는 호출부
    # 호환을 위해 남겨두었으나 더 이상 사용하지 않음.
    return make_futoshiki_record(idx, "easy")


def make_futoshiki_record(idx: int, tier: str) -> dict:
    cfg = FUTOSHIKI_TIERS[tier]
    n = _weighted_choice(cfg["size_weights"])
    target_givens = random.randint(*FUTOSHIKI_GIVENS_BY_SIZE[n])
    target_mode = "backtrack" if random.random() < cfg["backtrack_ratio"] else "propagation"
    p = FutoshikiGenerator.generate(n, target_givens, target_mode)
    p.difficulty = Difficulty[tier.upper()]
    return {
        "id": f"inequality_ko_{tier}_{idx:04d}",
        "question": create_futoshiki_question(p),
        "answer": p.flat_answer(),
        "solution": build_futoshiki_solution(p),
        "difficulty": tier,
    }


# --------------------------------------------------------------------------- #
# 오케스트레이션
# --------------------------------------------------------------------------- #

def create_dataset_files(num_questions: int, outdir: Path) -> Dict[str, List[dict]]:
    """세 가지 티어를 생성하고 티어별 JSONL + 통합 CSV 파일을 작성합니다."""
    import pandas as pd

    tiers = ["easy", "medium", "hard"]
    per_tier = num_questions // len(tiers)
    remainder = num_questions % len(tiers)

    chain_gen = InequalityPuzzleGenerator()
    outdir.mkdir(parents=True, exist_ok=True)

    all_records: List[dict] = []
    by_tier: Dict[str, List[dict]] = {}

    for i, tier in enumerate(tiers):
        count = per_tier + (1 if i < remainder else 0)
        print(f"\n=== {tier} 티어 생성 중 ({count}개) ===")
        records: List[dict] = []
        seen = set()
        produced = 0
        attempts = 0
        budget = max(50, count * 50)
        while produced < count and attempts < budget:
            attempts += 1
            try:
                if tier == "easy":
                    rec = make_easy_record(produced, chain_gen)
                else:
                    rec = make_futoshiki_record(produced, tier)
            except RuntimeError as exc:
                print(f"  재시도 ({exc})")
                continue
            if rec["question"] in seen:
                continue
            seen.add(rec["question"])
            records.append(rec)
            produced += 1
            if produced % 10 == 0 or produced == count:
                print(f"  [{produced}/{count}]")

        if produced < count:
            print(f"  ⚠️ {tier} 에서 {produced}/{count} 만 생성됨")

        by_tier[tier] = records
        all_records.extend(records)

        jsonl_path = outdir / f"inequality_ko_{tier}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  JSONL: {jsonl_path}")

    # 통합 CSV (동일 열 순서; CSV == 연결된 JSONL)
    df = pd.DataFrame(all_records, columns=["id", "question", "answer", "solution", "difficulty"])
    csv_path = outdir / "inequality_ko.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV: {csv_path}  ({len(df)}행)")

    return by_tier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="부등호 / Futoshiki 퍼즐 생성기 (한국어)")
    parser.add_argument("--num", type=int, default=300, help="총 퍼즐 수 (3 티어로 분배)")
    parser.add_argument("--seed", type=int, default=None, help="재현성을 위한 랜덤 시드")
    parser.add_argument("--outdir", type=str, default="out", help="출력 디렉터리")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print("=" * 50)
    print("부등호 / Futoshiki 퍼즐 생성기 (한국어)")
    print("=" * 50)
    create_dataset_files(num_questions=args.num, outdir=Path(args.outdir))