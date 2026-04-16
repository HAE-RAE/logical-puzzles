"""스도쿠 퍼즐 생성기 (한국어).

logical-puzzles-me/sudoku/* 모듈들을 단일 파일로 통합:
- DIFFICULTY_CONFIGS (탐색 노드 게이팅 + spotcheck_k: easy=3, medium=5, hard=6)
- generate_complete (기본 라틴 스퀘어 + 무작위 변환)
- _create_removal_groups (rot180 대칭 제거 그룹)
- count_solutions / has_valid_solutions / find_all_solutions (MRV DFS)
- solve_backtrack (SearchStats 반환)
- LogicSolver (L1: naked_single, hidden_single / L2: locked_candidates, naked_pair)
- rate() (DifficultyMeta 반환)
- Spotcheck: select_spotcheck_positions + make_spotcheck_code + make_code
  정답 형식: 스팟체크 위치의 숫자 K개를 공백으로 구분한 문자열.
"""

import argparse
import hashlib
import hmac
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


Grid = List[List[int]]

MAX_SOLUTIONS = 1


# ============================================================================
# 격자 유틸리티
# ============================================================================

def from_string(s: str) -> Grid:
    s = s.strip()
    if len(s) != 81:
        raise ValueError(f"문자열 길이는 81이어야 합니다 (입력: {len(s)})")
    grid: Grid = []
    for i in range(9):
        row = []
        for j in range(9):
            ch = s[i * 9 + j]
            if ch in '.0':
                row.append(0)
            elif '1' <= ch <= '9':
                row.append(int(ch))
            else:
                raise ValueError(f"잘못된 문자 '{ch}' (위치 {i*9+j})")
        grid.append(row)
    return grid


def to_string(g: Grid, blanks: str = '.') -> str:
    chars = []
    for row in g:
        for val in row:
            chars.append(blanks if val == 0 else str(val))
    return ''.join(chars)


def copy_grid(g: Grid) -> Grid:
    return [row[:] for row in g]


def get_cell_candidates(g: Grid, r: int, c: int) -> Set[int]:
    if g[r][c] != 0:
        return set()
    used = set()
    for val in g[r]:
        if val != 0:
            used.add(val)
    for row in g:
        if row[c] != 0:
            used.add(row[c])
    box_r, box_c = (r // 3) * 3, (c // 3) * 3
    for br in range(box_r, box_r + 3):
        for bc in range(box_c, box_c + 3):
            if g[br][bc] != 0:
                used.add(g[br][bc])
    return set(range(1, 10)) - used


def is_solved(g: Grid) -> bool:
    for row in g:
        if 0 in row:
            return False
    for row in g:
        if set(row) != set(range(1, 10)):
            return False
    for c in range(9):
        if set(g[r][c] for r in range(9)) != set(range(1, 10)):
            return False
    for box_idx in range(9):
        box_r = (box_idx // 3) * 3
        box_c = (box_idx % 3) * 3
        vals = []
        for r in range(box_r, box_r + 3):
            for c in range(box_c, box_c + 3):
                vals.append(g[r][c])
        if set(vals) != set(range(1, 10)):
            return False
    return True


def count_givens(puzzle: Grid) -> int:
    return sum(1 for r in range(9) for c in range(9) if puzzle[r][c] != 0)


# ============================================================================
# 격자 변환
# ============================================================================

def rotate_90_cw(g: Grid) -> Grid:
    result = [[0] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            result[c][8 - r] = g[r][c]
    return result


def rotate_180(g: Grid) -> Grid:
    result = [[0] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            result[8 - r][8 - c] = g[r][c]
    return result


def rotate_270_cw(g: Grid) -> Grid:
    result = [[0] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            result[8 - c][r] = g[r][c]
    return result


def mirror_horizontal(g: Grid) -> Grid:
    result = [[0] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            result[r][8 - c] = g[r][c]
    return result


def mirror_vertical(g: Grid) -> Grid:
    result = [[0] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            result[8 - r][c] = g[r][c]
    return result


def transpose(g: Grid) -> Grid:
    result = [[0] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            result[c][r] = g[r][c]
    return result


def anti_transpose(g: Grid) -> Grid:
    result = [[0] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            result[8 - c][8 - r] = g[r][c]
    return result


SYMMETRY_OPS = {
    'none': lambda g: copy_grid(g),
    'rot90': rotate_90_cw,
    'rot180': rotate_180,
    'rot270': rotate_270_cw,
    'mirror_h': mirror_horizontal,
    'mirror_v': mirror_vertical,
    'transpose': transpose,
    'anti_transpose': anti_transpose,
}


def apply_symmetry(g: Grid, symmetry: str) -> Grid:
    if symmetry not in SYMMETRY_OPS:
        raise ValueError(f"알 수 없는 대칭 변환: {symmetry}")
    return SYMMETRY_OPS[symmetry](g)


def relabel_digits(g: Grid, perm: dict) -> Grid:
    result = [[0] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            val = g[r][c]
            result[r][c] = 0 if val == 0 else perm.get(val, val)
    return result


def random_digit_permutation(rng: random.Random) -> dict:
    digits = list(range(1, 10))
    rng.shuffle(digits)
    return {i + 1: digits[i] for i in range(9)}


def shuffle_rows_in_band(g: Grid, band: int, rng: random.Random) -> Grid:
    result = copy_grid(g)
    rows_idx = [band * 3, band * 3 + 1, band * 3 + 2]
    rng.shuffle(rows_idx)
    for i, src_r in enumerate(rows_idx):
        result[band * 3 + i] = g[src_r]
    return result


def shuffle_cols_in_stack(g: Grid, stack: int, rng: random.Random) -> Grid:
    result = copy_grid(g)
    cols_idx = [stack * 3, stack * 3 + 1, stack * 3 + 2]
    rng.shuffle(cols_idx)
    for r in range(9):
        for i, src_c in enumerate(cols_idx):
            result[r][stack * 3 + i] = g[r][src_c]
    return result


def shuffle_bands(g: Grid, rng: random.Random) -> Grid:
    band_order = [0, 1, 2]
    rng.shuffle(band_order)
    result = [[0] * 9 for _ in range(9)]
    for i, band in enumerate(band_order):
        for offset in range(3):
            result[i * 3 + offset] = g[band * 3 + offset][:]
    return result


def shuffle_stacks(g: Grid, rng: random.Random) -> Grid:
    stack_order = [0, 1, 2]
    rng.shuffle(stack_order)
    result = copy_grid(g)
    for r in range(9):
        new_row = [0] * 9
        for i, stack in enumerate(stack_order):
            for offset in range(3):
                new_row[i * 3 + offset] = g[r][stack * 3 + offset]
        result[r] = new_row
    return result


def apply_random_transforms(g: Grid, rng: random.Random) -> Grid:
    result = copy_grid(g)
    perm = random_digit_permutation(rng)
    result = relabel_digits(result, perm)
    result = shuffle_bands(result, rng)
    result = shuffle_stacks(result, rng)
    for band in range(3):
        result = shuffle_rows_in_band(result, band, rng)
    for stack in range(3):
        result = shuffle_cols_in_stack(result, stack, rng)
    if rng.random() < 0.5:
        sym = rng.choice(['rot90', 'rot180', 'rot270', 'mirror_h',
                          'mirror_v', 'transpose', 'anti_transpose'])
        result = apply_symmetry(result, sym)
    return result


# ============================================================================
# 완성 격자 생성
# ============================================================================

def _base_solution() -> Grid:
    grid: Grid = []
    for r in range(9):
        row = []
        for c in range(9):
            val = (r * 3 + r // 3 + c) % 9 + 1
            row.append(val)
        grid.append(row)
    return grid


def generate_complete(seed: Optional[int] = None) -> Grid:
    """무작위 유효 완성 스도쿠 격자를 생성합니다."""
    rng = random.Random(seed)
    base = _base_solution()
    result = apply_random_transforms(base, rng)
    assert is_solved(result), "생성된 격자가 유효하지 않음"
    return result


# ============================================================================
# 풀이 개수 세기 (MRV DFS)
# ============================================================================

def count_solutions(puzzle: Grid, limit: int = 2) -> int:
    grid = copy_grid(puzzle)
    count = [0]

    def solve(g: Grid) -> bool:
        if count[0] >= limit:
            return True
        min_candidates = 10
        best_cell = None
        for r in range(9):
            for c in range(9):
                if g[r][c] == 0:
                    cands = get_cell_candidates(g, r, c)
                    if len(cands) == 0:
                        return False
                    if len(cands) < min_candidates:
                        min_candidates = len(cands)
                        best_cell = (r, c, cands)
        if best_cell is None:
            count[0] += 1
            return count[0] >= limit
        r, c, cands = best_cell
        for num in cands:
            g[r][c] = num
            if solve(g):
                return True
            g[r][c] = 0
        return False

    solve(grid)
    return count[0]


def has_valid_solutions(puzzle: Grid) -> bool:
    return count_solutions(puzzle, limit=MAX_SOLUTIONS + 1) == 1


def has_unique_solution(puzzle: Grid) -> bool:
    return count_solutions(puzzle, limit=2) == 1


def find_all_solutions(puzzle: Grid, limit: int = 2) -> List[Grid]:
    solutions: List[Grid] = []

    def solve(g: Grid) -> bool:
        if len(solutions) >= limit:
            return True
        min_candidates = 10
        best_cell = None
        for r in range(9):
            for c in range(9):
                if g[r][c] == 0:
                    cands = get_cell_candidates(g, r, c)
                    if len(cands) == 0:
                        return False
                    if len(cands) < min_candidates:
                        min_candidates = len(cands)
                        best_cell = (r, c, cands)
        if best_cell is None:
            solutions.append(copy_grid(g))
            return len(solutions) >= limit
        r, c, cands = best_cell
        for num in cands:
            g[r][c] = num
            if solve(g):
                return True
            g[r][c] = 0
        return False

    grid = copy_grid(puzzle)
    solve(grid)
    return solutions


# ============================================================================
# 백트래킹 솔버 (통계 포함)
# ============================================================================

@dataclass
class SearchStats:
    nodes: int
    max_depth: int
    avg_candidates: float


def solve_backtrack(puzzle: Grid) -> Tuple[Optional[Grid], SearchStats]:
    grid = copy_grid(puzzle)
    stats = {'nodes': 0, 'max_depth': 0, 'total_candidates': 0, 'candidate_count': 0}

    def backtrack(depth: int) -> bool:
        stats['nodes'] += 1
        stats['max_depth'] = max(stats['max_depth'], depth)
        min_cands = 10
        best_cell = None
        for r in range(9):
            for c in range(9):
                if grid[r][c] == 0:
                    cands = get_cell_candidates(grid, r, c)
                    if len(cands) == 0:
                        return False
                    if len(cands) < min_cands:
                        min_cands = len(cands)
                        best_cell = (r, c, cands)
        if best_cell is None:
            return is_solved(grid)
        r, c, cands = best_cell
        stats['total_candidates'] += len(cands)
        stats['candidate_count'] += 1
        for num in cands:
            grid[r][c] = num
            if backtrack(depth + 1):
                return True
            grid[r][c] = 0
        return False

    success = backtrack(0)
    avg_cands = (stats['total_candidates'] / stats['candidate_count']
                 if stats['candidate_count'] > 0 else 0.0)
    return (
        grid if success else None,
        SearchStats(
            nodes=stats['nodes'],
            max_depth=stats['max_depth'],
            avg_candidates=avg_cands,
        ),
    )


def search_stats(puzzle: Grid) -> SearchStats:
    _, stats = solve_backtrack(puzzle)
    return stats


# ============================================================================
# 논리 솔버 (L1 + L2)
# ============================================================================

@dataclass
class SolveSummary:
    solved: bool
    steps: int
    max_tech_level: str
    counts: dict
    guess_used: bool


class LogicSolver:
    def __init__(self, grid: Grid):
        self.grid = copy_grid(grid)
        self.candidates: List[List[Set[int]]] = []
        self._init_candidates()
        self.counts: dict = {}
        self.steps = 0
        self.max_tech_level = 'L0'

    def _init_candidates(self):
        self.candidates = []
        for r in range(9):
            row = []
            for c in range(9):
                if self.grid[r][c] == 0:
                    row.append(get_cell_candidates(self.grid, r, c))
                else:
                    row.append(set())
            self.candidates.append(row)

    def _update_level(self, level: str):
        levels = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5']
        if levels.index(level) > levels.index(self.max_tech_level):
            self.max_tech_level = level

    def _record_technique(self, name: str, level: str):
        self.counts[name] = self.counts.get(name, 0) + 1
        self._update_level(level)
        self.steps += 1

    def _set_value(self, r: int, c: int, val: int):
        self.grid[r][c] = val
        self.candidates[r][c] = set()
        for i in range(9):
            self.candidates[r][i].discard(val)
            self.candidates[i][c].discard(val)
        box_r, box_c = (r // 3) * 3, (c // 3) * 3
        for br in range(box_r, box_r + 3):
            for bc in range(box_c, box_c + 3):
                self.candidates[br][bc].discard(val)

    def naked_single(self) -> bool:
        found = False
        for r in range(9):
            for c in range(9):
                if self.grid[r][c] == 0 and len(self.candidates[r][c]) == 1:
                    val = next(iter(self.candidates[r][c]))
                    self._set_value(r, c, val)
                    self._record_technique('naked_single', 'L1')
                    found = True
        return found

    def hidden_single(self) -> bool:
        to_set: List[Tuple[int, int, int]] = []
        seen: Set[Tuple[int, int]] = set()
        for row in range(9):
            for num in range(1, 10):
                positions = [(row, c) for c in range(9)
                             if num in self.candidates[row][c]]
                if len(positions) == 1:
                    pos = positions[0]
                    if pos not in seen:
                        to_set.append((pos[0], pos[1], num))
                        seen.add(pos)
        for col in range(9):
            for num in range(1, 10):
                positions = [(r, col) for r in range(9)
                             if num in self.candidates[r][col]]
                if len(positions) == 1:
                    pos = positions[0]
                    if pos not in seen:
                        to_set.append((pos[0], pos[1], num))
                        seen.add(pos)
        for box_idx in range(9):
            box_r = (box_idx // 3) * 3
            box_c = (box_idx % 3) * 3
            for num in range(1, 10):
                positions = []
                for br in range(box_r, box_r + 3):
                    for bc in range(box_c, box_c + 3):
                        if num in self.candidates[br][bc]:
                            positions.append((br, bc))
                if len(positions) == 1:
                    pos = positions[0]
                    if pos not in seen:
                        to_set.append((pos[0], pos[1], num))
                        seen.add(pos)
        for r, c, val in to_set:
            if self.grid[r][c] == 0 and val in self.candidates[r][c]:
                self._set_value(r, c, val)
                self._record_technique('hidden_single', 'L1')
        return len(to_set) > 0

    def locked_candidates(self) -> bool:
        found = False
        for box_idx in range(9):
            box_r = (box_idx // 3) * 3
            box_c = (box_idx % 3) * 3
            for num in range(1, 10):
                positions = []
                for r in range(box_r, box_r + 3):
                    for c in range(box_c, box_c + 3):
                        if num in self.candidates[r][c]:
                            positions.append((r, c))
                if not positions:
                    continue
                if len(set(r for r, c in positions)) == 1:
                    row = positions[0][0]
                    for c in range(9):
                        if c < box_c or c >= box_c + 3:
                            if num in self.candidates[row][c]:
                                self.candidates[row][c].discard(num)
                                found = True
                if len(set(c for r, c in positions)) == 1:
                    col = positions[0][1]
                    for r in range(9):
                        if r < box_r or r >= box_r + 3:
                            if num in self.candidates[r][col]:
                                self.candidates[r][col].discard(num)
                                found = True
        if found:
            self._record_technique('locked', 'L2')
        return found

    def naked_pair(self) -> bool:
        found = False
        for r in range(9):
            for c1 in range(9):
                cands1 = self.candidates[r][c1]
                if len(cands1) != 2:
                    continue
                for c2 in range(c1 + 1, 9):
                    if self.candidates[r][c2] == cands1:
                        for c in range(9):
                            if c != c1 and c != c2:
                                before = len(self.candidates[r][c])
                                self.candidates[r][c] -= cands1
                                if len(self.candidates[r][c]) < before:
                                    found = True
        for c in range(9):
            for r1 in range(9):
                cands1 = self.candidates[r1][c]
                if len(cands1) != 2:
                    continue
                for r2 in range(r1 + 1, 9):
                    if self.candidates[r2][c] == cands1:
                        for r in range(9):
                            if r != r1 and r != r2:
                                before = len(self.candidates[r][c])
                                self.candidates[r][c] -= cands1
                                if len(self.candidates[r][c]) < before:
                                    found = True
        for box_idx in range(9):
            box_r = (box_idx // 3) * 3
            box_c = (box_idx % 3) * 3
            box_cells = [(r, c) for r in range(box_r, box_r + 3)
                         for c in range(box_c, box_c + 3)]
            for i, (r1, c1) in enumerate(box_cells):
                cands1 = self.candidates[r1][c1]
                if len(cands1) != 2:
                    continue
                for r2, c2 in box_cells[i + 1:]:
                    if self.candidates[r2][c2] == cands1:
                        for r, c in box_cells:
                            if (r, c) != (r1, c1) and (r, c) != (r2, c2):
                                before = len(self.candidates[r][c])
                                self.candidates[r][c] -= cands1
                                if len(self.candidates[r][c]) < before:
                                    found = True
        if found:
            self._record_technique('naked_pair', 'L2')
        return found

    def solve_l1(self) -> bool:
        progress = True
        while progress:
            progress = False
            progress |= self.naked_single()
            progress |= self.hidden_single()
        return is_solved(self.grid)

    def solve_l2(self) -> bool:
        progress = True
        while progress:
            progress = False
            progress |= self.naked_single()
            progress |= self.hidden_single()
            progress |= self.locked_candidates()
            progress |= self.naked_pair()
        return is_solved(self.grid)

    def solve_with_limit(self, max_level: str) -> SolveSummary:
        if max_level == 'L1':
            solved = self.solve_l1()
        else:
            solved = self.solve_l2()
        return SolveSummary(
            solved=solved,
            steps=self.steps,
            max_tech_level=self.max_tech_level,
            counts=self.counts.copy(),
            guess_used=False,
        )


def solve_with_limit(puzzle: Grid, max_level: str) -> SolveSummary:
    solver = LogicSolver(puzzle)
    return solver.solve_with_limit(max_level)


# ============================================================================
# 난이도 평가
# ============================================================================

@dataclass
class DifficultyMeta:
    label: str
    tech_profile: dict
    max_tech_level: str
    search_nodes: int
    steps: int


def rate(puzzle: Grid) -> DifficultyMeta:
    """논리 솔버 + 백트래킹을 사용한 퍼즐 난이도 평가."""
    summary_l1 = solve_with_limit(puzzle, 'L1')
    if summary_l1.solved:
        return DifficultyMeta(
            label='trivial',
            tech_profile=summary_l1.counts,
            max_tech_level=summary_l1.max_tech_level,
            search_nodes=0,
            steps=summary_l1.steps,
        )
    summary_l2 = solve_with_limit(puzzle, 'L2')
    if summary_l2.solved:
        stats = search_stats(puzzle)
        if stats.nodes <= 500:
            label = 'easy'
        elif stats.nodes <= 3000:
            label = 'medium'
        else:
            label = 'hard'
        return DifficultyMeta(
            label=label,
            tech_profile=summary_l2.counts,
            max_tech_level=summary_l2.max_tech_level,
            search_nodes=stats.nodes,
            steps=summary_l2.steps,
        )
    stats = search_stats(puzzle)
    tech_level = 'L3' if stats.nodes <= 3000 else 'L4'
    return DifficultyMeta(
        label='hard',
        tech_profile={},
        max_tech_level=tech_level,
        search_nodes=stats.nodes,
        steps=0,
    )


# ============================================================================
# 난이도 기반 생성
# ============================================================================

@dataclass
class DifficultyConfig:
    min_givens: int
    max_givens: int
    target_givens: int
    forbid_advanced: bool = False
    forbid_trivial: bool = False
    min_search_nodes: int = 0
    max_search_nodes: Optional[int] = None
    spotcheck_k: int = 6
    symmetry: str = 'rot180'
    minimal: bool = False
    forbidden_rate_labels: Set[str] = field(default_factory=set)


DIFFICULTY_CONFIGS = {
    'easy': DifficultyConfig(
        min_givens=35,
        max_givens=37,
        target_givens=36,
        forbid_trivial=False,
        max_search_nodes=30,
        spotcheck_k=3,
    ),
    'medium': DifficultyConfig(
        min_givens=32,
        max_givens=34,
        target_givens=33,
        forbid_trivial=True,
        min_search_nodes=25,
        max_search_nodes=120,
        spotcheck_k=5,
    ),
    'hard': DifficultyConfig(
        min_givens=29,
        max_givens=31,
        target_givens=30,
        forbid_trivial=True,
        min_search_nodes=90,
        spotcheck_k=6,
    ),
}


def _create_removal_groups(rng: random.Random, symmetry: str) -> List[List[Tuple[int, int]]]:
    positions = [(i, j) for i in range(9) for j in range(9)]
    if symmetry == 'rot180':
        pairs = []
        used = set()
        for i, j in positions:
            if (i, j) not in used:
                sym_i, sym_j = 8 - i, 8 - j
                if (i, j) != (sym_i, sym_j):
                    pairs.append([(i, j), (sym_i, sym_j)])
                    used.add((i, j))
                    used.add((sym_i, sym_j))
                else:
                    pairs.append([(i, j)])
                    used.add((i, j))
        removal_groups = pairs
    else:
        removal_groups = [[(i, j)] for i, j in positions]
    rng.shuffle(removal_groups)
    return removal_groups


def generate_difficulty_puzzle(
    difficulty: str,
    seed: Optional[int] = None,
) -> Tuple[str, List[str], dict]:
    """정확히 1개의 해를 가지는 스도쿠 퍼즐을 생성합니다."""
    if difficulty not in DIFFICULTY_CONFIGS:
        raise ValueError(f"잘못된 난이도: {difficulty}")

    config = DIFFICULTY_CONFIGS[difficulty]
    rng = random.Random(seed)

    solution_seed = rng.randint(0, 1_000_000_000)
    solution = generate_complete(solution_seed)
    puzzle = [row[:] for row in solution]

    removal_groups = _create_removal_groups(rng, config.symmetry)

    for group in removal_groups:
        temp_puzzle = [row[:] for row in puzzle]
        for i, j in group:
            temp_puzzle[i][j] = 0
        if count_givens(temp_puzzle) < config.min_givens:
            continue
        solution_count = count_solutions(temp_puzzle, limit=MAX_SOLUTIONS + 1)
        if solution_count == 1:
            puzzle = temp_puzzle
        current_givens = count_givens(puzzle)
        if current_givens <= config.target_givens:
            if not config.minimal:
                break

    final_givens = count_givens(puzzle)
    if final_givens > config.max_givens:
        raise RuntimeError(
            f"단서 수 {final_givens}개가 max_givens={config.max_givens}를 초과 "
            f"(seed={seed}, difficulty={difficulty})"
        )

    final_count = count_solutions(puzzle, limit=2)
    if final_count != 1:
        raise RuntimeError(
            f"최종 퍼즐의 해가 {final_count}개 (1개 예상). "
            f"seed={seed}, difficulty={difficulty}"
        )

    solutions = find_all_solutions(puzzle, limit=2)
    if not solutions:
        raise RuntimeError(
            f"count=1인데 find_all_solutions가 비어있음. seed={seed}"
        )

    rating = rate(puzzle)
    is_advanced = rating.max_tech_level in {'L3', 'L4', 'L5'}
    is_trivial = (rating.max_tech_level == 'L1' and rating.search_nodes == 0)

    if config.forbid_advanced and is_advanced:
        raise RuntimeError(
            f"forbid_advanced인데 max_tech={rating.max_tech_level}. seed={seed}"
        )
    if config.forbid_trivial and is_trivial:
        raise RuntimeError(
            f"forbid_trivial인데 L1으로 자명 (nodes=0). seed={seed}"
        )
    if config.forbidden_rate_labels and rating.label in config.forbidden_rate_labels:
        raise RuntimeError(
            f"rate().label={rating.label}은 금지됨. seed={seed}"
        )
    if rating.search_nodes < config.min_search_nodes:
        raise RuntimeError(
            f"search_nodes={rating.search_nodes} < min={config.min_search_nodes}. seed={seed}"
        )
    if config.max_search_nodes is not None and rating.search_nodes > config.max_search_nodes:
        raise RuntimeError(
            f"search_nodes={rating.search_nodes} > max={config.max_search_nodes}. seed={seed}"
        )

    puzzle_str = to_string(puzzle)
    solution_strs = [to_string(sol) for sol in solutions]

    metadata = {
        'difficulty': difficulty,
        'givens_count': count_givens(puzzle),
        'solution_count': final_count,
        'symmetry': config.symmetry,
        'seed': seed,
        'step_metrics': {
            'solver_steps': rating.steps,
            'search_nodes': rating.search_nodes,
            'max_tech_level': rating.max_tech_level,
            'tech_profile': rating.tech_profile,
            'rate_label': rating.label,
        },
    }
    return puzzle_str, solution_strs, metadata


# ============================================================================
# 스팟체크 (HMAC-SHA256 결정적 K-셀 선택)
# ============================================================================

def select_spotcheck_positions(canonical_hash: str, secret_hex: str, k: int) -> List[str]:
    """HMAC 기반 결정적 K개 위치 선택. 'r{R}c{C}' (1-기반) 형식 반환."""
    secret = bytes.fromhex(secret_hex)
    message = canonical_hash.encode('utf-8')
    mac = hmac.new(secret, message, hashlib.sha256).digest()
    seed = int.from_bytes(mac[:8], 'big')
    rng = random.Random(seed)
    all_positions = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(all_positions)
    selected = all_positions[:k]
    return [f"r{r+1}c{c+1}" for r, c in selected]


def _parse_position(pos: str) -> Tuple[int, int]:
    parts = pos[1:].split('c')
    return int(parts[0]) - 1, int(parts[1]) - 1


def make_spotcheck_code(solution: Grid, positions: List[str]) -> int:
    """레거시 스팟체크 코드: K개 위치 값의 합계 (디버깅용)."""
    total = 0
    for pos in positions:
        r, c = _parse_position(pos)
        total += solution[r][c]
    return total


def make_code(solution: Grid, positions: List[str]) -> int:
    """make_spotcheck_code의 별칭 (Repo A 네이밍)."""
    return make_spotcheck_code(solution, positions)


def make_spotcheck_answer(solution: Grid, positions: List[str]) -> str:
    """Repo A 정답 형식: 스팟체크 위치의 숫자 K개를 공백으로 구분한 문자열."""
    digits = []
    for pos in positions:
        r, c = _parse_position(pos)
        digits.append(str(solution[r][c]))
    return ' '.join(digits)


# ============================================================================
# 문제 텍스트 구성
# ============================================================================

def _format_grid(puzzle_str: str) -> str:
    cleaned = puzzle_str.replace('0', '.')
    rows = [cleaned[i:i + 9] for i in range(0, 81, 9)]
    return "\n".join(" ".join(row) for row in rows)


def _positions_as_rc_pairs(positions: List[str]) -> str:
    pairs = []
    for pos in positions:
        r, c = _parse_position(pos)
        pairs.append(f"({r+1}, {c+1})")
    return ", ".join(pairs)


def create_question(puzzle_str: str, positions: List[str]) -> str:
    grid = _format_grid(puzzle_str)
    rc_str = _positions_as_rc_pairs(positions)
    return (
        "당신은 스도쿠 풀이 전문가입니다. 다음 스도쿠 퍼즐을 완전히 풀어주세요.\n\n"
        "스도쿠 규칙:\n"
        "- 9×9 격자를 채우세요. 각 행, 열, 3×3 박스에 1-9 숫자가 들어갑니다\n"
        "- 각 숫자는 각 행, 열, 박스에 정확히 한 번만 나타나야 합니다\n\n"
        "퍼즐은 아래 9줄 격자로 제공됩니다.\n"
        "'.'은 빈 셀을 나타냅니다.\n\n"
        f"퍼즐 격자:\n{grid}\n\n"
        "먼저 퍼즐을 단계별로 완전히 풀어주세요.\n"
        "그런 다음, 다음 (행, 열) 좌표 (1-기반)의 값을 같은 순서로 제시하세요:\n"
        f"{rc_str}\n\n"
        "중요: 풀이 후 반드시 마지막 줄에 다음 형식으로 작성하세요:\n"
        "Answer: [해당 좌표의 숫자들을 공백으로 구분]\n\n"
        "예시 (값이 5, 3, 4, 6, 7, 8인 경우):\n"
        "Answer: 5 3 4 6 7 8"
    )


# ============================================================================
# 데이터셋 생성
# ============================================================================

def create_dataset_files(num_questions: int):
    """스도쿠 퍼즐 데이터셋 파일(CSV + JSONL)을 생성합니다."""
    import pandas as pd

    print(f"스도쿠 퍼즐 {num_questions}개 생성 중...")

    difficulties = ['easy', 'medium', 'hard']
    puzzles_per_diff = num_questions // len(difficulties)
    remainder = num_questions % len(difficulties)

    all_puzzles: List[Dict] = []
    puzzle_id = 0
    secret_hex = '0' * 64
    max_retries = 140

    for i, difficulty in enumerate(difficulties):
        count = puzzles_per_diff + (1 if i < remainder else 0)
        if count == 0:
            continue

        print(f"\n=== 난이도 '{difficulty}' 퍼즐 생성 중 ({count}개 필요) ===")
        generated = 0

        for j in range(count):
            puzzle_str = None
            solution_strs = None
            metadata = None
            last_err = None
            for retry in range(max_retries):
                seed = 42 + puzzle_id * 1000 + retry
                try:
                    puzzle_str, solution_strs, metadata = generate_difficulty_puzzle(
                        difficulty, seed
                    )
                    break
                except RuntimeError as e:
                    last_err = e
                    continue

            if puzzle_str is None:
                print(f"  [{j+1}/{count}] {max_retries}회 재시도 후 실패: {last_err}")
                puzzle_id += 1
                continue

            try:
                solution_grid = from_string(solution_strs[0])
                k = DIFFICULTY_CONFIGS[difficulty].spotcheck_k

                canonical_hash = f"sha256:{hashlib.sha256(puzzle_str.encode()).hexdigest()}"
                positions = select_spotcheck_positions(canonical_hash, secret_hex, k)
                answer_str = make_spotcheck_answer(solution_grid, positions)
                legacy_code = make_spotcheck_code(solution_grid, positions)

                question = create_question(puzzle_str, positions)

                puzzle_data = {
                    'id': f'sudoku_ko_{len(all_puzzles)}',
                    'question': question,
                    'answer': answer_str,
                    'solution': solution_strs[0],
                    'difficulty': difficulty,
                    'givens_count': metadata['givens_count'],
                    'puzzle': puzzle_str,
                    'spotcheck': {
                        'k': k,
                        'positions': positions,
                        'code': legacy_code,
                    },
                    'step_metrics': metadata.get('step_metrics', {}),
                }
                all_puzzles.append(puzzle_data)
                generated += 1
                print(
                    f"  [{j+1}/{count}] givens={metadata['givens_count']}, "
                    f"answer='{answer_str}' (k={k})"
                )
            except Exception as e:
                print(f"  [{j+1}/{count}] 후처리 실패: {e}")

            puzzle_id += 1

        if generated < count:
            print(f"  경고: {difficulty} 퍼즐 {generated}/{count}개만 생성")

    print(f"\n총 {len(all_puzzles)}개 퍼즐 생성 완료")

    df = pd.DataFrame(all_puzzles)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "sudoku_ko.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV 파일 생성 완료: {csv_path}")

    # JSONL 파일 저장
    json_dir = PROJECT_ROOT / "data" / "jsonl"
    json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = json_dir / "sudoku_ko.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_puzzles:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL 파일 생성 완료: {jsonl_path}")

    return df, all_puzzles


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="스도쿠 퍼즐 생성기 (한국어)")
    parser.add_argument("--num", type=int, default=12, help="생성할 문제 수")
    args = parser.parse_args()

    print("=" * 60)
    print("스도쿠 퍼즐 생성기 (한국어)")
    print("=" * 60)

    create_dataset_files(num_questions=args.num)
