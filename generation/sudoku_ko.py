"""스도쿠 퍼즐 생성기 및 풀이기 (한국어 버전)

격자 연산, 풀이기, 난이도 평가, 퍼즐 생성을 통합한 모듈.
sudoku_benchmark 패키지 기반.
3단계 난이도(easy/medium/hard) 생성 및 스팟체크 평가를 지원합니다.
"""

import random
import json
import hmac
import hashlib
from pathlib import Path
from typing import List, Tuple, Set, Optional
from dataclasses import dataclass


# ============================================================================
# 타입 정의
# ============================================================================

Grid = List[List[int]]  # 0 = 빈 셀


# ============================================================================
# 격자 기본 함수
# ============================================================================

def from_string(s: str) -> Grid:
    """81자 문자열을 9x9 격자로 변환합니다."""
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
    """9x9 격자를 81자 문자열로 변환합니다."""
    chars = []
    for row in g:
        for val in row:
            chars.append(blanks if val == 0 else str(val))
    return ''.join(chars)


def copy_grid(g: Grid) -> Grid:
    return [row[:] for row in g]


def get_cell_candidates(g: Grid, r: int, c: int) -> Set[int]:
    """셀 (r, c)의 후보 숫자를 반환합니다."""
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
    """격자가 유효하게 완성된 스도쿠인지 확인합니다."""
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


def get_symmetric_pair(r: int, c: int, symmetry: str) -> Tuple[int, int]:
    if symmetry == 'none':
        return (r, c)
    elif symmetry == 'rot90':
        return (c, 8 - r)
    elif symmetry == 'rot180':
        return (8 - r, 8 - c)
    elif symmetry == 'rot270':
        return (8 - c, r)
    elif symmetry == 'mirror_h':
        return (r, 8 - c)
    elif symmetry == 'mirror_v':
        return (8 - r, c)
    elif symmetry == 'transpose':
        return (c, r)
    elif symmetry == 'anti_transpose':
        return (8 - c, 8 - r)
    return (r, c)


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
    """무작위 변환 조합을 적용합니다."""
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
# 풀이 개수 세기
# ============================================================================

MAX_SOLUTIONS = 1


def count_solutions(puzzle: Grid, limit: int = 2) -> int:
    """풀이 개수를 셉니다 (limit에 도달하면 조기 종료)."""
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


def find_all_solutions(puzzle: Grid, limit: int = MAX_SOLUTIONS) -> List[Grid]:
    """limit까지 모든 풀이를 찾습니다."""
    solutions = []

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
# 통계 수집을 포함한 백트래킹 풀이기
# ============================================================================

@dataclass
class SearchStats:
    nodes: int
    max_depth: int
    avg_candidates: float


def solve_backtrack(puzzle: Grid) -> Tuple[Optional[Grid], SearchStats]:
    """백트래킹으로 퍼즐을 풀며 통계를 수집합니다."""
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
            avg_candidates=avg_cands
        )
    )


def search_stats(puzzle: Grid) -> SearchStats:
    """탐색 복잡도 통계만 반환합니다."""
    _, stats = solve_backtrack(puzzle)
    return stats


# ============================================================================
# 논리 풀이기 (L1-L2)
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
                    val = list(self.candidates[r][c])[0]
                    self._set_value(r, c, val)
                    self._record_technique('naked_single', 'L1')
                    found = True
        return found

    def hidden_single(self) -> bool:
        found = False
        for r in range(9):
            for num in range(1, 10):
                positions = [(r, c) for c in range(9) if num in self.candidates[r][c]]
                if len(positions) == 1:
                    _, c = positions[0]
                    self._set_value(r, c, num)
                    self._record_technique('hidden_single', 'L1')
                    found = True
        for c in range(9):
            for num in range(1, 10):
                positions = [(r, c) for r in range(9) if num in self.candidates[r][c]]
                if len(positions) == 1:
                    r, _ = positions[0]
                    self._set_value(r, c, num)
                    self._record_technique('hidden_single', 'L1')
                    found = True
        for box_idx in range(9):
            box_r = (box_idx // 3) * 3
            box_c = (box_idx % 3) * 3
            for num in range(1, 10):
                positions = []
                for r in range(box_r, box_r + 3):
                    for c in range(box_c, box_c + 3):
                        if num in self.candidates[r][c]:
                            positions.append((r, c))
                if len(positions) == 1:
                    r, c = positions[0]
                    self._set_value(r, c, num)
                    self._record_technique('hidden_single', 'L1')
                    found = True
        return found

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
            guess_used=False
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
    """논리 풀이기 + 백트래킹을 사용하여 퍼즐 난이도를 평가합니다."""
    summary_l1 = solve_with_limit(puzzle, 'L1')
    if summary_l1.solved:
        return DifficultyMeta(
            label='trivial',
            tech_profile=summary_l1.counts,
            max_tech_level=summary_l1.max_tech_level,
            search_nodes=0,
            steps=summary_l1.steps
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
            steps=summary_l2.steps
        )

    stats = search_stats(puzzle)
    tech_level = 'L3' if stats.nodes <= 3000 else 'L4'
    return DifficultyMeta(
        label='hard',
        tech_profile={},
        max_tech_level=tech_level,
        search_nodes=stats.nodes,
        steps=0
    )


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


def generate_complete(seed: int = None) -> Grid:
    """무작위 유효 완성 스도쿠 격자를 생성합니다."""
    rng = random.Random(seed)
    base = _base_solution()
    result = apply_random_transforms(base, rng)
    return result


# ============================================================================
# 빈 칸 파기를 통한 퍼즐 생성
# ============================================================================

def count_givens(puzzle: Grid) -> int:
    return sum(1 for r in range(9) for c in range(9) if puzzle[r][c] != 0)


def make_puzzle(solution: Grid, symmetry: str = 'rot180',
                ensure_minimal: bool = False, rng: random.Random = None) -> Grid:
    """완성 격자에서 단서를 제거하여 퍼즐을 생성합니다."""
    if rng is None:
        rng = random.Random()

    puzzle = copy_grid(solution)
    cells = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(cells)

    attempted = set()

    for r, c in cells:
        if (r, c) in attempted:
            continue

        r2, c2 = get_symmetric_pair(r, c, symmetry)
        attempted.add((r, c))
        attempted.add((r2, c2))

        if puzzle[r][c] == 0:
            continue
        if r != r2 or c != c2:
            if puzzle[r2][c2] == 0:
                continue

        val1 = puzzle[r][c]
        val2 = puzzle[r2][c2] if (r, c) != (r2, c2) else val1

        puzzle[r][c] = 0
        if (r, c) != (r2, c2):
            puzzle[r2][c2] = 0

        if count_solutions(puzzle, limit=2) != 1:
            puzzle[r][c] = val1
            if (r, c) != (r2, c2):
                puzzle[r2][c2] = val2

    if ensure_minimal:
        puzzle = _make_minimal(puzzle, symmetry, rng)

    return puzzle


def _make_minimal(puzzle: Grid, symmetry: str, rng: random.Random) -> Grid:
    result = copy_grid(puzzle)
    filled = [(r, c) for r in range(9) for c in range(9) if result[r][c] != 0]
    rng.shuffle(filled)

    for r, c in filled:
        if result[r][c] == 0:
            continue

        r2, c2 = get_symmetric_pair(r, c, symmetry)
        if (r, c) != (r2, c2) and result[r2][c2] == 0:
            continue

        val1 = result[r][c]
        val2 = result[r2][c2] if (r, c) != (r2, c2) else val1

        result[r][c] = 0
        if (r, c) != (r2, c2):
            result[r2][c2] = 0

        if count_solutions(result, limit=2) != 1:
            result[r][c] = val1
            if (r, c) != (r2, c2):
                result[r2][c2] = val2

    return result


# ============================================================================
# 난이도 기반 생성
# ============================================================================

@dataclass
class DifficultyConfig:
    min_givens: int
    max_givens: int
    target_givens: int
    symmetry: str
    minimal: bool


DIFFICULTY_CONFIGS = {
    'easy': DifficultyConfig(
        min_givens=40, max_givens=45, target_givens=42,
        symmetry='rot180', minimal=False
    ),
    'medium': DifficultyConfig(
        min_givens=33, max_givens=38, target_givens=35,
        symmetry='rot180', minimal=False
    ),
    'hard': DifficultyConfig(
        min_givens=28, max_givens=32, target_givens=30,
        symmetry='rot180', minimal=True
    )
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


def _prioritize_removal_groups(groups: List[List[Tuple[int, int]]],
                                puzzle: Grid) -> List[List[Tuple[int, int]]]:
    value_freq = {}
    for i in range(9):
        for j in range(9):
            val = puzzle[i][j]
            if val != 0:
                value_freq[val] = value_freq.get(val, 0) + 1

    def group_score(group):
        score = 0
        for i, j in group:
            val = puzzle[i][j]
            if val != 0:
                score += 10 - value_freq.get(val, 9)
        return score

    return sorted(groups, key=group_score, reverse=True)


def generate_difficulty_puzzle(
    difficulty: str,
    seed: Optional[int] = None
) -> Tuple[str, List[str], dict]:
    """
    지정된 난이도의 스도쿠 퍼즐을 생성합니다.

    반환값:
        Tuple (퍼즐_문자열, 풀이_문자열_리스트, 메타데이터)
    """
    if difficulty not in DIFFICULTY_CONFIGS:
        raise ValueError(f"잘못된 난이도: {difficulty}")

    config = DIFFICULTY_CONFIGS[difficulty]
    rng = random.Random(seed)

    solution_seed = rng.randint(0, 1_000_000_000)
    solution = generate_complete(solution_seed)
    puzzle = [row[:] for row in solution]

    removal_groups = _create_removal_groups(rng, config.symmetry)
    removal_groups = _prioritize_removal_groups(removal_groups, puzzle)

    for group in removal_groups:
        temp_puzzle = [row[:] for row in puzzle]
        for i, j in group:
            temp_puzzle[i][j] = 0

        solution_count = count_solutions(temp_puzzle, limit=MAX_SOLUTIONS + 1)

        if solution_count == 1:
            puzzle = temp_puzzle

        current_givens = count_givens(puzzle)

        if current_givens <= config.min_givens:
            break

        if current_givens <= config.target_givens:
            if not config.minimal:
                break

    solutions = find_all_solutions(puzzle, MAX_SOLUTIONS)
    if not solutions:
        solutions = [solution]

    puzzle_str = to_string(puzzle)
    solution_strs = [to_string(sol) for sol in solutions]

    metadata = {
        'difficulty': difficulty,
        'givens_count': count_givens(puzzle),
        'solution_count': len(solutions),
        'symmetry': config.symmetry,
        'seed': seed
    }

    return puzzle_str, solution_strs, metadata


def generate_puzzles_by_difficulty(
    difficulties: List[str],
    count_per_difficulty: int = 3,
    base_seed: int = 42
) -> List[dict]:
    """각 난이도별로 여러 퍼즐을 생성합니다."""
    puzzles = []
    puzzle_id = 0

    for difficulty in difficulties:
        print(f"난이도 '{difficulty}' 퍼즐 {count_per_difficulty}개 생성 중")

        for i in range(count_per_difficulty):
            seed = base_seed + puzzle_id * 1000
            puzzle_str, solution_strs, metadata = generate_difficulty_puzzle(difficulty, seed)

            record = {
                'id': f"sudoku_ko_{puzzle_id:04d}",
                'difficulty': difficulty,
                'puzzle': puzzle_str,
                'answer': solution_strs[0],
                'metadata': metadata
            }

            puzzles.append(record)
            puzzle_id += 1

            print(f"  퍼즐 {i+1} 생성 완료: 주어진 숫자 {metadata['givens_count']}개, "
                  f"풀이 {metadata['solution_count']}개")

    return puzzles


# ============================================================================
# 스팟체크 생성
# ============================================================================

def select_spotcheck_positions(canonical_hash: str, secret_hex: str, k: int) -> List[str]:
    """HMAC 기반 K개 위치 선택 (재현 가능)."""
    secret = bytes.fromhex(secret_hex)
    message = canonical_hash.encode('utf-8')
    mac = hmac.new(secret, message, hashlib.sha256).digest()

    seed = int.from_bytes(mac[:8], 'big')
    rng = random.Random(seed)

    all_positions = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(all_positions)
    selected = all_positions[:k]

    return [f"r{r+1}c{c+1}" for r, c in selected]


def make_spotcheck_code(solution: Grid, positions: List[str]) -> int:
    """K개 셀 값의 합계로 정답 코드를 생성합니다."""
    total = 0
    for pos in positions:
        parts = pos[1:].split('c')
        r = int(parts[0]) - 1
        c = int(parts[1]) - 1
        total += solution[r][c]
    return total


# ============================================================================
# 데이터셋 생성
# ============================================================================

def create_dataset_files(num_questions: int):
    """
    스도쿠 퍼즐 데이터셋 파일을 생성합니다.

    인자:
        num_questions: 생성할 문제 수

    반환값:
        Tuple[pd.DataFrame, List[Dict]]: (데이터프레임, JSON 리스트)
    """
    import pandas as pd

    print(f"스도쿠 퍼즐 {num_questions}개 생성 중...")

    difficulties = ['easy', 'medium', 'hard']
    puzzles_per_diff = num_questions // len(difficulties)
    remainder = num_questions % len(difficulties)

    spotcheck_k = {'easy': 3, 'medium': 5, 'hard': 6}
    all_puzzles = []
    puzzle_id = 0

    for i, difficulty in enumerate(difficulties):
        count = puzzles_per_diff + (1 if i < remainder else 0)

        if count == 0:
            continue

        print(f"\n=== 난이도 '{difficulty}' 퍼즐 생성 중 ({count}개 필요) ===")

        for j in range(count):
            seed = 42 + puzzle_id * 1000
            try:
                puzzle_str, solution_strs, metadata = generate_difficulty_puzzle(difficulty, seed)

                solution_grid = from_string(solution_strs[0])
                k = spotcheck_k.get(difficulty, 6)

                canonical_hash = f"sha256:{hashlib.sha256(puzzle_str.encode()).hexdigest()}"
                secret_hex = '0' * 64
                positions = select_spotcheck_positions(canonical_hash, secret_hex, k)
                code = make_spotcheck_code(solution_grid, positions)

                # 스팟체크 안내 텍스트 (한국어)
                positions_str = ", ".join(positions)
                question_text = (
                    f"당신은 스도쿠 풀이 전문가입니다. 다음 스도쿠 퍼즐을 완전히 풀어주세요.\n\n"
                    f"스도쿠 규칙:\n"
                    f"- 9×9 격자를 채우세요. 각 행, 열, 3×3 박스에 1-9 숫자가 들어갑니다\n"
                    f"- 각 숫자는 각 행, 열, 박스에 정확히 한 번만 나타나야 합니다\n\n"
                    f"퍼즐은 81자 문자열로 제공되며, 왼쪽에서 오른쪽으로, 위에서 아래로 읽습니다.\n"
                    f"'.' 또는 '0'은 빈 셀을 나타냅니다.\n\n"
                    f"퍼즐: {puzzle_str}\n\n"
                    f"먼저 퍼즐을 단계별로 완전히 풀어주세요.\n"
                    f"그런 다음, 다음 좌표의 값을 찾아 합계를 계산하세요:\n"
                    f"{positions_str}\n\n"
                    f"중요: 풀이 후 반드시 마지막 줄에 다음 형식으로 작성하세요:\n"
                    f"Answer: [합계 숫자]\n\n"
                    f"이 줄을 절대 생략하지 마세요. 이것이 응답에서 가장 중요한 부분입니다.\n\n"
                    f"예시 (값이 5,3,4,6,7,8인 경우):\n"
                    f"Answer: 33"
                )

                puzzle_data = {
                    'id': f'sudoku_ko_{len(all_puzzles)}',
                    'question': question_text,
                    'answer': str(code),
                    'solution': solution_strs[0],
                    'difficulty': difficulty,
                    'givens_count': metadata['givens_count'],
                    'spotcheck': {
                        'k': k,
                        'positions': positions,
                        'code': code
                    }
                }
                all_puzzles.append(puzzle_data)
                print(f"  [{j+1}/{count}] 주어진 숫자={metadata['givens_count']}, 코드={code}")
            except Exception as e:
                print(f"  [{j+1}/{count}] 실패: {e}")

            puzzle_id += 1

    print(f"\n총 {len(all_puzzles)}개 퍼즐 생성 완료")

    df = pd.DataFrame(all_puzzles)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # CSV 파일 저장
    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "sudoku_ko.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV 파일 생성 완료: {csv_path}")

    # JSONL 파일 저장
    json_dir = PROJECT_ROOT / "data" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = json_dir / "sudoku_ko.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_puzzles:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL 파일 생성 완료: {jsonl_path}")

    return df, all_puzzles


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="스도쿠 퍼즐 생성기 (한국어)")
    parser.add_argument("--num", type=int, default=12, help="생성할 문제 수")

    args = parser.parse_args()

    print("=" * 60)
    print("스도쿠 퍼즐 생성기 (한국어)")
    print("=" * 60)

    create_dataset_files(num_questions=args.num)
