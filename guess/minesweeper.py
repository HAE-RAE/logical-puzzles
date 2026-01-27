"""지뢰찾기(Minesweeper) 퍼즐 생성 및 평가 모듈

유일해가 보장된 지뢰찾기 퍼즐을 생성하고 LLM 평가를 위한 데이터셋을 만듭니다.
"""

import random
import json
from pathlib import Path
from typing import List, Tuple, Optional, Set
from collections import defaultdict
from itertools import product


# ============================================================================
# 타입 정의
# ============================================================================

Grid = List[List[int]]  # 0=safe, 1=mine
PuzzleGrid = List[List[Optional[int]]]  # None=hidden, 0-8=revealed number


# ============================================================================
# 그리드 기본 함수
# ============================================================================

def neighbors(r: int, c: int, rows: int, cols: int) -> List[Tuple[int, int]]:
    """셀 (r, c)의 8방향 이웃 좌표 반환"""
    result = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                result.append((nr, nc))
    return result


def compute_numbers(mine_grid: Grid) -> PuzzleGrid:
    """
    지뢰 배치로부터 숫자 힌트 계산

    Args:
        mine_grid: 1=지뢰, 0=안전

    Returns:
        숫자 그리드 (지뢰=None, 안전=이웃 지뢰 개수)
    """
    rows, cols = len(mine_grid), len(mine_grid[0])
    numbers = [[0] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if mine_grid[r][c] == 1:
                numbers[r][c] = None  # 지뢰 셀
            else:
                count = sum(mine_grid[nr][nc] == 1
                           for nr, nc in neighbors(r, c, rows, cols))
                numbers[r][c] = count

    return numbers


# ============================================================================
# 유일해 검증 솔버
# ============================================================================

def solve_puzzle(
    puzzle_grid: PuzzleGrid,
    rows: int,
    cols: int,
    total_mines: int,
    max_solutions: int = 2
) -> List[Grid]:
    """
    백트래킹으로 지뢰찾기 퍼즐 해결

    Args:
        puzzle_grid: None=숨김, 0-8=드러난 숫자
        rows, cols: 그리드 크기
        total_mines: 전체 지뢰 개수
        max_solutions: 최대 해 개수 (조기 종료)

    Returns:
        해 리스트 (각 해는 Grid 형식)
    """
    # 이웃 사전 계산
    nbs = [[neighbors(r, c, rows, cols) for c in range(cols)] for r in range(rows)]

    # 제약 조건 수집
    constraints = []
    for r in range(rows):
        for c in range(cols):
            v = puzzle_grid[r][c]
            if v is not None:
                constraints.append((r, c, v, nbs[r][c]))

    # 초기 할당: 드러난 셀은 안전(0)
    assignment = [[None] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if puzzle_grid[r][c] is not None:
                assignment[r][c] = 0

    # 변수 순서: 제약이 많은 셀부터
    constraints_per_cell = defaultdict(list)
    for idx, (_, _, _, nlist) in enumerate(constraints):
        for (nr, nc) in nlist:
            constraints_per_cell[(nr, nc)].append(idx)

    unknown_cells = [(r, c) for r in range(rows) for c in range(cols)
                     if assignment[r][c] is None]
    unknown_cells.sort(key=lambda rc: -len(constraints_per_cell[rc]))

    def check_constraints() -> bool:
        """현재 할당이 제약을 만족할 수 있는지 확인"""
        for (rr, cc, v, nlist) in constraints:
            assigned_mines = 0
            unknown_count = 0
            for (nr, nc) in nlist:
                if assignment[nr][nc] is None:
                    unknown_count += 1
                elif assignment[nr][nc] == 1:
                    assigned_mines += 1

            # 지뢰가 너무 많음
            if assigned_mines > v:
                return False
            # 남은 셀로 제약 만족 불가
            if assigned_mines + unknown_count < v:
                return False
        return True

    def check_global_mines() -> bool:
        """전체 지뢰 개수 제약 확인"""
        assigned_mines = sum(assignment[r][c] == 1
                           for r in range(rows) for c in range(cols)
                           if assignment[r][c] is not None)
        unknown_count = sum(assignment[r][c] is None
                          for r in range(rows) for c in range(cols))

        if assigned_mines > total_mines:
            return False
        if assigned_mines + unknown_count < total_mines:
            return False
        return True

    solutions = []

    def backtrack(i: int):
        """백트래킹 탐색"""
        if len(solutions) >= max_solutions:
            return

        if i == len(unknown_cells):
            # 최종 검증
            if not check_global_mines():
                return

            # 모든 제약 정확히 만족하는지 확인
            for (_, _, v, nlist) in constraints:
                actual = sum(assignment[nr][nc] == 1 for (nr, nc) in nlist)
                if actual != v:
                    return

            # 유효한 해 발견
            solutions.append([row[:] for row in assignment])
            return

        r, c = unknown_cells[i]

        # 0 시도 (안전)
        assignment[r][c] = 0
        if check_constraints() and check_global_mines():
            backtrack(i + 1)
        assignment[r][c] = None

        if len(solutions) >= max_solutions:
            return

        # 1 시도 (지뢰)
        assignment[r][c] = 1
        if check_constraints() and check_global_mines():
            backtrack(i + 1)
        assignment[r][c] = None

    backtrack(0)
    return solutions


def has_unique_solution(puzzle_grid: PuzzleGrid, rows: int, cols: int,
                        total_mines: int) -> bool:
    """퍼즐이 유일해를 가지는지 확인"""
    solutions = solve_puzzle(puzzle_grid, rows, cols, total_mines, max_solutions=2)
    return len(solutions) == 1


# ============================================================================
# 퍼즐 생성
# ============================================================================

def generate_random_mines(rows: int, cols: int, num_mines: int,
                         seed: int = None) -> Grid:
    """무작위 지뢰 배치 생성"""
    rng = random.Random(seed)

    # 모든 셀 위치
    cells = list(product(range(rows), range(cols)))
    rng.shuffle(cells)

    # 지뢰 배치
    mine_grid = [[0] * cols for _ in range(rows)]
    for i in range(num_mines):
        r, c = cells[i]
        mine_grid[r][c] = 1

    return mine_grid


def minimize_clues(mine_grid: Grid, rows: int, cols: int, num_mines: int,
                   max_attempts: int = 100) -> PuzzleGrid:
    """
    유일해를 유지하면서 힌트를 최소화

    Args:
        mine_grid: 지뢰 배치
        rows, cols: 그리드 크기
        num_mines: 지뢰 개수
        max_attempts: 최대 시도 횟수

    Returns:
        최소 힌트 퍼즐 (None=숨김, 0-8=드러남)
    """
    # 모든 숫자 계산
    full_numbers = compute_numbers(mine_grid)

    # 초기: 모든 안전 셀 드러남
    puzzle = [[full_numbers[r][c] for c in range(cols)] for r in range(rows)]

    # 안전 셀 위치들
    safe_cells = [(r, c) for r in range(rows) for c in range(cols)
                  if mine_grid[r][c] == 0]

    # 무작위 순서로 숨기기 시도
    random.shuffle(safe_cells)

    for r, c in safe_cells:
        if puzzle[r][c] is None:
            continue

        # 숨기기 시도
        original_value = puzzle[r][c]
        puzzle[r][c] = None

        # 유일해 검증
        if not has_unique_solution(puzzle, rows, cols, num_mines):
            # 유일해가 깨지면 복원
            puzzle[r][c] = original_value

    return puzzle


def generate_puzzle(rows: int = 6, cols: int = 6, num_mines: int = 8,
                   difficulty: str = 'easy', seed: int = None,
                   max_attempts: int = 20) -> dict:
    """
    유일해가 보장된 지뢰찾기 퍼즐 생성

    Args:
        rows, cols: 그리드 크기
        num_mines: 지뢰 개수
        difficulty: 난이도 ('easy', 'medium', 'hard')
        seed: 랜덤 시드
        max_attempts: 최대 시도 횟수

    Returns:
        퍼즐 데이터 (dict)
    """
    rng = random.Random(seed)

    for attempt in range(max_attempts):
        # 지뢰 배치
        mine_grid = generate_random_mines(rows, cols, num_mines,
                                         rng.randint(0, 1_000_000))

        # 힌트 최소화
        puzzle_grid = minimize_clues(mine_grid, rows, cols, num_mines)

        # 드러난 셀 개수 계산
        revealed = sum(1 for r in range(rows) for c in range(cols)
                      if puzzle_grid[r][c] is not None)

        # 최소 힌트 수 확인 (난이도 조절)
        min_clues = {
            'easy': rows * cols * 0.4,
            'medium': rows * cols * 0.3,
            'hard': rows * cols * 0.2
        }.get(difficulty, rows * cols * 0.3)

        if revealed >= min_clues:
            # 문자열 형식으로 변환
            puzzle_str = []
            solution_str = []

            for r in range(rows):
                puzzle_row = ''
                solution_row = ''
                for c in range(cols):
                    # 퍼즐: 숨김='#', 드러남=숫자
                    if puzzle_grid[r][c] is None:
                        puzzle_row += '#'
                    else:
                        puzzle_row += str(puzzle_grid[r][c])

                    # 솔루션: 지뢰=1, 안전=0
                    solution_row += str(mine_grid[r][c])

                puzzle_str.append(puzzle_row)
                solution_str.append(solution_row)

            return {
                'id': f'{difficulty}_{rows}x{cols}_{seed if seed else 0:04d}',
                'difficulty': difficulty,
                'rows': rows,
                'cols': cols,
                'mines': num_mines,
                'puzzle': puzzle_str,
                'solution': solution_str,
                'revealed_cells': revealed,
                'clue_density': revealed / (rows * cols),
                'mine_density': num_mines / (rows * cols)
            }

    raise Exception(f"Failed to generate puzzle after {max_attempts} attempts")


# ============================================================================
# 데이터셋 생성
# ============================================================================

def generate_dataset(output_path: str = None, num_puzzles_per_level: int = 5,
                    seed: int = 2025) -> List[dict]:
    """
    난이도별 평가 데이터셋 생성

    Args:
        output_path: 출력 파일 경로
        num_puzzles_per_level: 난이도당 퍼즐 개수
        seed: 랜덤 시드

    Returns:
        퍼즐 리스트
    """
    configs = [
        # (rows, cols, mines, difficulty)
        (6, 6, 8, 'easy'),
        (8, 8, 12, 'easy'),
        (8, 8, 15, 'medium'),
        (10, 10, 18, 'medium'),
        (10, 10, 22, 'hard'),
    ]

    puzzles = []

    print("="*70)
    print("지뢰찾기 평가 데이터셋 생성")
    print("="*70)

    for config_idx, (rows, cols, mines, difficulty) in enumerate(configs):
        print(f"\n[{config_idx+1}/{len(configs)}] {difficulty} {rows}×{cols} (지뢰 {mines}개)")

        for i in range(num_puzzles_per_level):
            puzzle_seed = seed + config_idx * 1000 + i
            try:
                puzzle = generate_puzzle(
                    rows=rows,
                    cols=cols,
                    num_mines=mines,
                    difficulty=difficulty,
                    seed=puzzle_seed
                )
                puzzles.append(puzzle)
                print(f"  ✓ {puzzle['id']}: {puzzle['revealed_cells']}개 힌트")
            except Exception as e:
                print(f"  ✗ 생성 실패 (seed={puzzle_seed}): {e}")

    # Re-assign ids to follow index-based naming convention
    for idx, puzzle in enumerate(puzzles):
        puzzle['id'] = f'minesweeper_{idx}'
    
    # 저장
    if output_path is None:
        output_dir = Path(__file__).parent.parent / 'data' / 'minesweeper'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'eval_dataset.jsonl'

    with open(output_path, 'w', encoding='utf-8') as f:
        for puzzle in puzzles:
            f.write(json.dumps(puzzle, ensure_ascii=False) + '\n')

    print(f"\n{'='*70}")
    print(f"✓ {len(puzzles)}개 퍼즐 생성 완료")
    print(f"✓ 저장 위치: {output_path}")
    print(f"{'='*70}")

    # 통계 출력
    by_difficulty = defaultdict(int)
    for p in puzzles:
        by_difficulty[p['difficulty']] += 1

    print("\n난이도별 분포:")
    for diff, count in sorted(by_difficulty.items()):
        print(f"  {diff:8s}: {count}개")

    return puzzles


# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == '__main__':
    # 데이터셋 생성
    generate_dataset(num_puzzles_per_level=5, seed=2025)
