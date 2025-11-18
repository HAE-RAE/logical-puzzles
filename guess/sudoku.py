"""스도쿠 퍼즐 생성 및 평가 모듈"""

import random
import json
import hmac
import hashlib
from pathlib import Path
from typing import List, Tuple, Set
from dataclasses import dataclass


# ============================================================================
# 타입 정의
# ============================================================================

Grid = List[List[int]]  # 0은 빈칸


# ============================================================================
# 그리드 기본 함수
# ============================================================================

def from_string(s: str) -> Grid:
    """81자 문자열을 9×9 그리드로 변환"""
    s = s.strip()
    if len(s) != 81:
        raise ValueError(f"문자열 길이는 81이어야 함 (현재: {len(s)})")

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
                raise ValueError(f"잘못된 문자 '{ch}' at position {i*9+j}")
        grid.append(row)
    return grid


def to_string(g: Grid, blanks: str = '.') -> str:
    """9×9 그리드를 81자 문자열로 변환"""
    chars = []
    for row in g:
        for val in row:
            if val == 0:
                chars.append(blanks)
            else:
                chars.append(str(val))
    return ''.join(chars)


def copy_grid(g: Grid) -> Grid:
    """그리드 복사"""
    return [row[:] for row in g]


def get_cell_candidates(g: Grid, r: int, c: int) -> Set[int]:
    """셀 (r, c)의 가능한 후보 숫자들"""
    if g[r][c] != 0:
        return set()

    used = set()

    # 행 검사
    for val in g[r]:
        if val != 0:
            used.add(val)

    # 열 검사
    for row in g:
        if row[c] != 0:
            used.add(row[c])

    # 박스 검사
    box_r, box_c = (r // 3) * 3, (c // 3) * 3
    for br in range(box_r, box_r + 3):
        for bc in range(box_c, box_c + 3):
            if g[br][bc] != 0:
                used.add(g[br][bc])

    return set(range(1, 10)) - used


def is_solved(g: Grid) -> bool:
    """완성된 유효한 스도쿠인지 확인"""
    # 빈칸 검사
    for row in g:
        if 0 in row:
            return False

    # 행 검사
    for row in g:
        if set(row) != set(range(1, 10)):
            return False

    # 열 검사
    for c in range(9):
        if set(g[r][c] for r in range(9)) != set(range(1, 10)):
            return False

    # 박스 검사
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
# 유일해 카운터
# ============================================================================

def count_solutions(puzzle: Grid, limit: int = 2) -> int:
    """해의 개수를 센다 (limit에 도달하면 조기 종료)"""
    grid = copy_grid(puzzle)
    count = [0]

    def solve(g: Grid) -> bool:
        if count[0] >= limit:
            return True

        # MRV: 후보가 가장 적은 빈칸 선택
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


# ============================================================================
# 완성 격자 생성
# ============================================================================

def generate_complete(seed: int = None) -> Grid:
    """무작위 완성 스도쿠 격자 생성"""
    rng = random.Random(seed)

    # 기본 라틴 스퀘어
    grid: Grid = []
    for r in range(9):
        row = []
        for c in range(9):
            val = (r * 3 + r // 3 + c) % 9 + 1
            row.append(val)
        grid.append(row)

    # 무작위 변환 적용
    # 숫자 순열
    digits = list(range(1, 10))
    rng.shuffle(digits)
    perm = {i + 1: digits[i] for i in range(9)}

    for r in range(9):
        for c in range(9):
            grid[r][c] = perm[grid[r][c]]

    # 행 셔플 (밴드 내)
    for band in range(3):
        rows_idx = [band * 3, band * 3 + 1, band * 3 + 2]
        rng.shuffle(rows_idx)
        temp = [grid[i] for i in rows_idx]
        for i, row_data in enumerate(temp):
            grid[band * 3 + i] = row_data

    # 열 셔플 (스택 내)
    for stack in range(3):
        cols_idx = [stack * 3, stack * 3 + 1, stack * 3 + 2]
        rng.shuffle(cols_idx)
        for r in range(9):
            temp = [grid[r][c] for c in cols_idx]
            for i, val in enumerate(temp):
                grid[r][stack * 3 + i] = val

    return grid


# ============================================================================
# 단서 제거 (Digging)
# ============================================================================

def get_symmetric_pair(r: int, c: int, symmetry: str) -> Tuple[int, int]:
    """주어진 대칭에서 (r, c)의 쌍이 되는 좌표"""
    if symmetry == 'none':
        return (r, c)
    elif symmetry == 'rot180':
        return (8 - r, 8 - c)
    else:
        return (r, c)


def make_puzzle(solution: Grid, symmetry: str = 'rot180',
                ensure_minimal: bool = False, rng: random.Random = None) -> Grid:
    """완성 격자에서 단서를 제거하여 퍼즐 생성"""
    if rng is None:
        rng = random.Random()

    puzzle = copy_grid(solution)

    # 모든 셀 좌표
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

    return puzzle


# ============================================================================
# 난이도 평가
# ============================================================================

@dataclass
class DifficultyInfo:
    """난이도 정보"""
    label: str
    givens_count: int
    search_nodes: int


def rate_difficulty(puzzle: Grid) -> DifficultyInfo:
    """퍼즐의 난이도를 평가 (간단한 버전)"""
    givens = sum(1 for r in range(9) for c in range(9) if puzzle[r][c] != 0)

    # 백트래킹 복잡도 측정
    grid = copy_grid(puzzle)
    nodes = [0]

    def backtrack(g: Grid) -> bool:
        nodes[0] += 1
        if nodes[0] > 5000:  # 너무 많으면 중단
            return False

        for r in range(9):
            for c in range(9):
                if g[r][c] == 0:
                    cands = get_cell_candidates(g, r, c)
                    if not cands:
                        return False
                    for num in cands:
                        g[r][c] = num
                        if backtrack(g):
                            return True
                        g[r][c] = 0
                    return False
        return True

    backtrack(grid)

    # 난이도 결정
    if givens >= 36:
        label = 'Easy'
    elif givens >= 30:
        label = 'Medium' if nodes[0] <= 50 else 'Hard'
    elif givens >= 26:
        label = 'Hard' if nodes[0] <= 500 else 'Expert'
    elif givens >= 22:
        label = 'Expert' if nodes[0] <= 2000 else 'Extreme'
    else:
        label = 'Extreme'

    return DifficultyInfo(
        label=label,
        givens_count=givens,
        search_nodes=nodes[0]
    )


# ============================================================================
# Spotcheck 생성
# ============================================================================

def select_spotcheck_positions(canonical_hash: str, secret_hex: str, k: int) -> List[str]:
    """HMAC 기반으로 K개 좌표 선택"""
    secret = bytes.fromhex(secret_hex)
    message = canonical_hash.encode('utf-8')
    mac = hmac.new(secret, message, hashlib.sha256).digest()

    seed = int.from_bytes(mac[:8], 'big')
    rng = random.Random(seed)

    all_positions = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(all_positions)
    selected = all_positions[:k]

    return [f"r{r+1}c{c+1}" for r, c in selected]


def make_spotcheck_code(solution: Grid, positions: List[str]) -> str:
    """주어진 좌표의 정답을 K자리 숫자열로 생성"""
    code_chars = []
    for pos in positions:
        parts = pos[1:].split('c')
        r = int(parts[0]) - 1
        c = int(parts[1]) - 1
        code_chars.append(str(solution[r][c]))
    return ''.join(code_chars)


# ============================================================================
# 퍼즐 생성 메인 함수
# ============================================================================

def generate_puzzle(difficulty: str = 'Any', symmetry: str = 'rot180',
                   seed: int = None, k: int = 6) -> dict:
    """단일 퍼즐 생성"""
    rng = random.Random(seed)

    max_attempts = 100
    for attempt in range(max_attempts):
        # 완성 격자 생성
        solution_grid = generate_complete(rng.randint(0, 1_000_000_000))

        # 퍼즐 생성
        puzzle_grid = make_puzzle(solution_grid, symmetry, False, rng)

        # 유일해 검증
        if count_solutions(puzzle_grid, limit=2) != 1:
            continue

        # 난이도 평가
        diff_info = rate_difficulty(puzzle_grid)

        # 난이도 필터
        if difficulty != 'Any' and diff_info.label != difficulty:
            continue

        # 문자열 변환
        puzzle_str = to_string(puzzle_grid)
        solution_str = to_string(solution_grid)

        # Spotcheck 생성
        canonical_hash = f"sha256:{hashlib.sha256(puzzle_str.encode()).hexdigest()}"
        secret_hex = '0' * 64  # 간단한 시크릿
        positions = select_spotcheck_positions(canonical_hash, secret_hex, k)
        code = make_spotcheck_code(solution_grid, positions)

        return {
            'id': f's9_{seed if seed else 0:06d}',
            'puzzle': puzzle_str,
            'solution': solution_str,
            'givens_count': diff_info.givens_count,
            'symmetry': symmetry,
            'difficulty': {
                'label': diff_info.label,
                'search_nodes': diff_info.search_nodes
            },
            'spotcheck': {
                'k': k,
                'positions': positions,
                'code': code
            },
            'canonical_hash': canonical_hash
        }

    raise Exception(f"{difficulty} 퍼즐 생성 실패 ({max_attempts}회 시도)")


def generate_5diff_dataset(output_path: str = None, k: int = 6, seed: int = 2025):
    """5개 난이도별 평가 데이터셋 생성"""
    difficulties = ['Easy', 'Hard', 'Expert', 'Extreme', 'Easy']  # Medium 대신 Easy 2개

    puzzles = []

    print("="*70)
    print("5개 난이도별 스도쿠 평가 데이터 생성")
    print("="*70)

    for i, diff in enumerate(difficulties):
        print(f"\n[{i+1}/5] {diff} 퍼즐 생성 중...")
        try:
            puzzle = generate_puzzle(
                difficulty=diff,
                symmetry='rot180',
                seed=seed + i * 1000,
                k=k
            )
            puzzles.append(puzzle)
            print(f"✓ {diff:8s}: {puzzle['id']}")
            print(f"  Givens: {puzzle['givens_count']}, Code: {puzzle['spotcheck']['code']}")
        except Exception as e:
            print(f"✗ {diff} 생성 실패: {e}")
            return None

    # 저장
    if output_path is None:
        output_dir = Path(__file__).parent.parent / 'data' / 'sudoku'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'eval_5diff_k6.jsonl'

    with open(output_path, 'w', encoding='utf-8') as f:
        for puzzle in puzzles:
            f.write(json.dumps(puzzle, ensure_ascii=False) + '\n')

    print(f"\n{'='*70}")
    print(f"✓ 5개 퍼즐 생성 완료")
    print(f"✓ 저장 위치: {output_path}")
    print(f"{'='*70}")

    return puzzles


# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == '__main__':
    # 5개 난이도별 데이터셋 생성
    generate_5diff_dataset()
