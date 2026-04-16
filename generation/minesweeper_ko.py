"""지뢰찾기(Minesweeper) 퍼즐 생성기 - 한국어 버전

logical-puzzles-me/minesweeper/generator.py 기반 이식:
- solve_puzzle 내부 _stats['nodes'] 역추적 노드 계측
- DIFFICULTY_CONFIGS: min_solver_nodes, max_effective_reveal_ratio, reveal_order
- 이웃 제약 가중 기반 셀 정보 점수
- 정답 포맷: 좌표 목록 "(r1,c1), (r2,c2), ..." (Repo A 호환)
"""

import random
import json
from itertools import product
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set


MAX_SOLUTIONS = 1


def neighbors(r: int, c: int, R: int, C: int) -> List[Tuple[int, int]]:
    result = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C:
                result.append((nr, nc))
    return result


def compute_numbers(mask: List[List[int]]) -> List[List[Optional[int]]]:
    R, C = len(mask), len(mask[0])
    nums = [[0] * C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            if mask[r][c] == 1:
                nums[r][c] = None
            else:
                count = sum(mask[nr][nc] == 1 for nr, nc in neighbors(r, c, R, C))
                nums[r][c] = count
    return nums


def solve_puzzle(
    puzzle_nums: List[List[Optional[int]]],
    R: int,
    C: int,
    max_solutions: int = 2,
    total_mines: Optional[int] = None,
    _stats: Optional[Dict] = None,
) -> List[List[List[int]]]:
    nbs = [[neighbors(r, c, R, C) for c in range(C)] for r in range(R)]

    constraints = []
    for r in range(R):
        for c in range(C):
            v = puzzle_nums[r][c]
            if v is not None:
                constraints.append((r, c, v, nbs[r][c]))

    assignment = [[None] * C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            if puzzle_nums[r][c] is not None:
                assignment[r][c] = 0

    constraints_per_cell = defaultdict(list)
    for idx, (_, _, _, nlist) in enumerate(constraints):
        for (nr, nc) in nlist:
            constraints_per_cell[(nr, nc)].append(idx)

    unknown_cells = [(r, c) for r in range(R) for c in range(C)
                     if assignment[r][c] is None]
    unknown_cells.sort(key=lambda rc: -len(constraints_per_cell[rc]))

    def check_constraints() -> bool:
        for (_, _, v, nlist) in constraints:
            assigned_mines = 0
            unknown_count = 0
            for (nr, nc) in nlist:
                if assignment[nr][nc] is None:
                    unknown_count += 1
                elif assignment[nr][nc] == 1:
                    assigned_mines += 1
            if assigned_mines > v:
                return False
            if assigned_mines + unknown_count < v:
                return False
        return True

    def check_global_mines() -> bool:
        if total_mines is None:
            return True
        assigned_mines = sum(assignment[r][c] == 1
                             for r in range(R) for c in range(C)
                             if assignment[r][c] is not None)
        unknown_count = sum(assignment[r][c] is None
                            for r in range(R) for c in range(C))
        if assigned_mines > total_mines:
            return False
        if assigned_mines + unknown_count < total_mines:
            return False
        return True

    solutions = []

    def backtrack(i: int):
        if _stats is not None:
            _stats['nodes'] = _stats.get('nodes', 0) + 1
        if len(solutions) >= max_solutions:
            return

        if i == len(unknown_cells):
            if not check_global_mines():
                return
            for (_, _, v, nlist) in constraints:
                actual = sum(assignment[nr][nc] == 1 for (nr, nc) in nlist)
                if actual != v:
                    return
            solutions.append([row[:] for row in assignment])
            return

        r, c = unknown_cells[i]

        for val in (0, 1):
            assignment[r][c] = val
            if check_global_mines() and check_constraints():
                backtrack(i + 1)
            assignment[r][c] = None

            if len(solutions) >= max_solutions:
                return

    backtrack(0)
    return solutions


def mask_to_bitstring(mask: List[List[int]]) -> str:
    return "".join("".join(str(cell) for cell in row) for row in mask)


def mask_to_coord_list(mask: List[List[int]]) -> List[Tuple[int, int]]:
    coords = []
    for r, row in enumerate(mask):
        for c, cell in enumerate(row):
            if cell == 1:
                coords.append((r, c))
    return coords


def coords_to_answer_string(coords: List[Tuple[int, int]]) -> str:
    return ", ".join(f"({r},{c})" for r, c in sorted(coords))


class DifficultyPuzzleGenerator:
    DIFFICULTY_CONFIGS = {
        'easy': {
            'grid_size': (6, 6),
            'mine_ratio': 0.15,
            'reveal_ratio': 0.28,
            'max_effective_reveal_ratio': 0.55,
            'min_solver_nodes': 20,
            'reveal_order': 'balanced',
            'description': '6x6 격자, 중간 수준 모호성',
        },
        'medium': {
            'grid_size': (7, 7),
            'mine_ratio': 0.16,
            'reveal_ratio': 0.22,
            'max_effective_reveal_ratio': 0.48,
            'min_solver_nodes': 50,
            'reveal_order': 'low_info',
            'description': '7x7 격자, 낮은 정보량 공개',
        },
        'hard': {
            'grid_size': (8, 8),
            'mine_ratio': 0.17,
            'reveal_ratio': 0.18,
            'max_effective_reveal_ratio': 0.42,
            'min_solver_nodes': 100,
            'reveal_order': 'low_info',
            'description': '8x8 격자, 희소한 낮은 정보량 공개',
        },
    }

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.seed = seed

    def _rank_cells_by_information(self, nums, mask, R, C):
        safe_cells = [(r, c) for r in range(R) for c in range(C) if mask[r][c] == 0]

        def cell_info_score(pos):
            r, c = pos
            num = nums[r][c]
            neighbor_count = len(neighbors(r, c, R, C))
            if num == 0:
                return neighbor_count * 2
            if num == neighbor_count:
                return neighbor_count * 2
            return abs(num - neighbor_count / 2) * 2 + 1

        safe_cells.sort(key=cell_info_score, reverse=True)
        return safe_cells

    def _order_ranked_cells(self, ranked_cells, difficulty):
        order = self.DIFFICULTY_CONFIGS[difficulty].get('reveal_order', 'high_info')
        if order == 'high_info':
            return list(ranked_cells)
        if order == 'low_info':
            return list(reversed(ranked_cells))
        if order == 'balanced':
            cells = list(ranked_cells)
            out = []
            lo, hi = 0, len(cells) - 1
            take_high = True
            while lo <= hi:
                if take_high:
                    out.append(cells[lo]); lo += 1
                else:
                    out.append(cells[hi]); hi -= 1
                take_high = not take_high
            return out
        return list(ranked_cells)

    def _count_solutions_fast(self, puzzle, R, C, total_mines):
        sols = solve_puzzle(puzzle, R, C, max_solutions=MAX_SOLUTIONS + 1, total_mines=total_mines)
        return len(sols)

    def generate_puzzle_with_difficulty(self, difficulty, puzzle_id, max_attempts=300):
        if difficulty not in self.DIFFICULTY_CONFIGS:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        config = self.DIFFICULTY_CONFIGS[difficulty]
        R, C = config['grid_size']
        total_cells = R * C
        num_mines = max(1, int(total_cells * config['mine_ratio']))
        safe_cells_count = total_cells - num_mines
        target_reveals = max(2, int(safe_cells_count * config['reveal_ratio']))
        max_reveals = safe_cells_count

        for _ in range(max_attempts):
            cells = [(r, c) for r in range(R) for c in range(C)]
            mine_positions = set(self.rng.sample(cells, num_mines))
            mask = [[1 if (r, c) in mine_positions else 0 for c in range(C)] for r in range(R)]
            nums = compute_numbers(mask)

            ranked_cells = self._order_ranked_cells(
                self._rank_cells_by_information(nums, mask, R, C),
                difficulty,
            )

            puzzle = [[None] * C for _ in range(R)]
            revealed: Set[Tuple[int, int]] = set()

            initial_reveals = min(target_reveals // 2, len(ranked_cells))
            for i in range(initial_reveals):
                r, c = ranked_cells[i]
                puzzle[r][c] = nums[r][c]
                revealed.add((r, c))

            solution_count = self._count_solutions_fast(puzzle, R, C, num_mines)

            reveal_idx = initial_reveals
            while solution_count > MAX_SOLUTIONS and reveal_idx < len(ranked_cells):
                r, c = ranked_cells[reveal_idx]
                puzzle[r][c] = nums[r][c]
                revealed.add((r, c))
                reveal_idx += 1
                solution_count = self._count_solutions_fast(puzzle, R, C, num_mines)

                if len(revealed) >= max_reveals:
                    break

            if solution_count == 1:
                step_stats = {'nodes': 0}
                solutions_with_stats = solve_puzzle(
                    puzzle, R, C, max_solutions=2, total_mines=num_mines,
                    _stats=step_stats,
                )
                solutions = solutions_with_stats[:1]

                effective_reveal_ratio = len(revealed) / safe_cells_count
                if effective_reveal_ratio > config.get('max_effective_reveal_ratio', 1.0):
                    continue
                if step_stats['nodes'] < config.get('min_solver_nodes', 0):
                    continue

                puzzle_display = []
                for row in puzzle:
                    row_str = ''.join(
                        str(cell) if cell is not None else '#'
                        for cell in row
                    )
                    puzzle_display.append(row_str)

                bitstring = mask_to_bitstring(solutions[0])
                coord_list = mask_to_coord_list(solutions[0])
                answer_str = coords_to_answer_string(coord_list)

                return {
                    'id': puzzle_id,
                    'difficulty': difficulty,
                    'rows': R,
                    'cols': C,
                    'total_mines': num_mines,
                    'puzzle': puzzle_display,
                    'answer': answer_str,
                    'solution': bitstring,
                    'answer_type': 'coord_list',
                    'description': f"{R}x{C} 격자, 지뢰 {num_mines}개",
                    'cells_revealed': len(revealed),
                    'step_metrics': {
                        'solver_backtrack_nodes': step_stats['nodes'],
                        'unrevealed_cells': total_cells - len(revealed),
                        'grid_size': total_cells,
                        'effective_reveal_ratio': effective_reveal_ratio,
                        'configured_reveal_ratio': config['reveal_ratio'],
                        'max_effective_reveal_ratio': config.get('max_effective_reveal_ratio', 1.0),
                    },
                }

        return None


def format_puzzle_grid_labeled(puzzle_rows: List[str]) -> str:
    if not puzzle_rows:
        return ""
    C = len(puzzle_rows[0])
    header = "  " + " ".join(f"c{c}" for c in range(C))
    lines = [header]
    for r, row in enumerate(puzzle_rows):
        cells = " ".join(row[c] for c in range(C))
        lines.append(f"r{r} {cells}")
    return "\n".join(lines)


def create_prompt(puzzle_data: Dict) -> str:
    grid = format_puzzle_grid_labeled(puzzle_data['puzzle'])
    rows = puzzle_data['rows']
    cols = puzzle_data['cols']
    mines = puzzle_data['total_mines']

    mine_info = f"2. 전체 지뢰 수: {mines}개가 격자에 숨겨져 있습니다"
    task_info = f"모든 {mines}개 지뢰의 정확한 위치를 찾으세요."
    uniqueness_info = "\n6. 이 퍼즐은 정확히 하나의 고유한 해를 가집니다"

    return f"""다음 규칙에 따라 지뢰찾기 퍼즐을 풀어주세요:

게임 규칙:
1. 격자 크기: {rows}행 × {cols}열 (0-인덱스)
{mine_info}
3. 각 공개된 숫자(0-8)는 인접한 8개 셀 중 지뢰가 있는 셀의 수를 나타냅니다
4. '#'은 지뢰이거나 안전한 셀일 수 있는 숨겨진 셀입니다
5. 인접 셀은 가로, 세로, 대각선 8방향을 모두 포함합니다{uniqueness_info}

중요한 신뢰성 안내:
- 아래 퍼즐은 machine-verified 되었고 내부적으로 일관됩니다.
- 아래의 행/열 라벨과 공백으로 구분된 셀 표기가 authoritative 합니다.
- 모든 행은 이미 올바른 길이를 가집니다.
- 국소적인 모순이 보이면 퍼즐을 잘못되었다고 판단하지 말고 추론을 다시 점검하세요.

퍼즐:
아래 격자는 행/열 라벨을 포함합니다.
각 퍼즐 행은 rN으로 시작하며 각 셀은 공백으로 구분됩니다.
{grid}

과제:
{task_info}

단계별로 풀어주세요:
1. 각 숫자 셀을 분석하여 인접한 어떤 셀이 지뢰여야 하는지 추론합니다
2. 셀 간 제약을 전파합니다
3. 모든 지뢰 좌표를 확정합니다

출력 형식:
- 어떤 셀이 지뢰인지 추론 과정을 서술하세요
- 모든 지뢰 좌표를 (행, 열) 쌍으로 행 순서대로 정렬하여 나열하세요
- 마지막 줄은 "Answer: (r1,c1), (r2,c2), ..." 형식으로 작성하세요
"""


def create_dataset_files(num_questions: int):
    import pandas as pd

    print(f"{num_questions}개의 지뢰찾기 퍼즐을 생성합니다...")

    generator = DifficultyPuzzleGenerator(seed=42)

    difficulties = ['easy', 'medium', 'hard']
    puzzles_per_diff = num_questions // len(difficulties)
    remainder = num_questions % len(difficulties)

    all_puzzles = []

    for i, difficulty in enumerate(difficulties):
        count = puzzles_per_diff + (1 if i < remainder else 0)
        if count == 0:
            continue

        print(f"\n=== {difficulty} 퍼즐 생성 중 ({count}개 필요) ===")

        for j in range(count):
            puzzle_id = f"minesweeper_ko_{len(all_puzzles)}"

            puzzle_generated = False
            for seed_offset in range(10):
                generator.rng = random.Random(
                    generator.seed + seed_offset + j * 100 + i * 10_000_000
                )

                result = generator.generate_puzzle_with_difficulty(
                    difficulty=difficulty,
                    puzzle_id=puzzle_id,
                )

                if result:
                    result['question'] = create_prompt(result)
                    all_puzzles.append(result)
                    print(f"  [{j+1}/{count}] {result['description']}, answer={result['answer']}")
                    puzzle_generated = True
                    break

            if not puzzle_generated:
                print(f"  [{j+1}/{count}] 생성 실패")

    print(f"\n총 {len(all_puzzles)}개의 퍼즐을 생성했습니다")

    df = pd.DataFrame(all_puzzles)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "minesweeper_ko.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV 파일 생성 완료: {csv_path}")

    # JSONL
    json_dir = PROJECT_ROOT / "data" / "jsonl"
    json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = json_dir / "minesweeper_ko.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_puzzles:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL 파일 생성 완료: {jsonl_path}")

    return df, all_puzzles


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="지뢰찾기(Minesweeper) 퍼즐 생성기 - 한국어")
    parser.add_argument("--num", type=int, default=12, help="생성할 문제 수")

    args = parser.parse_args()

    create_dataset_files(num_questions=args.num)
