"""Minesweeper Puzzle Generator (EN).

Ported from logical-puzzles-me/minesweeper/generator.py:
- solve_puzzle with _stats['nodes'] backtrack instrumentation
- DIFFICULTY_CONFIGS with min_solver_nodes, max_effective_reveal_ratio, reveal_order
- Neighbor-aware cell info ranking
- Answer format: coordinate-list "(r1,c1), (r2,c2), ..." (Repo A compatible)
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
            # v3 recalibration: reveal 0.55 (from 0.45), nodes 0 (from 5) for very
            # LLM-friendly easy. gemini v2 dropped to 47% — making clearly easier.
            'grid_size': (5, 5),
            'mine_ratio': 0.12,
            'reveal_ratio': 0.55,
            'max_effective_reveal_ratio': 0.75,
            'min_solver_nodes': 0,
            'reveal_order': 'balanced',
            'description': '5x5 grid, mostly-revealed cells',
        },
        'medium': {
            # v3 recalibration: 7x7 → 6x6 + mine 0.14 + reveal 0.30 + nodes 25 so
            # gpt-5.4-mini (v2: 3%) can reach ~30-50% range and easy≥medium holds.
            'grid_size': (6, 6),
            'mine_ratio': 0.14,
            'reveal_ratio': 0.30,
            'max_effective_reveal_ratio': 0.55,
            'min_solver_nodes': 25,
            'reveal_order': 'low_info',
            'description': '6x6 grid, moderate reveals',
        },
        'hard': {
            # v2 recalibration: slightly higher mine density + lower reveal + higher node bar.
            # Stayed at 8x8 (not 9x9) because 9x9 + mine 0.20 exhausted generation time in pre-test.
            'grid_size': (8, 8),
            'mine_ratio': 0.18,
            'reveal_ratio': 0.16,
            'max_effective_reveal_ratio': 0.38,
            'min_solver_nodes': 120,
            'reveal_order': 'low_info',
            'description': '8x8 grid, denser mines and sparser reveals',
        },
    }

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.seed = seed

    def _rank_cells_by_information(self, nums: List[List[Optional[int]]],
                                     mask: List[List[int]], R: int, C: int) -> List[Tuple[int, int]]:
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

    def _order_ranked_cells(self, ranked_cells: List[Tuple[int, int]], difficulty: str) -> List[Tuple[int, int]]:
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

    def _count_solutions_fast(self, puzzle, R, C, total_mines) -> int:
        sols = solve_puzzle(puzzle, R, C, max_solutions=MAX_SOLUTIONS + 1, total_mines=total_mines)
        return len(sols)

    def generate_puzzle_with_difficulty(self, difficulty: str, puzzle_id: str,
                                        max_attempts: int = 300) -> Optional[Dict]:
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
                sft_solution = _build_minesweeper_solution_en(
                    puzzle_rows=puzzle_display,
                    R=R,
                    C=C,
                    total_mines=num_mines,
                    coord_list=coord_list,
                    answer_str=answer_str,
                    difficulty=difficulty,
                    bitstring=bitstring,
                    solver_nodes=step_stats['nodes'],
                )

                return {
                    'id': puzzle_id,
                    'difficulty': difficulty,
                    'rows': R,
                    'cols': C,
                    'total_mines': num_mines,
                    'puzzle': puzzle_display,
                    'answer': answer_str,
                    'solution': sft_solution,
                    'bitstring': bitstring,
                    'answer_type': 'coord_list',
                    'description': f"{R}x{C} grid with {num_mines} mines",
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


SFT_SOLUTION_RUBRIC_EN = (
    "STEP0=meta · STEP1=given · STEP2=worked solution · "
    "STEP3=answer and verification"
)


def _build_minesweeper_solution_en(
    puzzle_rows: List[str],
    R: int,
    C: int,
    total_mines: int,
    coord_list: List[Tuple[int, int]],
    answer_str: str,
    difficulty: str,
    bitstring: str,
    solver_nodes: int,
) -> str:
    """SFT teacher trace: minesweeper mine-confirmation SEGs."""
    coord_sorted = sorted(coord_list)

    clue_lookup: Dict[Tuple[int, int], int] = {}
    for r, row in enumerate(puzzle_rows):
        for c, ch in enumerate(row):
            if ch.isdigit():
                clue_lookup[(r, c)] = int(ch)

    def adjacent_clues_for(rr: int, cc: int) -> List[Tuple[int, int, int]]:
        out = []
        for nr, nc in neighbors(rr, cc, R, C):
            if (nr, nc) in clue_lookup:
                out.append((nr, nc, clue_lookup[(nr, nc)]))
        return out

    lines: List[str] = [
        SFT_SOLUTION_RUBRIC_EN,
        "[STEP 0] Problem meta",
        f"  - Difficulty: {difficulty}",
        f"  - Grid: {R} rows x {C} cols",
        f"  - Mines: {total_mines} · revealed number cells: {len(clue_lookup)}",
        f"  - Solver backtrack nodes: {solver_nodes}",
        "  - Final answer is confirmed in [STEP 3]",
        "[STEP 1] Given",
        "  - Rule: each revealed number = count of mines in its 8 neighbors.",
        "  - Rule: '#' is a hidden cell (mine or safe).",
        "  - Puzzle rows:",
    ]
    for r, row in enumerate(puzzle_rows):
        lines.append(f"    r{r}: {' '.join(row)}")

    lines.append("[STEP 2] Worked solution")
    lines.append(
        f"  · Summary: propagate number constraints · "
        f"{sum(row.count('#') for row in puzzle_rows)} hidden cells / "
        f"{total_mines} mines -> unique model · {len(coord_sorted)} SEGs"
    )
    for i, (r, c) in enumerate(coord_sorted, 1):
        clue_hits = adjacent_clues_for(r, c)
        if clue_hits:
            clue_text = ", ".join(f"({cr},{cc})={cv}" for cr, cc, cv in clue_hits)
            explain = f"adjacent clues {clue_text} force this cell to be a mine"
        else:
            explain = "forced by the global remaining-mines count constraint"
        lines.append(f"    [SEG {i}] mine at ({r},{c}): {explain}")

    lines.extend([
        "[STEP 3] Answer and verification",
        f"  - Final answer: {answer_str}",
        f"  - Total mines = {total_mines} must match the count of confirmed cells.",
        "  - For every revealed number, its neighborhood must contain exactly that many of the listed mines.",
        f"  - Internal bitstring: {bitstring[:48]}{'…' if len(bitstring) > 48 else ''}",
    ])
    return "\n".join(lines)


def format_puzzle_grid_labeled(puzzle_rows: List[str]) -> str:
    """Render grid with r/c labels and spaced cells per Repo A prompt style."""
    if not puzzle_rows:
        return ""
    C = len(puzzle_rows[0])
    header = "   " + " ".join(f"c{c}" for c in range(C))
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

    mine_info = f"2. Total mines: {mines} hidden in the grid"
    task_info = f"Determine the exact location of ALL {mines} mines."
    uniqueness_info = "\n6. This puzzle has exactly one unique solution"

    return f"""You are solving a minesweeper puzzle with the following rules:

GAME RULES:
1. Grid size: {rows} rows × {cols} columns (0-indexed)
{mine_info}
3. Each revealed number (0-8) indicates how many of its 8 neighbors contain mines
4. '#' represents a hidden cell that could be either a mine or safe
5. Adjacent cells include all 8 directions: horizontal, vertical, and diagonal{uniqueness_info}

IMPORTANT RELIABILITY NOTE:
- The puzzle below is machine-verified and internally consistent.
- The row/column labels and spaced cells below are authoritative.
- Every displayed row already has the correct length.
- Do not reject the puzzle as malformed; if a local contradiction appears, revisit your deduction instead.

PUZZLE:
The grid below includes explicit row/column labels.
Each puzzle row starts with rN and each cell is separated by spaces.
{grid}

YOUR TASK:
{task_info}

WORK THROUGH THIS STEP BY STEP:
1. Analyze each numbered cell to deduce which neighbors must be mines
2. Propagate constraints between cells
3. Identify the full set of mine coordinates

OUTPUT FORMAT:
- Show your reasoning about which cells are mines
- List all mine coordinates as (row, col) pairs sorted by row then column
- End your response with a line of the form: "Answer: (r1,c1), (r2,c2), ..."
"""


def create_dataset_files(num_questions: int):
    import pandas as pd

    print(f"Generating {num_questions} minesweeper puzzles...")

    generator = DifficultyPuzzleGenerator(seed=42)

    difficulties = ['easy', 'medium', 'hard']
    puzzles_per_diff = num_questions // len(difficulties)
    remainder = num_questions % len(difficulties)

    all_puzzles = []

    # Per-difficulty dedup: identical puzzle grids (same `puzzle` rows tuple) are
    # rejected → 같은 input 이 한 difficulty bucket 안에서 두 번 나오지 않게.
    for i, difficulty in enumerate(difficulties):
        count = puzzles_per_diff + (1 if i < remainder else 0)
        if count == 0:
            continue

        print(f"\n=== Generating {difficulty} puzzles ({count} needed) ===")

        seen_keys: set = set()
        diff_success = 0
        attempt = 0
        # 각 attempt 가 seed_offset 10개를 시도. 4×4 easy 같이 grid 공간이 좁은
        # 경우 무한 루프 방지로 50× 한도.
        max_attempts = count * 50

        while diff_success < count and attempt < max_attempts:
            attempt += 1
            puzzle_id = f"minesweeper_en_{difficulty}_{diff_success:04d}"

            puzzle_generated = False
            for seed_offset in range(10):
                generator.rng = random.Random(
                    generator.seed + seed_offset + attempt * 100 + i * 10_000_000
                )

                result = generator.generate_puzzle_with_difficulty(
                    difficulty=difficulty,
                    puzzle_id=puzzle_id,
                )

                if result:
                    # dedup key: 모델이 보는 puzzle grid
                    key = tuple(result['puzzle']) if isinstance(result['puzzle'], list) else result['puzzle']
                    if key in seen_keys:
                        puzzle_generated = False
                        continue
                    seen_keys.add(key)

                    result['question'] = create_prompt(result)
                    puzzle_data = {
                        "id": result["id"],
                        "question": result["question"],
                        "answer": result["answer"],
                        "solution": result["solution"],
                        "difficulty": result["difficulty"],
                    }
                    all_puzzles.append(puzzle_data)
                    print(f"  [{diff_success+1}/{count}] {result['description']}, answer={result['answer']}")
                    puzzle_generated = True
                    diff_success += 1
                    break

        if diff_success < count:
            print(
                f"  ⚠️ dedup retry budget exhausted at "
                f"{diff_success}/{count} for {difficulty}"
            )

    print(f"\nGenerated {len(all_puzzles)} puzzles total")

    df = pd.DataFrame(all_puzzles)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "minesweeper_en.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV file created: {csv_path}")

    # JSONL
    json_dir = PROJECT_ROOT / "data" / "jsonl"
    json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = json_dir / "minesweeper_en.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_puzzles:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL file created: {jsonl_path}")

    return df, all_puzzles


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Minesweeper Puzzle Generator (EN)")
    parser.add_argument("--num", type=int, default=12, help="Number of questions to generate")

    args = parser.parse_args()

    create_dataset_files(num_questions=args.num)
