"""Minesweeper Puzzle Generator with Difficulty Levels

Constructive generation: progressively reveals cells while monitoring
solution count, guaranteeing valid puzzle generation.
Includes scoring via weighted coordinate sum format.
"""

MAX_SOLUTIONS = 1  # Only allow exactly 1 solution

import random
import re
import json
from itertools import product
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set


# ============================================================
# Core utility functions
# ============================================================

def neighbors(r: int, c: int, R: int, C: int) -> List[Tuple[int, int]]:
    """Get 8-directional neighbors of cell (r, c) within R x C grid."""
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
    """
    Compute number hints for each cell.
    mask: 1=mine, 0=safe
    Returns: grid where mines are None, safe cells have neighbor mine count (0-8)
    """
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
    total_mines: Optional[int] = None
) -> List[List[List[int]]]:
    """
    Solve minesweeper puzzle using backtracking with constraint propagation.

    Args:
        puzzle_nums: Grid where revealed cells have numbers 0-8, hidden cells are None
        R, C: Grid dimensions
        max_solutions: Stop after finding this many solutions
        total_mines: Optional global constraint on total mine count

    Returns:
        List of solutions (mine masks), up to max_solutions
    """
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
        for (rr, cc, v, nlist) in constraints:
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


def random_mask(R: int, C: int, M: int, rng: random.Random) -> List[List[int]]:
    """Generate random mine placement."""
    cells = list(product(range(R), range(C)))
    mines = set(rng.sample(cells, M))
    return [[1 if (r, c) in mines else 0 for c in range(C)] for r in range(R)]


def mask_to_solution_string(mask: List[List[int]]) -> str:
    """Convert mine mask to a single bitstring format."""
    return "".join("".join(str(cell) for cell in row) for row in mask)


def puzzle_to_string(puzzle: List[List[Optional[int]]]) -> List[str]:
    """Convert puzzle to string format with # for hidden cells."""
    return [''.join(str(cell) if cell is not None else '#' for cell in row)
            for row in puzzle]


# ============================================================
# Scorer functions (weighted coordinate sum)
# ============================================================

def bitstring_to_coordinates(solution_str: str, R: int, C: int) -> Set[Tuple[int, int]]:
    """Convert solution bitstring to coordinate set."""
    coords = set()
    for i, cell in enumerate(solution_str):
        if cell == '1':
            r, c = divmod(i, C)
            coords.add((r, c))
    return coords


def compute_total_sum(coords: Set[Tuple[int, int]], C: int) -> int:
    """Compute weighted coordinate sum: sum(row * C + col) for each mine."""
    if not coords:
        return 0
    return sum(r * C + c for r, c in coords)


def parse_total_sum(output: str) -> Optional[int]:
    """Parse LLM output to extract total sum as single integer."""
    output = re.sub(r'```[a-z]*\n?', '', output)
    output = re.sub(r'```', '', output)
    output = output.strip()

    answer_matches = re.findall(
        r'(?:Answer|Output|Final\s*Answer)\s*[:\s]*(\d+)',
        output, re.IGNORECASE
    )
    if answer_matches:
        return int(answer_matches[-1])

    lines = output.strip().split('\n')
    last_lines = lines[-5:]
    for line in reversed(last_lines):
        match = re.search(r'\b(\d{2,})\b', line.strip())
        if match:
            return int(match.group(1))

    matches = re.findall(r'\b(\d{2,})\b', output)
    if matches:
        return int(matches[-1])

    return None


def score_from_solution(solution_str: str, R: int, C: int, pred_output: str) -> Dict:
    """Score LLM prediction against solution using weighted sum."""
    truth_coords = bitstring_to_coordinates(solution_str, R, C)
    truth_sum = compute_total_sum(truth_coords, C)
    pred_sum = parse_total_sum(pred_output)

    if pred_sum is None:
        return {'score': 0.0, 'truth_sum': truth_sum, 'pred_sum': None, 'error': 'Failed to parse'}

    score = 1.0 if truth_sum == pred_sum else 0.0
    return {'score': score, 'truth_sum': truth_sum, 'pred_sum': pred_sum, 'error': None}


# ============================================================
# Prompt template functions
# ============================================================

def format_puzzle_grid(puzzle_rows: List[str]) -> str:
    """Format puzzle grid as multi-line string."""
    return '\n'.join(puzzle_rows)


def create_prompt(puzzle_data: Dict) -> str:
    """Create prompt for minesweeper puzzle evaluation."""
    puzzle_grid = format_puzzle_grid(puzzle_data['puzzle'])
    rows = puzzle_data['rows']
    cols = puzzle_data['cols']
    mines = puzzle_data.get('total_mines', puzzle_data.get('mines', 0))
    difficulty = puzzle_data.get('difficulty', 'medium')

    if difficulty == 'easy':
        mine_info = f"2. Total mines: {mines} hidden in the grid"
        task_info = f"Determine the exact location of ALL {mines} mines."
    else:
        mine_info = "2. Some cells contain hidden mines"
        task_info = "Determine the exact location of ALL mines."

    if difficulty == 'easy':
        uniqueness_info = "\n6. This puzzle has exactly one unique solution"
    else:
        uniqueness_info = ""

    prompt = f"""You are solving a minesweeper puzzle with the following rules:

GAME RULES:
1. Grid size: {rows} rows x {cols} columns (0-indexed)
{mine_info}
3. Each revealed number (0-8) indicates how many of its 8 neighbors contain mines
4. '#' represents a hidden cell that could be either a mine or safe
5. Adjacent cells include all 8 directions: horizontal, vertical, and diagonal{uniqueness_info}

PUZZLE:
{puzzle_grid}

YOUR TASK:
{task_info}

OUTPUT FORMAT (STRICT):
- Find ALL mine coordinates using 0-based indexing (row 0 to {rows-1}, col 0 to {cols-1})
- For each mine at (row, col), compute its linear index: row * {cols} + col
- Sum all linear indices to get a single number
- Output ONLY this single integer

Example (with {cols} columns per row):
If mines are at (1,2) and (3,0):
- Linear indices: (1*{cols}+2) = {1*cols+2}, (3*{cols}+0) = {3*cols}
- Sum = {1*cols+2} + {3*cols} = {1*cols+2 + 3*cols}
- Output: {1*cols+2 + 3*cols}

Answer:"""

    return prompt


# ============================================================
# Difficulty-based puzzle generator
# ============================================================

class DifficultyPuzzleGenerator:
    """Generate puzzles with varying difficulty levels using progressive revelation."""

    DIFFICULTY_CONFIGS = {
        'easy': {
            'grid_size': (5, 5),
            'mine_ratio': 0.22,
            'reveal_ratio': 0.45,
            'description': 'Small grid, balanced reveals'
        },
        'medium': {
            'grid_size': (6, 6),
            'mine_ratio': 0.33,
            'reveal_ratio': 0.25,
            'description': 'Medium grid, more mines, no mine count hint'
        },
        'hard': {
            'grid_size': (7, 7),
            'mine_ratio': 0.28,
            'reveal_ratio': 0.25,
            'description': 'Larger grid, no mine count hint'
        }
    }

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.seed = seed

    def _rank_cells_by_information(self, nums: List[List[Optional[int]]],
                                     mask: List[List[int]], R: int, C: int) -> List[Tuple[int, int]]:
        """Rank safe cells by information value for progressive revelation."""
        safe_cells = [(r, c) for r in range(R) for c in range(C) if mask[r][c] == 0]

        def cell_info_score(pos):
            r, c = pos
            num = nums[r][c]
            adjacency_bonus = 1 if num > 0 else 0
            return num * 2 + adjacency_bonus

        safe_cells.sort(key=cell_info_score, reverse=True)
        return safe_cells

    def _count_solutions_fast(self, puzzle: List[List[Optional[int]]],
                              R: int, C: int, total_mines: int) -> int:
        solutions = solve_puzzle(puzzle, R, C, max_solutions=MAX_SOLUTIONS + 1, total_mines=total_mines)
        return len(solutions)

    def generate_puzzle_with_difficulty(
        self,
        difficulty: str,
        puzzle_id: str,
        max_attempts: int = 100
    ) -> Optional[Dict]:
        """Generate a puzzle with exactly 1 solution."""
        if difficulty not in self.DIFFICULTY_CONFIGS:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        config = self.DIFFICULTY_CONFIGS[difficulty]
        R, C = config['grid_size']
        total_cells = R * C
        num_mines = max(1, int(total_cells * config['mine_ratio']))
        target_reveals = max(2, int((total_cells - num_mines) * config['reveal_ratio']))
        max_reveals = int(target_reveals * 1.5)

        for attempt in range(max_attempts):
            cells = [(r, c) for r in range(R) for c in range(C)]
            mine_positions = set(self.rng.sample(cells, num_mines))
            mask = [[1 if (r, c) in mine_positions else 0 for c in range(C)] for r in range(R)]

            nums = compute_numbers(mask)
            ranked_cells = self._rank_cells_by_information(nums, mask, R, C)

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
                solutions = solve_puzzle(puzzle, R, C, max_solutions=1, total_mines=num_mines)

                puzzle_display = []
                for row in puzzle:
                    row_str = ''.join(str(cell) if cell is not None else '#' for cell in row)
                    puzzle_display.append(row_str)

                answer_bitstring = mask_to_solution_string(solutions[0])

                # Compute weighted coordinate sum as the answer
                coords = bitstring_to_coordinates(answer_bitstring, R, C)
                answer_sum = compute_total_sum(coords, C)

                return {
                    'id': puzzle_id,
                    'difficulty': difficulty,
                    'rows': R,
                    'cols': C,
                    'total_mines': num_mines,
                    'puzzle': puzzle_display,
                    'answer': str(answer_sum),
                    'solution': answer_bitstring,
                    'answer_type': 'weighted_sum',
                    'description': f"{R}x{C} grid with {num_mines} mines",
                    'cells_revealed': len(revealed)
                }

        return None

    def verify_solutions(self, problem: Dict) -> bool:
        """Verify that the puzzle has exactly 1 solution."""
        R, C = problem['rows'], problem['cols']
        total_mines = problem['total_mines']

        puzzle = []
        for row_str in problem['puzzle']:
            row = []
            for char in row_str:
                if char == '#':
                    row.append(None)
                else:
                    row.append(int(char))
            puzzle.append(row)

        solutions = solve_puzzle(puzzle, R, C, max_solutions=2, total_mines=total_mines)
        return len(solutions) == 1


# ============================================================
# Dataset generation
# ============================================================

def create_dataset_files(num_questions: int):
    """
    Create dataset files for minesweeper puzzles.

    Args:
        num_questions: Number of questions to generate

    Returns:
        Tuple[pd.DataFrame, List[Dict]]: (dataframe, json list)
    """
    import pandas as pd

    print(f"Generating {num_questions} minesweeper puzzles...")

    generator = DifficultyPuzzleGenerator(seed=42)

    difficulties = ['easy', 'medium', 'hard']
    puzzles_per_diff = num_questions // len(difficulties)
    remainder = num_questions % len(difficulties)

    all_puzzles = []

    for i, difficulty in enumerate(difficulties):
        count = puzzles_per_diff + (1 if i < remainder else 0)

        if count == 0:
            continue

        print(f"\n=== Generating {difficulty} puzzles ({count} needed) ===")

        for j in range(count):
            puzzle_id = f"minesweeper_en_{len(all_puzzles)}"

            # Try with different seeds
            puzzle_generated = False
            for seed_offset in range(10):
                generator.rng = random.Random(generator.seed + seed_offset + j * 100 + i * 1000)

                result = generator.generate_puzzle_with_difficulty(
                    difficulty=difficulty,
                    puzzle_id=puzzle_id
                )

                if result:
                    # Add question (prompt)
                    result['question'] = create_prompt(result)
                    all_puzzles.append(result)
                    print(f"  [{j+1}/{count}] {result['description']}, answer_sum={result['answer']}")
                    puzzle_generated = True
                    break

            if not puzzle_generated:
                print(f"  [{j+1}/{count}] Failed to generate")

    print(f"\nGenerated {len(all_puzzles)} puzzles total")

    df = pd.DataFrame(all_puzzles)

    # Save files
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # CSV
    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "minesweeper_en.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV file created: {csv_path}")

    # JSONL
    json_dir = PROJECT_ROOT / "data" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = json_dir / "minesweeper_en.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_puzzles:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL file created: {jsonl_path}")

    return df, all_puzzles


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Minesweeper Puzzle Generator")
    parser.add_argument("--num", type=int, default=12, help="Number of questions to generate")

    args = parser.parse_args()

    create_dataset_files(num_questions=args.num)
