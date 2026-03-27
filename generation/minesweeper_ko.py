"""지뢰찾기 퍼즐 생성기 (난이도별)

구성적 생성: 해의 수를 모니터링하면서 점진적으로 셀을 공개하여
유효한 퍼즐 생성을 보장합니다.
가중 좌표 합 형식의 채점 기능을 포함합니다.
"""

MAX_SOLUTIONS = 1  # 정확히 1개의 해만 허용

import random
import re
import json
from itertools import product
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set


# ============================================================
# 핵심 유틸리티 함수
# ============================================================

def neighbors(r: int, c: int, R: int, C: int) -> List[Tuple[int, int]]:
    """R x C 격자 내에서 셀 (r, c)의 8방향 이웃을 반환합니다."""
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
    각 셀의 숫자 힌트를 계산합니다.
    mask: 1=지뢰, 0=안전
    반환값: 지뢰 셀은 None, 안전 셀은 인접 지뢰 수(0-8)를 가진 격자
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
    백트래킹과 제약 전파를 사용하여 지뢰찾기 퍼즐을 풉니다.

    인자:
        puzzle_nums: 공개된 셀은 숫자 0-8, 숨겨진 셀은 None인 격자
        R, C: 격자 크기
        max_solutions: 이 수만큼의 해를 찾으면 중단
        total_mines: 전체 지뢰 수에 대한 선택적 전역 제약

    반환값:
        해(지뢰 마스크) 리스트, 최대 max_solutions개
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
    """무작위 지뢰 배치를 생성합니다."""
    cells = list(product(range(R), range(C)))
    mines = set(rng.sample(cells, M))
    return [[1 if (r, c) in mines else 0 for c in range(C)] for r in range(R)]


def mask_to_solution_string(mask: List[List[int]]) -> str:
    """지뢰 마스크를 단일 비트 문자열 형식으로 변환합니다."""
    return "".join("".join(str(cell) for cell in row) for row in mask)


def puzzle_to_string(puzzle: List[List[Optional[int]]]) -> List[str]:
    """퍼즐을 숨겨진 셀에 #을 사용하는 문자열 형식으로 변환합니다."""
    return [''.join(str(cell) if cell is not None else '#' for cell in row)
            for row in puzzle]


# ============================================================
# 채점 함수 (가중 좌표 합)
# ============================================================

def bitstring_to_coordinates(solution_str: str, R: int, C: int) -> Set[Tuple[int, int]]:
    """해 비트 문자열을 좌표 집합으로 변환합니다."""
    coords = set()
    for i, cell in enumerate(solution_str):
        if cell == '1':
            r, c = divmod(i, C)
            coords.add((r, c))
    return coords


def compute_total_sum(coords: Set[Tuple[int, int]], C: int) -> int:
    """가중 좌표 합을 계산합니다: 각 지뢰에 대해 sum(행 * C + 열)."""
    if not coords:
        return 0
    return sum(r * C + c for r, c in coords)


def parse_total_sum(output: str) -> Optional[int]:
    """LLM 출력에서 합계를 단일 정수로 파싱합니다."""
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
    """가중 합을 사용하여 LLM 예측을 해답과 비교 채점합니다."""
    truth_coords = bitstring_to_coordinates(solution_str, R, C)
    truth_sum = compute_total_sum(truth_coords, C)
    pred_sum = parse_total_sum(pred_output)

    if pred_sum is None:
        return {'score': 0.0, 'truth_sum': truth_sum, 'pred_sum': None, 'error': '파싱 실패'}

    score = 1.0 if truth_sum == pred_sum else 0.0
    return {'score': score, 'truth_sum': truth_sum, 'pred_sum': pred_sum, 'error': None}


# ============================================================
# 프롬프트 템플릿 함수
# ============================================================

def format_puzzle_grid(puzzle_rows: List[str]) -> str:
    """퍼즐 격자를 여러 줄 문자열로 포맷합니다."""
    return '\n'.join(puzzle_rows)


def create_prompt(puzzle_data: Dict) -> str:
    """지뢰찾기 퍼즐 평가용 프롬프트를 생성합니다."""
    puzzle_grid = format_puzzle_grid(puzzle_data['puzzle'])
    rows = puzzle_data['rows']
    cols = puzzle_data['cols']
    mines = puzzle_data.get('total_mines', puzzle_data.get('mines', 0))
    difficulty = puzzle_data.get('difficulty', 'medium')

    if difficulty == 'easy':
        mine_info = f"2. 전체 지뢰 수: {mines}개가 격자에 숨겨져 있습니다"
        task_info = f"모든 {mines}개 지뢰의 정확한 위치를 찾으세요."
    else:
        mine_info = "2. 일부 셀에 지뢰가 숨겨져 있습니다"
        task_info = "모든 지뢰의 정확한 위치를 찾으세요."

    if difficulty == 'easy':
        uniqueness_info = "\n6. 이 퍼즐은 정확히 하나의 고유한 해를 가집니다"
    else:
        uniqueness_info = ""

    prompt = f"""다음 규칙에 따라 지뢰찾기 퍼즐을 풀어주세요:

게임 규칙:
1. 격자 크기: {rows}행 x {cols}열 (0-인덱스)
{mine_info}
3. 각 공개된 숫자(0-8)는 인접한 8개 셀 중 지뢰가 있는 셀의 수를 나타냅니다
4. '#'은 지뢰이거나 안전한 셀일 수 있는 숨겨진 셀입니다
5. 인접 셀은 가로, 세로, 대각선 8방향을 모두 포함합니다{uniqueness_info}

퍼즐:
{puzzle_grid}

과제:
{task_info}

출력 형식 (엄격히 준수):
- 0-기반 인덱스로 모든 지뢰 좌표를 찾으세요 (행 0 ~ {rows-1}, 열 0 ~ {cols-1})
- 각 지뢰의 (행, 열)에 대해 선형 인덱스를 계산하세요: 행 * {cols} + 열
- 모든 선형 인덱스의 합을 구하세요
- 이 합계 정수만 출력하세요

예시 ({cols}열 기준):
지뢰가 (1,2)와 (3,0)에 있다면:
- 선형 인덱스: (1*{cols}+2) = {1*cols+2}, (3*{cols}+0) = {3*cols}
- 합 = {1*cols+2} + {3*cols} = {1*cols+2 + 3*cols}
- 출력: {1*cols+2 + 3*cols}

Answer:"""

    return prompt


# ============================================================
# 난이도별 퍼즐 생성기
# ============================================================

class DifficultyPuzzleGenerator:
    """점진적 공개 방식을 사용하여 다양한 난이도의 퍼즐을 생성합니다."""

    DIFFICULTY_CONFIGS = {
        'easy': {
            'grid_size': (5, 5),
            'mine_ratio': 0.22,
            'reveal_ratio': 0.45,
            'description': '작은 격자, 균형 잡힌 공개'
        },
        'medium': {
            'grid_size': (6, 6),
            'mine_ratio': 0.33,
            'reveal_ratio': 0.25,
            'description': '중간 격자, 더 많은 지뢰, 지뢰 수 힌트 없음'
        },
        'hard': {
            'grid_size': (7, 7),
            'mine_ratio': 0.28,
            'reveal_ratio': 0.25,
            'description': '큰 격자, 지뢰 수 힌트 없음'
        }
    }

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.seed = seed

    def _rank_cells_by_information(self, nums: List[List[Optional[int]]],
                                     mask: List[List[int]], R: int, C: int) -> List[Tuple[int, int]]:
        """점진적 공개를 위해 안전 셀을 정보 가치 순으로 정렬합니다."""
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
        """정확히 1개의 해를 가진 퍼즐을 생성합니다."""
        if difficulty not in self.DIFFICULTY_CONFIGS:
            raise ValueError(f"알 수 없는 난이도: {difficulty}")

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

                # 가중 좌표 합을 정답으로 계산
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
                    'description': f"{R}x{C} 격자, 지뢰 {num_mines}개",
                    'cells_revealed': len(revealed)
                }

        return None

    def verify_solutions(self, problem: Dict) -> bool:
        """퍼즐이 정확히 1개의 해를 가지는지 검증합니다."""
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
# 데이터셋 생성
# ============================================================

def create_dataset_files(num_questions: int):
    """
    지뢰찾기 퍼즐 데이터셋 파일을 생성합니다.

    인자:
        num_questions: 생성할 문제 수

    반환값:
        Tuple[pd.DataFrame, List[Dict]]: (데이터프레임, JSON 리스트)
    """
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

            # 다른 시드로 시도
            puzzle_generated = False
            for seed_offset in range(10):
                generator.rng = random.Random(generator.seed + seed_offset + j * 100 + i * 1000)

                result = generator.generate_puzzle_with_difficulty(
                    difficulty=difficulty,
                    puzzle_id=puzzle_id
                )

                if result:
                    # 질문(프롬프트) 추가
                    result['question'] = create_prompt(result)
                    all_puzzles.append(result)
                    print(f"  [{j+1}/{count}] {result['description']}, answer_sum={result['answer']}")
                    puzzle_generated = True
                    break

            if not puzzle_generated:
                print(f"  [{j+1}/{count}] 생성 실패")

    print(f"\n총 {len(all_puzzles)}개의 퍼즐이 생성되었습니다")

    df = pd.DataFrame(all_puzzles)

    # 파일 저장
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # CSV
    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "minesweeper_ko.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV 파일 생성됨: {csv_path}")

    # JSONL
    json_dir = PROJECT_ROOT / "data" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = json_dir / "minesweeper_ko.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_puzzles:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL 파일 생성됨: {jsonl_path}")

    return df, all_puzzles


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="지뢰찾기 퍼즐 생성기")
    parser.add_argument("--num", type=int, default=12, help="생성할 문제 수")

    args = parser.parse_args()

    create_dataset_files(num_questions=args.num)
