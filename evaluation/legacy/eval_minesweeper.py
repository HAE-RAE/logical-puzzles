"""지뢰찾기 LLM 평가 스크립트

OpenAI 모델을 사용하여 지뢰찾기 퍼즐을 평가합니다.
"""

import json
import os
import sys
import time
import re
from pathlib import Path
from typing import List, Tuple
from openai import OpenAI
from dotenv import load_dotenv
import argparse
from collections import defaultdict

# guess 디렉터리를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'guess'))

# .env 로드
load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


# ============================================================================
# 프롬프트 생성
# ============================================================================

def create_prompt(puzzle_data: dict) -> str:
    """
    지뢰찾기 평가 프롬프트 생성

    프롬프트 설계:
    1. 태스크 설명: 지뢰찾기 게임 규칙
    2. 입력 형식: '#'=숨김, 숫자=이웃 지뢰 개수
    3. 출력 형식: 좌표 리스트 (r,c) 형식
    4. 명확한 제약: 0-based indexing, 정확한 지뢰 개수
    """
    rows = puzzle_data['rows']
    cols = puzzle_data['cols']
    num_mines = puzzle_data['mines']
    puzzle_lines = puzzle_data['puzzle']

    # 그리드를 보기 좋게 포맷팅
    puzzle_str = '\n'.join(puzzle_lines)

    prompt = f"""You are solving a Minesweeper puzzle. The grid is {rows}x{cols} with {num_mines} mines total.

Grid notation:
- '#' represents a hidden cell (unknown if mine or safe)
- Numbers (0-8) show count of adjacent mines in 8 directions
- Your task: identify ALL mine locations

Puzzle grid:
{puzzle_str}

Instructions:
1. Analyze the puzzle using logical reasoning
2. Deduce the exact location of all {num_mines} mines
3. Output your final answer starting with "Final answer:"
4. List ALL mine coordinates in format: (r,c) (r,c) ...
5. Use 0-based indexing (row 0 to {rows-1}, column 0 to {cols-1})
6. The answer must contain exactly {num_mines} coordinates

Example output format:
Final answer: (0,1) (0,3) (1,2) (2,4) (3,0) (4,2)

Now solve the puzzle and provide your final answer:"""

    return prompt


# ============================================================================
# 응답 파싱
# ============================================================================

def parse_coordinates(output: str, expected_count: int) -> List[Tuple[int, int]]:
    """
    LLM 출력에서 좌표 추출

    파싱 전략:
    1. "Final answer:" 이후 부분만 파싱 (노이즈 제거)
    2. (r,c) 패턴 매칭
    3. 중복 제거
    4. 개수 검증
    """
    # "Final answer:" 이후 부분 추출
    final_answer_pattern = r'(?:final answer|answer):\s*(.*)'
    final_match = re.search(final_answer_pattern, output, re.IGNORECASE | re.DOTALL)

    if final_match:
        search_text = final_match.group(1)
        print(f"  [파싱] 'Final answer:' 발견")
    else:
        # fallback: 마지막 200자
        search_text = output[-200:] if len(output) > 200 else output
        print(f"  [파싱] 'Final answer:' 없음, 마지막 부분 파싱")

    # (r,c) 패턴 찾기
    pattern = r'\((\d+),\s*(\d+)\)'
    matches = re.findall(pattern, search_text)

    coords = [(int(r), int(c)) for r, c in matches]

    # 중복 제거 (순서 유지)
    coords = list(dict.fromkeys(coords))

    print(f"  [파싱] {len(coords)}개 좌표 추출 (기대: {expected_count}개)")

    return coords[:expected_count] if len(coords) >= expected_count else coords


# ============================================================================
# API 호출
# ============================================================================

def call_openai_api(model: str, prompt: str, max_retries: int = 3) -> tuple:
    """OpenAI API 호출 (재시도 로직 포함)"""
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            }

            # o1 계열은 max_completion_tokens 사용
            if model.startswith("o1"):
                kwargs["max_completion_tokens"] = 32000
            else:
                kwargs["max_tokens"] = 10000
                kwargs["temperature"] = 0

            response = client.chat.completions.create(**kwargs)
            output = response.choices[0].message.content.strip()

            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }

            return output, usage

        except Exception as e:
            error_msg = str(e)
            print(f"    ✗ API 오류 (시도 {attempt + 1}/{max_retries}): {error_msg[:100]}")

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"    ⏳ {wait_time}초 후 재시도...")
                time.sleep(wait_time)
            else:
                raise Exception(f"API 호출 실패 ({max_retries}회 시도): {error_msg}")


# ============================================================================
# 평가 메트릭
# ============================================================================

def coords_to_grid(coords: List[Tuple[int, int]], rows: int, cols: int) -> List[List[int]]:
    """좌표 리스트를 그리드로 변환"""
    grid = [[0] * cols for _ in range(rows)]
    for r, c in coords:
        if 0 <= r < rows and 0 <= c < cols:
            grid[r][c] = 1
    return grid


def solution_to_grid(solution_lines: List[str]) -> List[List[int]]:
    """솔루션 문자열을 그리드로 변환"""
    return [[int(ch) for ch in line] for line in solution_lines]


def compute_metrics(predicted_coords: List[Tuple[int, int]],
                   solution_lines: List[str],
                   rows: int, cols: int) -> dict:
    """
    평가 메트릭 계산

    메트릭:
    - exact_match: 완전 일치 (1.0 or 0.0)
    - precision: 예측한 지뢰 중 실제 지뢰 비율
    - recall: 실제 지뢰 중 찾은 비율
    - f1: precision과 recall의 조화평균
    """
    pred_grid = coords_to_grid(predicted_coords, rows, cols)
    true_grid = solution_to_grid(solution_lines)

    # 정확도 계산
    tp = sum(pred_grid[r][c] == 1 and true_grid[r][c] == 1
             for r in range(rows) for c in range(cols))
    fp = sum(pred_grid[r][c] == 1 and true_grid[r][c] == 0
             for r in range(rows) for c in range(cols))
    fn = sum(pred_grid[r][c] == 0 and true_grid[r][c] == 1
             for r in range(rows) for c in range(cols))
    tn = sum(pred_grid[r][c] == 0 and true_grid[r][c] == 0
             for r in range(rows) for c in range(cols))

    # Exact match
    exact_match = 1.0 if pred_grid == true_grid else 0.0

    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'exact_match': exact_match,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


# ============================================================================
# 퍼즐 평가
# ============================================================================

def evaluate_puzzle(puzzle_data: dict, model: str) -> dict:
    """단일 퍼즐 평가"""
    puzzle_id = puzzle_data['id']
    rows = puzzle_data['rows']
    cols = puzzle_data['cols']
    num_mines = puzzle_data['mines']
    difficulty = puzzle_data['difficulty']

    print(f"  퍼즐: {puzzle_id}")
    print(f"  난이도: {difficulty}, 크기: {rows}×{cols}, 지뢰: {num_mines}개")

    # 프롬프트 생성
    prompt = create_prompt(puzzle_data)

    try:
        # API 호출
        output, usage = call_openai_api(model, prompt)

        print(f"  응답 길이: {len(output)}자, 토큰: {usage['total_tokens']}")

        # 좌표 파싱
        predicted_coords = parse_coordinates(output, num_mines)

        # 평가
        metrics = compute_metrics(predicted_coords, puzzle_data['solution'], rows, cols)

        status = '✓ 정답' if metrics['exact_match'] == 1.0 else f"✗ 오답 (F1: {metrics['f1']:.2f})"
        print(f"  {status}\n")

        return {
            'puzzle_id': puzzle_id,
            'difficulty': difficulty,
            'rows': rows,
            'cols': cols,
            'mines': num_mines,
            'model': model,
            'predicted_coords': predicted_coords,
            'metrics': metrics,
            'output_length': len(output),
            'usage': usage,
            'full_output': output
        }

    except Exception as e:
        error_msg = str(e)
        print(f"  ✗ 평가 실패: {error_msg}\n")
        return {
            'puzzle_id': puzzle_id,
            'difficulty': difficulty,
            'model': model,
            'predicted_coords': [],
            'metrics': {'exact_match': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
            'output_length': 0,
            'usage': {},
            'error': error_msg
        }


# ============================================================================
# 메인 함수
# ============================================================================

def main():
    """메인 평가 함수"""
    parser = argparse.ArgumentParser(description='지뢰찾기 LLM 평가')
    parser.add_argument('--model', type=str, default='gpt-4o',
                       choices=['gpt-4o', 'gpt-4o-mini', 'o1', 'o1-mini'],
                       help='평가할 모델')
    parser.add_argument('--data', type=str,
                       default='../data/minesweeper/eval_dataset.jsonl',
                       help='평가 데이터 경로')
    parser.add_argument('--output', type=str, default=None,
                       help='결과 저장 경로')
    args = parser.parse_args()

    # 경로 설정
    script_dir = Path(__file__).parent
    data_file = script_dir / args.data

    if not data_file.exists():
        print(f"✗ 데이터 파일이 없습니다: {data_file}")
        print(f"먼저 퍼즐을 생성하세요:")
        print(f"  cd ../guess")
        print(f"  python minesweeper.py")
        return

    # 출력 경로 설정
    if args.output is None:
        results_dir = script_dir.parent / 'data' / 'minesweeper' / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f'eval_results_{args.model}.json'
    else:
        output_path = Path(args.output)

    # 평가 데이터 읽기
    puzzles = []
    with open(data_file, 'r') as f:
        for line in f:
            if line.strip():
                puzzles.append(json.loads(line))

    print(f"{'='*70}")
    print(f"지뢰찾기 LLM 평가")
    print(f"{'='*70}")
    print(f"모델: {args.model}")
    print(f"데이터: {data_file.name}")
    print(f"퍼즐 수: {len(puzzles)}개")
    print(f"{'='*70}\n")

    all_results = []

    for i, puzzle in enumerate(puzzles):
        print(f"[{i+1}/{len(puzzles)}] {puzzle['id']}")
        print("-"*70)

        result = evaluate_puzzle(puzzle, args.model)
        all_results.append(result)

        # API 제한 방지
        time.sleep(1)

    # 결과 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 통계 출력
    print("\n" + "="*70)
    print("평가 결과 요약")
    print("="*70)

    total = len(all_results)
    exact_match_count = sum(1 for r in all_results if r['metrics']['exact_match'] == 1.0)
    avg_precision = sum(r['metrics']['precision'] for r in all_results) / total
    avg_recall = sum(r['metrics']['recall'] for r in all_results) / total
    avg_f1 = sum(r['metrics']['f1'] for r in all_results) / total
    total_tokens = sum(r.get('usage', {}).get('total_tokens', 0) for r in all_results)

    print(f"\n{args.model}:")
    print(f"  완전 정답: {exact_match_count}/{total} ({exact_match_count/total*100:.1f}%)")
    print(f"  평균 Precision: {avg_precision*100:.1f}%")
    print(f"  평균 Recall: {avg_recall*100:.1f}%")
    print(f"  평균 F1 Score: {avg_f1*100:.1f}%")
    print(f"  총 토큰 사용: {total_tokens:,}")

    # 난이도별 통계
    print("\n난이도별 결과:")
    by_difficulty = defaultdict(list)
    for r in all_results:
        by_difficulty[r['difficulty']].append(r)

    for diff in ['easy', 'medium', 'hard']:
        if diff in by_difficulty:
            items = by_difficulty[diff]
            exact = sum(1 for r in items if r['metrics']['exact_match'] == 1.0)
            avg_f1 = sum(r['metrics']['f1'] for r in items) / len(items)
            print(f"  {diff:8s}: {exact}/{len(items)} 정답 (평균 F1: {avg_f1*100:.1f}%)")

    print(f"\n{'='*70}")
    print(f"✓ 결과 저장: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
