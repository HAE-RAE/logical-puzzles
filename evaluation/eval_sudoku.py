"""스도쿠 LLM 평가 스크립트"""

import json
import os
import sys
import time
import re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import argparse

# guess 디렉터리를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'guess'))
from sudoku import from_string

# .env 로드
load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def puzzle_string_to_grid_list(puzzle_str: str) -> list:
    """81자 문자열을 9x9 리스트로 변환"""
    grid = []
    for i in range(9):
        row = list(puzzle_str[i*9:(i+1)*9])
        grid.append(row)
    return grid


def create_prompt(puzzle_str: str, positions: list) -> str:
    """평가 프롬프트 생성"""
    grid = puzzle_string_to_grid_list(puzzle_str)

    # 리스트 형식으로 문자열 생성
    grid_str = "[\n"
    for i, row in enumerate(grid):
        grid_str += f"  {row}"
        if i < 8:
            grid_str += ","
        grid_str += "\n"
    grid_str += "]"

    positions_str = ', '.join(positions)
    k = len(positions)

    prompt = f"""스도쿠 퍼즐을 풀고, 지정된 정답 좌표의 값들을 순서대로 출력하세요.

퍼즐 (9x9 리스트, '.'은 빈칸):
{grid_str}

정답 좌표 (1-기반, r=행, c=열):
{positions_str}

문제해결 방식:
1. 먼저 스도쿠를 완전히 푸세요.
2. 완성된 9x9 그리드에서 정답 좌표 값들을 순서대로 출력하세요.

출력 형식:
정답: [{k}자리 숫자]

예시: 정답: {'1' * k}"""

    return prompt


def extract_answer_from_output(output: str, k: int) -> str:
    """출력에서 '정답:' 뒤의 숫자만 추출"""
    # "정답:" 뒤의 숫자 찾기
    patterns = [
        rf'정답:\s*([1-9]{{{k}}})',
        rf'정답:\s*\[?([1-9]{{{k}}})\]?',
    ]

    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            return match.group(1)

    # "정답:" 뒤의 모든 숫자 추출 (fallback)
    answer_match = re.search(r'정답:\s*(.+)', output)
    if answer_match:
        answer_part = answer_match.group(1)
        digits = re.findall(r'[1-9]', answer_part)
        if len(digits) >= k:
            return ''.join(digits[:k])

    # 최종 fallback: 모든 숫자 추출
    digits = re.findall(r'[1-9]', output)
    if len(digits) >= k:
        return ''.join(digits[:k])

    return ''.join(digits)


def call_openai_api(model: str, prompt: str, max_retries: int = 3):
    """OpenAI API 호출 (재시도 로직 포함)"""
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "당신은 논리 퍼즐 전문가입니다."},
                    {"role": "user", "content": prompt}
                ]
            }

            # o1 계열은 max_completion_tokens, 다른 모델은 max_tokens
            if model.startswith("o1"):
                kwargs["max_completion_tokens"] = 100000
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
            print(f"    ✗ API 오류 (시도 {attempt + 1}/{max_retries}): {error_msg}")

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"    ⏳ {wait_time}초 후 재시도...")
                time.sleep(wait_time)
            else:
                raise Exception(f"API 호출 실패 ({max_retries}회 시도): {error_msg}")


def evaluate_puzzle(puzzle_data: dict, model: str) -> dict:
    """단일 퍼즐 평가"""
    puzzle_id = puzzle_data['id']
    difficulty = puzzle_data['difficulty']['label']
    puzzle_str = puzzle_data['puzzle']
    positions = puzzle_data['spotcheck']['positions']
    expected_code = puzzle_data['spotcheck']['code']
    k = puzzle_data['spotcheck']['k']

    print(f"  퍼즐: {puzzle_id} ({difficulty})")
    print(f"  모델: {model}")

    # 프롬프트 생성
    prompt = create_prompt(puzzle_str, positions)

    try:
        # API 호출
        output, usage = call_openai_api(model, prompt)

        print(f"  응답 길이: {len(output)}자, 토큰: {usage['total_tokens']}")

        # 정답 추출
        predicted_code = extract_answer_from_output(output, k)

        print(f"  기대 정답: {expected_code}")
        print(f"  예측 정답: {predicted_code} (길이: {len(predicted_code)})")

        # 평가
        if len(predicted_code) < k:
            hamming_dist = k
            accuracy = 0.0
            correct = False
        else:
            predicted_code = predicted_code[:k]
            correct = predicted_code == expected_code
            hamming_dist = sum(1 for a, b in zip(predicted_code, expected_code) if a != b)
            accuracy = 1.0 - (hamming_dist / k)

        status = '✓ 정답' if correct else f'✗ 오답 (해밍: {hamming_dist})'
        print(f"  {status}, 정확도: {accuracy*100:.1f}%\n")

        return {
            'puzzle_id': puzzle_id,
            'difficulty': difficulty,
            'model': model,
            'k': k,
            'expected': expected_code,
            'predicted': predicted_code,
            'correct': correct,
            'hamming': hamming_dist,
            'accuracy': accuracy,
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
            'k': k,
            'expected': expected_code,
            'predicted': '',
            'correct': False,
            'hamming': k,
            'accuracy': 0.0,
            'output_length': 0,
            'usage': {},
            'error': error_msg
        }


def main():
    """메인 평가 함수"""
    parser = argparse.ArgumentParser(description='스도쿠 LLM 평가')
    parser.add_argument('--model', type=str, default='gpt-4o',
                       choices=['gpt-4o', 'gpt-4o-mini', 'o1', 'o1-mini'],
                       help='평가할 모델')
    parser.add_argument('--data', type=str,
                       default='../data/sudoku/eval_5diff_k6.jsonl',
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
        print(f"  python sudoku.py")
        return

    # 출력 경로 설정
    if args.output is None:
        results_dir = script_dir.parent / 'data' / 'sudoku' / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f'eval_results_{args.model}.json'
    else:
        output_path = Path(args.output)

    # 평가 데이터 읽기
    with open(data_file, 'r') as f:
        puzzles = [json.loads(line) for line in f]

    print(f"{'='*70}")
    print(f"스도쿠 LLM 평가")
    print(f"{'='*70}")
    print(f"모델: {args.model}")
    print(f"데이터: {data_file.name}")
    print(f"퍼즐 수: {len(puzzles)}개")
    print(f"{'='*70}\n")

    all_results = []

    for i, puzzle in enumerate(puzzles):
        print(f"[{i+1}/{len(puzzles)}] {puzzle['id']} ({puzzle['difficulty']['label']})")
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
    correct_count = sum(1 for r in all_results if r['correct'])
    avg_accuracy = sum(r['accuracy'] for r in all_results) / total if total > 0 else 0
    avg_hamming = sum(r['hamming'] for r in all_results) / total if total > 0 else 0
    total_tokens = sum(r.get('usage', {}).get('total_tokens', 0) for r in all_results)

    print(f"\n{args.model}:")
    print(f"  완전 정답: {correct_count}/{total} ({correct_count/total*100:.1f}%)")
    print(f"  평균 정확도: {avg_accuracy*100:.1f}%")
    print(f"  평균 해밍 거리: {avg_hamming:.2f}")
    print(f"  총 토큰 사용: {total_tokens:,}")

    # 난이도별 통계
    print("\n난이도별 결과:")
    from collections import defaultdict
    by_difficulty = defaultdict(list)
    for r in all_results:
        by_difficulty[r['difficulty']].append(r)

    for diff in ['Easy', 'Medium', 'Hard', 'Expert', 'Extreme']:
        if diff in by_difficulty:
            items = by_difficulty[diff]
            correct = sum(1 for r in items if r['correct'])
            avg_acc = sum(r['accuracy'] for r in items) / len(items)
            print(f"  {diff:8s}: {correct}/{len(items)} 정답 (평균 {avg_acc*100:.1f}%)")

    print(f"\n{'='*70}")
    print(f"✓ 결과 저장: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
