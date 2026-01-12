"""Cypher LLM 평가 스크립트"""

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

# .env 로드 (프로젝트 루트의 env 파일 사용)
env_path = Path(__file__).parent.parent / 'env'
load_dotenv(dotenv_path=env_path)

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def load_cypher_data(jsonl_path: str) -> list:
    """CYPHER JSONL 파일 로드"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def create_prompt(question: str) -> str:
    """평가 프롬프트 생성"""
    system_prompt = """너는 암호 해독 전문가입니다. 주어진 미션 로그와 암호화 가이드를 분석하여 암호문을 복호화해야 합니다.

중요:
- 로그 지문과 가이드를 분석하여 암호 키워드를 먼저 도출하세요.
- 가이드에 명시된 알고리즘(Vigenere, Transposition, Substitution, Reverse 등)을 정확한 역순으로 적용하여 복호화하세요.
- Columnar Transposition(전치 암호)의 경우, 키워드의 알파벳 순서에 따라 열을 재배치하는 방식임을 유의하세요.
- 최종 답변은 반드시 '원문: [답]' 형식으로 제시하세요.

출력 형식:
원문: [복호화된 텍스트]"""

    return system_prompt


def extract_answer_from_output(output: str) -> str:
    """LLM 응답에서 원문 추출"""
    patterns = [
        r'원문[:\s]*([A-Z]+)',
        r'답[:\s]*([A-Z]+)',
        r'정답[:\s]*([A-Z]+)',
        r'answer[:\s]*([A-Z]+)',
        r'plaintext[:\s]*([A-Z]+)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            return matches[-1].strip().upper()

    # 마지막 대문자 단어 추출
    words = re.findall(r'\b[A-Z]{3,}\b', output)
    if words:
        return words[-1]

    return ''


def call_openai_api(model: str, system_prompt: str, user_prompt: str, max_retries: int = 3):
    """OpenAI API 호출"""
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }

            # o1 시리즈와 GPT-5 reasoning 모델은 max_completion_tokens 사용
            if model.startswith("o1") or model.startswith("gpt-5"):
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
            print(f"    [X] API 오류 (시도 {attempt + 1}/{max_retries}): {error_msg}")

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"    [{wait_time}초 후 재시도...]")
                time.sleep(wait_time)
            else:
                raise Exception(f"API 호출 실패 ({max_retries}회 시도): {error_msg}")


def evaluate_puzzle(puzzle_data: dict, model: str, system_prompt: str) -> dict:
    """단일 퍼즐 평가"""
    question = puzzle_data['question']
    expected_answer = puzzle_data['answer']

    print(f"  문제: {puzzle_data.get('difficulty', 'MEDIUM')}")
    print(f"  모델: {model}")

    try:
        # API 호출
        output, usage = call_openai_api(model, system_prompt, question)

        print(f"  응답 길이: {len(output)}자, 토큰: {usage['total_tokens']}")

        # 정답 추출
        predicted_answer = extract_answer_from_output(output)

        print(f"  기대 정답: {expected_answer}")
        print(f"  예측 정답: {predicted_answer}")

        # 평가
        correct = predicted_answer == expected_answer
        status = '[O] 정답' if correct else '[X] 오답'
        print(f"  {status}\n")

        return {
            'question': question,
            'expected': expected_answer,
            'predicted': predicted_answer,
            'correct': correct,
            'output_length': len(output),
            'usage': usage,
            'full_output': output,
            'difficulty': puzzle_data.get('difficulty', 'Unknown'),
            'description': puzzle_data.get('description', '')
        }

    except Exception as e:
        error_msg = str(e)
        print(f"  [X] 평가 실패: {error_msg}\n")
        return {
            'question': question,
            'expected': expected_answer,
            'predicted': '',
            'correct': False,
            'output_length': 0,
            'usage': {},
            'error': error_msg
        }


def main():
    """메인 평가 함수"""
    parser = argparse.ArgumentParser(description='Cypher LLM 평가')
    parser.add_argument('--model', type=str, default='gpt-4o',
                       choices=[
                           # GPT-4o series
                           'gpt-4o', 'gpt-4o-mini',
                           # GPT-4.1 series (April 2025)
                           'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano',
                           # GPT-5 series (August 2025)
                           'gpt-5', 'gpt-5-mini', 'gpt-5-nano',
                           # o1 series
                           'o1', 'o1-mini'
                       ],
                       help='평가할 모델')
    parser.add_argument('--data', type=str,
                       default='../data/json/CYPHER_MEDIUM_v1.jsonl',
                       help='평가 데이터 경로')
    parser.add_argument('--output', type=str, default=None,
                       help='결과 저장 경로')
    args = parser.parse_args()

    # 경로 설정
    script_dir = Path(__file__).parent
    data_file = script_dir / args.data

    if not data_file.exists():
        print(f"[X] 데이터 파일이 없습니다: {data_file}")
        print(f"먼저 퍼즐을 생성하세요:")
        print(f"  cd ../guess")
        print(f"  python cypher.py")
        return

    # 출력 경로 설정
    if args.output is None:
        results_dir = script_dir.parent / 'data' / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f'eval_results_cypher_{args.model}.json'
    else:
        output_path = Path(args.output)

    # 평가 데이터 읽기
    puzzles = load_cypher_data(str(data_file))

    print(f"{'='*70}")
    print(f"Cypher LLM 평가")
    print(f"{'='*70}")
    print(f"모델: {args.model}")
    print(f"데이터: {data_file.name}")
    print(f"퍼즐 수: {len(puzzles)}개")
    print(f"{'='*70}\n")

    system_prompt = create_prompt('')
    all_results = []

    for i, puzzle in enumerate(puzzles):
        print(f"[{i+1}/{len(puzzles)}]")
        print("-"*70)

        result = evaluate_puzzle(puzzle, args.model, system_prompt)
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
    accuracy = (correct_count / total * 100) if total > 0 else 0
    total_tokens = sum(r.get('usage', {}).get('total_tokens', 0) for r in all_results)

    print(f"\n{args.model}:")
    print(f"  정답: {correct_count}/{total} ({accuracy:.1f}%)")
    print(f"  총 토큰 사용: {total_tokens:,}")

    print(f"\n{'='*70}")
    print(f"[완료] 결과 저장: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
