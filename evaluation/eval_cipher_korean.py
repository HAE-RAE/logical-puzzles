"""Hangul-based Cipher Evaluation Script"""

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
from cypher_hangul import KoreanTextCipher

# .env 로드
env_path = Path(__file__).parent.parent / 'env'
load_dotenv(dotenv_path=env_path)

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def load_data(jsonl_path: str) -> list:
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def create_prompt() -> str:
    return """너는 한글 암호 해독 전문가입니다. 주어진 미션 로그와 암호화 가이드를 분석하여 한글 암호문을 복호화해야 합니다.

중요:
- 로그 지문을 분석하여 암호 키워드를 먼저 찾아내세요.
- 한글의 초성(ㄱ, ㄴ...), 중성(ㅏ, ㅑ...) 구조를 활용한 알고리즘을 정확한 역순으로 적용하세요.
- 최종 답변은 반드시 '원문: [한글정답]' 형식으로 제시하세요.

출력 형식:
원문: [복호화된 한글 텍스트]"""

def extract_answer(output: str) -> str:
    patterns = [
        r'원문[:\s]*([가-힣\s]+)',
        r'정답[:\s]*([가-힣\s]+)',
        r'답[:\s]*([가-힣\s]+)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, output)
        if matches:
            return matches[-1].strip().replace(" ", "")
    
    # 마지막 한글 단어 추출
    words = re.findall(r'[가-힣]{2,}', output)
    if words:
        return words[-1]
    return ''

def call_openai_api(model: str, system_prompt: str, user_prompt: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
            if model.startswith("o1") or model.startswith("gpt-5"):
                kwargs["max_completion_tokens"] = 100000
            else:
                kwargs["max_tokens"] = 10000
                kwargs["temperature"] = 0

            response = client.chat.completions.create(**kwargs)
            output = response.choices[0].message.content.strip()
            return output, {
                'total_tokens': response.usage.total_tokens
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument('--data', type=str, default='../data/json/CYPHER_HANGUL.jsonl')
    args = parser.parse_args()

    data_file = Path(__file__).parent / args.data
    if not data_file.exists():
        print(f"Data not found: {data_file}")
        return

    puzzles = load_data(str(data_file))
    system_prompt = create_prompt()
    results = []

    print(f"Evaluating {len(puzzles)} Hangul puzzles with {args.model}...")

    for i, puzzle in enumerate(puzzles):
        print(f"[{i+1}/{len(puzzles)}] {puzzle['difficulty']}")
        try:
            output, usage = call_openai_api(args.model, system_prompt, puzzle['question'])
            predicted = extract_answer(output)
            correct = predicted == puzzle['answer']
            results.append({
                'difficulty': puzzle['difficulty'],
                'expected': puzzle['answer'],
                'predicted': predicted,
                'correct': correct
            })
            print(f"  Expected: {puzzle['answer']}, Predicted: {predicted} -> {'O' if correct else 'X'}")
        except Exception as e:
            print(f"  Error: {e}")

    # Print statistics
    accuracy = sum(1 for r in results if r['correct']) / len(results) * 100
    print(f"\nFinal Accuracy: {accuracy:.1f}%")

    # Save results
    output_dir = Path(__file__).parent.parent / 'data' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'eval_hangul_{args.model}.json'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
