"""Causal DAG Reasoning Puzzle Evaluation - Korean Version
[진행도] ☑ 완료
[파일명] eval_causal_dag_korean.py
[목적] 한국어 인과관계 DAG 추론 퍼즐 평가

한국어 시간-인과 추론 퍼즐에 대한 LLM 성능 평가
시간에 따른 인과 효과 체인을 추적하는 능력 측정
"""

import os
import sys
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


# ============================================================================
# API Configuration
# ============================================================================

def get_openai_client():
    """OpenAI 클라이언트 초기화"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 환경 변수에 없습니다")
    return OpenAI(api_key=api_key)


def call_openai_api(model: str, prompt: str, temperature: float = 0.0) -> Tuple[str, Dict]:
    """
    재시도 로직이 있는 OpenAI API 호출
    
    Args:
        model: 모델 식별자 (예: 'gpt-4', 'gpt-3.5-turbo')
        prompt: 질문 프롬프트
        temperature: 샘플링 온도
    
    Returns:
        (응답_텍스트, 사용량_딕셔너리) 튜플
    """
    client = get_openai_client()
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "당신은 논리적 추론 전문가입니다. 문제를 신중하게 분석하고 정확한 숫자 답변을 제공하세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=2000
            )
            
            output = response.choices[0].message.content
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            return output, usage
        
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  API 오류: {e}. {wait_time}초 후 재시도...")
                time.sleep(wait_time)
            else:
                raise
    
    raise Exception("모든 재시도 후 API 호출 실패")


# ============================================================================
# Answer Parsing
# ============================================================================

def parse_answer(response: str) -> Optional[int]:
    """
    모델 응답에서 숫자 답변 추출
    
    다음 형식 처리:
    - "45"
    - "답변: 45" 또는 "정답: 45"
    - "사건이 45분에 발생합니다"
    - "45분"
    - "\\boxed{45}"
    
    Args:
        response: 모델의 텍스트 응답
    
    Returns:
        추출된 정수 또는 파싱 실패 시 None
    """
    # 정확한 정수 매치 시도
    response = response.strip()
    
    # 패턴 1: 단순 숫자
    if response.isdigit():
        return int(response)
    
    # 패턴 2: LaTeX boxed 형식: \boxed{45} 또는 \\boxed{45}
    match = re.search(r'\\+boxed\{(\d+)\}', response)
    if match:
        return int(match.group(1))
    
    # 패턴 3: "답변: 45" 또는 "정답: 45"
    match = re.search(r'[답정][변답]\s*[:：]\s*(\d+)', response)
    if match:
        return int(match.group(1))
    
    # 패턴 4: 영어 Answer 패턴도 지원
    match = re.search(r'[Aa]nswer\s*[:：]\s*(\d+)', response)
    if match:
        return int(match.group(1))
    
    # 패턴 5: "사건 X가 45분에 처음 발생합니다" 또는 "45분에 발생합니다"
    matches = list(re.finditer(r'(\d+)\s*분에\s*(?:처음\s*)?발생', response))
    if matches:
        return int(matches[-1].group(1))
    
    # 패턴 6: "45분" (마지막 발생)
    matches = list(re.finditer(r'(\d+)\s*분', response))
    if matches:
        return int(matches[-1].group(1))
    
    # 패턴 7: 영어 패턴도 지원 "occurs at minute 45"
    matches = list(re.finditer(r'(?:first\s+)?occurs?\s+at\s+minute\s+(\d+)', response, re.IGNORECASE))
    if matches:
        return int(matches[-1].group(1))
    
    # 패턴 8: 응답의 마지막 숫자
    numbers = re.findall(r'\b(\d+)\b', response)
    if numbers:
        # 마지막 숫자 사용 (종종 최종 답변)
        return int(numbers[-1])
    
    return None


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_puzzle(puzzle_data: Dict, model: str, verbose: bool = True) -> Dict:
    """
    단일 퍼즐 평가
    
    Args:
        puzzle_data: 'question'과 'answer'가 있는 퍼즐 딕셔너리
        model: 모델 식별자
        verbose: 상세 출력 여부
    
    Returns:
        평가 결과 딕셔너리
    """
    question = puzzle_data['question']
    correct_answer = int(puzzle_data['answer'])
    difficulty = puzzle_data.get('difficulty', 'Unknown')
    
    if verbose:
        print(f"\n  난이도: {difficulty}")
        print(f"  정답: {correct_answer}분")
    
    try:
        # API 호출
        start_time = time.time()
        response, usage = call_openai_api(model, question)
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"  응답 시간: {elapsed:.2f}초")
            print(f"  토큰: {usage['total_tokens']}")
        
        # 답변 파싱
        predicted_answer = parse_answer(response)
        
        if predicted_answer is None:
            if verbose:
                print(f"  ✗ 답변 파싱 실패")
                print(f"  원본 응답: {response[:200]}...")
            
            return {
                'success': False,
                'correct': False,
                'predicted': None,
                'actual': correct_answer,
                'difficulty': difficulty,
                'error': 'parsing_failed',
                'response': response,
                'usage': usage,
                'time': elapsed
            }
        
        # 정확성 확인
        is_correct = (predicted_answer == correct_answer)
        
        if verbose:
            status = "✓" if is_correct else "✗"
            print(f"  {status} 예측: {predicted_answer}, 실제: {correct_answer}")
        
        return {
            'success': True,
            'correct': is_correct,
            'predicted': predicted_answer,
            'actual': correct_answer,
            'difficulty': difficulty,
            'error': None,
            'response': response,
            'usage': usage,
            'time': elapsed
        }
    
    except Exception as e:
        if verbose:
            print(f"  ✗ 오류: {e}")
        
        return {
            'success': False,
            'correct': False,
            'predicted': None,
            'actual': correct_answer,
            'difficulty': difficulty,
            'error': str(e),
            'response': None,
            'usage': None,
            'time': 0
        }


def evaluate_dataset(dataset: List[Dict], model: str, 
                    output_dir: Optional[Path] = None,
                    max_workers: int = 1) -> Dict:
    """
    전체 데이터셋 평가 (병렬 평가 지원)
    
    Args:
        dataset: 퍼즐 딕셔너리 리스트
        model: 모델 식별자
        output_dir: 결과 저장 디렉터리 (선택사항)
        max_workers: 병렬 워커 수 (기본값: 1, 순차 처리)
    
    Returns:
        요약 통계
    """
    print(f"\n{'='*70}")
    print(f"인과 DAG 퍼즐 평가 (한국어)")
    print(f"모델: {model}")
    print(f"데이터셋 크기: {len(dataset)}")
    print(f"워커: {max_workers}")
    print(f"{'='*70}")
    
    results = []
    
    if max_workers == 1:
        # 순차 평가
        for i, puzzle in enumerate(dataset):
            print(f"\n[{i+1}/{len(dataset)}] 퍼즐 평가 중...")
            result = evaluate_puzzle(puzzle, model, verbose=True)
            results.append(result)
            
            # 속도 제한
            if i < len(dataset) - 1:
                time.sleep(0.5)
    else:
        # 병렬 평가
        print_lock = Lock()
        completed_count = [0]  # 클로저를 위한 가변 객체
        
        def evaluate_with_progress(puzzle_with_idx):
            idx, puzzle = puzzle_with_idx
            result = evaluate_puzzle(puzzle, model, verbose=False)
            
            with print_lock:
                completed_count[0] += 1
                status = "✓" if result['correct'] else "✗"
                print(f"[{completed_count[0]}/{len(dataset)}] {status} "
                      f"퍼즐 {idx+1} - 난이도: {result['difficulty']} - "
                      f"예측: {result['predicted']}, 실제: {result['actual']}")
            
            return result
        
        # 병렬 실행
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, puzzle in enumerate(dataset):
                future = executor.submit(evaluate_with_progress, (i, puzzle))
                futures.append(future)
                
                # 속도 제한 회피를 위해 제출을 약간 지연
                time.sleep(0.05)
            
            # 원래 순서대로 결과 수집
            for future in futures:
                results.append(future.result())
    
    # 통계 계산
    total = len(results)
    successful = sum(1 for r in results if r['success'])
    correct = sum(1 for r in results if r['correct'])
    parsing_failed = sum(1 for r in results if r.get('error') == 'parsing_failed')
    
    accuracy = correct / total if total > 0 else 0
    success_rate = successful / total if total > 0 else 0
    
    # 난이도별 통계
    difficulties = set(r['difficulty'] for r in results)
    difficulty_stats = {}
    
    for diff in difficulties:
        diff_results = [r for r in results if r['difficulty'] == diff]
        diff_total = len(diff_results)
        diff_correct = sum(1 for r in diff_results if r['correct'])
        difficulty_stats[diff] = {
            'total': diff_total,
            'correct': diff_correct,
            'accuracy': diff_correct / diff_total if diff_total > 0 else 0
        }
    
    # 토큰 사용량
    total_tokens = sum(r['usage']['total_tokens'] for r in results if r['usage'])
    avg_tokens = total_tokens / successful if successful > 0 else 0
    
    summary = {
        'model': model,
        'timestamp': datetime.now().isoformat(),
        'total_puzzles': total,
        'successful_evaluations': successful,
        'correct_answers': correct,
        'parsing_failures': parsing_failed,
        'accuracy': accuracy,
        'success_rate': success_rate,
        'difficulty_breakdown': difficulty_stats,
        'total_tokens': total_tokens,
        'avg_tokens_per_puzzle': avg_tokens,
        'results': results
    }
    
    # 요약 출력
    print(f"\n{'='*70}")
    print("평가 요약")
    print(f"{'='*70}")
    print(f"총 퍼즐:          {total}")
    print(f"성공:            {successful} ({success_rate*100:.1f}%)")
    print(f"정답:            {correct} ({accuracy*100:.1f}%)")
    print(f"파싱 실패:        {parsing_failed}")
    print(f"\n난이도별 정확도:")
    for diff in ['Easy', 'Medium', 'Hard']:
        if diff in difficulty_stats:
            stats = difficulty_stats[diff]
            print(f"  {diff:10s}: {stats['correct']}/{stats['total']} "
                  f"({stats['accuracy']*100:.1f}%)")
    print(f"\n평균 토큰:        {avg_tokens:.0f}")
    
    # 결과 저장
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 요약 JSON
        summary_path = output_dir / f"summary_korean_{model.replace('/', '_')}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n결과 저장: {summary_path}")
        
        # 상세 결과 JSONL
        details_path = output_dir / f"details_korean_{model.replace('/', '_')}.jsonl"
        with open(details_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"상세 결과: {details_path}")
    
    return summary


# ============================================================================
# Main
# ============================================================================

def load_dataset(dataset_path: Path) -> List[Dict]:
    """JSONL 파일에서 데이터셋 로드"""
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def main():
    """메인 평가 스크립트"""
    import argparse
    
    parser = argparse.ArgumentParser(description='인과 DAG 퍼즐 평가 (한국어)')
    parser.add_argument('--model', type=str, default='gpt-4',
                       help='평가할 OpenAI 모델')
    parser.add_argument('--dataset', type=str, 
                       help='데이터셋 JSONL 파일 경로')
    parser.add_argument('--generate', type=int,
                       help='데이터셋 로드 대신 N개 퍼즐 생성')
    parser.add_argument('--difficulty', type=str, default='Medium',
                       choices=['Easy', 'Medium', 'Hard'],
                       help='생성된 퍼즐의 난이도')
    parser.add_argument('--output', type=str,
                       help='결과 출력 디렉터리')
    parser.add_argument('--workers', type=int, default=1,
                       help='병렬 워커 수 (기본값: 1)')
    
    args = parser.parse_args()
    
    # 데이터셋 로드 또는 생성
    if args.generate:
        print(f"{args.generate}개의 {args.difficulty} 퍼즐 생성 중...")
        sys.path.append(str(Path(__file__).parent.parent / 'guess'))
        from causal_dag_korean import generate_dataset
        
        puzzles_per_diff = args.generate
        dataset = []
        if args.difficulty:
            from causal_dag_korean import CausalPuzzleGenerator, create_question
            generator = CausalPuzzleGenerator()
            for i in range(puzzles_per_diff):
                puzzle = generator.generate_puzzle(args.difficulty, seed=i)
                dataset.append({
                    'question': create_question(puzzle),
                    'answer': str(puzzle.answer),
                    'difficulty': args.difficulty
                })
        else:
            dataset = generate_dataset(puzzles_per_diff=puzzles_per_diff // 3, 
                                      verbose=False)
    elif args.dataset:
        print(f"{args.dataset}에서 데이터셋 로드 중...")
        dataset = load_dataset(Path(args.dataset))
    else:
        print("오류: --generate 또는 --dataset 중 하나를 지정해야 합니다")
        return
    
    # 출력 디렉터리 설정
    if args.output:
        output_dir = Path(args.output)
    else:
        PROJECT_ROOT = Path(__file__).parent.parent
        output_dir = PROJECT_ROOT / "evaluation_results" / "causal_dag_korean"
    
    # 평가 실행
    summary = evaluate_dataset(dataset, args.model, output_dir, max_workers=args.workers)
    
    print(f"\n{'='*70}")
    print("✓ 평가 완료!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
