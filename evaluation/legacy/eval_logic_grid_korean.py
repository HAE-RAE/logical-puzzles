#!/usr/bin/env python3
"""
Evaluation script for Logic Grid Puzzles - Korean Version
[진행도] ☑ 완료
[파일명] eval_logic_grid_korean.py
[목적] 한국어 논리 그리드 퍼즐 평가
"""

import json
import argparse
import os
import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

# Import OpenAI
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

client = OpenAI()


def load_puzzles(dataset_path: str) -> List[dict]:
    """Load puzzles from JSONL file"""
    puzzles = []
    with open(dataset_path, 'r') as f:
        for line in f:
            puzzles.append(json.loads(line.strip()))
    return puzzles


def create_prompt(puzzle: dict) -> str:
    """퍼즐 데이터에서 미리 생성된 프롬프트 가져오기"""
    return puzzle['question']


def parse_answer(response: str, people: List[str], categories: List[str]) -> Optional[Dict[str, Dict[str, str]]]:
    """Parse the LLM's response to extract the answer"""
    try:
        # Try to find JSON in the response
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            answer_json = json_match.group(1)
            answer = json.loads(answer_json)
            
            # Validate structure
            if not isinstance(answer, dict):
                return None
            
            # Check all people are present
            for person in people:
                if person not in answer:
                    return None
                if not isinstance(answer[person], dict):
                    return None
                
                # Check all categories are present
                for cat in categories:
                    if cat not in answer[person]:
                        return None
            
            return answer
        
        # Try to find JSON without markdown
        json_match = re.search(r'\{[^{}]*"[^"]+"\s*:\s*\{[^{}]+\}[^{}]*\}', response, re.DOTALL)
        if json_match:
            answer = json.loads(json_match.group(0))
            
            # Validate
            if isinstance(answer, dict):
                valid = True
                for person in people:
                    if person not in answer or not isinstance(answer[person], dict):
                        valid = False
                        break
                    for cat in categories:
                        if cat not in answer[person]:
                            valid = False
                            break
                
                if valid:
                    return answer
        
        return None
    
    except (json.JSONDecodeError, AttributeError):
        return None


def check_answer(predicted: Dict[str, Dict[str, str]], actual: Dict[str, Dict[str, str]]) -> Tuple[bool, float]:
    """
    Check if predicted answer matches actual answer.
    Returns (is_correct, partial_score)
    """
    if predicted is None:
        return False, 0.0
    
    total_assignments = 0
    correct_assignments = 0
    
    for person, attrs in actual.items():
        if person not in predicted:
            total_assignments += len(attrs)
            continue
        
        for cat, val in attrs.items():
            total_assignments += 1
            if cat in predicted[person] and predicted[person][cat] == val:
                correct_assignments += 1
    
    partial_score = correct_assignments / total_assignments if total_assignments > 0 else 0.0
    is_correct = (correct_assignments == total_assignments)
    
    return is_correct, partial_score


def evaluate_puzzle(puzzle: dict, model: str = "gpt-4o", temperature: float = 0.0) -> dict:
    """Evaluate a single puzzle"""
    prompt = create_prompt(puzzle)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "당신은 논리 퍼즐 해결 전문가입니다. 모든 제약 조건을 신중하게 분석하고 정확한 답변을 제공하세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=2000
        )
        
        response_text = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        
        # Parse answer
        predicted = parse_answer(
            response_text,
            puzzle['people'],
            list(puzzle['attributes'].keys())
        )
        
        # Check correctness
        is_correct, partial_score = check_answer(predicted, puzzle['answer'])
        
        return {
            'puzzle_id': puzzle['id'],
            'difficulty': puzzle['difficulty'],
            'correct': is_correct,
            'partial_score': partial_score,
            'predicted': predicted,
            'actual': puzzle['answer'],
            'response': response_text,
            'tokens': tokens_used
        }
    
    except Exception as e:
        print(f"Error evaluating puzzle {puzzle['id']}: {str(e)}")
        return {
            'puzzle_id': puzzle['id'],
            'difficulty': puzzle['difficulty'],
            'correct': False,
            'partial_score': 0.0,
            'predicted': None,
            'actual': puzzle['answer'],
            'response': f"ERROR: {str(e)}",
            'tokens': 0
        }


def evaluate_dataset(
    dataset_path: str,
    model: str = "gpt-4o",
    output_dir: str = "evaluation_results/logic_grid",
    temperature: float = 0.0,
    workers: int = 1
):
    """Evaluate entire dataset"""
    print(f"Loading dataset from {dataset_path}...")
    puzzles = load_puzzles(dataset_path)
    print(f"Loaded {len(puzzles)} puzzles")
    
    print(f"Model: {model}")
    print(f"Workers: {workers}")
    print(f"Temperature: {temperature}")
    print()
    
    # Run evaluation
    print("Running evaluation...")
    if workers > 1:
        with Pool(workers) as p:
            results = list(tqdm(
                p.imap(partial(evaluate_puzzle, model=model, temperature=temperature), puzzles),
                total=len(puzzles)
            ))
    else:
        results = []
        for puzzle in tqdm(puzzles):
            result = evaluate_puzzle(puzzle, model=model, temperature=temperature)
            results.append(result)
    
    # Print progress
    for result in results:
        status = "✓" if result['correct'] else "✗"
        print(f"  [{result['puzzle_id']}] {status} (score: {result['partial_score']:.2f})")
    
    print("\nCalculating statistics...")
    
    # Calculate statistics
    stats = {
        'total': len(results),
        'correct': sum(1 for r in results if r['correct']),
        'by_difficulty': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'total_tokens': sum(r['tokens'] for r in results),
        'avg_partial_score': sum(r['partial_score'] for r in results) / len(results)
    }
    
    for result in results:
        diff = result['difficulty']
        stats['by_difficulty'][diff]['total'] += 1
        if result['correct']:
            stats['by_difficulty'][diff]['correct'] += 1
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving results to {output_dir}...")
    
    # Save summary
    summary_path = os.path.join(output_dir, f"summary_{model}.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'model': model,
            'dataset': dataset_path,
            'statistics': {
                'total': stats['total'],
                'correct': stats['correct'],
                'accuracy': stats['correct'] / stats['total'],
                'avg_partial_score': stats['avg_partial_score'],
                'by_difficulty': dict(stats['by_difficulty']),
                'total_tokens': stats['total_tokens'],
                'avg_tokens': stats['total_tokens'] / stats['total']
            }
        }, f, indent=2)
    print(f"  - Summary: {summary_path}")
    
    # Save detailed results
    details_path = os.path.join(output_dir, f"details_{model}.jsonl")
    with open(details_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    print(f"  - Details: {details_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Model: {model}")
    print(f"Total Puzzles: {stats['total']}")
    print(f"Correct: {stats['correct']}")
    print(f"Overall Accuracy: {stats['correct'] / stats['total'] * 100:.2f}%")
    print(f"Average Partial Score: {stats['avg_partial_score'] * 100:.2f}%")
    print()
    print("Difficulty Breakdown:")
    for diff in ['Easy', 'Medium', 'Hard']:
        if diff in stats['by_difficulty']:
            d = stats['by_difficulty'][diff]
            acc = d['correct'] / d['total'] * 100 if d['total'] > 0 else 0
            print(f"  {diff:8s}: {d['correct']:3d}/{d['total']:3d} ({acc:5.2f}%)")
    print()
    print("Token Usage:")
    print(f"  Total: {stats['total_tokens']:,}")
    print(f"  Average per puzzle: {stats['total_tokens'] / stats['total']:.1f}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Logic Grid Puzzles")
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to the puzzle dataset (JSONL)')
    parser.add_argument('--model', type=str, default='gpt-4o',
                       help='OpenAI model to use')
    parser.add_argument('--output-dir', type=str, default='evaluation_results/logic_grid',
                       help='Output directory for results')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Sampling temperature')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers')
    
    args = parser.parse_args()
    
    evaluate_dataset(
        dataset_path=args.dataset,
        model=args.model,
        output_dir=args.output_dir,
        temperature=args.temperature,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
