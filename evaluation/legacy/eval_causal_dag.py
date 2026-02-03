"""Causal DAG Reasoning Puzzle Evaluation

Evaluates LLM performance on temporal-causal reasoning puzzles.
Measures ability to trace cause-effect chains through time.
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
    """Initialize OpenAI client"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return OpenAI(api_key=api_key)


def call_openai_api(model: str, prompt: str, temperature: float = 0.0) -> Tuple[str, Dict]:
    """
    Call OpenAI API with retry logic
    
    Args:
        model: Model identifier (e.g., 'gpt-4', 'gpt-3.5-turbo')
        prompt: Question prompt
        temperature: Sampling temperature
    
    Returns:
        Tuple of (response_text, usage_dict)
    """
    client = get_openai_client()
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a logical reasoning expert. Analyze the problem carefully and provide precise numerical answers."},
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
                print(f"  API error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    
    raise Exception("API call failed after all retries")


# ============================================================================
# Answer Parsing
# ============================================================================

def parse_answer(response: str) -> Optional[int]:
    """
    Extract numerical answer from model response
    
    Handles formats like:
    - "45"
    - "Answer: 45"
    - "The event occurs at minute 45"
    - "45 minutes"
    - "\\boxed{45}"
    
    Args:
        response: Model's text response
    
    Returns:
        Extracted integer or None if parsing fails
    """
    # Try exact integer match
    response = response.strip()
    
    # Pattern 1: Just a number
    if response.isdigit():
        return int(response)
    
    # Pattern 2: LaTeX boxed format: \boxed{45} or \\boxed{45}
    match = re.search(r'\\+boxed\{(\d+)\}', response)
    if match:
        return int(match.group(1))
    
    # Pattern 3: "Answer: 45" or "answer: 45"
    match = re.search(r'[Aa]nswer\s*[:：]\s*(\d+)', response)
    if match:
        return int(match.group(1))
    
    # Pattern 4: Final answer patterns (last occurrence)
    # "event X first occurs at minute 45" or "occurs at minute 45"
    matches = list(re.finditer(r'(?:first\s+)?occurs?\s+at\s+minute\s+(\d+)', response, re.IGNORECASE))
    if matches:
        return int(matches[-1].group(1))
    
    # Pattern 5: "at minute 45" (last occurrence)
    matches = list(re.finditer(r'at\s+minute\s+(\d+)', response, re.IGNORECASE))
    if matches:
        return int(matches[-1].group(1))
    
    # Pattern 6: "minute 45" (last occurrence)
    matches = list(re.finditer(r'[Mm]inute\s+(\d+)', response))
    if matches:
        return int(matches[-1].group(1))
    
    # Pattern 7: "45 minutes"
    match = re.search(r'(\d+)\s+[Mm]inutes?', response)
    if match:
        return int(match.group(1))
    
    # Pattern 8: Last number in response
    numbers = re.findall(r'\b(\d+)\b', response)
    if numbers:
        # Take last number (often the final answer)
        return int(numbers[-1])
    
    return None


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_puzzle(puzzle_data: Dict, model: str, verbose: bool = True) -> Dict:
    """
    Evaluate a single puzzle
    
    Args:
        puzzle_data: Puzzle dictionary with 'question' and 'answer'
        model: Model identifier
        verbose: Print detailed output
    
    Returns:
        Evaluation result dictionary
    """
    question = puzzle_data['question']
    correct_answer = int(puzzle_data['answer'])
    difficulty = puzzle_data.get('difficulty', 'Unknown')
    
    if verbose:
        print(f"\n  Difficulty: {difficulty}")
        print(f"  Correct Answer: {correct_answer} minutes")
    
    try:
        # Call API
        start_time = time.time()
        response, usage = call_openai_api(model, question)
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"  Response time: {elapsed:.2f}s")
            print(f"  Tokens: {usage['total_tokens']}")
        
        # Parse answer
        predicted_answer = parse_answer(response)
        
        if predicted_answer is None:
            if verbose:
                print(f"  ✗ Failed to parse answer")
                print(f"  Raw response: {response[:200]}...")
            
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
        
        # Check correctness
        is_correct = (predicted_answer == correct_answer)
        
        if verbose:
            status = "✓" if is_correct else "✗"
            print(f"  {status} Predicted: {predicted_answer}, Actual: {correct_answer}")
        
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
            print(f"  ✗ Error: {e}")
        
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
    Evaluate entire dataset (supports parallel evaluation)
    
    Args:
        dataset: List of puzzle dictionaries
        model: Model identifier
        output_dir: Directory to save results (optional)
        max_workers: Number of parallel workers (default: 1 for sequential)
    
    Returns:
        Summary statistics
    """
    print(f"\n{'='*70}")
    print(f"Evaluating Causal DAG Puzzles")
    print(f"Model: {model}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Workers: {max_workers}")
    print(f"{'='*70}")
    
    results = []
    
    if max_workers == 1:
        # Sequential evaluation
        for i, puzzle in enumerate(dataset):
            print(f"\n[{i+1}/{len(dataset)}] Evaluating puzzle...")
            result = evaluate_puzzle(puzzle, model, verbose=True)
            results.append(result)
            
            # Rate limiting
            if i < len(dataset) - 1:
                time.sleep(0.5)
    else:
        # Parallel evaluation
        print_lock = Lock()
        completed_count = [0]  # Mutable for closure
        
        def evaluate_with_progress(puzzle_with_idx):
            idx, puzzle = puzzle_with_idx
            result = evaluate_puzzle(puzzle, model, verbose=False)
            
            with print_lock:
                completed_count[0] += 1
                status = "✓" if result['correct'] else "✗"
                print(f"[{completed_count[0]}/{len(dataset)}] {status} "
                      f"Puzzle {idx+1} - Difficulty: {result['difficulty']} - "
                      f"Predicted: {result['predicted']}, Actual: {result['actual']}")
            
            return result
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, puzzle in enumerate(dataset):
                future = executor.submit(evaluate_with_progress, (i, puzzle))
                futures.append(future)
                
                # Stagger submissions slightly to avoid rate limits
                time.sleep(0.05)
            
            # Collect results in original order
            for future in futures:
                results.append(future.result())
    
    # Calculate statistics
    total = len(results)
    successful = sum(1 for r in results if r['success'])
    correct = sum(1 for r in results if r['correct'])
    parsing_failed = sum(1 for r in results if r.get('error') == 'parsing_failed')
    
    accuracy = correct / total if total > 0 else 0
    success_rate = successful / total if total > 0 else 0
    
    # Per-difficulty statistics
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
    
    # Token usage
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
    
    # Print summary
    print(f"\n{'='*70}")
    print("Evaluation Summary")
    print(f"{'='*70}")
    print(f"Total Puzzles:    {total}")
    print(f"Successful:       {successful} ({success_rate*100:.1f}%)")
    print(f"Correct:          {correct} ({accuracy*100:.1f}%)")
    print(f"Parsing Failed:   {parsing_failed}")
    print(f"\nPer-Difficulty Accuracy:")
    for diff in ['Easy', 'Medium', 'Hard']:
        if diff in difficulty_stats:
            stats = difficulty_stats[diff]
            print(f"  {diff:10s}: {stats['correct']}/{stats['total']} "
                  f"({stats['accuracy']*100:.1f}%)")
    print(f"\nAverage Tokens:   {avg_tokens:.0f}")
    
    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Summary JSON
        summary_path = output_dir / f"summary_{model.replace('/', '_')}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {summary_path}")
        
        # Detailed results JSONL
        details_path = output_dir / f"details_{model.replace('/', '_')}.jsonl"
        with open(details_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"Detailed results: {details_path}")
    
    return summary


# ============================================================================
# Main
# ============================================================================

def load_dataset(dataset_path: Path) -> List[Dict]:
    """Load dataset from JSONL file"""
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Causal DAG puzzles')
    parser.add_argument('--model', type=str, default='gpt-4',
                       help='OpenAI model to evaluate')
    parser.add_argument('--dataset', type=str, 
                       help='Path to dataset JSONL file')
    parser.add_argument('--generate', type=int,
                       help='Generate N puzzles instead of loading dataset')
    parser.add_argument('--difficulty', type=str, default='Medium',
                       choices=['Easy', 'Medium', 'Hard'],
                       help='Difficulty level for generated puzzles')
    parser.add_argument('--output', type=str,
                       help='Output directory for results')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers (default: 1)')
    
    args = parser.parse_args()
    
    # Load or generate dataset
    if args.generate:
        print(f"Generating {args.generate} {args.difficulty} puzzles...")
        sys.path.append(str(Path(__file__).parent.parent / 'guess'))
        from causal_dag import generate_dataset
        
        puzzles_per_diff = args.generate
        dataset = []
        if args.difficulty:
            from causal_dag import CausalPuzzleGenerator, create_question
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
        print(f"Loading dataset from {args.dataset}...")
        dataset = load_dataset(Path(args.dataset))
    else:
        print("Error: Must specify either --generate or --dataset")
        return
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        PROJECT_ROOT = Path(__file__).parent.parent
        output_dir = PROJECT_ROOT / "evaluation_results" / "causal_dag"
    
    # Run evaluation
    summary = evaluate_dataset(dataset, args.model, output_dir, max_workers=args.workers)
    
    print(f"\n{'='*70}")
    print("✓ Evaluation complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

