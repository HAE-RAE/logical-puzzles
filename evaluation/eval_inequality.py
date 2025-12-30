"""Inequality Puzzle LLM Evaluation Script

Evaluates LLM performance on inequality puzzles using OpenAI models.
"""

import json
import os
import sys
import time
import re
from pathlib import Path
from typing import Optional, List
from openai import OpenAI
from dotenv import load_dotenv
import argparse
from collections import defaultdict

# Add guess directory to import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'guess'))
from inequality import InequalityPuzzleGenerator

# Load .env
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


# ============================================================================
# Prompt Creation
# ============================================================================

def create_prompt(puzzle_data: dict) -> str:
    """
    Create evaluation prompt for inequality puzzle.

    The prompt includes:
    1. Task description with rules
    2. The puzzle in visual format
    3. Expected output format
    """
    question = puzzle_data['question']

    prompt = f"""{question}

Think step by step:
1. Identify which positions already have numbers
2. Determine what numbers are left to place
3. Use the inequality constraints to narrow down possibilities
4. Find the unique arrangement that satisfies all constraints

After solving, provide your final answer as a sequence of digits.
Format: Answer: [sequence of digits]"""

    return prompt


# ============================================================================
# Response Parsing
# ============================================================================

def extract_answer(output: str, expected_length: int) -> Optional[str]:
    """
    Extract digit sequence answer from LLM output.

    Parsing strategy:
    1. Look for "Answer: [digits]" pattern
    2. Extract the digit sequence
    3. Validate length matches expected
    """
    # Pattern 1: "Answer: 12345" or "Answer: 1 2 3 4 5"
    patterns = [
        r'Answer:\s*([1-9\s]+)',
        r'answer:\s*([1-9\s]+)',
        r'solution[:\s]+([1-9\s]+)',
        r'sequence[:\s]+([1-9\s]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            # Extract digits only, remove spaces
            digits = re.findall(r'[1-9]', match.group(1))
            if len(digits) >= expected_length:
                return ''.join(digits[:expected_length])

    # Fallback: Find the last sequence of N digits
    # Look for patterns like "3 1 4 2" or "3142"
    digit_sequences = re.findall(r'[1-9][\s,]*[1-9][\s,]*[1-9][\s,]*[1-9]?[\s,]*[1-9]?[\s,]*[1-9]?[\s,]*[1-9]?', output)
    if digit_sequences:
        last_seq = digit_sequences[-1]
        digits = re.findall(r'[1-9]', last_seq)
        if len(digits) >= expected_length:
            return ''.join(digits[:expected_length])

    # Last resort: Extract all digits from the end of output
    last_part = output[-200:] if len(output) > 200 else output
    all_digits = re.findall(r'[1-9]', last_part)
    if len(all_digits) >= expected_length:
        return ''.join(all_digits[-expected_length:])

    return None


# ============================================================================
# API Call
# ============================================================================

def call_openai_api(model: str, prompt: str, max_retries: int = 3) -> tuple:
    """Call OpenAI API with retry logic."""
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert puzzle solver specializing in logical constraint puzzles."},
                    {"role": "user", "content": prompt}
                ]
            }

            # o1 series uses max_completion_tokens, others use max_tokens
            if model.startswith("o1"):
                kwargs["max_completion_tokens"] = 16000
            else:
                kwargs["max_tokens"] = 4000
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
            print(f"    ✗ API error (attempt {attempt + 1}/{max_retries}): {error_msg[:100]}")

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"    ⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise Exception(f"API call failed ({max_retries} attempts): {error_msg}")


# ============================================================================
# Puzzle Evaluation
# ============================================================================

def evaluate_puzzle(puzzle_data: dict, model: str) -> dict:
    """Evaluate a single puzzle."""
    problem = puzzle_data.get('solution', puzzle_data.get('problem', 'unknown'))
    difficulty = puzzle_data.get('difficulty', 'unknown')
    expected_answer = puzzle_data['answer']
    size = puzzle_data.get('size', len(expected_answer))

    print(f"  Puzzle: {problem}")
    print(f"  Difficulty: {difficulty}, Size: {size}")

    # Create prompt
    prompt = create_prompt(puzzle_data)

    try:
        # API call
        output, usage = call_openai_api(model, prompt)

        print(f"  Response length: {len(output)} chars, Tokens: {usage['total_tokens']}")

        # Extract answer
        predicted_answer = extract_answer(output, size)

        print(f"  Expected: {expected_answer}")
        print(f"  Predicted: {predicted_answer}")

        # Evaluate
        if predicted_answer is None:
            correct = False
            status = '✗ Failed to extract answer'
        else:
            correct = str(predicted_answer) == str(expected_answer)
            status = '✓ Correct' if correct else '✗ Incorrect'

        print(f"  {status}\n")

        return {
            'problem': problem,
            'difficulty': difficulty,
            'size': size,
            'model': model,
            'expected': expected_answer,
            'predicted': predicted_answer,
            'correct': correct,
            'output_length': len(output),
            'usage': usage,
            'full_output': output
        }

    except Exception as e:
        error_msg = str(e)
        print(f"  ✗ Evaluation failed: {error_msg}\n")
        return {
            'problem': problem,
            'difficulty': difficulty,
            'size': size,
            'model': model,
            'expected': expected_answer,
            'predicted': None,
            'correct': False,
            'output_length': 0,
            'usage': {},
            'error': error_msg
        }


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Inequality Puzzle LLM Evaluation')
    parser.add_argument('--model', type=str, default='gpt-4o',
                       choices=['gpt-4o', 'gpt-4o-mini', 'o1', 'o1-mini'],
                       help='Model to evaluate')
    parser.add_argument('--data', type=str,
                       default='../data/json/INEQUALITY_v1.jsonl',
                       help='Path to evaluation data')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save results')
    args = parser.parse_args()

    # Path setup
    script_dir = Path(__file__).parent
    data_file = script_dir / args.data

    if not data_file.exists():
        print(f"✗ Data file not found: {data_file}")
        print(f"Generate puzzles first:")
        print(f"  cd ../guess")
        print(f"  python inequality.py")
        return

    # Output path setup
    if args.output is None:
        results_dir = script_dir.parent / 'data' / 'inequality' / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f'eval_results_{args.model}.json'
    else:
        output_path = Path(args.output)

    # Load evaluation data
    puzzles = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                puzzles.append(json.loads(line))

    print(f"{'='*70}")
    print(f"Inequality Puzzle LLM Evaluation")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Data: {data_file.name}")
    print(f"Puzzles: {len(puzzles)}")
    print(f"{'='*70}\n")

    all_results = []

    for i, puzzle in enumerate(puzzles):
        print(f"[{i+1}/{len(puzzles)}] {puzzle.get('solution', puzzle.get('problem', 'Unknown'))}")
        print("-"*70)

        result = evaluate_puzzle(puzzle, args.model)
        all_results.append(result)

        # API rate limiting
        time.sleep(1)

    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Print statistics
    print("\n" + "="*70)
    print("Evaluation Results Summary")
    print("="*70)

    total = len(all_results)
    correct_count = sum(1 for r in all_results if r['correct'])
    accuracy = correct_count / total * 100 if total > 0 else 0
    total_tokens = sum(r.get('usage', {}).get('total_tokens', 0) for r in all_results)

    print(f"\n{args.model}:")
    print(f"  Correct: {correct_count}/{total} ({accuracy:.1f}%)")
    print(f"  Total tokens: {total_tokens:,}")

    # Statistics by difficulty
    print("\nResults by difficulty:")
    by_difficulty = defaultdict(list)
    for r in all_results:
        by_difficulty[r['difficulty']].append(r)

    for diff in ['EASY', 'MEDIUM', 'HARD', 'EXPERT']:
        if diff in by_difficulty:
            items = by_difficulty[diff]
            correct = sum(1 for r in items if r['correct'])
            acc = correct / len(items) * 100
            print(f"  {diff:8s}: {correct}/{len(items)} correct ({acc:.1f}%)")

    # Statistics by size
    print("\nResults by puzzle size:")
    by_size = defaultdict(list)
    for r in all_results:
        by_size[r['size']].append(r)

    for size in sorted(by_size.keys()):
        items = by_size[size]
        correct = sum(1 for r in items if r['correct'])
        acc = correct / len(items) * 100
        print(f"  Size {size}: {correct}/{len(items)} correct ({acc:.1f}%)")

    print(f"\n{'='*70}")
    print(f"✓ Results saved: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
