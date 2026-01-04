"""Number Baseball (Bulls and Cows) LLM Evaluation Script

Evaluates LLM performance on number baseball puzzles using OpenAI models.
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
from number_baseball import BullsAndCows, Hint, validate_problem

# Load .env
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


# ============================================================================
# Prompt Creation
# ============================================================================

def create_prompt(puzzle_data: dict) -> str:
    """
    Create evaluation prompt for number baseball puzzle.

    The prompt includes:
    1. Task description with rules
    2. The hints in clear format
    3. Expected output format
    """
    question = puzzle_data['question']

    prompt = f"""{question}

Think step by step:
1. Start by analyzing what each hint tells you
2. Eliminate impossible digits for each position
3. Use the combination of all hints to narrow down
4. Find the unique number that satisfies ALL hints

After solving, provide your final answer.
Format: Answer: [the secret number]"""

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
    3. Validate length and uniqueness of digits
    """
    # Pattern 1: "Answer: 123" or "Answer: 1234"
    patterns = [
        r'Answer:\s*(\d+)',
        r'answer:\s*(\d+)',
        r'secret number[:\s]+(\d+)',
        r'number is[:\s]+(\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            digits = match.group(1)
            if len(digits) == expected_length:
                return digits

    # Fallback: Find standalone N-digit numbers at the end of output
    last_part = output[-500:] if len(output) > 500 else output
    numbers = re.findall(rf'\b(\d{{{expected_length}}})\b', last_part)
    if numbers:
        # Return the last N-digit number found
        return numbers[-1]

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
                    {"role": "system", "content": "You are an expert puzzle solver specializing in logical deduction games like Bulls and Cows (Number Baseball)."},
                    {"role": "user", "content": prompt}
                ]
            }

            # o1 series uses max_completion_tokens, others use max_tokens
            if model.startswith("o1"):
                kwargs["max_completion_tokens"] = 32000
            else:
                kwargs["max_tokens"] = 8000
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
    difficulty = puzzle_data.get('difficulty', 'unknown')
    num_digits = puzzle_data.get('num_digits', 3)
    expected_answer = puzzle_data['answer']
    num_hints = len(puzzle_data.get('hints', []))

    print(f"  Difficulty: {difficulty}, Digits: {num_digits}, Hints: {num_hints}")
    print(f"  Answer: {expected_answer}")

    # Create prompt
    prompt = create_prompt(puzzle_data)

    try:
        # API call
        output, usage = call_openai_api(model, prompt)

        print(f"  Response length: {len(output)} chars, Tokens: {usage['total_tokens']}")

        # Extract answer
        predicted_answer = extract_answer(output, num_digits)

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
            'difficulty': difficulty,
            'num_digits': num_digits,
            'num_hints': num_hints,
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
            'difficulty': difficulty,
            'num_digits': num_digits,
            'num_hints': num_hints,
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
    parser = argparse.ArgumentParser(description='Number Baseball LLM Evaluation')
    parser.add_argument('--model', type=str, default='gpt-4o',
                       choices=['gpt-4o', 'gpt-4o-mini', 'o1', 'o1-mini'],
                       help='Model to evaluate')
    parser.add_argument('--data', type=str,
                       default='../data/json/NUMBER_BASEBALL_v1.jsonl',
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
        print(f"  python number_baseball.py")
        return

    # Output path setup
    if args.output is None:
        results_dir = script_dir.parent / 'data' / 'number_baseball' / 'results'
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
    print(f"Number Baseball (Bulls and Cows) LLM Evaluation")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Data: {data_file.name}")
    print(f"Puzzles: {len(puzzles)}")
    print(f"{'='*70}\n")

    all_results = []

    for i, puzzle in enumerate(puzzles):
        print(f"[{i+1}/{len(puzzles)}] {puzzle.get('difficulty', 'Unknown')}")
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

    # Statistics by number of digits
    print("\nResults by number of digits:")
    by_digits = defaultdict(list)
    for r in all_results:
        by_digits[r['num_digits']].append(r)

    for digits in sorted(by_digits.keys()):
        items = by_digits[digits]
        correct = sum(1 for r in items if r['correct'])
        acc = correct / len(items) * 100
        print(f"  {digits} digits: {correct}/{len(items)} correct ({acc:.1f}%)")

    # Statistics by number of hints
    print("\nResults by number of hints:")
    by_hints = defaultdict(list)
    for r in all_results:
        by_hints[r['num_hints']].append(r)

    for hints in sorted(by_hints.keys()):
        items = by_hints[hints]
        correct = sum(1 for r in items if r['correct'])
        acc = correct / len(items) * 100
        print(f"  {hints} hints: {correct}/{len(items)} correct ({acc:.1f}%)")

    print(f"\n{'='*70}")
    print(f"✓ Results saved: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
