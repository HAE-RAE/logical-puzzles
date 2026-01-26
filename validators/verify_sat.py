#!/usr/bin/env python3
"""
Verification script for SAT Puzzles
Checks if generated solutions actually satisfy all clauses
"""

import json
import random
from typing import Dict, List, Tuple


def load_puzzles(jsonl_path: str) -> List[dict]:
    """Load puzzles from JSONL"""
    puzzles = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            puzzles.append(json.loads(line.strip()))
    return puzzles


def eval_literal(var: str, is_positive: bool, assignment: Dict[str, bool]) -> bool:
    """Evaluate a literal given an assignment"""
    var_value = assignment[var]
    return var_value if is_positive else not var_value


def eval_clause(clause_data: List, assignment: Dict[str, bool]) -> bool:
    """Evaluate a clause (disjunction of literals)"""
    # Clause data is a list of [var, is_positive] pairs
    # e.g., [['A', True], ['B', False]] means (A OR NOT B)
    
    # A clause is true if at least one literal is true
    return any(eval_literal(lit[0], lit[1], assignment) for lit in clause_data)


def verify_solution_satisfies_clauses(puzzle: dict, verbose: bool = False) -> bool:
    """Verify that the provided solution satisfies all clauses"""
    solution = puzzle['answer']
    clauses_data = puzzle['clauses']
    
    if verbose:
        print(f"  Checking {len(clauses_data)} clauses...")
    
    # The clauses are stored as a flat list in the JSON
    # We need to reconstruct them properly
    # Actually, looking at the generator, it stores each clause's literals in sequence
    
    # Parse clauses from the flat structure
    # Based on the generator, each clause is serialized as its literals
    # Let's check the structure first
    
    failed_clauses = []
    
    # Try to parse - the structure might be wrong, let's check actual data
    for i, clause_data in enumerate(clauses_data):
        # clause_data should be a list of [var, is_positive, var, is_positive, ...]
        if not eval_clause(clause_data, solution):
            failed_clauses.append(i)
            if verbose:
                print(f"  ❌ Clause {i + 1} FAILED: {clause_data}")
    
    if failed_clauses:
        print(f"  ❌ {len(failed_clauses)} clause(s) failed")
        return False
    
    if verbose:
        print(f"  ✅ All clauses satisfied")
    
    return True


def count_solutions(puzzle: dict) -> int:
    """
    Count the number of valid solutions for the puzzle.
    Uses brute force for small problems.
    """
    variables = puzzle['variables']
    clauses_data = puzzle['clauses']
    
    # For large problems, skip (too slow)
    if len(variables) > 10:
        return -1  # Unknown
    
    num_solutions = 0
    
    # Try all possible assignments
    for i in range(2 ** len(variables)):
        # Generate assignment from binary representation
        assignment = {}
        for j, var in enumerate(variables):
            assignment[var] = bool((i >> j) & 1)
        
        # Check if all clauses are satisfied
        all_satisfied = True
        for clause_data in clauses_data:
            if not eval_clause(clause_data, assignment):
                all_satisfied = False
                break
        
        if all_satisfied:
            num_solutions += 1
            if num_solutions > 1:
                return num_solutions  # Early exit
    
    return num_solutions


def load_evaluation_results(details_path: str) -> List[dict]:
    """Load evaluation results"""
    results = []
    with open(details_path, 'r') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def verify_evaluation(puzzle: dict, eval_result: dict, verbose: bool = False) -> bool:
    """Verify that evaluation was done correctly"""
    actual = puzzle['answer']
    predicted = eval_result['predicted']
    
    if predicted is None:
        if verbose:
            print("  ⚠️  LLM failed to parse answer")
        return True  # Evaluation is correct (marked as wrong)
    
    # Manually check correctness
    correct = all(predicted.get(var) == actual[var] for var in puzzle['variables'])
    
    # Check if evaluation result matches our manual check
    if correct != eval_result['correct']:
        print(f"  ❌ EVALUATION ERROR!")
        print(f"     Manual check: {correct}")
        print(f"     Eval result: {eval_result['correct']}")
        return False
    
    return True


def main():
    print("="*70)
    print("SAT PUZZLE VERIFICATION")
    print("="*70)
    
    # Load puzzles
    print("\n1. Loading puzzles...")
    puzzles = load_puzzles('data/sat/sat_puzzles.jsonl')
    print(f"   Loaded {len(puzzles)} puzzles")
    
    # Check clause data structure first
    print("\n2. Checking clause data structure...")
    sample_puzzle = puzzles[0]
    print(f"   Sample puzzle ID: {sample_puzzle['id']}")
    print(f"   Variables: {sample_puzzle['variables']}")
    print(f"   Number of clauses: {len(sample_puzzle['clauses'])}")
    if sample_puzzle['clauses']:
        print(f"   First clause data: {sample_puzzle['clauses'][0]}")
    
    # Verify solutions satisfy clauses
    print("\n3. Verifying solutions satisfy clauses...")
    clause_errors = 0
    
    sample_puzzles = random.sample(puzzles, min(10, len(puzzles)))
    for puzzle in sample_puzzles:
        print(f"\n   Puzzle: {puzzle['id']} ({puzzle['difficulty']})")
        if not verify_solution_satisfies_clauses(puzzle, verbose=False):
            clause_errors += 1
            print(f"   ❌ Solution does not satisfy clauses!")
            # Show details
            print(f"   Answer: {puzzle['answer']}")
            verify_solution_satisfies_clauses(puzzle, verbose=True)
        else:
            print(f"   ✅ Solution satisfies all clauses")
    
    if clause_errors == 0:
        print(f"\n   ✅ All {len(sample_puzzles)} sampled puzzles have valid solutions")
    else:
        print(f"\n   ❌ {clause_errors}/{len(sample_puzzles)} puzzles have invalid solutions")
        return
    
    # Check uniqueness for small puzzles
    print("\n4. Checking solution uniqueness...")
    small_puzzles = [p for p in puzzles if len(p['variables']) <= 6]
    print(f"   Found {len(small_puzzles)} puzzles with ≤6 variables")
    
    if small_puzzles:
        sample_small = random.sample(small_puzzles, min(5, len(small_puzzles)))
        
        for puzzle in sample_small:
            print(f"\n   Puzzle: {puzzle['id']} ({len(puzzle['variables'])} vars)")
            num_sols = count_solutions(puzzle)
            
            if num_sols == 1:
                print(f"   ✅ Unique solution")
            elif num_sols > 1:
                print(f"   ⚠️  Multiple solutions: {num_sols}")
            else:
                print(f"   ⚠️  No solutions found!")
    
    # Verify evaluation
    print("\n5. Verifying evaluation results...")
    try:
        eval_results = load_evaluation_results('evaluation_results/sat/details_gpt-4o.jsonl')
        print(f"   Loaded {len(eval_results)} evaluation results")
        
        # Create puzzle lookup
        puzzle_dict = {p['id']: p for p in puzzles}
        
        # Check a sample of correct and incorrect
        correct_results = [r for r in eval_results if r['correct']]
        incorrect_results = [r for r in eval_results if not r['correct']]
        
        print(f"\n   Checking sample of CORRECT evaluations (marked as ✓)...")
        sample_correct = random.sample(correct_results, min(3, len(correct_results)))
        eval_errors = 0
        
        for result in sample_correct:
            puzzle = puzzle_dict[result['puzzle_id']]
            print(f"\n   Puzzle: {result['puzzle_id']} ({puzzle['difficulty']})")
            if not verify_evaluation(puzzle, result, verbose=True):
                eval_errors += 1
            else:
                print(f"   ✅ Evaluation correct")
        
        print(f"\n   Checking sample of INCORRECT evaluations (marked as ✗)...")
        sample_incorrect = random.sample(incorrect_results, min(5, len(incorrect_results)))
        
        for result in sample_incorrect:
            puzzle = puzzle_dict[result['puzzle_id']]
            print(f"\n   Puzzle: {result['puzzle_id']} ({puzzle['difficulty']})")
            print(f"   Partial score: {result['partial_score']:.2f}")
            
            if result['predicted']:
                # Show what was wrong
                diffs = []
                for var in puzzle['variables']:
                    actual_val = puzzle['answer'][var]
                    pred_val = result['predicted'].get(var, '???')
                    if actual_val != pred_val:
                        diffs.append(f"{var}: {pred_val} ≠ {actual_val}")
                
                if diffs:
                    print(f"   Differences ({len(diffs)}):")
                    for diff in diffs[:3]:  # Show first 3
                        print(f"     - {diff}")
            
            if not verify_evaluation(puzzle, result, verbose=False):
                eval_errors += 1
            else:
                print(f"   ✅ Evaluation correct (LLM was wrong)")
        
        if eval_errors == 0:
            print(f"\n   ✅ All sampled evaluations are correct")
        else:
            print(f"\n   ❌ {eval_errors} evaluation errors found")
    
    except FileNotFoundError:
        print("\n   ⚠️  Evaluation results not found, skipping verification")
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
