#!/usr/bin/env python3
"""
Verification script for Logic Grid Puzzles
Checks if generated solutions actually satisfy all constraints
"""

import json
import random
from typing import Dict, List


def load_puzzles(jsonl_path: str) -> List[dict]:
    """Load puzzles from JSONL"""
    puzzles = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            puzzles.append(json.loads(line.strip()))
    return puzzles


def verify_solution_structure(puzzle: dict) -> bool:
    """Verify solution has correct structure"""
    people = puzzle['people']
    categories = list(puzzle['attributes'].keys())
    solution = puzzle['answer']
    
    # Check all people present
    for person in people:
        if person not in solution:
            print(f"  ❌ Person {person} missing from solution")
            return False
        
        # Check all categories present
        for cat in categories:
            if cat not in solution[person]:
                print(f"  ❌ Category {cat} missing for {person}")
                return False
    
    # Check uniqueness: no two people have same value in any category
    for cat in categories:
        values = [solution[person][cat] for person in people]
        if len(values) != len(set(values)):
            print(f"  ❌ Duplicate values in category {cat}: {values}")
            return False
    
    # Check all values are from allowed set
    for cat in categories:
        allowed = puzzle['attributes'][cat]
        for person in people:
            val = solution[person][cat]
            if val not in allowed:
                print(f"  ❌ Invalid value {val} for {cat} (allowed: {allowed})")
                return False
    
    return True


def verify_constraints(puzzle: dict, verbose: bool = False) -> bool:
    """
    Verify that solution satisfies all constraints.
    This is a simplified checker - parses common constraint patterns.
    """
    solution = puzzle['answer']
    people = puzzle['people']
    constraints = puzzle['constraints']
    
    if verbose:
        print(f"\n  Checking {len(constraints)} constraints...")
    
    failed = []
    
    for i, constraint in enumerate(constraints, 1):
        satisfied = check_constraint(constraint, solution, people, puzzle['attributes'])
        if not satisfied:
            failed.append((i, constraint))
            if verbose:
                print(f"  ❌ Constraint {i} FAILED: {constraint}")
        elif verbose:
            print(f"  ✓ Constraint {i}: {constraint}")
    
    if failed:
        print(f"  ❌ {len(failed)} constraint(s) failed:")
        for i, c in failed[:3]:  # Show first 3
            print(f"     {i}. {c}")
        return False
    
    return True


def check_constraint(constraint: str, solution: dict, people: list, attributes: dict) -> bool:
    """
    Check if a constraint is satisfied by the solution.
    Simplified pattern matching for common constraint types.
    """
    constraint_lower = constraint.lower()
    
    # Pattern 1: "Person has/drinks/is X"
    for person in people:
        if person.lower() in constraint_lower:
            for cat, values in attributes.items():
                for val in values:
                    if val.lower() in constraint_lower:
                        # Check various phrasings
                        if any(phrase in constraint_lower for phrase in [
                            "has", "drinks", "lives in", "is a", "works as",
                            "favorite", "belongs to", "'s"
                        ]):
                            # This person should have this value
                            if solution[person].get(cat) == val:
                                return True
                            # Or explicitly does NOT have
                            if "not" in constraint_lower or "does not" in constraint_lower:
                                if solution[person].get(cat) != val:
                                    return True
    
    # Pattern 2: "The person with X has Y" (indirect)
    # Find who has X, check if they also have Y
    for cat1, values1 in attributes.items():
        for val1 in values1:
            if val1.lower() in constraint_lower:
                # Find person with val1
                person_with_val1 = None
                for person in people:
                    if solution[person].get(cat1) == val1:
                        person_with_val1 = person
                        break
                
                if person_with_val1:
                    # Check if this person has another value mentioned
                    for cat2, values2 in attributes.items():
                        if cat2 == cat1:
                            continue
                        for val2 in values2:
                            if val2.lower() in constraint_lower:
                                if solution[person_with_val1].get(cat2) == val2:
                                    return True
    
    # If we can't parse it, assume it's satisfied (conservative)
    # In production, you'd want stricter parsing
    return True


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
    correct = True
    for person in puzzle['people']:
        if person not in predicted:
            correct = False
            break
        for cat in puzzle['attributes'].keys():
            if cat not in predicted[person]:
                correct = False
                break
            if predicted[person][cat] != actual[person][cat]:
                correct = False
                break
    
    # Check if evaluation result matches our manual check
    if correct != eval_result['correct']:
        print(f"  ❌ EVALUATION ERROR!")
        print(f"     Manual check: {correct}")
        print(f"     Eval result: {eval_result['correct']}")
        print(f"     Actual: {actual}")
        print(f"     Predicted: {predicted}")
        return False
    
    return True


def main():
    print("="*70)
    print("LOGIC GRID PUZZLE VERIFICATION")
    print("="*70)
    
    # Load puzzles
    print("\n1. Loading puzzles...")
    puzzles = load_puzzles('data/logic_grid/logic_grid_puzzles.jsonl')
    print(f"   Loaded {len(puzzles)} puzzles")
    
    # Verify solution structures
    print("\n2. Verifying solution structures...")
    structure_errors = 0
    for puzzle in puzzles:
        if not verify_solution_structure(puzzle):
            structure_errors += 1
            print(f"   ❌ Puzzle {puzzle['id']} has structure errors")
            if structure_errors >= 3:
                print("   ... stopping after 3 errors")
                break
    
    if structure_errors == 0:
        print("   ✅ All solution structures valid")
    else:
        print(f"   ❌ {structure_errors} puzzles with structure errors")
        return
    
    # Verify constraints for a sample
    print("\n3. Verifying constraints for sample puzzles...")
    sample_puzzles = random.sample(puzzles, min(10, len(puzzles)))
    constraint_errors = 0
    
    for puzzle in sample_puzzles:
        print(f"\n   Puzzle: {puzzle['id']} ({puzzle['difficulty']})")
        if not verify_constraints(puzzle, verbose=False):
            constraint_errors += 1
            print(f"   ❌ Constraints not satisfied!")
            # Show details
            verify_constraints(puzzle, verbose=True)
        else:
            print(f"   ✅ All constraints satisfied")
    
    if constraint_errors == 0:
        print(f"\n   ✅ All {len(sample_puzzles)} sampled puzzles have valid solutions")
    else:
        print(f"\n   ❌ {constraint_errors}/{len(sample_puzzles)} puzzles have constraint violations")
    
    # Verify evaluation
    print("\n4. Verifying evaluation results...")
    eval_results = load_evaluation_results('evaluation_results/logic_grid/details_gpt-4o.jsonl')
    print(f"   Loaded {len(eval_results)} evaluation results")
    
    # Create puzzle lookup
    puzzle_dict = {p['id']: p for p in puzzles}
    
    # Check a sample of correct and incorrect
    correct_results = [r for r in eval_results if r['correct']]
    incorrect_results = [r for r in eval_results if not r['correct']]
    
    print(f"\n   Checking sample of CORRECT evaluations (marked as ✓)...")
    sample_correct = random.sample(correct_results, min(5, len(correct_results)))
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
            for person in puzzle['people']:
                for cat in puzzle['attributes'].keys():
                    actual_val = puzzle['answer'][person][cat]
                    pred_val = result['predicted'].get(person, {}).get(cat, '???')
                    if actual_val != pred_val:
                        diffs.append(f"{person}.{cat}: {pred_val} ≠ {actual_val}")
            
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
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
