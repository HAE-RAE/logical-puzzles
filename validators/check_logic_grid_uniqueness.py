#!/usr/bin/env python3
"""
Measure whether the logic_grid GENERATOR actually produces unique-solution puzzles.

Background: logic_grid_en.py:_verify_unique_solution() does NOT verify uniqueness —
it only returns `len(constraints) >= 0.6 * N*K` (a count heuristic; no CSP solving).
README/docstring claim "Unique Solution Guaranteed / CSP backtracking", but that path
is unimplemented. This script measures the real multi-solution rate.

Method (no NL parsing — reliable):
  - Subclass LogicGridGenerator, override the two constraint builders to return STRUCTURED
    predicates instead of strings:
        direct   -> ('D', person, category, value)
        indirect -> ('I', cat1, val1, cat2, val2)   # the entity with cat1=val1 has cat2=val2
    The generator's accept/reject heuristic depends ONLY on constraint COUNT (not content),
    so structured constraints preserve which puzzles the generator would emit.
  - Run a real backtracking CSP solver (MRV + AllDifferent + indirect propagation) that
    counts solutions up to 2 -> determines uniqueness exactly.

Usage:
    uv run python validators/check_logic_grid_uniqueness.py --n 300 --seed 0
"""
import sys
import argparse
import random
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation.logic_grid_en import LogicGridGenerator, Difficulty  # noqa: E402


# --------------------------------------------------------------------------- #
# Instrumented generator: emit STRUCTURED constraints (faithful sampling).
# Count of constraints is preserved -> the count-only uniqueness heuristic in the
# base generate() behaves identically, so the same puzzles are accepted.
# --------------------------------------------------------------------------- #
class StructuredLogicGridGenerator(LogicGridGenerator):
    def _generate_direct_constraints(self, people, solution, count):
        out, used = [], set()
        attempts = 0
        while len(out) < count and attempts < count * 10:
            attempts += 1
            person = random.choice(people)
            category = random.choice(list(solution[person].keys()))
            value = solution[person][category]
            fact = (person, category, value)
            if fact in used:
                continue
            used.add(fact)
            out.append(('D', person, category, value))
        return out

    def _generate_indirect_constraints(self, people, categories, solution, count):
        out, used = [], set()
        attempts = 0
        while len(out) < count and attempts < count * 10:
            attempts += 1
            person = random.choice(people)
            if len(categories) < 2:
                break
            cat1, cat2 = random.sample(categories, 2)
            val1 = solution[person][cat1]
            val2 = solution[person][cat2]
            link = tuple(sorted([f"{cat1}:{val1}", f"{cat2}:{val2}"]))
            if link in used:
                continue
            used.add(link)
            out.append(('I', cat1, val1, cat2, val2))
        return out


# --------------------------------------------------------------------------- #
# Exact solution counter (backtracking CSP, MRV, stop after `cap` solutions).
#   cells: x[(person,cat)] in attributes[cat]; each category is a bijection.
# --------------------------------------------------------------------------- #
def count_solutions(people, attributes, constraints, cap=2, node_limit=5_000_000):
    cats = list(attributes.keys())
    direct = {(p, c): v for (k, p, c, v) in (x for x in constraints if x[0] == 'D')}
    indirect = [(c1, v1, c2, v2) for (k, c1, v1, c2, v2) in
                (x for x in constraints if x[0] == 'I')]

    cells = [(p, c) for c in cats for p in people]
    assign = {}
    used = {c: set() for c in cats}            # values already used within a category
    nodes = [0]
    sols = [0]
    aborted = [False]

    # contradictory direct facts on the same cell -> 0 solutions immediately
    for (p, c), v in direct.items():
        pass

    def legal_values(p, c):
        if (p, c) in direct:
            v = direct[(p, c)]
            return [v] if v not in used[c] else []
        return [v for v in attributes[c] if v not in used[c]]

    def indirect_consistent():
        # check all indirect constraints against the current PARTIAL assignment
        for (c1, v1, c2, v2) in indirect:
            pstar = None
            for p in people:
                if assign.get((p, c1)) == v1:
                    pstar = p
                    break
            if pstar is not None and (pstar, c2) in assign:
                if assign[(pstar, c2)] != v2:
                    return False
            # if v2 is placed on some q in c2 but that q already has c1 != v1 -> q can't be pstar;
            # and if c1 is fully assigned with nobody holding v1 -> impossible (caught by alldiff at leaf)
        return True

    def select_cell():
        # MRV: unassigned cell with fewest legal values
        best, best_dom = None, None
        for (p, c) in cells:
            if (p, c) in assign:
                continue
            dom = legal_values(p, c)
            if best is None or len(dom) < len(best_dom):
                best, best_dom = (p, c), dom
                if len(dom) <= 1:
                    break
        return best, best_dom

    def backtrack():
        if aborted[0] or sols[0] >= cap:
            return
        nodes[0] += 1
        if nodes[0] > node_limit:
            aborted[0] = True
            return
        if len(assign) == len(cells):
            sols[0] += 1
            return
        cell, dom = select_cell()
        if not dom:
            return
        p, c = cell
        for v in dom:
            assign[(p, c)] = v
            used[c].add(v)
            if indirect_consistent():
                backtrack()
            del assign[(p, c)]
            used[c].discard(v)
            if aborted[0] or sols[0] >= cap:
                return

    backtrack()
    return sols[0], aborted[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="puzzles per difficulty")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    gen = StructuredLogicGridGenerator()

    print(f"logic_grid 생성기 유일해 실측 (per difficulty N={args.n}, seed={args.seed})\n")
    print(f"{'difficulty':9}{'1해(유일)':>10}{'≥2해(비유일)':>13}{'aborted':>9}{'비유일%':>9}")
    for diff in [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]:
        dist = Counter()
        n_abort = 0
        for _ in range(args.n):
            pz = gen.generate(diff)
            s, ab = count_solutions(pz.people, pz.attributes, pz.constraints)
            if ab:
                n_abort += 1
                dist['abort'] += 1
            else:
                dist['unique' if s == 1 else 'multi'] += 1
        uniq = dist['unique']
        multi = dist['multi']
        decided = uniq + multi
        pct = (multi / decided * 100) if decided else float('nan')
        print(f"{diff.name.lower():9}{uniq:>10}{multi:>13}{n_abort:>9}{pct:>8.1f}%")


if __name__ == "__main__":
    main()
