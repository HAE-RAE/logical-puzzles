"""
Tower of Hanoi Rule-Based Problem Generator (v3 - Recalibrated)

Difficulty Levels (calibrated for gemini-3-flash-preview, reasoning OFF):
- easy:   n=3-4 disks, formula + light tracing   → target 75% (65-85%)
- medium: n=5-7 disks, tracing + simulation heavy → target 50% (40-60%)
- hard:   n=8-10 disks, pure simulation / inverse → target 25% (15-35%)

Design rationale:
- Reasoning-off Gemini Flash can reliably apply formulas (min_moves, disk_move_count)
  but struggles with sequential tracing (kth_disk) and fails at state simulation
  (disks_on_peg_after_k, where_is_disk_after_k) as n grows.
- Easy mixes ~40% formula questions (~95% acc) with ~60% tracing (~60% acc) → ~75%
- Medium mixes ~15% formula (~85%) with ~85% tracing+simulation (~44%) → ~50%
- Hard is ~100% simulation/inverse at high n (~25% acc) → ~25%
"""

import random
import json
import hashlib
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class HanoiConfig:
    difficulty: str = "medium"
    seed: Optional[int] = None
    min_disks: int = 5
    max_disks: int = 7

    def __post_init__(self):
        if self.difficulty == "easy":
            self.min_disks, self.max_disks = 3, 4
        elif self.difficulty == "medium":
            self.min_disks, self.max_disks = 5, 7
        elif self.difficulty == "hard":
            self.min_disks, self.max_disks = 8, 10


Move = Tuple[int, int, int]
Context = Dict[str, Any]


def build_hanoi_moves(n: int, src: int, aux: int, dst: int, acc: List[Move]) -> None:
    if n == 0:
        return
    build_hanoi_moves(n - 1, src, dst, aux, acc)
    acc.append((n, src, dst))
    build_hanoi_moves(n - 1, aux, src, dst, acc)


def get_hanoi_moves(n: int, src: int, aux: int, dst: int) -> List[Move]:
    moves: List[Move] = []
    build_hanoi_moves(n, src, aux, dst, moves)
    return moves


def simulate_pegs(n: int, src: int, aux: int, dst: int, moves: List[Move], steps: int) -> Dict[int, List[int]]:
    pegs: Dict[int, List[int]] = {
        src: list(range(n, 0, -1)),
        aux: [],
        dst: [],
    }
    for idx in range(min(steps, len(moves))):
        disk, from_peg, to_peg = moves[idx]
        popped = pegs[from_peg].pop()
        assert popped == disk, f"Internal error: expected disk {disk}, got {popped}"
        pegs[to_peg].append(disk)
    return pegs


def _weighted_choice(rng, templates):
    weights = [t[2] for t in templates]
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0
    for t in templates:
        cumulative += t[2]
        if r <= cumulative:
            return t[0], t[1], t[3], t[4]
    return templates[-1][0], templates[-1][1], templates[-1][3], templates[-1][4]


def _format_peg_state(pegs: Dict[int, List[int]]) -> str:
    parts = []
    for peg in sorted(pegs.keys()):
        disks = pegs[peg]
        if disks:
            parts.append(f"Peg {peg}: [{', '.join(str(d) for d in disks)}]")
        else:
            parts.append(f"Peg {peg}: []")
    return ", ".join(parts)


SFT_SOLUTION_RUBRIC_EN = (
    "STEP0=meta · STEP1=given · STEP2=worked solution · "
    "STEP3=answer and verification"
)

_HANOI_QTYPE_HINT_EN = {
    "min_moves": "apply 2^n-1 minimum-move formula",
    "kth_disk": "build optimal sequence, identify disk at move k",
    "kth_from_to": "build optimal sequence, identify source/target pegs at move k",
    "kth_full_triplet": "build optimal sequence, identify full (disk, from, to) at move k",
    "largest_disk_move": "pinpoint the unique move of the largest disk",
    "disk_move_count": "apply 2^(n-d) move-count formula",
    "disks_on_peg_after_k": "simulate k moves, report peg contents",
    "where_is_disk_after_k": "simulate k moves, locate specific disk",
    "inverse_find_n": "deduce n from a known move",
    "disk_k_total_moves": "identify disk at step k, then count its total moves",
    "first_last_move": "trace first and last occurrence of a disk",
    "count_disks_on_peg_after_k": "simulate k moves, count disks on a peg",
    "smallest_disk_on_peg_after_k": "simulate k moves, find smallest disk on peg",
    "full_state_after_k": "simulate k moves, report complete state of all pegs",
}


def _hanoi_worked_body_lines_en(solution: str) -> Tuple[List[str], str]:
    seg_lines: List[str] = []
    final_answer = ""
    seg_idx = 1
    for raw in solution.rstrip().splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("final answer") or low.startswith("final:"):
            after = line.split(":", 1)
            final_answer = after[1].strip() if len(after) == 2 else line
            continue
        body = line
        if low.startswith("step "):
            parts = line.split(":", 1)
            if len(parts) == 2:
                body = parts[1].strip()
        seg_lines.append(f"    [SEG {seg_idx}] {body}")
        seg_idx += 1
    return seg_lines, final_answer


def _wrap_sft_hanoi_solution_en(
    solution: str,
    *,
    n: Optional[int] = None,
    total_moves: Optional[int] = None,
    qtype: Optional[str] = None,
    answer: Optional[str] = None,
) -> str:
    seg_lines, final_answer = _hanoi_worked_body_lines_en(solution)
    if answer is None:
        answer = final_answer or "(see prompt)"
    hint = _HANOI_QTYPE_HINT_EN.get(qtype or "", "trace the optimal solution")
    meta_bits = []
    if n is not None:
        meta_bits.append(f"n={n}")
    if total_moves is not None:
        meta_bits.append(f"total moves={total_moves}")
    if qtype:
        meta_bits.append(f"qtype={qtype}")
    meta_line = " · ".join(meta_bits) if meta_bits else "standard rules"
    summary = (
        f"  · Summary: {hint} · {meta_line} · {len(seg_lines)} SEGs"
    )
    step2 = "\n".join([summary, *seg_lines]) if seg_lines else summary
    return (
        f"{SFT_SOLUTION_RUBRIC_EN}\n"
        f"[STEP 0] Problem meta\n"
        f"  - Optimal Tower of Hanoi (2^n-1 moves) and standard rules\n"
        f"  - Final answer is confirmed in [STEP 3]\n"
        f"[STEP 1] Given\n"
        f"  - n, peg labels, and k (as in the problem statement)\n"
        f"[STEP 2] Worked solution\n{step2}\n"
        f"[STEP 3] Answer and verification\n"
        f"  - Final answer: {answer}\n"
        f"  - Cross-check 2^ formulas / simulation against the [SEG] trace."
    )

def _build_templates_easy(ctx: Context, rng) -> list:
    n = ctx["n"]
    src, aux, dst = ctx["src"], ctx["aux"], ctx["dst"]
    total = ctx["total_moves"]
    k = ctx["k"]
    disk_k, from_k, to_k = ctx["disk_k"], ctx["from_k"], ctx["to_k"]
    moves = ctx["moves"]
    pegs_after_k = ctx["pegs_after_k"]

    largest = n
    largest_idx = next(idx for idx, (d, _, _) in enumerate(moves) if d == largest)
    l_disk, l_from, l_to = moves[largest_idx]

    disk_target = rng.randint(1, n)
    disk_count = sum(1 for d, _, _ in moves if d == disk_target)

    disk_query = rng.randint(1, n)
    peg_of_disk = None
    for peg, stack in pegs_after_k.items():
        if disk_query in stack:
            peg_of_disk = peg
            break

    return [
        (
            f"In a Tower of Hanoi puzzle with {n} disks, all disks start on Peg {src}.\n"
            f"The goal is to move all disks to Peg {dst} using Peg {aux} as auxiliary,\n"
            f"following the usual rules (move one disk at a time, never place a larger disk on a smaller one).\n"
            f"What is the minimum number of moves needed to complete the puzzle?",
            f"({total}, {total}, {total})",
            2,  
            "min_moves",
            f"Step 1: The minimum moves for n disks = 2^n - 1\n"
            f"Step 2: n = {n}, so 2^{n} - 1 = {total}\n"
            f"Final answer: {total}"
        ),
        (
            f"In the optimal solution for a Tower of Hanoi puzzle with {n} disks,\n"
            f"how many times does Disk {disk_target} move in total?",
            f"({disk_target}, {disk_count}, {disk_count})",
            2,  
            "disk_move_count",
            f"Step 1: In optimal solution, Disk d moves 2^(n-d) times\n"
            f"Step 2: Disk {disk_target} with n={n}: moves = 2^({n}-{disk_target}) = {2**(n - disk_target)}\n"
            f"Step 3: Verify by counting: {disk_count}\n"
            f"Final answer: {disk_count}"
        ),
        (
            f"In the optimal solution of a Tower of Hanoi puzzle with {n} disks,\n"
            f"all disks start on Peg {src} and must reach Peg {dst} (Peg {aux} is auxiliary).\n"
            f"On which move number does the largest disk (Disk {n}) move?",
            f"({l_disk}, {l_from}, {l_to})",
            1, 
            "largest_disk_move",
            f"Step 1: The largest disk (Disk {n}) moves exactly once in the optimal solution\n"
            f"Step 2: It moves on step {largest_idx + 1}: Peg {l_from} → Peg {l_to}\n"
            f"Final answer: Move {largest_idx + 1}"
        ),
        (
            f"Consider the optimal solution of a Tower of Hanoi puzzle with {n} disks.\n"
            f"All disks start on Peg {src} and must be moved to Peg {dst} (Peg {aux} is auxiliary).\n"
            f"In this optimal sequence, which disk is moved on the {k}-th move?",
            f"({disk_k}, {from_k}, {to_k})",
            3, 
            "kth_disk",
            f"Step 1: Generate optimal move sequence for {n} disks: Peg {src} → Peg {dst}\n"
            f"Step 2: Total moves = {total}\n"
            f"Step 3: The {k}-th move is Disk {disk_k} from Peg {from_k} to Peg {to_k}\n"
            f"Final answer: Disk {disk_k}"
        ),
        (
            f"In the optimal {n}-disk Tower of Hanoi solution from Peg {src} to Peg {dst}\n"
            f"(with Peg {aux} as auxiliary), from which peg to which peg does the disk move on the {k}-th move?",
            f"({disk_k}, {from_k}, {to_k})",
            3,  
            "kth_from_to",
            f"Step 1: Generate optimal move sequence for {n} disks\n"
            f"Step 2: The {k}-th move: Disk {disk_k}, Peg {from_k} → Peg {to_k}\n"
            f"Final answer: Peg {from_k} → Peg {to_k}"
        ),
        (
            f"In an optimal Tower of Hanoi solution with {n} disks, all disks start on Peg {src}\n"
            f"and must be moved to Peg {dst}, using Peg {aux} as auxiliary.\n"
            f"After exactly {k} moves, on which peg is Disk {disk_query} located?",
            f"({disk_query}, {peg_of_disk}, {peg_of_disk})",
            1, 
            "where_is_disk_after_k",
            f"Step 1: Generate optimal sequence for {n} disks\n"
            f"Step 2: Simulate {k} moves from initial state\n"
            f"Step 3: State after {k} moves: {_format_peg_state(pegs_after_k)}\n"
            f"Step 4: Disk {disk_query} is on Peg {peg_of_disk}\n"
            f"Final answer: Peg {peg_of_disk}"
        ),
    ]


def _build_templates_medium(ctx: Context, rng) -> list:
    n = ctx["n"]
    src, aux, dst = ctx["src"], ctx["aux"], ctx["dst"]
    total = ctx["total_moves"]
    k = ctx["k"]
    disk_k, from_k, to_k = ctx["disk_k"], ctx["from_k"], ctx["to_k"]
    moves = ctx["moves"]
    pegs_after_k = ctx["pegs_after_k"]

    peg_target = rng.choice([src, aux, dst])
    disks_on_peg = sorted(pegs_after_k[peg_target])

    disk_query = rng.randint(1, n)
    peg_of_disk = None
    for peg, stack in pegs_after_k.items():
        if disk_query in stack:
            peg_of_disk = peg
            break

    disk_count_k = sum(1 for d, _, _ in moves if d == disk_k)

    disk_target = rng.randint(1, n)
    disk_count_target = sum(1 for d, _, _ in moves if d == disk_target)

    largest = n
    largest_idx = next(idx for idx, (d, _, _) in enumerate(moves) if d == largest)
    l_disk, l_from, l_to = moves[largest_idx]

    return [
        (
            f"In the optimal solution for a Tower of Hanoi puzzle with {n} disks,\n"
            f"how many times does Disk {disk_target} move in total?",
            f"({disk_target}, {disk_count_target}, {disk_count_target})",
            3,  
            "disk_move_count",
            f"Step 1: In optimal Hanoi with {n} disks, Disk d moves 2^(n-d) times\n"
            f"Step 2: Disk {disk_target}: 2^({n}-{disk_target}) = {2**(n - disk_target)}\n"
            f"Final answer: {2**(n - disk_target)}"
        ),
        (
            f"In the optimal solution of a Tower of Hanoi puzzle with {n} disks,\n"
            f"all disks start on Peg {src} and must reach Peg {dst} (Peg {aux} is auxiliary).\n"
            f"On which move number does the largest disk (Disk {n}) move?",
            f"({l_disk}, {l_from}, {l_to})",
            1,  
            "largest_disk_move",
            f"Step 1: The largest disk (Disk {n}) moves exactly once\n"
            f"Step 2: It moves on step {largest_idx + 1}: Peg {l_from} → Peg {l_to}\n"
            f"Final answer: Move {largest_idx + 1}"
        ),
        (
            f"In an optimal Tower of Hanoi puzzle with {n} disks, all disks start on Peg {src}\n"
            f"and must be moved to Peg {dst} using Peg {aux} as auxiliary.\n"
            f"Describe the {k}-th move in the form (disk, from_peg, to_peg).",
            f"({disk_k}, {from_k}, {to_k})",
            2,  
            "kth_full_triplet",
            f"Step 1: Generate optimal sequence for {n} disks: Peg {src} → Peg {dst}, auxiliary Peg {aux}\n"
            f"Step 2: Total moves = 2^{n} - 1 = {total}\n"
            f"Step 3: The {k}-th move is (Disk {disk_k}, Peg {from_k}, Peg {to_k})\n"
            f"Final answer: ({disk_k}, {from_k}, {to_k})"
        ),
       
        (
            f"In an optimal Tower of Hanoi solution with {n} disks, all disks start on Peg {src}\n"
            f"and must be moved to Peg {dst}, using Peg {aux} as auxiliary.\n"
            f"After exactly {k} moves, on which peg is Disk {disk_query} located?",
            f"({disk_query}, {peg_of_disk}, {peg_of_disk})",
            2, 
            "where_is_disk_after_k",
            f"Step 1: Generate optimal sequence for {n} disks\n"
            f"Step 2: Simulate {k} moves from initial state\n"
            f"Step 3: State after {k} moves: {_format_peg_state(pegs_after_k)}\n"
            f"Step 4: Disk {disk_query} is on Peg {peg_of_disk}\n"
            f"Final answer: Peg {peg_of_disk}"
        ),
        (
            f"In a Tower of Hanoi puzzle with {n} disks (Peg {src} → Peg {dst}, Peg {aux} auxiliary),\n"
            f"consider the optimal sequence of moves. After exactly {k} moves have been performed,\n"
            f"which disks are on Peg {peg_target}? List all disk numbers in ascending order.",
            f"({', '.join(str(d) for d in disks_on_peg) if disks_on_peg else 'none'}, {peg_target}, {peg_target})",
            2,  
            "disks_on_peg_after_k",
            f"Step 1: Generate optimal sequence for {n} disks\n"
            f"Step 2: Simulate {k} moves\n"
            f"Step 3: State after {k} moves: {_format_peg_state(pegs_after_k)}\n"
            f"Step 4: Peg {peg_target} has: {disks_on_peg if disks_on_peg else 'no disks'}\n"
            f"Final answer: {disks_on_peg if disks_on_peg else 'none'}"
        ),
        (
            f"In an optimal Tower of Hanoi solution with {n} disks (Peg {src} → Peg {dst}, Peg {aux} auxiliary),\n"
            f"look at the {k}-th move of the sequence.\n"
            f"How many times does the disk moved at step {k} move in the entire optimal solution?",
            f"({disk_k}, {disk_count_k}, {disk_count_k})",
            2,  
            "disk_k_total_moves",
            f"Step 1: The {k}-th move involves Disk {disk_k}\n"
            f"Step 2: Count all occurrences of Disk {disk_k} in the full sequence\n"
            f"Step 3: Disk {disk_k} moves {disk_count_k} times total\n"
            f"Final answer: {disk_count_k}"
        ),
    ]

def _build_templates_hard(ctx: Context, rng) -> list:
    n = ctx["n"]
    src, aux, dst = ctx["src"], ctx["aux"], ctx["dst"]
    total = ctx["total_moves"]
    k = ctx["k"]
    disk_k, from_k, to_k = ctx["disk_k"], ctx["from_k"], ctx["to_k"]
    moves = ctx["moves"]
    pegs_after_k = ctx["pegs_after_k"]

    k2 = rng.randint(1, total)
    disk_k2, from_k2, to_k2 = moves[k2 - 1]
    pegs_after_k2 = simulate_pegs(n, src, aux, dst, moves, k2)

    peg_target = rng.choice([src, aux, dst])
    disks_on_peg = sorted(pegs_after_k2[peg_target])

    disk_query = rng.randint(1, n)
    peg_of_disk = None
    for peg, stack in pegs_after_k2.items():
        if disk_query in stack:
            peg_of_disk = peg
            break

    disk_count_k = sum(1 for d, _, _ in moves if d == disk_k)

    disk_target = rng.randint(1, n)
    disk_count_target = sum(1 for d, _, _ in moves if d == disk_target)

    first_move_of_disk = {}
    last_move_of_disk = {}
    for idx, (d, f, t) in enumerate(moves):
        if d not in first_move_of_disk:
            first_move_of_disk[d] = (idx + 1, f, t)
        last_move_of_disk[d] = (idx + 1, f, t)

    target_disk_fl = rng.randint(1, n)
    first_info = first_move_of_disk[target_disk_fl]
    last_info = last_move_of_disk[target_disk_fl]

    peg_count_target = rng.choice([src, aux, dst])
    count_on_peg = len(pegs_after_k2[peg_count_target])

    return [
        (
            f"In the optimal solution for a Tower of Hanoi puzzle with {n} disks,\n"
            f"how many times does Disk {disk_target} move in total?",
            f"({disk_target}, {disk_count_target}, {disk_count_target})",
            3,  
            "disk_move_count",
            f"Step 1: In optimal Hanoi with {n} disks, Disk d moves 2^(n-d) times\n"
            f"Step 2: Disk {disk_target}: 2^({n}-{disk_target}) = {2**(n - disk_target)}\n"
            f"Final answer: {2**(n - disk_target)}"
        ),
        (
            f"In a certain optimal Tower of Hanoi puzzle, all disks start on Peg {src}\n"
            f"and the goal is to move them to Peg {dst} using Peg {aux} as auxiliary.\n"
            f"It is known that on move {k}, Disk {disk_k} moves from Peg {from_k} to Peg {to_k}.\n"
            f"How many disks are in this Tower of Hanoi puzzle?",
            f"({n}, {n}, {n})",
            2, 
            "inverse_find_n",
            f"Step 1: We know move {k} is Disk {disk_k}: Peg {from_k} → Peg {to_k}\n"
            f"Step 2: The largest disk number seen is {disk_k}, so n >= {disk_k}\n"
            f"Step 3: Total moves = 2^n - 1 >= {k}, so n >= ceil(log2({k}+1))\n"
            f"Step 4: The puzzle has {n} disks (verified: move {k} matches)\n"
            f"Final answer: {n}"
        ),
        (
            f"In an optimal Tower of Hanoi solution with {n} disks (Peg {src} → Peg {dst}, Peg {aux} auxiliary),\n"
            f"after exactly {k2} moves, which disks are on Peg {peg_target}?\n"
            f"List all disk numbers in ascending order.",
            f"({', '.join(str(d) for d in disks_on_peg) if disks_on_peg else 'none'}, {peg_target}, {peg_target})",
            2,  
            "disks_on_peg_after_k",
            f"Step 1: Generate optimal sequence for {n} disks: Peg {src} → Peg {dst}\n"
            f"Step 2: Simulate {k2} moves step by step\n"
            f"Step 3: State after {k2} moves: {_format_peg_state(pegs_after_k2)}\n"
            f"Step 4: Peg {peg_target}: {disks_on_peg if disks_on_peg else 'empty'}\n"
            f"Final answer: {disks_on_peg if disks_on_peg else 'none'}"
        ),
        (
            f"In an optimal Tower of Hanoi solution with {n} disks (Peg {src} → Peg {dst}, Peg {aux} auxiliary),\n"
            f"after exactly {k2} moves, on which peg is Disk {disk_query} located?",
            f"({disk_query}, {peg_of_disk}, {peg_of_disk})",
            3, 
            "where_is_disk_after_k",
            f"Step 1: Generate optimal sequence for {n} disks\n"
            f"Step 2: Simulate {k2} moves from initial state\n"
            f"Step 3: State after {k2} moves: {_format_peg_state(pegs_after_k2)}\n"
            f"Step 4: Disk {disk_query} is on Peg {peg_of_disk}\n"
            f"Final answer: Peg {peg_of_disk}"
        ),
        (
            f"In an optimal Tower of Hanoi puzzle with {n} disks (Peg {src} → Peg {dst}, Peg {aux} auxiliary),\n"
            f"describe the {k2}-th move in the form (disk, from_peg, to_peg).",
            f"({disk_k2}, {from_k2}, {to_k2})",
            2, 
            "kth_full_triplet",
            f"Step 1: Generate optimal sequence for {n} disks: Peg {src} → Peg {dst}\n"
            f"Step 2: Total moves = 2^{n} - 1 = {total}\n"
            f"Step 3: The {k2}-th move is (Disk {disk_k2}, Peg {from_k2}, Peg {to_k2})\n"
            f"Final answer: ({disk_k2}, {from_k2}, {to_k2})"
        ),
        (
            f"In an optimal Tower of Hanoi solution with {n} disks (Peg {src} → Peg {dst}, Peg {aux} auxiliary),\n"
            f"on which move number does Disk {target_disk_fl} first move, and on which move number does it last move?",
            f"({first_info[0]}, {last_info[0]}, {target_disk_fl})",
            2,  
            "first_last_move",
            f"Step 1: Trace Disk {target_disk_fl} through the entire sequence\n"
            f"Step 2: First move of Disk {target_disk_fl}: step {first_info[0]} (Peg {first_info[1]} → Peg {first_info[2]})\n"
            f"Step 3: Last move of Disk {target_disk_fl}: step {last_info[0]} (Peg {last_info[1]} → Peg {last_info[2]})\n"
            f"Final answer: first = {first_info[0]}, last = {last_info[0]}"
        ),
        (
            f"In an optimal Tower of Hanoi solution with {n} disks (Peg {src} → Peg {dst}, Peg {aux} auxiliary),\n"
            f"the {k}-th move involves a certain disk.\n"
            f"How many times does that disk move in the entire optimal solution?",
            f"({disk_k}, {disk_count_k}, {disk_count_k})",
            2, 
            "disk_k_total_moves",
            f"Step 1: The {k}-th move involves Disk {disk_k}\n"
            f"Step 2: In optimal {n}-disk Hanoi, Disk {disk_k} moves 2^({n}-{disk_k}) = {2**(n-disk_k)} times\n"
            f"Step 3: Verified by counting: {disk_count_k}\n"
            f"Final answer: {disk_count_k}"
        ),
    ]


DIFFICULTY_TEMPLATE_BUILDERS = {
    "easy": _build_templates_easy,
    "medium": _build_templates_medium,
    "hard": _build_templates_hard,
}


def generate_puzzle(difficulty: str = "medium", seed: Optional[int] = None) -> Dict[str, Any]:
    if seed is None:
        seed = random.randint(1, 1000000)

    rng = random.Random(seed)
    config = HanoiConfig(difficulty=difficulty, seed=seed)
    n = rng.randint(config.min_disks, config.max_disks)
    src, aux, dst = rng.sample([0, 1, 2], 3)

    moves = get_hanoi_moves(n, src, aux, dst)
    total_moves = len(moves)

    k = rng.randint(1, total_moves)
    disk_k, from_k, to_k = moves[k - 1]
    pegs_after_k = simulate_pegs(n, src, aux, dst, moves, k)

    ctx: Context = {
        "n": n,
        "src": src,
        "aux": aux,
        "dst": dst,
        "moves": moves,
        "total_moves": total_moves,
        "k": k,
        "disk_k": disk_k,
        "from_k": from_k,
        "to_k": to_k,
        "pegs_after_k": pegs_after_k,
    }

    builder = DIFFICULTY_TEMPLATE_BUILDERS[difficulty]
    templates = builder(ctx, rng)
    question, answer, qtype, solution = _weighted_choice(rng, templates)

    puzzle_hash = hashlib.md5(f"{seed}_{difficulty}_{qtype}".encode()).hexdigest()[:8]

    return {
        "question": question,
        "answer": answer,
        "solution": _wrap_sft_hanoi_solution_en(
            solution, n=n, total_moves=total_moves, qtype=qtype, answer=answer
        ),
        "difficulty": difficulty,
        "type": qtype,
        "n": n,
        "src": src,
        "aux": aux,
        "dst": dst,
        "seed": seed,
        "id": f"hanoi_en_{difficulty}_{qtype}_{puzzle_hash}",
    }


def generate_dataset(num_per_difficulty: int = 100, seed: int = 2025) -> List[Dict[str, Any]]:
    puzzles = []
    difficulties = ["easy", "medium", "hard"]

    puzzle_seed = seed
    for difficulty in difficulties:
        for diff_idx in range(num_per_difficulty):
            puzzle = generate_puzzle(difficulty=difficulty, seed=puzzle_seed)
            puzzle["id"] = f"hanoi_en_{difficulty}_{diff_idx:04d}"
            puzzles.append(puzzle)
            puzzle_seed += 1

    return puzzles


def save_dataset(puzzles: List[Dict], base_dir: str = "./data"):
    base_path = Path(base_dir)
    csv_dir = base_path / "csv"
    json_dir = base_path / "jsonl"

    csv_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    csv_path = csv_dir / "hanoi_en.csv"
    jsonl_paths = {}
    for diff in ["easy", "medium", "hard"]:
        p = json_dir / f"hanoi_en_{diff}.jsonl"
        subset = [pz for pz in puzzles if pz["difficulty"] == diff]
        with open(p, "w", encoding="utf-8") as f:
            for puzzle in subset:
                row = {
                    "id": puzzle["id"],
                    "question": puzzle["question"],
                    "answer": puzzle["answer"],
                    "solution": puzzle["solution"],
                    "difficulty": puzzle["difficulty"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved {len(subset)} puzzles to {p}")
        jsonl_paths[diff] = p

    csv_columns = ["id", "question", "answer", "solution", "difficulty", "type", "n"]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for puzzle in puzzles:
            writer.writerow({
                "id": puzzle["id"],
                "question": puzzle["question"],
                "answer": puzzle["answer"],
                "solution": puzzle["solution"],
                "difficulty": puzzle["difficulty"],
                "type": puzzle["type"],
                "n": puzzle["n"],
            })

    print(f"Saved {len(puzzles)} puzzles to {csv_path}")

    stats = {}
    n_stats = {}
    for puzzle in puzzles:
        key = f"{puzzle['difficulty']}_{puzzle['type']}"
        stats[key] = stats.get(key, 0) + 1
        nkey = f"{puzzle['difficulty']}_n={puzzle['n']}"
        n_stats[nkey] = n_stats.get(nkey, 0) + 1

    print("\n=== Dataset Statistics ===")
    print("\nBy difficulty + question type:")
    for key, count in sorted(stats.items()):
        print(f"  {key}: {count}")

    print("\nBy difficulty + disk count:")
    for key, count in sorted(n_stats.items()):
        print(f"  {key}: {count}")

    return csv_path, jsonl_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hanoi Puzzle Generator v3")
    parser.add_argument("--num", type=int, default=100, help="Number of puzzles per difficulty level")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--output", type=str, default="./data", help="Output base directory")
    parser.add_argument("--demo", action="store_true", help="Print demo puzzles")

    args = parser.parse_args()

    if args.demo:
        print("=" * 60)
        print("Hanoi Puzzle Demo (v3 - Recalibrated)")
        print("=" * 60)
        for difficulty in ["easy", "medium", "hard"]:
            puzzle = generate_puzzle(difficulty=difficulty, seed=42)
            print(f"\n[{difficulty} | n={puzzle['n']} | type={puzzle['type']}]")
            print("-" * 40)
            print(puzzle["question"])
            print(f"\nAnswer: {puzzle['answer']}")
            print("=" * 60)
    else:
        puzzles = generate_dataset(num_per_difficulty=args.num, seed=args.seed)
        save_dataset(puzzles, args.output)