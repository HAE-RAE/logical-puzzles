"""
Tower of Hanoi Rule-Based Problem Generator (v16 - Score Recalibrated)

Difficulty Levels (calibrated from gemini-3-flash-preview evals, reasoning=medium):
- easy:   n=5-6 disks, formula + inverse tracing        → target 75% (65-85%)
- medium: n=6-8 disks, state/inverse tracing            → target 50% (40-60%)
- hard:   n=12-15 disks, count-per-peg after continuation → target 25% (15-35%)

Calibration notes:
- The 2026-04-30 run scored easy=72%, medium=70%, hard=63%.
- Easy is already in range, so it is intentionally stable.
- The first v4 recalibration overshot: medium=91%, hard=97%, because simple
  3-tuple state queries were easy for Gemini.
- The 2026-04-30 22:58 run scored easy=41%, medium=83%, hard=93%.
- v11 kept easy/medium stable; hard move-window sums still scored 96%.
- v12 move-window counts still scored ~96%; v13 used where_is heavily but still
  scored ~97%. v14 removes easy checksum/where_is from hard, raises n to 12–15,
  and uses three_disk_pegs_after_k (three peg labels must match at once).
- v14 still scored 88%: the analytical formula disk_d_peg = f(k, d) lets the
  model compute each disk's peg in O(1) without simulation. Additionally, 42%
  of questions had at least one trivially-positioned disk (front-1/3 k means
  large disks haven't moved; back-1/3 k means they're at dst).
- v15 uses intermediate-state continuation: present the explicit peg layout at a
  hidden step k_hide, then ask "after j more optimal moves, where is disk d?"
  Scored 81%: model reverse-engineers k from the board state (the board→k
  bijection is algorithmically recoverable), then applies the formula.
  Implied per-disk accuracy: 0.75^(1/3) ≈ 91%.
- v16 keeps the hidden-k board presentation but asks for COUNT of disks on each
  peg instead of 3 individual disk positions.  Correctly answering requires all n
  disk assignments to be right: P ≈ 0.91^n ≈ 0.24–0.32 for n=12–15.
  inverse_find_n is removed (it added 22% easy questions at ~99% accuracy that
  inflated the overall score).
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
            self.min_disks, self.max_disks = 5, 6
        elif self.difficulty == "medium":
            self.min_disks, self.max_disks = 6, 8
        elif self.difficulty == "hard":
            self.min_disks, self.max_disks = 12, 15


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


def _peg_of_disk(pegs: Dict[int, List[int]], disk: int) -> int:
    for peg, stack in pegs.items():
        if disk in stack:
            return peg
    raise ValueError(f"Disk {disk} not found in peg state")


def _top_disks_by_peg(pegs: Dict[int, List[int]]) -> Tuple[int, int, int]:
    # Disk labels start at 1, so 0 is an unambiguous empty-peg sentinel.
    return tuple(pegs[peg][-1] if pegs[peg] else 0 for peg in [0, 1, 2])


def _peg_sums_by_peg(pegs: Dict[int, List[int]]) -> Tuple[int, int, int]:
    return tuple(sum(pegs[peg]) for peg in [0, 1, 2])


def _peg_weighted_checksums(pegs: Dict[int, List[int]]) -> Tuple[int, int, int]:
    return tuple(
        sum((idx + 1) * disk for idx, disk in enumerate(pegs[peg]))
        for peg in [0, 1, 2]
    )


def _peg_square_checksums(pegs: Dict[int, List[int]]) -> Tuple[int, int, int]:
    return tuple(
        sum((idx + 1) * disk * disk for idx, disk in enumerate(pegs[peg]))
        for peg in [0, 1, 2]
    )


def _peg_profile(pegs: Dict[int, List[int]], peg: int) -> Tuple[int, int, int]:
    stack = pegs[peg]
    return (
        len(stack),
        sum(stack),
        sum((idx + 1) * disk for idx, disk in enumerate(stack)),
    )


def _abs_tuple_delta(left: Tuple[int, int, int], right: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return tuple(abs(a - b) for a, b in zip(left, right))


def _disk_at_depth_by_peg(pegs: Dict[int, List[int]], depth_from_top: int) -> Tuple[int, int, int]:
    values = []
    for peg in [0, 1, 2]:
        stack = pegs[peg]
        values.append(stack[-depth_from_top] if len(stack) >= depth_from_top else 0)
    return tuple(values)


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
    "three_disk_locations_after_k": "simulate k moves, locate three queried disks",
    "top_disks_after_k": "simulate k moves, report top disk on each peg",
    "next_move_after_k": "simulate to k, then identify the next optimal move",
    "peg_sums_after_k": "simulate k moves, compute disk-label sums on all pegs",
    "peg_profile_after_k": "simulate k moves, compute count/sum/checksum for one peg",
    "peg_weighted_checksums_after_k": "simulate k moves, compute position-weighted checksums",
    "peg_square_checksums_after_k": "simulate k moves, compute square-weighted checksums",
    "two_state_sum_delta": "simulate two states, compare peg sums",
    "two_state_checksum_delta": "simulate two states, compare weighted checksums",
    "three_time_disk_location": "simulate three different times, locate one disk each time",
    "disk_at_depth_after_k": "simulate k moves, report the disk at a fixed stack depth on each peg",
    "move_window_disk123_counts": "count how often disks 1–3 move inside a long interval (legacy; hard no longer uses)",
    "three_disk_pegs_after_k": "simulate k moves, locate three distinct disks on pegs at the same time",
    "intermediate_state_continuation": "given an explicit mid-solution board state, simulate j more moves and locate disks",
    "count_per_peg_after_continuation": "given an explicit mid-solution board state, simulate j more moves and count disks per peg",
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
            f"In a certain optimal Tower of Hanoi puzzle, all disks start on Peg {src}\n"
            f"and the goal is to move them to Peg {dst} using Peg {aux} as auxiliary.\n"
            f"It is known that on move {k}, Disk {disk_k} moves from Peg {from_k} to Peg {to_k}.\n"
            f"How many disks are in this Tower of Hanoi puzzle?",
            f"({n}, {n}, {n})",
            16,
            "inverse_find_n",
            f"Step 1: We know move {k} is Disk {disk_k}: Peg {from_k} → Peg {to_k}\n"
            f"Step 2: Total moves = 2^n - 1 and this move pattern is verified by n={n}\n"
            f"Final answer: {n}"
        ),
        (
            f"In a Tower of Hanoi puzzle with {n} disks, all disks start on Peg {src}.\n"
            f"The goal is to move all disks to Peg {dst} using Peg {aux} as auxiliary,\n"
            f"following the usual rules (move one disk at a time, never place a larger disk on a smaller one).\n"
            f"What is the minimum number of moves needed to complete the puzzle?\n"
            f"Answer as (moves, moves, moves).",
            f"({total}, {total}, {total})",
            1,
            "min_moves",
            f"Step 1: The minimum moves for n disks = 2^n - 1\n"
            f"Step 2: n = {n}, so 2^{n} - 1 = {total}\n"
            f"Final answer: {total}"
        ),
        (
            f"In the optimal solution for a Tower of Hanoi puzzle with {n} disks,\n"
            f"how many times does Disk {disk_target} move in total?\n"
            f"Answer as (disk, count, count).",
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
            f"Describe the move made by the largest disk (Disk {n}).\n"
            f"Answer as (disk, from_peg, to_peg).",
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
            f"In this optimal sequence, which disk is moved on the {k}-th move?\n"
            f"Answer as (disk, from_peg, to_peg).",
            f"({disk_k}, {from_k}, {to_k})",
            2,
            "kth_disk",
            f"Step 1: Generate optimal move sequence for {n} disks: Peg {src} → Peg {dst}\n"
            f"Step 2: Total moves = {total}\n"
            f"Step 3: The {k}-th move is Disk {disk_k} from Peg {from_k} to Peg {to_k}\n"
            f"Final answer: Disk {disk_k}"
        ),
        (
            f"In the optimal {n}-disk Tower of Hanoi solution from Peg {src} to Peg {dst}\n"
            f"(with Peg {aux} as auxiliary), from which peg to which peg does the disk move on the {k}-th move?\n"
            f"Answer as (disk, from_peg, to_peg).",
            f"({disk_k}, {from_k}, {to_k})",
            2,
            "kth_from_to",
            f"Step 1: Generate optimal move sequence for {n} disks\n"
            f"Step 2: The {k}-th move: Disk {disk_k}, Peg {from_k} → Peg {to_k}\n"
            f"Final answer: Peg {from_k} → Peg {to_k}"
        ),
        (
            f"In an optimal Tower of Hanoi solution with {n} disks, all disks start on Peg {src}\n"
            f"and must be moved to Peg {dst}, using Peg {aux} as auxiliary.\n"
            f"After exactly {k} moves, on which peg is Disk {disk_query} located?\n"
            f"Answer as (disk, peg, peg).",
            f"({disk_query}, {peg_of_disk}, {peg_of_disk})",
            2,
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
    peg_profile = _peg_profile(pegs_after_k, peg_target)

    disk_query = rng.randint(1, n)
    peg_of_disk = _peg_of_disk(pegs_after_k, disk_query)

    disk_count_k = sum(1 for d, _, _ in moves if d == disk_k)

    peg_sums = _peg_sums_by_peg(pegs_after_k)
    weighted = _peg_weighted_checksums(pegs_after_k)

    k_a, k_b = sorted(rng.sample(range(1, total + 1), 2))
    pegs_after_a = simulate_pegs(n, src, aux, dst, moves, k_a)
    pegs_after_b = simulate_pegs(n, src, aux, dst, moves, k_b)
    sum_delta = _abs_tuple_delta(
        _peg_sums_by_peg(pegs_after_a),
        _peg_sums_by_peg(pegs_after_b),
    )

    return [
        (
            f"In a certain optimal Tower of Hanoi puzzle, all disks start on Peg {src}\n"
            f"and the goal is to move them to Peg {dst} using Peg {aux} as auxiliary.\n"
            f"It is known that on move {k}, Disk {disk_k} moves from Peg {from_k} to Peg {to_k}.\n"
            f"How many disks are in this Tower of Hanoi puzzle?",
            f"({n}, {n}, {n})",
            10,
            "inverse_find_n",
            f"Step 1: We know move {k} is Disk {disk_k}: Peg {from_k} → Peg {to_k}\n"
            f"Step 2: The puzzle has {n} disks (verified: move {k} matches)\n"
            f"Final answer: {n}"
        ),
        (
            f"In an optimal Tower of Hanoi solution with {n} disks, all disks start on Peg {src}\n"
            f"and must be moved to Peg {dst}, using Peg {aux} as auxiliary.\n"
            f"After exactly {k} moves, on which peg is Disk {disk_query} located?",
            f"({disk_query}, {peg_of_disk}, {peg_of_disk})",
            10,
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
            f"compute the sum of disk numbers on Peg 0, Peg 1, and Peg 2 respectively.\n"
            f"Answer as (sum_on_0, sum_on_1, sum_on_2).",
            f"({peg_sums[0]}, {peg_sums[1]}, {peg_sums[2]})",
            0,
            "peg_sums_after_k",
            f"Step 1: Generate optimal sequence for {n} disks\n"
            f"Step 2: Simulate {k} moves\n"
            f"Step 3: State after {k} moves: {_format_peg_state(pegs_after_k)}\n"
            f"Step 4: Peg sums are {peg_sums}\n"
            f"Final answer: ({peg_sums[0]}, {peg_sums[1]}, {peg_sums[2]})"
        ),
        (
            f"In an optimal Tower of Hanoi solution with {n} disks (Peg {src} → Peg {dst}, Peg {aux} auxiliary),\n"
            f"after exactly {k} moves, report Peg {peg_target}'s profile as (count, sum, weighted_sum).\n"
            f"The weighted_sum is computed from bottom to top as 1*bottom_disk + 2*next_disk + ...",
            f"({peg_profile[0]}, {peg_profile[1]}, {peg_profile[2]})",
            0,
            "peg_profile_after_k",
            f"Step 1: Generate optimal sequence for {n} disks\n"
            f"Step 2: Simulate {k} moves\n"
            f"Step 3: State after {k} moves: {_format_peg_state(pegs_after_k)}\n"
            f"Step 4: Peg {peg_target} profile is {peg_profile}\n"
            f"Final answer: ({peg_profile[0]}, {peg_profile[1]}, {peg_profile[2]})"
        ),
        (
            f"In an optimal Tower of Hanoi solution with {n} disks (Peg {src} → Peg {dst}, Peg {aux} auxiliary),\n"
            f"after exactly {k} moves, compute each peg's position-weighted checksum.\n"
            f"For each peg, read disks bottom to top and compute 1*bottom + 2*next + ...\n"
            f"Answer as (checksum_peg_0, checksum_peg_1, checksum_peg_2).",
            f"({weighted[0]}, {weighted[1]}, {weighted[2]})",
            0,
            "peg_weighted_checksums_after_k",
            f"Step 1: Generate optimal sequence for {n} disks\n"
            f"Step 2: Simulate {k} moves\n"
            f"Step 3: State after {k} moves: {_format_peg_state(pegs_after_k)}\n"
            f"Step 4: Weighted checksums are {weighted}\n"
            f"Final answer: ({weighted[0]}, {weighted[1]}, {weighted[2]})"
        ),
        (
            f"In an optimal Tower of Hanoi solution with {n} disks (Peg {src} → Peg {dst}, Peg {aux} auxiliary),\n"
            f"compare the states after {k_a} moves and after {k_b} moves.\n"
            f"For Peg 0, Peg 1, and Peg 2 respectively, compute the absolute change in the sum of disk numbers.\n"
            f"Answer as (delta_sum_0, delta_sum_1, delta_sum_2).",
            f"({sum_delta[0]}, {sum_delta[1]}, {sum_delta[2]})",
            0,
            "two_state_sum_delta",
            f"Step 1: Simulate {k_a} moves: {_format_peg_state(pegs_after_a)}\n"
            f"Step 2: Simulate {k_b} moves: {_format_peg_state(pegs_after_b)}\n"
            f"Step 3: Absolute peg-sum deltas are {sum_delta}\n"
            f"Final answer: ({sum_delta[0]}, {sum_delta[1]}, {sum_delta[2]})"
        ),
        (
            f"In an optimal Tower of Hanoi solution with {n} disks (Peg {src} → Peg {dst}, Peg {aux} auxiliary),\n"
            f"look at the {k}-th move of the sequence.\n"
            f"How many times does the disk moved at step {k} move in the entire optimal solution?",
            f"({disk_k}, {disk_count_k}, {disk_count_k})",
            0,
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
    moves = ctx["moves"]

    # --- count-per-peg after continuation (v16) ---
    # Present an explicit board state at a hidden step k_hide (25%-75% range)
    # and ask: after j_more more moves, how many disks are on each peg?
    # Correctly answering requires knowing the peg of EVERY disk (not just 3),
    # so P(correct) ≈ p_per_disk^n ≈ 0.91^(12..15) ≈ 0.24–0.32.
    k_hide_lo = max(1, total // 4)
    k_hide_hi = max(k_hide_lo + 1, 3 * total // 4)
    k_hide = rng.randint(k_hide_lo, k_hide_hi)

    j_lo = max(1, total // 20)
    j_hi = max(j_lo + 1, total // 6)
    j_more = rng.randint(j_lo, j_hi)
    k_final = min(k_hide + j_more, total)

    state_hide = simulate_pegs(n, src, aux, dst, moves, k_hide)
    state_final = simulate_pegs(n, src, aux, dst, moves, k_final)

    counts_final = tuple(len(state_final[p]) for p in [0, 1, 2])

    def _fmt_board(state: Dict[int, List[int]]) -> str:
        parts = []
        for peg in sorted(state.keys()):
            disks = state[peg]
            label = ", ".join(str(d) for d in disks) if disks else "empty"
            parts.append(f"  Peg {peg}: [{label}]  (bottom → top)")
        return "\n".join(parts)

    board_desc = _fmt_board(state_hide)

    q_count = (
        f"A Tower of Hanoi puzzle has {n} disks (labeled 1 = smallest … {n} = largest).\n"
        f"The goal is to move all disks from Peg {src} to Peg {dst} optimally,\n"
        f"using Peg {aux} as the auxiliary peg.\n\n"
        f"At some point during the optimal solution the puzzle looks like this:\n"
        f"{board_desc}\n\n"
        f"From this configuration, {j_more} more moves of the optimal solution are performed.\n"
        f"After those {j_more} additional moves,\n"
        f"how many disks are on Peg 0, Peg 1, and Peg 2 respectively?\n"
        f"Answer as (count_on_peg_0, count_on_peg_1, count_on_peg_2)."
    )
    a_count = f"({counts_final[0]}, {counts_final[1]}, {counts_final[2]})"
    sol_count = (
        f"Step 1: Read the given board state (step k_hide hidden)\n"
        f"Step 2: Reverse-engineer k_hide from the board structure → k_hide = {k_hide}\n"
        f"Step 3: k_final = {k_hide} + {j_more} = {k_final}\n"
        f"Step 4: State after {k_final} total moves: {_format_peg_state(state_final)}\n"
        f"Step 5: Count per peg → Peg 0: {counts_final[0]}, "
        f"Peg 1: {counts_final[1]}, Peg 2: {counts_final[2]}\n"
        f"Final answer: ({counts_final[0]}, {counts_final[1]}, {counts_final[2]})"
    )

    return [
        (q_count, a_count, 100, "count_per_peg_after_continuation", sol_count),
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

    parser = argparse.ArgumentParser(description="Hanoi Puzzle Generator v15")
    parser.add_argument("--num", type=int, default=100, help="Number of puzzles per difficulty level")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--output", type=str, default="./data", help="Output base directory")
    parser.add_argument("--demo", action="store_true", help="Print demo puzzles")

    args = parser.parse_args()

    if args.demo:
        print("=" * 60)
        print("Hanoi Puzzle Demo (v15 - Score Recalibrated)")
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