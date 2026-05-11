import random
import json
import hashlib
import csv
from pathlib import Path


def build_hanoi_moves(n, src, aux, dst, acc):
    if n == 0:
        return
    build_hanoi_moves(n - 1, src, dst, aux, acc)
    acc.append((n, src, dst))
    build_hanoi_moves(n - 1, aux, src, dst, acc)


def get_hanoi_moves(n, src, aux, dst):
    moves = []
    build_hanoi_moves(n, src, aux, dst, moves)
    return moves


def simulate_pegs(n, src, aux, dst, moves, steps):
    pegs = {src: list(range(n, 0, -1)), aux: [], dst: []}
    for idx in range(min(steps, len(moves))):
        disk, from_peg, to_peg = moves[idx]
        pegs[from_peg].pop()
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


def _format_peg_state(pegs):
    parts = []
    for peg in sorted(pegs.keys()):
        disks = pegs[peg]
        if disks:
            parts.append(f"Peg {peg}: [{', '.join(str(d) for d in disks)}]")
        else:
            parts.append(f"Peg {peg}: []")
    return ", ".join(parts)


def _hanoi_worked_body_lines_en(solution):
    seg_lines = []
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


def _wrap_sft_hanoi_solution_en(solution, n, total_moves, qtype, answer):
    seg_lines, final_answer = _hanoi_worked_body_lines_en(solution)
    if answer is None:
        answer = final_answer or "(see prompt)"
    hint = "trace the optimal solution"
    meta_bits = []
    if n is not None:
        meta_bits.append(f"n={n}")
    if total_moves is not None:
        meta_bits.append(f"total moves={total_moves}")
    if qtype:
        meta_bits.append(f"qtype={qtype}")
    meta_line = " · ".join(meta_bits) if meta_bits else "standard rules"
    summary = f"  · Summary: {hint} · {meta_line} · {len(seg_lines)} SEGs"
    step2 = "\n".join([summary, *seg_lines]) if seg_lines else summary
    return (
        f"STEP0=meta · STEP1=given · STEP2=worked solution · STEP3=answer and verification\n"
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


def _build_templates_easy(ctx, rng):
    n = ctx["n"]
    src = ctx["src"]
    aux = ctx["aux"]
    dst = ctx["dst"]
    total = ctx["total_moves"]
    k = ctx["k"]
    disk_k = ctx["disk_k"]
    from_k = ctx["from_k"]
    to_k = ctx["to_k"]
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
            1,
            "min_moves",
            f"Step 1: The minimum moves for n disks = 2^n - 1\n"
            f"Step 2: n = {n}, so 2^{n} - 1 = {total}\n"
            f"Final answer: {total}"
        ),
        (
            f"In the optimal solution for a Tower of Hanoi puzzle with {n} disks,\n"
            f"how many times does Disk {disk_target} move in total?",
            f"({disk_target}, {disk_count}, {disk_count})",
            1,
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
            4,
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
            4,
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
            10,
            "where_is_disk_after_k",
            f"Step 1: Generate optimal sequence for {n} disks\n"
            f"Step 2: Simulate {k} moves from initial state\n"
            f"Step 3: State after {k} moves: {_format_peg_state(pegs_after_k)}\n"
            f"Step 4: Disk {disk_query} is on Peg {peg_of_disk}\n"
            f"Final answer: ({disk_query}, {peg_of_disk}, {peg_of_disk})"
        )
    ]


def _build_templates_medium(ctx, rng):
    n = ctx["n"]
    src = ctx["src"]
    aux = ctx["aux"]
    dst = ctx["dst"]
    total = ctx["total_moves"]
    k = ctx["k"]
    disk_k = ctx["disk_k"]
    from_k = ctx["from_k"]
    to_k = ctx["to_k"]
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
    ans_disks = f"({', '.join(str(d) for d in disks_on_peg)}, {peg_target}, {peg_target})" if disks_on_peg else f"(none, {peg_target}, {peg_target})"
    return [
        (
            f"In the optimal solution for a Tower of Hanoi puzzle with {n} disks,\n"
            f"how many times does Disk {disk_target} move in total?",
            f"({disk_target}, {disk_count_target}, {disk_count_target})",
            5,
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
            5,
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
            5,
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
            4,
            "where_is_disk_after_k",
            f"Step 1: Generate optimal sequence for {n} disks\n"
            f"Step 2: Simulate {k} moves from initial state\n"
            f"Step 3: State after {k} moves: {_format_peg_state(pegs_after_k)}\n"
            f"Step 4: Disk {disk_query} is on Peg {peg_of_disk}\n"
            f"Final answer: ({disk_query}, {peg_of_disk}, {peg_of_disk})"
        ),
        (
            f"In a Tower of Hanoi puzzle with {n} disks (Peg {src} → Peg {dst}, Peg {aux} auxiliary),\n"
            f"consider the optimal sequence of moves. After exactly {k} moves have been performed,\n"
            f"which disks are on Peg {peg_target}? List all disk numbers in ascending order.",
            ans_disks,
            3,
            "disks_on_peg_after_k",
            f"Step 1: Generate optimal sequence for {n} disks\n"
            f"Step 2: Simulate {k} moves\n"
            f"Step 3: State after {k} moves: {_format_peg_state(pegs_after_k)}\n"
            f"Step 4: Peg {peg_target} has: {disks_on_peg if disks_on_peg else 'no disks'}\n"
            f"Final answer: {ans_disks}"
        ),
        (
            f"In an optimal Tower of Hanoi solution with {n} disks (Peg {src} → Peg {dst}, Peg {aux} auxiliary),\n"
            f"look at the {k}-th move of the sequence.\n"
            f"How many times does the disk moved at step {k} move in the entire optimal solution?",
            f"({disk_k}, {disk_count_k}, {disk_count_k})",
            5,
            "disk_k_total_moves",
            f"Step 1: The {k}-th move involves Disk {disk_k}\n"
            f"Step 2: Count all occurrences of Disk {disk_k} in the full sequence\n"
            f"Step 3: Disk {disk_k} moves {disk_count_k} times total\n"
            f"Final answer: ({disk_k}, {disk_count_k}, {disk_count_k})"
        )
    ]


def _build_templates_hard(ctx, rng):
    n = ctx["n"]
    src = ctx["src"]
    aux = ctx["aux"]
    dst = ctx["dst"]
    total = ctx["total_moves"]
    k = ctx["k"]
    disk_k = ctx["disk_k"]
    from_k = ctx["from_k"]
    to_k = ctx["to_k"]
    moves = ctx["moves"]
    pegs_after_k = ctx["pegs_after_k"]

    k2 = rng.randint(total // 3, total - 1)
    disk_k2, from_k2, to_k2 = moves[k2 - 1]
    pegs_after_k2 = simulate_pegs(n, src, aux, dst, moves, k2)
    peg_target = rng.choice([src, aux, dst])
    disks_on_peg = sorted(pegs_after_k2[peg_target])
    disk_query = rng.randint(1, min(4, n))
    peg_of_disk = None
    for peg, stack in pegs_after_k2.items():
        if disk_query in stack:
            peg_of_disk = peg
            break

    first_move_of_disk = {}
    last_move_of_disk = {}
    for idx, (d, f, t) in enumerate(moves):
        if d not in first_move_of_disk:
            first_move_of_disk[d] = (idx + 1, f, t)
        last_move_of_disk[d] = (idx + 1, f, t)

    target_disk_fl = rng.randint(1, n)
    first_info = first_move_of_disk[target_disk_fl]
    last_info = last_move_of_disk[target_disk_fl]

    ans_disks = f"({', '.join(str(d) for d in disks_on_peg)}, {peg_target}, {peg_target})" if disks_on_peg else f"(none, {peg_target}, {peg_target})"

    return [
        (
            f"In a certain optimal Tower of Hanoi puzzle, all disks start on Peg {src}\n"
            f"and the goal is to move them to Peg {dst} using Peg {aux} as auxiliary.\n"
            f"It is known that on move {k}, Disk {disk_k} moves from Peg {from_k} to Peg {to_k}.\n"
            f"How many disks are in this Tower of Hanoi puzzle?",
            f"({n}, {n}, {n})",
            1,
            "inverse_find_n",
            f"Step 1: We know move {k} is Disk {disk_k}: Peg {from_k} → Peg {to_k}\n"
            f"Step 2: The largest disk number seen is {disk_k}, so n >= {disk_k}\n"
            f"Step 3: Total moves = 2^n - 1 >= {k}, so n >= ceil(log2({k}+1))\n"
            f"Step 4: The puzzle has {n} disks (verified: move {k} matches)\n"
            f"Final answer: ({n}, {n}, {n})"
        ),
        (
            f"In an optimal Tower of Hanoi solution with {n} disks (Peg {src} → Peg {dst}, Peg {aux} auxiliary),\n"
            f"after exactly {k2} moves, which disks are on Peg {peg_target}?\n"
            f"List all disk numbers in ascending order.",
            ans_disks,
            15,
            "disks_on_peg_after_k",
            f"Step 1: Generate optimal sequence for {n} disks: Peg {src} → Peg {dst}\n"
            f"Step 2: Simulate {k2} moves step by step\n"
            f"Step 3: State after {k2} moves: {_format_peg_state(pegs_after_k2)}\n"
            f"Step 4: Peg {peg_target}: {disks_on_peg if disks_on_peg else 'empty'}\n"
            f"Final answer: {ans_disks}"
        ),
        (
            f"In an optimal Tower of Hanoi solution with {n} disks (Peg {src} → Peg {dst}, Peg {aux} auxiliary),\n"
            f"after exactly {k2} moves, on which peg is Disk {disk_query} located?",
            f"({disk_query}, {peg_of_disk}, {peg_of_disk})",
            15,
            "where_is_disk_after_k",
            f"Step 1: Generate optimal sequence for {n} disks\n"
            f"Step 2: Simulate {k2} moves from initial state\n"
            f"Step 3: State after {k2} moves: {_format_peg_state(pegs_after_k2)}\n"
            f"Step 4: Disk {disk_query} is on Peg {peg_of_disk}\n"
            f"Final answer: ({disk_query}, {peg_of_disk}, {peg_of_disk})"
        ),
        (
            f"In an optimal Tower of Hanoi puzzle with {n} disks (Peg {src} → Peg {dst}, Peg {aux} auxiliary),\n"
            f"describe the {k2}-th move in the form (disk, from_peg, to_peg).",
            f"({disk_k2}, {from_k2}, {to_k2})",
            3,
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
            1,
            "first_last_move",
            f"Step 1: Trace Disk {target_disk_fl} through the entire sequence\n"
            f"Step 2: First move of Disk {target_disk_fl}: step {first_info[0]} (Peg {first_info[1]} → Peg {first_info[2]})\n"
            f"Step 3: Last move of Disk {target_disk_fl}: step {last_info[0]} (Peg {last_info[1]} → Peg {last_info[2]})\n"
            f" ({first_info[0]}, {last_info[0]}, {target_disk_fl})"
        )
    ]


def generate_all_datasets(num_per_difficulty=100, seed=2025):
    puzzles = []
    difficulties = {
        "easy": {"n_weights": ([5, 6], [0.5, 0.5]), "builder": _build_templates_easy},
        "medium": {"n_weights": ([6, 7], [0.5, 0.5]), "builder": _build_templates_medium},
        "hard": {"n_weights": ([12, 13, 14], [0.3, 0.4, 0.3]), "builder": _build_templates_hard}
    }

    rng = random.Random(seed)
    for diff, config in difficulties.items():
        seen_questions = set()
        seen_signatures = set()
        idx = 0
        attempts = 0
        max_attempts = num_per_difficulty * 50
        while len([p for p in puzzles if p["difficulty"] == diff]) < num_per_difficulty and attempts < max_attempts:
            attempts += 1
            n_choices, n_weights = config["n_weights"]
            n = rng.choices(n_choices, weights=n_weights)[0]
            src, aux, dst = rng.sample([0, 1, 2], 3)
            moves = get_hanoi_moves(n, src, aux, dst)
            total_moves = len(moves)

            if diff == "hard":
                k = rng.randint(total_moves // 3, total_moves - 1)
            else:
                k = rng.randint(1, total_moves)

            disk_k, from_k, to_k = moves[k - 1]
            pegs_after_k = simulate_pegs(n, src, aux, dst, moves, k)

            ctx = {
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

            templates = config["builder"](ctx, rng)
            question, answer, qtype, solution = _weighted_choice(rng, templates)

            signature = (qtype, question, answer)
            if question not in seen_questions and signature not in seen_signatures:
                seen_questions.add(question)
                seen_signatures.add(signature)
                puzzle_hash = hashlib.md5(f"{seed}_{diff}_{idx}_{qtype}".encode()).hexdigest()[:8]
                puzzles.append({
                    "id": f"hanoi_en_{diff}_{idx:04d}_{puzzle_hash}",
                    "question": question,
                    "answer": answer,
                    "solution": _wrap_sft_hanoi_solution_en(solution, n, total_moves, qtype, answer),
                    "difficulty": diff
                })
                idx += 1
            seed += 1

    return puzzles


def save_all_datasets(puzzles, base_dir="data"):
    base_path = Path(base_dir)
    csv_dir = base_path / "csv"
    json_dir = base_path / "jsonl"
    csv_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    for diff in ["easy", "medium", "hard"]:
        diff_puzzles = [p for p in puzzles if p["difficulty"] == diff]
        json_path = json_dir / f"hanoi_en_{diff}.jsonl"
        with open(json_path, "w", encoding="utf-8") as f:
            for puzzle in diff_puzzles:
                f.write(json.dumps(puzzle, ensure_ascii=False) + "\n")

    csv_path = csv_dir / "hanoi_en.csv"
    csv_columns = ["id", "question", "answer", "solution", "difficulty"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for puzzle in puzzles:
            writer.writerow(puzzle)


if __name__ == "__main__":
    generated_puzzles = generate_all_datasets(num_per_difficulty=100, seed=1)
    save_all_datasets(generated_puzzles, "data")