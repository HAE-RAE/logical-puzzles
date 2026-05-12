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
    pegs = {0: [], 1: [], 2: []}
    pegs[src] = list(range(n, 0, -1))
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
        f"  - Cross-check formulas / simulation against the [SEG] trace."
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
            5,
            "min_moves",
            f"Step 1: The minimum moves for n disks = 2^n - 1\n"
            f"Step 2: n = {n}, so 2^{n} - 1 = {total}\n"
            f"Final answer: {total}"
        ),
        (
            f"In the optimal solution for a Tower of Hanoi puzzle with {n} disks,\n"
            f"how many times does Disk {disk_target} move in total?",
            f"({disk_target}, {disk_count}, {disk_count})",
            5,
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
            4,
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
            2,
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
            2,
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
            f"Final answer: ({disk_query}, {peg_of_disk}, {peg_of_disk})"
        )
    ]

def _build_templates_medium(ctx, rng):
    n = ctx["n"]
    src = ctx["src"]
    aux = ctx["aux"]
    dst = ctx["dst"]
    total = ctx["total_moves"]
    moves = ctx["moves"]

    k = rng.randint(15, 25)
    if k > total:
        k = total

    sum_odd_disks = sum(d for i, (d, f, t) in enumerate(moves[:k]) if (i + 1) % 2 == 1)
    sum_even_to = sum(t for i, (d, f, t) in enumerate(moves[:k]) if (i + 1) % 2 == 0)
    ans_len_2 = f"({sum_odd_disks}, {sum_even_to})"

    temp_pegs = {0: [], 1: [], 2: []}
    temp_pegs[src] = list(range(n, 0, -1))
    for d, f, t in moves[:k]:
        temp_pegs[f].pop()
        temp_pegs[t].append(d)

    top0 = temp_pegs[0][-1] if temp_pegs[0] else 0
    top1 = temp_pegs[1][-1] if temp_pegs[1] else 0
    top2 = temp_pegs[2][-1] if temp_pegs[2] else 0
    ans_len_3 = f"({top0}, {top1}, {top2})"

    c1 = sum(1 for d, f, t in moves[:k] if d == 1)
    c2 = sum(1 for d, f, t in moves[:k] if d == 2)
    c3 = sum(1 for d, f, t in moves[:k] if d == 3)
    ans_len_4 = f"({c1}, {c2}, {c3}, {k})"

    return [
        (
            f"In an optimal Tower of Hanoi sequence for {n} disks (start Peg {src}, dest Peg {dst}, aux Peg {aux}).\n"
            f"Consider the first {k} moves, indexed with 1-based step numbers (i = 1, 2, ..., {k}).\n"
            f"Compute two values:\n"
            f"(a) S_odd = the SUM of disk numbers moved at ODD steps (i = 1, 3, 5, ...).\n"
            f"(b) S_even = the SUM of destination peg numbers at EVEN steps (i = 2, 4, 6, ...).\n"
            f"Provide the answer in the exact format: (S_odd, S_even).",
            ans_len_2,
            10,
            "odd_disk_even_dest_sums",
            f"Step 1: Use 1-based step index i.\n"
            f"Step 2: For odd i, sum the disk number moved -> S_odd = {sum_odd_disks}.\n"
            f"Step 3: For even i, sum the destination peg number -> S_even = {sum_even_to}.\n"
            f"Final answer: {ans_len_2}"
        ),
        (
            f"In an optimal Tower of Hanoi sequence for {n} disks (start Peg {src}, dest Peg {dst}, aux Peg {aux}).\n"
            f"After exactly {k} moves, identify the disk number that is at the TOP of Peg 0, Peg 1, and Peg 2.\n"
            f"If a peg is empty, use 0 for that peg.\n"
            f"Provide the answer in the exact format: (top_disk_peg0, top_disk_peg1, top_disk_peg2).",
            ans_len_3,
            10,
            "top_disks_after_k_moves",
            f"Step 1: Simulate the first {k} moves to get exact peg states.\n"
            f"Step 2: Find the top disk on Peg 0 -> {top0}, Peg 1 -> {top1}, Peg 2 -> {top2}.\n"
            f"Final answer: {ans_len_3}"
        ),
        (
            f"In an optimal Tower of Hanoi sequence for {n} disks (start Peg {src}, dest Peg {dst}, aux Peg {aux}).\n"
            f"Consider the first {k} moves.\n"
            f"Count how many times Disk 1 is moved, how many times Disk 2 is moved, and how many times Disk 3 is moved. Append the value of 'k' at the end.\n"
            f"Provide the answer in the exact format: (count_disk1, count_disk2, count_disk3, k).",
            ans_len_4,
            10,
            "disk_1_2_3_counts",
            f"Step 1: Iterate through the first {k} moves.\n"
            f"Step 2: Track moves for Disks 1, 2, and 3.\n"
            f"Step 3: Disk 1: {c1}, Disk 2: {c2}, Disk 3: {c3}, k: {k}.\n"
            f"Final answer: {ans_len_4}"
        )
    ]

def _build_templates_hard(ctx, rng):
    n = ctx["n"]
    src = ctx["src"]
    aux = ctx["aux"]
    dst = ctx["dst"]
    total = ctx["total_moves"]
    moves = ctx["moves"]
    
    k = rng.randint(total // 2, total - 5)
    if k < 1:
        k = total

    H1 = 0
    H2 = 0
    for i, (d, f, t) in enumerate(moves[:k]):
        step = i + 1
        H1 = (H1 * 33 + d * step + f) % 1000003
        H2 = (H2 * 17 + d * t + step) % 1000003
    ans_len_2 = f"({H1}, {H2})"

    temp_pegs = {0: [], 1: [], 2: []}
    temp_pegs[src] = list(range(n, 0, -1))
    
    empty_dst_count = 0
    odd_size_dst_count = 0
    
    for i, (d, f, t) in enumerate(moves[:k]):
        if len(temp_pegs[t]) == 0:
            empty_dst_count += 1
        if len(temp_pegs[t]) % 2 != 0:
            odd_size_dst_count += 1
            
        temp_pegs[f].pop()
        temp_pegs[t].append(d)

    sum_sq_0 = sum(x**2 for x in temp_pegs[0])
    sum_sq_1 = sum(x**2 for x in temp_pegs[1])
    sum_sq_2 = sum(x**2 for x in temp_pegs[2])
    ans_len_3 = f"({sum_sq_0}, {sum_sq_1}, {sum_sq_2})"

    c_mult_3 = sum(1 for d, f, t in moves[:k] if d % 3 == 0)
    ans_len_4 = f"({empty_dst_count}, {odd_size_dst_count}, {c_mult_3}, {k})"

    return [
        (
            f"In an optimal Tower of Hanoi sequence for {n} disks (start Peg {src}, dest Peg {dst}, aux Peg {aux}).\n"
            f"Consider the first {k} moves, indexed with 1-based step numbers (i = 1, 2, ..., {k}).\n"
            f"At each step i, let D_i, F_i, T_i be the disk moved, the from-peg, and the to-peg respectively.\n"
            f"Compute two running hash values. Initially H1 = 0 and H2 = 0. For each step i from 1 to {k}, update them exactly as follows:\n"
            f"(a) H1 = (H1 * 33 + D_i * i + F_i) modulo 1000003.\n"
            f"(b) H2 = (H2 * 17 + D_i * T_i + i) modulo 1000003.\n"
            f"Provide the answer in the exact format: (H1, H2).",
            ans_len_2,
            10,
            "polynomial_running_hash",
            f"Step 1: Iterate the first {k} moves with 1-based index i.\n"
            f"Step 2: Accumulate H1 = (H1 * 33 + D_i * i + F_i) % 1000003 -> {H1}.\n"
            f"Step 3: Accumulate H2 = (H2 * 17 + D_i * T_i + i) % 1000003 -> {H2}.\n"
            f"Final answer: {ans_len_2}"
        ),
        (
            f"In an optimal Tower of Hanoi sequence for {n} disks (start Peg {src}, dest Peg {dst}, aux Peg {aux}).\n"
            f"After exactly {k} moves, compute the sum of the SQUARES of the disk numbers residing on each peg.\n"
            f"(e.g., if a peg has disks 2 and 3, its value is 2^2 + 3^2 = 13. If a peg is empty, its value is 0).\n"
            f"Provide the answer in the exact format: (sum_sq_peg0, sum_sq_peg1, sum_sq_peg2).",
            ans_len_3,
            10,
            "sum_of_squares_all_pegs",
            f"Step 1: Simulate the first {k} moves precisely to get the full stack of each peg.\n"
            f"Step 2: Calculate sum of squares for Peg 0 -> {sum_sq_0}, Peg 1 -> {sum_sq_1}, Peg 2 -> {sum_sq_2}.\n"
            f" {ans_len_3}"
        ),
        (
            f"In an optimal Tower of Hanoi sequence for {n} disks (start Peg {src}, dest Peg {dst}, aux Peg {aux}).\n"
            f"Consider the first {k} moves. Compute four values:\n"
            f"(a) empty_dst_count = number of moves where the destination peg was COMPLETELY EMPTY immediately before the disk was placed on it.\n"
            f"(b) odd_size_dst_count = number of moves where the destination peg had an ODD number of disks on it immediately before the disk was placed on it.\n"
            f"(c) c_mult_3 = number of times the moved disk's number is an exact multiple of 3.\n"
            f"(d) the exact value of k.\n"
            f"Provide the answer in the exact format: (empty_dst_count, odd_size_dst_count, c_mult_3, k).",
            ans_len_4,
            10,
            "conditional_state_counts",
            f"Step 1: Iterate the first {k} moves.\n"
            f"Step 2: Track the length of the destination peg before each move.\n"
            f"Step 3: empty_dst_count = {empty_dst_count}, odd_size_dst_count = {odd_size_dst_count}, c_mult_3 = {c_mult_3}.\n"
            f" {ans_len_4}"
        )
    ]

def generate_all_datasets(num_per_difficulty=100, seed=2025):
    puzzles = []
    
    difficulties = {
        "easy": {"n_weights": ([3, 4], [0.6, 0.4]), "builder": _build_templates_easy},
        "medium": {"n_weights": ([7, 8, 9, 10], [0.25, 0.25, 0.25, 0.25]), "builder": _build_templates_medium},
        "hard": {"n_weights": ([10, 11, 12, 13], [0.25, 0.25, 0.25, 0.25]), "builder": _build_templates_hard}
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
