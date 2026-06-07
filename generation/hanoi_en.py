import random
import json
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
    moves = ctx["moves"]
    k = ctx["k"]

    # Running triple-hash: requires sequential iteration, no closed-form shortcut.
    H1 = H2 = H3 = 0
    for i, (d, f, t) in enumerate(moves[:k]):
        step = i + 1
        H1 = (H1 * 33 + d * step + f) % 1000003
        H2 = (H2 * 17 + d * t + step) % 1000003
        H3 = (H3 * 7 + f * t + d) % 1000003
    answer = f"({H1}, {H2}, {H3})"

    solution = (
        f"Step 1: Generate the optimal move sequence for {n} disks: Peg {src} → Peg {dst} via Peg {aux}.\n"
        f"Step 2: Iterate steps i = 1 to {k}, updating H1/H2/H3 at each step.\n"
        f"Step 3: Final values — H1={H1}, H2={H2}, H3={H3}.\n"
        f"Final answer: {answer}"
    )

    return [
        (
            f"In an optimal Tower of Hanoi solution with {n} disks, all disks start on Peg {src}\n"
            f"and must be moved to Peg {dst}, using Peg {aux} as auxiliary.\n"
            f"Consider the first {k} moves. At each step i (1-indexed), let D_i be the disk moved,\n"
            f"F_i the source peg, and T_i the destination peg.\n"
            f"Starting with H1 = H2 = H3 = 0, update for each step i from 1 to {k}:\n"
            f"  H1 = (H1 * 33 + D_i * i + F_i) mod 1000003\n"
            f"  H2 = (H2 * 17 + D_i * T_i + i) mod 1000003\n"
            f"  H3 = (H3 * 7 + F_i * T_i + D_i) mod 1000003\n"
            f"Provide the final values in the exact format: (H1, H2, H3).",
            answer,
            1,
            "triple_hash",
            solution,
        )
    ]

def _build_templates_medium(ctx, rng):
    n = ctx["n"]
    src = ctx["src"]
    aux = ctx["aux"]
    dst = ctx["dst"]
    moves = ctx["moves"]
    k = ctx["k"]

    # Running triple-hash: same structure as easy but with higher k (calibrated for ~50% accuracy).
    # Exp error model recalibrated at k=24→88%: C=0.00541, a=0.1292.
    # k=34 (avg of 31-37) → error≈56% → accuracy≈56%; acceptable for ~50% target.
    H1 = H2 = H3 = 0
    for i, (d, f, t) in enumerate(moves[:k]):
        step = i + 1
        H1 = (H1 * 33 + d * step + f) % 1000003
        H2 = (H2 * 17 + d * t + step) % 1000003
        H3 = (H3 * 7 + f * t + d) % 1000003
    answer = f"({H1}, {H2}, {H3})"

    solution = (
        f"Step 1: Generate the optimal move sequence for {n} disks: Peg {src} → Peg {dst} via Peg {aux}.\n"
        f"Step 2: Iterate steps i = 1 to {k}, updating H1/H2/H3 at each step.\n"
        f"Step 3: Final values — H1={H1}, H2={H2}, H3={H3}.\n"
        f"Final answer: {answer}"
    )

    return [
        (
            f"In an optimal Tower of Hanoi solution with {n} disks, all disks start on Peg {src}\n"
            f"and must be moved to Peg {dst}, using Peg {aux} as auxiliary.\n"
            f"Consider the first {k} moves. At each step i (1-indexed), let D_i be the disk moved,\n"
            f"F_i the source peg, and T_i the destination peg.\n"
            f"Starting with H1 = H2 = H3 = 0, update for each step i from 1 to {k}:\n"
            f"  H1 = (H1 * 33 + D_i * i + F_i) mod 1000003\n"
            f"  H2 = (H2 * 17 + D_i * T_i + i) mod 1000003\n"
            f"  H3 = (H3 * 7 + F_i * T_i + D_i) mod 1000003\n"
            f"Provide the final values in the exact format: (H1, H2, H3).",
            answer,
            1,
            "triple_hash",
            solution,
        )
    ]

def _build_templates_hard(ctx, rng):
    n = ctx["n"]
    src = ctx["src"]
    aux = ctx["aux"]
    dst = ctx["dst"]
    total = ctx["total_moves"]
    moves = ctx["moves"]
    
    lo_k = max(1, int(total * 0.68))
    hi_k = max(lo_k, total - 5)
    k = rng.randint(lo_k, hi_k)

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
            12,
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
            6,
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
            6,
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
        # All difficulties use triple-hash (easy/medium) or dual-hash+simulation (hard).
        # Recalibrated exp model at k=24→88%: C=0.00541, a=0.1292.
        # easy:   n=5-7, k=25-30; midpoint after the 86% and 60% runs.
        # medium: n=7-9, k=31-37 (avg 34) → accuracy≈56%. Target: 50%.
        # hard:   n=12-15, k=68-100% of total (set inside builder). Target: 25%.
        "easy": {"n_weights": ([5, 6, 7], [0.15, 0.45, 0.40]), "builder": _build_templates_easy},
        "medium": {"n_weights": ([7, 8, 9], [0.34, 0.33, 0.33]), "builder": _build_templates_medium},
        "hard": {"n_weights": ([12, 13, 14, 15], [0.25, 0.25, 0.25, 0.25]), "builder": _build_templates_hard}
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

            # Recalibrated exp model at k=24→88%: C=0.00541, a=0.1292.
            # Easy:   k=27-33 (avg 30) → error≈26% → accuracy≈74%. Target: 75%.
            # Medium: k=31-37 (avg 34) → error≈44% → accuracy≈56%. Target: 50%.
            # Hard:   k set inside _build_templates_hard (68-100% of total).
            if diff == "easy":
                k = rng.randint(25, min(30, total_moves))
            elif diff == "medium":
                k = rng.randint(31, min(37, total_moves))
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
                puzzles.append({
                    "id": f"hanoi_en_{diff}_{idx:04d}",
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
