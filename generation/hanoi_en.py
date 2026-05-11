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
    moves = ctx["moves"]
    
    k = rng.randint(1, total)
    disk_k, from_k, to_k = moves[k-1]
    pegs_k = simulate_pegs(n, src, aux, dst, moves, k)
    
    ans_len_2 = f"({total}, {disk_k})"
    ans_len_3 = f"({disk_k}, {from_k}, {to_k})"
    
    c0 = len(pegs_k.get(0, []))
    c1 = len(pegs_k.get(1, []))
    c2 = len(pegs_k.get(2, []))
    ans_len_4 = f"({c0}, {c1}, {c2}, {k})"

    return [
        (
            f"In an optimal Tower of Hanoi puzzle with {n} disks (start Peg {src}, dest Peg {dst}, aux Peg {aux}).\n"
            f"What is the total minimum number of moves required to solve the puzzle, and which disk is moved exactly at step {k}?\n"
            f"Provide the answer in the exact format: (total_moves, disk_moved_at_step_{k}).",
            ans_len_2,
            10,
            "total_moves_and_kth_disk",
            f"Step 1: Total minimum moves = 2^{n} - 1 = {total}.\n"
            f"Step 2: Generate optimal sequence. At step {k}, Disk {disk_k} is moved.\n"
            f"Final answer: {ans_len_2}"
        ),
        (
            f"In an optimal Tower of Hanoi puzzle with {n} disks (start Peg {src}, dest Peg {dst}, aux Peg {aux}).\n"
            f"Describe the move occurring exactly at step number {k}. Provide the answer in the exact format: (disk_moved, from_peg, to_peg).",
            ans_len_3,
            10,
            "kth_full_triplet",
            f"Step 1: Generate optimal sequence up to step {k}.\n"
            f"Step 2: At step {k}, the move involves Disk {disk_k} moving from Peg {from_k} to Peg {to_k}.\n"
            f"Final answer: {ans_len_3}"
        ),
        (
            f"In an optimal Tower of Hanoi puzzle with {n} disks (start Peg {src}, dest Peg {dst}, aux Peg {aux}).\n"
            f"After exactly {k} moves, how many disks are located on Peg 0, Peg 1, and Peg 2 respectively? Append the value of 'k' at the end.\n"
            f"Provide the answer in the exact format: (count_peg0, count_peg1, count_peg2, k).",
            ans_len_4,
            10,
            "peg_counts_and_k",
            f"Step 1: Simulate {k} moves from the start.\n"
            f"Step 2: State after {k} moves: {_format_peg_state(pegs_k)}\n"
            f"Step 3: Count for Peg 0: {c0}, Peg 1: {c1}, Peg 2: {c2}. The value of k is {k}.\n"
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
    
    # [난이도 상향 1] 시뮬레이션 길이를 전체 이동의 절반 이후로 대폭 밀어버림
    k = rng.randint(total // 2, total - 5)
    if k < 1:
        k = total

    # [난이도 상향 2] 단순 덧셈이 아닌 '누적 다항 해시(Polynomial Hash)' 적용
    # 이전 상태에 곱셈이 들어가므로 텍스트 추론만으로는 연산 오차가 무조건 발생함
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
        # [난이도 상향 4] 이동 '직전' 목적지 기둥의 상태를 조건으로 카운트
        if len(temp_pegs[t]) == 0:
            empty_dst_count += 1
        if len(temp_pegs[t]) % 2 != 0:
            odd_size_dst_count += 1
            
        temp_pegs[f].pop()
        temp_pegs[t].append(d)

    # [난이도 상향 3] 각 기둥에 있는 모든 디스크 번호의 '제곱의 합'
    # 스택 내부의 모든 요소를 완벽히 추적하지 않으면 오답 처리됨
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
            f"Final answer: {ans_len_3}"
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








def generate_all_datasets(num_per_difficulty=100, seed=2025):
    puzzles = []
    
    difficulties = {
        "easy": {"n_weights": ([4, 5, 6, 7], [0.25, 0.25, 0.25, 0.25]), "builder": _build_templates_easy},
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

            ctx = {
                "n": n,
                "src": src,
                "aux": aux,
                "dst": dst,
                "moves": moves,
                "total_moves": total_moves,
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