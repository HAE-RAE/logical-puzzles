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

# hard reuses the exact same running triple-hash builder as easy/medium.
# The sole difficulty lever is k (query move count) — tuned only via the k range
# in generate_all_datasets below.
# (The previous version mixed closed-form-solvable "final-state sum of squares" /
#  "conditional counts" types into hard only, which caused hard>medium inversion on
#  models that shortcut those closed-form types → removed by going single-type.)
_build_templates_hard = _build_templates_medium

def generate_all_datasets(num_per_difficulty=100, seed=2025):
    puzzles = []
    
    difficulties = {
        # All difficulties use the SAME type = running triple-hash. Sole difficulty lever = k.
        # Exp error model (k=24→88%): C=0.00541, a=0.1292 (flash reference, indicative only).
        # easy:   k=25-30 → accuracy≈75% (measured en 0.79).
        # medium: k=31-37 (avg 34) → accuracy≈56% (measured en 0.52). Target: 50%.
        # hard:   k=42-48 (avg 45). Target: ~20%. (measured: k42-47→23%/19%, k42-48→~18% band; user-set 42-48.)
        #         (n is no longer a difficulty lever — the first k moves are ~n-independent; n only varies question text.)
        "easy": {"n_weights": ([5, 6, 7], [0.15, 0.45, 0.40]), "builder": _build_templates_easy},
        "medium": {"n_weights": ([7, 8, 9], [0.34, 0.33, 0.33]), "builder": _build_templates_medium},
        "hard": {"n_weights": ([8, 9, 10, 11], [0.25, 0.25, 0.25, 0.25]), "builder": _build_templates_hard}
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

            # Sole difficulty lever = k. All three bands use the same triple_hash type, k only (monotonic).
            # Exp error model (k=24→88%): C=0.00541, a=0.1292 (flash reference, indicative only).
            # Easy:   k=25-30 (avg 27) → accuracy≈75%.
            # Medium: k=31-37 (avg 34) → accuracy≈56%. Target: 50%.
            # Hard:   k=42-48 (avg 45). Target: ~20% (measured band ~15-25%; user-set 42-48).
            if diff == "easy":
                k = rng.randint(25, min(30, total_moves))
            elif diff == "medium":
                k = rng.randint(31, min(37, total_moves))
            else:
                k = rng.randint(42, min(48, total_moves))
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
