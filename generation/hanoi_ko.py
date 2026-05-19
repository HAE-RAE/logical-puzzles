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
            parts.append(f"기둥 {peg}: [{', '.join(str(d) for d in disks)}]")
        else:
            parts.append(f"기둥 {peg}: []")
    return ", ".join(parts)

def _hanoi_worked_body_lines_ko(solution):
    seg_lines = []
    final_answer = ""
    seg_idx = 1
    for raw in solution.rstrip().splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("최종 답") or low.startswith("정답:"):
            after = line.split(":", 1)
            final_answer = after[1].strip() if len(after) == 2 else line
            continue
        body = line
        if low.startswith("단계 "):
            parts = line.split(":", 1)
            if len(parts) == 2:
                body = parts[1].strip()
        seg_lines.append(f"    [SEG {seg_idx}] {body}")
        seg_idx += 1
    return seg_lines, final_answer

def _wrap_sft_hanoi_solution_ko(solution, n, total_moves, qtype, answer):
    seg_lines, final_answer = _hanoi_worked_body_lines_ko(solution)
    if answer is None:
        answer = final_answer or "(문제 참조)"
    hint = "최적 해법을 추적"
    meta_bits = []
    if n is not None:
        meta_bits.append(f"n={n}")
    if total_moves is not None:
        meta_bits.append(f"총 이동={total_moves}")
    if qtype:
        meta_bits.append(f"유형={qtype}")
    meta_line = " · ".join(meta_bits) if meta_bits else "표준 규칙"
    summary = f"  · 요약: {hint} · {meta_line} · {len(seg_lines)} SEGs"
    step2 = "\n".join([summary, *seg_lines]) if seg_lines else summary
    return (
        f"STEP0=메타 · STEP1=주어진 조건 · STEP2=풀이 과정 · STEP3=정답 및 검증\n"
        f"[STEP 0] 문제 메타\n"
        f"  - 최적 하노이의 탑 (2^n-1 이동) 및 표준 규칙\n"
        f"  - 최종 답은 [STEP 3]에서 확인\n"
        f"[STEP 1] 주어진 조건\n"
        f"  - n, 기둥 번호, k (문제에 명시된 대로)\n"
        f"[STEP 2] 풀이 과정\n{step2}\n"
        f"[STEP 3] 정답 및 검증\n"
        f"  - 최종 답: {answer}\n"
        f"  - 공식 / 시뮬레이션과 [SEG] 추적 결과 교차 검증."
    )


def _build_templates_easy(ctx, rng):
    n = ctx["n"]
    src = ctx["src"]
    aux = ctx["aux"]
    dst = ctx["dst"]
    moves = ctx["moves"]
    k = ctx["k"]

    # 누적 삼중 해시: 순차 반복이 필수, 닫힌 형식 공식 없음.
    H1 = H2 = H3 = 0
    for i, (d, f, t) in enumerate(moves[:k]):
        step = i + 1
        H1 = (H1 * 33 + d * step + f) % 1000003
        H2 = (H2 * 17 + d * t + step) % 1000003
        H3 = (H3 * 7 + f * t + d) % 1000003
    answer = f"({H1}, {H2}, {H3})"

    solution = (
        f"단계 1: {n}개 원판의 최적 이동 수열 생성: 기둥 {src} → 기둥 {dst}, 보조 기둥 {aux}.\n"
        f"단계 2: i = 1부터 {k}까지 각 단계에서 H1/H2/H3 갱신.\n"
        f"단계 3: 최종 값 — H1={H1}, H2={H2}, H3={H3}.\n"
        f"정답: {answer}"
    )

    return [
        (
            f"{n}개의 원판이 있는 하노이의 탑 최적 해법에서, 모든 원판은 기둥 {src}에서 시작하여\n"
            f"기둥 {dst}(으)로 이동해야 합니다 (기둥 {aux}은(는) 보조).\n"
            f"처음 {k}번의 이동을 생각해 봅시다. 각 단계 i (1부터 시작)에서 D_i를 이동한 원판,\n"
            f"F_i를 출발 기둥, T_i를 도착 기둥이라 합니다.\n"
            f"H1 = H2 = H3 = 0에서 시작하여, 각 단계 i = 1부터 {k}까지 다음과 같이 갱신합니다:\n"
            f"  H1 = (H1 * 33 + D_i * i + F_i) mod 1000003\n"
            f"  H2 = (H2 * 17 + D_i * T_i + i) mod 1000003\n"
            f"  H3 = (H3 * 7 + F_i * T_i + D_i) mod 1000003\n"
            f"최종 값을 정확히 다음 형식으로 답하시오: (H1, H2, H3).",
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

    # 누적 삼중 해시: easy와 동일한 구조이나 더 높은 k로 ~50% 정확도 목표.
    # 재보정 지수 모델 (k=24→88%): C=0.00541, a=0.1292.
    # k=34 (31-37 avg) → 오차≈44% → 정확도≈56%. 목표: 50%.
    H1 = H2 = H3 = 0
    for i, (d, f, t) in enumerate(moves[:k]):
        step = i + 1
        H1 = (H1 * 33 + d * step + f) % 1000003
        H2 = (H2 * 17 + d * t + step) % 1000003
        H3 = (H3 * 7 + f * t + d) % 1000003
    answer = f"({H1}, {H2}, {H3})"

    solution = (
        f"단계 1: {n}개 원판의 최적 이동 수열 생성: 기둥 {src} → 기둥 {dst}, 보조 기둥 {aux}.\n"
        f"단계 2: i = 1부터 {k}까지 각 단계에서 H1/H2/H3 갱신.\n"
        f"단계 3: 최종 값 — H1={H1}, H2={H2}, H3={H3}.\n"
        f"정답: {answer}"
    )

    return [
        (
            f"{n}개의 원판이 있는 하노이의 탑 최적 해법에서, 모든 원판은 기둥 {src}에서 시작하여\n"
            f"기둥 {dst}(으)로 이동해야 합니다 (기둥 {aux}은(는) 보조).\n"
            f"처음 {k}번의 이동을 생각해 봅시다. 각 단계 i (1부터 시작)에서 D_i를 이동한 원판,\n"
            f"F_i를 출발 기둥, T_i를 도착 기둥이라 합니다.\n"
            f"H1 = H2 = H3 = 0에서 시작하여, 각 단계 i = 1부터 {k}까지 다음과 같이 갱신합니다:\n"
            f"  H1 = (H1 * 33 + D_i * i + F_i) mod 1000003\n"
            f"  H2 = (H2 * 17 + D_i * T_i + i) mod 1000003\n"
            f"  H3 = (H3 * 7 + F_i * T_i + D_i) mod 1000003\n"
            f"최종 값을 정확히 다음 형식으로 답하시오: (H1, H2, H3).",
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
            f"{n}개의 원판이 있는 하노이의 탑 최적 이동 수열 (출발 기둥 {src}, 목적 기둥 {dst}, 보조 기둥 {aux}).\n"
            f"처음 {k}번의 이동을 1-기반 단계 번호 (i = 1, 2, ..., {k})로 생각합니다.\n"
            f"각 단계 i에서 D_i, F_i, T_i를 각각 이동한 원판, 출발 기둥, 도착 기둥이라 합니다.\n"
            f"두 누적 해시 값을 계산합니다. 초기값은 H1 = 0, H2 = 0. 각 단계 i = 1부터 {k}까지 다음과 같이 정확히 갱신합니다:\n"
            f"(a) H1 = (H1 * 33 + D_i * i + F_i) modulo 1000003.\n"
            f"(b) H2 = (H2 * 17 + D_i * T_i + i) modulo 1000003.\n"
            f"정확히 다음 형식으로 답하시오: (H1, H2).",
            ans_len_2,
            12,
            "polynomial_running_hash",
            f"단계 1: 1-기반 인덱스 i로 처음 {k}번의 이동을 반복합니다.\n"
            f"단계 2: H1 = (H1 * 33 + D_i * i + F_i) % 1000003 누적 -> {H1}.\n"
            f"단계 3: H2 = (H2 * 17 + D_i * T_i + i) % 1000003 누적 -> {H2}.\n"
            f"정답: {ans_len_2}"
        ),
        (
            f"{n}개의 원판이 있는 하노이의 탑 최적 이동 수열 (출발 기둥 {src}, 목적 기둥 {dst}, 보조 기둥 {aux}).\n"
            f"정확히 {k}번의 이동 후, 각 기둥에 있는 원판 번호의 제곱의 합을 구하시오.\n"
            f"(예: 기둥에 원판 2와 3이 있으면 2^2 + 3^2 = 13. 기둥이 비어 있으면 0.)\n"
            f"정확히 다음 형식으로 답하시오: (기둥0_제곱합, 기둥1_제곱합, 기둥2_제곱합).",
            ans_len_3,
            6,
            "sum_of_squares_all_pegs",
            f"단계 1: 처음 {k}번의 이동을 정확히 시뮬레이션하여 각 기둥의 전체 스택을 파악합니다.\n"
            f"단계 2: 기둥 0의 제곱합 -> {sum_sq_0}, 기둥 1 -> {sum_sq_1}, 기둥 2 -> {sum_sq_2}.\n"
            f" {ans_len_3}"
        ),
        (
            f"{n}개의 원판이 있는 하노이의 탑 최적 이동 수열 (출발 기둥 {src}, 목적 기둥 {dst}, 보조 기둥 {aux}).\n"
            f"처음 {k}번의 이동에서 네 가지 값을 계산하시오:\n"
            f"(a) 빈_목적_횟수 = 원판이 놓이기 직전에 도착 기둥이 완전히 비어 있었던 이동 횟수.\n"
            f"(b) 홀수_크기_목적_횟수 = 원판이 놓이기 직전에 도착 기둥의 원판 수가 홀수였던 이동 횟수.\n"
            f"(c) 3의배수_횟수 = 이동한 원판 번호가 정확히 3의 배수인 이동 횟수.\n"
            f"(d) k의 정확한 값.\n"
            f"정확히 다음 형식으로 답하시오: (빈_목적_횟수, 홀수_크기_목적_횟수, 3의배수_횟수, k).",
            ans_len_4,
            6,
            "conditional_state_counts",
            f"단계 1: 처음 {k}번의 이동을 반복합니다.\n"
            f"단계 2: 각 이동 전 도착 기둥의 원판 수를 추적합니다.\n"
            f"단계 3: 빈_목적_횟수 = {empty_dst_count}, 홀수_크기_목적_횟수 = {odd_size_dst_count}, 3의배수_횟수 = {c_mult_3}.\n"
            f" {ans_len_4}"
        )
    ]


def generate_all_datasets(num_per_difficulty=100, seed=2025):
    puzzles = []

    difficulties = {
        # 모든 난이도가 삼중 해시(easy/medium) 또는 이중 해시+시뮬레이션(hard) 사용.
        # 재보정 지수 모델 (k=24→88%): C=0.00541, a=0.1292.
        # easy:   n=6-8,   k=27-33 (avg 30) → 정확도≈74%. 목표: 75%.
        # medium: n=7-9,   k=31-37 (avg 34) → 정확도≈56%. 목표: 50%.
        # hard:   n=12-15, k=68-100% (builder 내부 설정). 목표: 25%.
        "easy": {"n_weights": ([6, 7, 8], [0.34, 0.33, 0.33]), "builder": _build_templates_easy},
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

            # 재보정 지수 모델 (k=24→88%): C=0.00541, a=0.1292.
            # easy:   k=27-33 (avg 30) → 오차≈26% → 정확도≈74%. 목표: 75%.
            # medium: k=31-37 (avg 34) → 오차≈44% → 정확도≈56%. 목표: 50%.
            # hard:   k는 _build_templates_hard 내부에서 설정 (총 이동의 68-100%).
            if diff == "easy":
                k = rng.randint(27, min(33, total_moves))
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
                    "id": f"hanoi_ko_{diff}_{idx:04d}",
                    "question": question,
                    "answer": answer,
                    "solution": _wrap_sft_hanoi_solution_ko(solution, n, total_moves, qtype, answer),
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
        json_path = json_dir / f"hanoi_ko_{diff}.jsonl"
        with open(json_path, "w", encoding="utf-8") as f:
            for puzzle in diff_puzzles:
                f.write(json.dumps(puzzle, ensure_ascii=False) + "\n")

    csv_path = csv_dir / "hanoi_ko.csv"
    csv_columns = ["id", "question", "answer", "solution", "difficulty"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for puzzle in puzzles:
            writer.writerow(puzzle)


if __name__ == "__main__":
    generated_puzzles = generate_all_datasets(num_per_difficulty=100, seed=1)
    save_all_datasets(generated_puzzles, "data")
