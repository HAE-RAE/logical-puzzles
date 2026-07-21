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
        f"STEP0=문제 메타 · STEP1=주어진 조건 · STEP2=풀이 전개 · STEP3=답·검산\n"
        f"[STEP 0] 문제 메타\n"
        f"  - 최적 하노이의 탑 (2^n-1 이동) 및 표준 규칙\n"
        f"  - 최종 답은 [STEP 3]에서 확인\n"
        f"[STEP 1] 주어진 조건\n"
        f"  - n, 기둥 번호, k (문제에 명시된 대로)\n"
        f"[STEP 2] 풀이 전개\n{step2}\n"
        f"[STEP 3] 답·검산\n"
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
    # k=34 (31-37 avg) → 오차≈56% → 정확도≈56%; ~50% 목표에 허용 가능.
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


# hard 는 easy/medium 과 동일한 롤링 삼중 해시(triple_hash) 빌더를 그대로 사용한다.
# 난이도 레버는 오직 k (질의 이동 수) — 아래 generate_all_datasets 의 k 범위로만 조절한다.
# (구 버전은 hard 에만 닫힌형으로 풀리는 "최종상태 제곱합"/"조건부 카운트" 유형을 섞어,
#  그 지름길을 쓰는 모델에서 hard>medium 역전이 발생 → 단일 유형으로 통일해 제거했다.)
_build_templates_hard = _build_templates_medium

def generate_all_datasets(num_per_difficulty=100, seed=2025):
    puzzles = []

    difficulties = {
        # 전 난이도 동일 유형 = 롤링 삼중 해시(triple_hash). 유일 난이도 레버 = k(질의 이동 수).
        # 지수 오차 모델 (k=24→88%): C=0.00541, a=0.1292 (flash 기준, 참고용).
        # easy:   k=25-30 → 정확도≈75%(측정 en 0.79).
        # medium: k=31-37 (avg 34) → 정확도≈56%(측정 en 0.52). 목표: 50%.
        # hard:   k=42-48 (avg 45). 목표: ~20%. (실측: k42-47→23%/19%, k42-48→~18% 밴드; 사용자 지정 42-48.)
        #         (n 은 더 이상 난이도 레버가 아님 — 첫 k 이동은 n 에 거의 무관, question 다양성용.)
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

            # 유일 난이도 레버 = k. 세 밴드 모두 동일 triple_hash 유형, k 만 다름 (연속·단조).
            # 지수 오차 모델 (k=24→88%): C=0.00541, a=0.1292 (flash 기준, 참고용).
            # Easy:   k=25-30 (avg 27) → 정확도≈75%.
            # Medium: k=31-37 (avg 34) → 정확도≈56%. 목표: 50%.
            # Hard:   k=42-48 (avg 45). 목표: ~20% (실측 밴드 ~15-25%; 사용자 지정 42-48).
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
