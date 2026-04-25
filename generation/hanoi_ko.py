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
        assert popped == disk
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
            parts.append(f"기둥 {peg}: [{', '.join(str(d) for d in disks)}]")
        else:
            parts.append(f"기둥 {peg}: []")
    return ", ".join(parts)


SFT_SOLUTION_RUBRIC_KO = (
    "STEP0=메타 · STEP1=주어진 조건 · STEP2=풀이 과정 · "
    "STEP3=정답 및 검증"
)

_HANOI_QTYPE_HINT_KO = {
    "min_moves": "2^n-1 최소 이동 공식 적용",
    "kth_disk": "최적 수열을 생성하여 k번째 이동의 원판 식별",
    "kth_from_to": "최적 수열을 생성하여 k번째 이동의 출발/도착 기둥 식별",
    "kth_full_triplet": "최적 수열을 생성하여 k번째 이동의 (원판, 출발, 도착) 식별",
    "largest_disk_move": "가장 큰 원판의 유일한 이동 시점 파악",
    "disk_move_count": "2^(n-d) 이동 횟수 공식 적용",
    "disks_on_peg_after_k": "k번 이동 후 기둥 상태 시뮬레이션",
    "where_is_disk_after_k": "k번 이동 후 특정 원판 위치 시뮬레이션",
    "inverse_find_n": "알려진 이동 정보로부터 n 역추론",
    "disk_k_total_moves": "k번째 이동의 원판을 식별한 뒤 총 이동 횟수 계산",
    "first_last_move": "특정 원판의 첫 번째와 마지막 이동 추적",
}


def _hanoi_worked_body_lines_ko(solution: str) -> Tuple[List[str], str]:
    seg_lines: List[str] = []
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


def _wrap_sft_hanoi_solution_ko(
    solution: str,
    *,
    n: Optional[int] = None,
    total_moves: Optional[int] = None,
    qtype: Optional[str] = None,
    answer: Optional[str] = None,
) -> str:
    seg_lines, final_answer = _hanoi_worked_body_lines_ko(solution)
    if answer is None:
        answer = final_answer or "(문제 참조)"
    hint = _HANOI_QTYPE_HINT_KO.get(qtype or "", "최적 해법을 추적")
    meta_bits = []
    if n is not None:
        meta_bits.append(f"n={n}")
    if total_moves is not None:
        meta_bits.append(f"총 이동={total_moves}")
    if qtype:
        meta_bits.append(f"유형={qtype}")
    meta_line = " · ".join(meta_bits) if meta_bits else "표준 규칙"
    summary = (
        f"  · 요약: {hint} · {meta_line} · {len(seg_lines)} SEGs"
    )
    step2 = "\n".join([summary, *seg_lines]) if seg_lines else summary
    return (
        f"{SFT_SOLUTION_RUBRIC_KO}\n"
        f"[STEP 0] 문제 메타\n"
        f"  - 최적 하노이의 탑 (2^n-1 이동) 및 표준 규칙\n"
        f"  - 최종 답은 [STEP 3]에서 확인\n"
        f"[STEP 1] 주어진 조건\n"
        f"  - n, 기둥 번호, k (문제에 명시된 대로)\n"
        f"[STEP 2] 풀이 과정\n{step2}\n"
        f"[STEP 3] 정답 및 검증\n"
        f"  - 최종 답: {answer}\n"
        f"  - 2^공식 / 시뮬레이션과 [SEG] 추적 결과 교차 검증."
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
            f"{n}개의 원판이 있는 하노이의 탑 퍼즐에서, 모든 원판은 기둥 {src}에 놓여 있습니다.\n"
            f"목표는 기둥 {aux}을(를) 보조 기둥으로 사용하여 모든 원판을 기둥 {dst}(으)로 옮기는 것입니다.\n"
            f"일반적인 규칙(한 번에 하나의 원판만 이동, 큰 원판을 작은 원판 위에 놓을 수 없음)을 따릅니다.\n"
            f"퍼즐을 완성하는 데 필요한 최소 이동 횟수는 얼마입니까?",
            f"({total}, {total}, {total})",
            2,
            "min_moves",
            f"단계 1: n개 원판의 최소 이동 횟수 = 2^n - 1\n"
            f"단계 2: n = {n}이므로 2^{n} - 1 = {total}\n"
            f"정답: {total}"
        ),
        (
            f"{n}개의 원판이 있는 하노이의 탑 퍼즐의 최적 해법에서,\n"
            f"원판 {disk_target}은(는) 총 몇 번 이동합니까?",
            f"({disk_target}, {disk_count}, {disk_count})",
            2,
            "disk_move_count",
            f"단계 1: 최적 해법에서 원판 d는 2^(n-d)번 이동\n"
            f"단계 2: 원판 {disk_target}, n={n}: 이동 횟수 = 2^({n}-{disk_target}) = {2**(n - disk_target)}\n"
            f"단계 3: 카운팅으로 검증: {disk_count}\n"
            f"정답: {disk_count}"
        ),
        (
            f"{n}개의 원판이 있는 하노이의 탑 퍼즐의 최적 해법에서,\n"
            f"모든 원판은 기둥 {src}에서 시작하여 기둥 {dst}(으)로 이동해야 합니다 (기둥 {aux}은(는) 보조).\n"
            f"가장 큰 원판(원판 {n})은 몇 번째 이동에서 움직입니까?",
            f"({l_disk}, {l_from}, {l_to})",
            1,
            "largest_disk_move",
            f"단계 1: 가장 큰 원판(원판 {n})은 최적 해법에서 정확히 1번 이동\n"
            f"단계 2: {largest_idx + 1}번째 이동에서 움직임: 기둥 {l_from} → 기둥 {l_to}\n"
            f"정답: {largest_idx + 1}번째 이동"
        ),
        (
            f"{n}개의 원판이 있는 하노이의 탑 퍼즐의 최적 해법을 생각해 봅시다.\n"
            f"모든 원판은 기둥 {src}에서 시작하여 기둥 {dst}(으)로 이동해야 합니다 (기둥 {aux}은(는) 보조).\n"
            f"이 최적 수열에서 {k}번째 이동에서 어떤 원판이 움직입니까?",
            f"({disk_k}, {from_k}, {to_k})",
            3,
            "kth_disk",
            f"단계 1: {n}개 원판의 최적 이동 수열 생성: 기둥 {src} → 기둥 {dst}\n"
            f"단계 2: 총 이동 횟수 = {total}\n"
            f"단계 3: {k}번째 이동은 원판 {disk_k}이(가) 기둥 {from_k}에서 기둥 {to_k}(으)로 이동\n"
            f"정답: 원판 {disk_k}"
        ),
        (
            f"기둥 {src}에서 기둥 {dst}(으)로의 {n}개 원판 하노이의 탑 최적 해법에서\n"
            f"(기둥 {aux}은(는) 보조), {k}번째 이동에서 원판은 어느 기둥에서 어느 기둥으로 이동합니까?",
            f"({disk_k}, {from_k}, {to_k})",
            3,
            "kth_from_to",
            f"단계 1: {n}개 원판의 최적 이동 수열 생성\n"
            f"단계 2: {k}번째 이동: 원판 {disk_k}, 기둥 {from_k} → 기둥 {to_k}\n"
            f"정답: 기둥 {from_k} → 기둥 {to_k}"
        ),
        (
            f"{n}개의 원판이 있는 하노이의 탑 최적 해법에서, 모든 원판은 기둥 {src}에서 시작하여\n"
            f"기둥 {dst}(으)로 이동해야 합니다 (기둥 {aux}은(는) 보조).\n"
            f"정확히 {k}번의 이동 후, 원판 {disk_query}은(는) 어느 기둥에 위치합니까?",
            f"({disk_query}, {peg_of_disk}, {peg_of_disk})",
            1,
            "where_is_disk_after_k",
            f"단계 1: {n}개 원판의 최적 수열 생성\n"
            f"단계 2: 초기 상태에서 {k}번 이동 시뮬레이션\n"
            f"단계 3: {k}번 이동 후 상태: {_format_peg_state(pegs_after_k)}\n"
            f"단계 4: 원판 {disk_query}은(는) 기둥 {peg_of_disk}에 위치\n"
            f"정답: 기둥 {peg_of_disk}"
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
            f"{n}개의 원판이 있는 하노이의 탑 퍼즐의 최적 해법에서,\n"
            f"원판 {disk_target}은(는) 총 몇 번 이동합니까?",
            f"({disk_target}, {disk_count_target}, {disk_count_target})",
            3,
            "disk_move_count",
            f"단계 1: 최적 하노이에서 n={n}일 때, 원판 d는 2^(n-d)번 이동\n"
            f"단계 2: 원판 {disk_target}: 2^({n}-{disk_target}) = {2**(n - disk_target)}\n"
            f"정답: {2**(n - disk_target)}"
        ),
        (
            f"{n}개의 원판이 있는 하노이의 탑 퍼즐의 최적 해법에서,\n"
            f"모든 원판은 기둥 {src}에서 시작하여 기둥 {dst}(으)로 이동해야 합니다 (기둥 {aux}은(는) 보조).\n"
            f"가장 큰 원판(원판 {n})은 몇 번째 이동에서 움직입니까?",
            f"({l_disk}, {l_from}, {l_to})",
            1,
            "largest_disk_move",
            f"단계 1: 가장 큰 원판(원판 {n})은 정확히 1번 이동\n"
            f"단계 2: {largest_idx + 1}번째 이동에서 움직임: 기둥 {l_from} → 기둥 {l_to}\n"
            f"정답: {largest_idx + 1}번째 이동"
        ),
        (
            f"{n}개의 원판이 있는 하노이의 탑 최적 해법에서, 모든 원판은 기둥 {src}에서 시작하여\n"
            f"기둥 {dst}(으)로 이동해야 합니다 (기둥 {aux}은(는) 보조).\n"
            f"{k}번째 이동을 (원판, 출발기둥, 도착기둥)의 형태로 답하시오.",
            f"({disk_k}, {from_k}, {to_k})",
            2,
            "kth_full_triplet",
            f"단계 1: {n}개 원판의 최적 수열 생성: 기둥 {src} → 기둥 {dst}, 보조 기둥 {aux}\n"
            f"단계 2: 총 이동 횟수 = 2^{n} - 1 = {total}\n"
            f"단계 3: {k}번째 이동은 (원판 {disk_k}, 기둥 {from_k}, 기둥 {to_k})\n"
            f"정답: ({disk_k}, {from_k}, {to_k})"
        ),
        (
            f"{n}개의 원판이 있는 하노이의 탑 최적 해법에서, 모든 원판은 기둥 {src}에서 시작하여\n"
            f"기둥 {dst}(으)로 이동해야 합니다 (기둥 {aux}은(는) 보조).\n"
            f"정확히 {k}번의 이동 후, 원판 {disk_query}은(는) 어느 기둥에 위치합니까?",
            f"({disk_query}, {peg_of_disk}, {peg_of_disk})",
            2,
            "where_is_disk_after_k",
            f"단계 1: {n}개 원판의 최적 수열 생성\n"
            f"단계 2: 초기 상태에서 {k}번 이동 시뮬레이션\n"
            f"단계 3: {k}번 이동 후 상태: {_format_peg_state(pegs_after_k)}\n"
            f"단계 4: 원판 {disk_query}은(는) 기둥 {peg_of_disk}에 위치\n"
            f"정답: 기둥 {peg_of_disk}"
        ),
        (
            f"{n}개의 원판이 있는 하노이의 탑 퍼즐(기둥 {src} → 기둥 {dst}, 기둥 {aux} 보조)의\n"
            f"최적 이동 수열에서, 정확히 {k}번의 이동이 수행된 후\n"
            f"기둥 {peg_target}에 있는 원판들은 무엇입니까? 오름차순으로 나열하시오.",
            f"({', '.join(str(d) for d in disks_on_peg) if disks_on_peg else '없음'}, {peg_target}, {peg_target})",
            2,
            "disks_on_peg_after_k",
            f"단계 1: {n}개 원판의 최적 수열 생성\n"
            f"단계 2: {k}번 이동 시뮬레이션\n"
            f"단계 3: {k}번 이동 후 상태: {_format_peg_state(pegs_after_k)}\n"
            f"단계 4: 기둥 {peg_target}에 있는 원판: {disks_on_peg if disks_on_peg else '없음'}\n"
            f"정답: {disks_on_peg if disks_on_peg else '없음'}"
        ),
        (
            f"{n}개의 원판이 있는 하노이의 탑 최적 해법(기둥 {src} → 기둥 {dst}, 기둥 {aux} 보조)에서,\n"
            f"수열의 {k}번째 이동을 살펴보십시오.\n"
            f"{k}번째에서 이동한 원판은 전체 최적 해법에서 총 몇 번 이동합니까?",
            f"({disk_k}, {disk_count_k}, {disk_count_k})",
            2,
            "disk_k_total_moves",
            f"단계 1: {k}번째 이동은 원판 {disk_k}과(와) 관련\n"
            f"단계 2: 전체 수열에서 원판 {disk_k}의 모든 출현 횟수 카운팅\n"
            f"단계 3: 원판 {disk_k}은(는) 총 {disk_count_k}번 이동\n"
            f"정답: {disk_count_k}"
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
            f"{n}개의 원판이 있는 하노이의 탑 퍼즐의 최적 해법에서,\n"
            f"원판 {disk_target}은(는) 총 몇 번 이동합니까?",
            f"({disk_target}, {disk_count_target}, {disk_count_target})",
            3,
            "disk_move_count",
            f"단계 1: 최적 하노이에서 n={n}일 때, 원판 d는 2^(n-d)번 이동\n"
            f"단계 2: 원판 {disk_target}: 2^({n}-{disk_target}) = {2**(n - disk_target)}\n"
            f"정답: {2**(n - disk_target)}"
        ),
        (
            f"어떤 최적 하노이의 탑 퍼즐에서, 모든 원판은 기둥 {src}에서 시작하여\n"
            f"기둥 {dst}(으)로 이동해야 하며, 기둥 {aux}을(를) 보조로 사용합니다.\n"
            f"{k}번째 이동에서 원판 {disk_k}이(가) 기둥 {from_k}에서 기둥 {to_k}(으)로 이동한다고 알려져 있습니다.\n"
            f"이 하노이의 탑 퍼즐에는 원판이 몇 개 있습니까?",
            f"({n}, {n}, {n})",
            2,
            "inverse_find_n",
            f"단계 1: {k}번째 이동이 원판 {disk_k}: 기둥 {from_k} → 기둥 {to_k}임을 알고 있음\n"
            f"단계 2: 확인된 가장 큰 원판 번호는 {disk_k}이므로 n >= {disk_k}\n"
            f"단계 3: 총 이동 횟수 = 2^n - 1 >= {k}이므로 n >= ceil(log2({k}+1))\n"
            f"단계 4: 퍼즐은 {n}개의 원판을 가짐 (검증: {k}번째 이동 일치)\n"
            f"정답: {n}"
        ),
        (
            f"{n}개의 원판이 있는 하노이의 탑 최적 해법(기둥 {src} → 기둥 {dst}, 기둥 {aux} 보조)에서,\n"
            f"정확히 {k2}번의 이동 후, 기둥 {peg_target}에 있는 원판들은 무엇입니까?\n"
            f"모든 원판 번호를 오름차순으로 나열하시오.",
            f"({', '.join(str(d) for d in disks_on_peg) if disks_on_peg else '없음'}, {peg_target}, {peg_target})",
            2,
            "disks_on_peg_after_k",
            f"단계 1: {n}개 원판의 최적 수열 생성: 기둥 {src} → 기둥 {dst}\n"
            f"단계 2: {k2}번 이동을 단계별로 시뮬레이션\n"
            f"단계 3: {k2}번 이동 후 상태: {_format_peg_state(pegs_after_k2)}\n"
            f"단계 4: 기둥 {peg_target}: {disks_on_peg if disks_on_peg else '비어 있음'}\n"
            f"정답: {disks_on_peg if disks_on_peg else '없음'}"
        ),
        (
            f"{n}개의 원판이 있는 하노이의 탑 최적 해법(기둥 {src} → 기둥 {dst}, 기둥 {aux} 보조)에서,\n"
            f"정확히 {k2}번의 이동 후, 원판 {disk_query}은(는) 어느 기둥에 위치합니까?",
            f"({disk_query}, {peg_of_disk}, {peg_of_disk})",
            3,
            "where_is_disk_after_k",
            f"단계 1: {n}개 원판의 최적 수열 생성\n"
            f"단계 2: 초기 상태에서 {k2}번 이동 시뮬레이션\n"
            f"단계 3: {k2}번 이동 후 상태: {_format_peg_state(pegs_after_k2)}\n"
            f"단계 4: 원판 {disk_query}은(는) 기둥 {peg_of_disk}에 위치\n"
            f"정답: 기둥 {peg_of_disk}"
        ),
        (
            f"{n}개의 원판이 있는 하노이의 탑 최적 해법(기둥 {src} → 기둥 {dst}, 기둥 {aux} 보조)에서,\n"
            f"{k2}번째 이동을 (원판, 출발기둥, 도착기둥)의 형태로 답하시오.",
            f"({disk_k2}, {from_k2}, {to_k2})",
            2,
            "kth_full_triplet",
            f"단계 1: {n}개 원판의 최적 수열 생성: 기둥 {src} → 기둥 {dst}\n"
            f"단계 2: 총 이동 횟수 = 2^{n} - 1 = {total}\n"
            f"단계 3: {k2}번째 이동은 (원판 {disk_k2}, 기둥 {from_k2}, 기둥 {to_k2})\n"
            f"정답: ({disk_k2}, {from_k2}, {to_k2})"
        ),
        (
            f"{n}개의 원판이 있는 하노이의 탑 최적 해법(기둥 {src} → 기둥 {dst}, 기둥 {aux} 보조)에서,\n"
            f"원판 {target_disk_fl}이(가) 처음 이동하는 것은 몇 번째이고, 마지막으로 이동하는 것은 몇 번째입니까?",
            f"({first_info[0]}, {last_info[0]}, {target_disk_fl})",
            2,
            "first_last_move",
            f"단계 1: 전체 수열에서 원판 {target_disk_fl} 추적\n"
            f"단계 2: 원판 {target_disk_fl}의 첫 이동: {first_info[0]}번째 (기둥 {first_info[1]} → 기둥 {first_info[2]})\n"
            f"단계 3: 원판 {target_disk_fl}의 마지막 이동: {last_info[0]}번째 (기둥 {last_info[1]} → 기둥 {last_info[2]})\n"
            f"정답: 첫 번째 = {first_info[0]}, 마지막 = {last_info[0]}"
        ),
        (
            f"{n}개의 원판이 있는 하노이의 탑 최적 해법(기둥 {src} → 기둥 {dst}, 기둥 {aux} 보조)에서,\n"
            f"{k}번째 이동에서 움직이는 원판이 있습니다.\n"
            f"그 원판은 전체 최적 해법에서 총 몇 번 이동합니까?",
            f"({disk_k}, {disk_count_k}, {disk_count_k})",
            2,
            "disk_k_total_moves",
            f"단계 1: {k}번째 이동은 원판 {disk_k}과(와) 관련\n"
            f"단계 2: 최적 {n}개 원판 하노이에서 원판 {disk_k}은(는) 2^({n}-{disk_k}) = {2**(n-disk_k)}번 이동\n"
            f"단계 3: 카운팅으로 검증: {disk_count_k}\n"
            f"정답: {disk_count_k}"
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
        "solution": _wrap_sft_hanoi_solution_ko(
            solution, n=n, total_moves=total_moves, qtype=qtype, answer=answer
        ),
        "difficulty": difficulty,
        "type": qtype,
        "n": n,
        "src": src,
        "aux": aux,
        "dst": dst,
        "seed": seed,
        "id": f"hanoi_ko_{difficulty}_{qtype}_{puzzle_hash}",
    }


def generate_dataset(num_per_difficulty: int = 100, seed: int = 2025) -> List[Dict[str, Any]]:
    puzzles = []
    difficulties = ["easy", "medium", "hard"]

    puzzle_seed = seed
    for difficulty in difficulties:
        for diff_idx in range(num_per_difficulty):
            puzzle = generate_puzzle(difficulty=difficulty, seed=puzzle_seed)
            puzzle["id"] = f"hanoi_ko_{difficulty}_{diff_idx:04d}"
            puzzles.append(puzzle)
            puzzle_seed += 1

    return puzzles


def save_dataset(puzzles: List[Dict], base_dir: str = "./data"):
    base_path = Path(base_dir)
    csv_dir = base_path / "csv"
    json_dir = base_path / "jsonl"

    csv_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    csv_path = csv_dir / "hanoi_ko.csv"
    jsonl_paths = {}
    for diff in ["easy", "medium", "hard"]:
        p = json_dir / f"hanoi_ko_{diff}.jsonl"
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

    print("\n=== 데이터셋 통계 ===")
    print("\n난이도 + 문제 유형별:")
    for key, count in sorted(stats.items()):
        print(f"  {key}: {count}")

    print("\n난이도 + 원판 수별:")
    for key, count in sorted(n_stats.items()):
        print(f"  {key}: {count}")

    return csv_path, jsonl_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="하노이의 탑 퍼즐 생성기 v3")
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output", type=str, default="./data")
    parser.add_argument("--demo", action="store_true")

    args = parser.parse_args()

    if args.demo:
        for difficulty in ["easy", "medium", "hard"]:
            puzzle = generate_puzzle(difficulty=difficulty, seed=42)
            print(f"\n[{difficulty} | n={puzzle['n']} | type={puzzle['type']}]")
            print(puzzle["question"])
            print(f"정답: {puzzle['answer']}")
            print()
    else:
        puzzles = generate_dataset(num_per_difficulty=args.num, seed=args.seed)
        save_dataset(puzzles, args.output)