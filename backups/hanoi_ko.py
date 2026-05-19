"""
하노이의 탑 규칙 기반 문제 생성기 (v16 - 점수 재보정)

난이도 수준 (gemini-3-flash-preview, reasoning=medium 평가 기준):
- easy:   원판 5-6개, 공식 + 역추적                  → 목표 75% (65-85%)
- medium: 원판 6-8개, 상태/역추적                    → 목표 50% (40-60%)
- hard:   원판 12-15개, 이어풀기 후 기둥별 원판 수    → 목표 25% (15-35%)

보정 메모:
- 2026-04-30 실행에서 easy=72%, medium=70%, hard=63%를 기록.
- Easy는 목표 범위 안이라 의도적으로 안정적으로 유지.
- 첫 v4 재보정은 단순한 3-튜플 상태 질의가 Gemini에 쉬워 medium=91%, hard=97%로 과상승.
- v13 where_is/체크섬 혼합도 hard ~97%. v14 three_disk_pegs 88%. v15 81%.
- v15 실패 원인: 중간 상태에서 k를 역추론할 수 있는 알고리즘을 모델이 알고 있어
  결국 f(k, d) 공식으로 환원됨. 세 원판만 맞추면 돼 per-disk 91%.
- v16: 같은 중간 상태 제시 방식을 유지하되, 3개 원판 위치가 아닌
  "각 기둥의 원판 개수"를 물음. n개 전체가 올바르게 배치되어야 하므로
  P(정답) ≈ 0.91^n ≈ 25-32%. inverse_find_n 제거(99% 정확도로 점수 팽창).
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
        assert popped == disk, f"내부 오류: 원판 {disk}을(를) 예상했지만 {popped}이(가) 나왔습니다"
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


def _peg_of_disk(pegs: Dict[int, List[int]], disk: int) -> int:
    for peg, stack in pegs.items():
        if disk in stack:
            return peg
    raise ValueError(f"원판 {disk}을(를) 기둥 상태에서 찾을 수 없습니다")


def _top_disks_by_peg(pegs: Dict[int, List[int]]) -> Tuple[int, int, int]:
    # 원판 번호는 1부터 시작하므로, 빈 기둥은 0으로 명확히 표시할 수 있다.
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
    "count_disks_on_peg_after_k": "k번 이동 후 특정 기둥의 원판 개수 시뮬레이션",
    "smallest_disk_on_peg_after_k": "k번 이동 후 특정 기둥의 가장 작은 원판 찾기",
    "full_state_after_k": "k번 이동 후 모든 기둥 상태 보고",
    "three_disk_locations_after_k": "k번 이동 후 세 원판의 위치 찾기",
    "top_disks_after_k": "k번 이동 후 각 기둥의 맨 위 원판 보고",
    "next_move_after_k": "k번까지 시뮬레이션한 뒤 다음 최적 이동 식별",
    "peg_sums_after_k": "k번 이동 후 각 기둥의 원판 번호 합 계산",
    "peg_profile_after_k": "k번 이동 후 한 기둥의 개수/합/체크섬 계산",
    "peg_weighted_checksums_after_k": "k번 이동 후 위치 가중 체크섬 계산",
    "peg_square_checksums_after_k": "k번 이동 후 제곱 가중 체크섬 계산",
    "two_state_sum_delta": "두 상태를 시뮬레이션하여 기둥별 합 변화 비교",
    "two_state_checksum_delta": "두 상태를 시뮬레이션하여 가중 체크섬 변화 비교",
    "three_time_disk_location": "서로 다른 세 시점에서 한 원판의 위치 시뮬레이션",
    "disk_at_depth_after_k": "k번 이동 후 각 기둥의 특정 깊이 원판 보고",
    "move_window_disk123_counts": "(레거시) 긴 구간에서 원판 1·2·3 이동 횟수 — hard에서 미사용",
    "three_disk_pegs_after_k": "k번 이동 후 서로 다른 세 원판이 각각 어느 기둥에 있는지 한 번에 보고",
    "intermediate_state_continuation": "중간 기둥 배치를 제시하고 j번 더 이동 후 원판 위치 추적",
    "count_per_peg_after_continuation": "중간 기둥 배치를 제시하고 j번 더 이동 후 각 기둥의 원판 수 계산",
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
            f"어떤 최적 하노이의 탑 퍼즐에서, 모든 원판은 기둥 {src}에서 시작하여\n"
            f"기둥 {dst}(으)로 이동해야 하며, 기둥 {aux}을(를) 보조로 사용합니다.\n"
            f"{k}번째 이동에서 원판 {disk_k}이(가) 기둥 {from_k}에서 기둥 {to_k}(으)로 이동한다고 알려져 있습니다.\n"
            f"이 하노이의 탑 퍼즐에는 원판이 몇 개 있습니까?",
            f"({n}, {n}, {n})",
            16,
            "inverse_find_n",
            f"단계 1: {k}번째 이동이 원판 {disk_k}: 기둥 {from_k} → 기둥 {to_k}임을 알고 있음\n"
            f"단계 2: 총 이동 횟수 = 2^n - 1이며 n={n}일 때 이 이동 패턴이 검증됨\n"
            f"정답: {n}"
        ),
        (
            f"{n}개의 원판이 있는 하노이의 탑 퍼즐에서, 모든 원판은 기둥 {src}에 놓여 있습니다.\n"
            f"목표는 기둥 {aux}을(를) 보조 기둥으로 사용하여 모든 원판을 기둥 {dst}(으)로 옮기는 것입니다.\n"
            f"일반적인 규칙(한 번에 하나의 원판만 이동, 큰 원판을 작은 원판 위에 놓을 수 없음)을 따릅니다.\n"
            f"퍼즐을 완성하는 데 필요한 최소 이동 횟수는 얼마입니까?\n"
            f"답은 (이동횟수, 이동횟수, 이동횟수) 형식으로 쓰시오.",
            f"({total}, {total}, {total})",
            1,
            "min_moves",
            f"단계 1: n개 원판의 최소 이동 횟수 = 2^n - 1\n"
            f"단계 2: n = {n}이므로 2^{n} - 1 = {total}\n"
            f"정답: {total}"
        ),
        (
            f"{n}개의 원판이 있는 하노이의 탑 퍼즐의 최적 해법에서,\n"
            f"원판 {disk_target}은(는) 총 몇 번 이동합니까?\n"
            f"답은 (원판, 횟수, 횟수) 형식으로 쓰시오.",
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
            f"가장 큰 원판(원판 {n})이 수행하는 이동을 설명하시오.\n"
            f"답은 (원판, 출발기둥, 도착기둥) 형식으로 쓰시오.",
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
            f"이 최적 수열에서 {k}번째 이동에서 어떤 원판이 움직입니까?\n"
            f"답은 (원판, 출발기둥, 도착기둥) 형식으로 쓰시오.",
            f"({disk_k}, {from_k}, {to_k})",
            2,
            "kth_disk",
            f"단계 1: {n}개 원판의 최적 이동 수열 생성: 기둥 {src} → 기둥 {dst}\n"
            f"단계 2: 총 이동 횟수 = {total}\n"
            f"단계 3: {k}번째 이동은 원판 {disk_k}이(가) 기둥 {from_k}에서 기둥 {to_k}(으)로 이동\n"
            f"정답: 원판 {disk_k}"
        ),
        (
            f"기둥 {src}에서 기둥 {dst}(으)로의 {n}개 원판 하노이의 탑 최적 해법에서\n"
            f"(기둥 {aux}은(는) 보조), {k}번째 이동에서 원판은 어느 기둥에서 어느 기둥으로 이동합니까?\n"
            f"답은 (원판, 출발기둥, 도착기둥) 형식으로 쓰시오.",
            f"({disk_k}, {from_k}, {to_k})",
            2,
            "kth_from_to",
            f"단계 1: {n}개 원판의 최적 이동 수열 생성\n"
            f"단계 2: {k}번째 이동: 원판 {disk_k}, 기둥 {from_k} → 기둥 {to_k}\n"
            f"정답: 기둥 {from_k} → 기둥 {to_k}"
        ),
        (
            f"{n}개의 원판이 있는 하노이의 탑 최적 해법에서, 모든 원판은 기둥 {src}에서 시작하여\n"
            f"기둥 {dst}(으)로 이동해야 합니다 (기둥 {aux}은(는) 보조).\n"
            f"정확히 {k}번의 이동 후, 원판 {disk_query}은(는) 어느 기둥에 위치합니까?\n"
            f"답은 (원판, 기둥, 기둥) 형식으로 쓰시오.",
            f"({disk_query}, {peg_of_disk}, {peg_of_disk})",
            2,
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
            f"어떤 최적 하노이의 탑 퍼즐에서, 모든 원판은 기둥 {src}에서 시작하여\n"
            f"기둥 {dst}(으)로 이동해야 하며, 기둥 {aux}을(를) 보조로 사용합니다.\n"
            f"{k}번째 이동에서 원판 {disk_k}이(가) 기둥 {from_k}에서 기둥 {to_k}(으)로 이동한다고 알려져 있습니다.\n"
            f"이 하노이의 탑 퍼즐에는 원판이 몇 개 있습니까?",
            f"({n}, {n}, {n})",
            10,
            "inverse_find_n",
            f"단계 1: {k}번째 이동이 원판 {disk_k}: 기둥 {from_k} → 기둥 {to_k}임을 알고 있음\n"
            f"단계 2: 퍼즐은 {n}개의 원판을 가짐 (검증: {k}번째 이동 일치)\n"
            f"정답: {n}"
        ),
        (
            f"{n}개의 원판이 있는 하노이의 탑 최적 해법에서, 모든 원판은 기둥 {src}에서 시작하여\n"
            f"기둥 {dst}(으)로 이동해야 합니다 (기둥 {aux}은(는) 보조).\n"
            f"정확히 {k}번의 이동 후, 원판 {disk_query}은(는) 어느 기둥에 위치합니까?",
            f"({disk_query}, {peg_of_disk}, {peg_of_disk})",
            10,
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
            f"기둥 0, 기둥 1, 기둥 2에 있는 원판 번호의 합을 각각 계산하시오.\n"
            f"답은 (기둥0_합, 기둥1_합, 기둥2_합) 형식으로 쓰시오.",
            f"({peg_sums[0]}, {peg_sums[1]}, {peg_sums[2]})",
            0,
            "peg_sums_after_k",
            f"단계 1: {n}개 원판의 최적 수열 생성\n"
            f"단계 2: {k}번 이동 시뮬레이션\n"
            f"단계 3: {k}번 이동 후 상태: {_format_peg_state(pegs_after_k)}\n"
            f"단계 4: 기둥별 합은 {peg_sums}\n"
            f"정답: ({peg_sums[0]}, {peg_sums[1]}, {peg_sums[2]})"
        ),
        (
            f"{n}개의 원판이 있는 하노이의 탑 최적 해법(기둥 {src} → 기둥 {dst}, 기둥 {aux} 보조)에서,\n"
            f"정확히 {k}번 이동한 후 기둥 {peg_target}의 프로필을 (개수, 합, 가중합)으로 보고하시오.\n"
            f"가중합은 아래에서 위로 1*맨아래원판 + 2*다음원판 + ... 으로 계산합니다.",
            f"({peg_profile[0]}, {peg_profile[1]}, {peg_profile[2]})",
            0,
            "peg_profile_after_k",
            f"단계 1: {n}개 원판의 최적 수열 생성\n"
            f"단계 2: {k}번 이동 시뮬레이션\n"
            f"단계 3: {k}번 이동 후 상태: {_format_peg_state(pegs_after_k)}\n"
            f"단계 4: 기둥 {peg_target}의 프로필은 {peg_profile}\n"
            f"정답: ({peg_profile[0]}, {peg_profile[1]}, {peg_profile[2]})"
        ),
        (
            f"{n}개의 원판이 있는 하노이의 탑 최적 해법(기둥 {src} → 기둥 {dst}, 기둥 {aux} 보조)에서,\n"
            f"정확히 {k}번 이동한 후 각 기둥의 위치 가중 체크섬을 계산하시오.\n"
            f"각 기둥에서 원판을 아래에서 위로 읽고 1*맨아래 + 2*다음 + ... 으로 계산합니다.\n"
            f"답은 (기둥0_체크섬, 기둥1_체크섬, 기둥2_체크섬) 형식으로 쓰시오.",
            f"({weighted[0]}, {weighted[1]}, {weighted[2]})",
            0,
            "peg_weighted_checksums_after_k",
            f"단계 1: {n}개 원판의 최적 수열 생성\n"
            f"단계 2: {k}번 이동 시뮬레이션\n"
            f"단계 3: {k}번 이동 후 상태: {_format_peg_state(pegs_after_k)}\n"
            f"단계 4: 가중 체크섬은 {weighted}\n"
            f"정답: ({weighted[0]}, {weighted[1]}, {weighted[2]})"
        ),
        (
            f"{n}개의 원판이 있는 하노이의 탑 최적 해법(기둥 {src} → 기둥 {dst}, 기둥 {aux} 보조)에서,\n"
            f"{k_a}번 이동 후 상태와 {k_b}번 이동 후 상태를 비교하시오.\n"
            f"기둥 0, 기둥 1, 기둥 2에 대해 원판 번호 합의 절댓값 변화를 각각 계산하시오.\n"
            f"답은 (기둥0_합변화, 기둥1_합변화, 기둥2_합변화) 형식으로 쓰시오.",
            f"({sum_delta[0]}, {sum_delta[1]}, {sum_delta[2]})",
            0,
            "two_state_sum_delta",
            f"단계 1: {k_a}번 이동 시뮬레이션: {_format_peg_state(pegs_after_a)}\n"
            f"단계 2: {k_b}번 이동 시뮬레이션: {_format_peg_state(pegs_after_b)}\n"
            f"단계 3: 기둥별 합의 절댓값 변화는 {sum_delta}\n"
            f"정답: ({sum_delta[0]}, {sum_delta[1]}, {sum_delta[2]})"
        ),
        (
            f"{n}개의 원판이 있는 하노이의 탑 최적 해법(기둥 {src} → 기둥 {dst}, 기둥 {aux} 보조)에서,\n"
            f"수열의 {k}번째 이동을 살펴보십시오.\n"
            f"{k}번째에서 이동한 원판은 전체 최적 해법에서 총 몇 번 이동합니까?",
            f"({disk_k}, {disk_count_k}, {disk_count_k})",
            0,
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
    moves = ctx["moves"]

    # --- 이어풀기 후 기둥별 원판 수 (v16) ---
    # k_hide를 25%~75% 구간에서 숨겨 기둥 배치를 제시하고
    # j_more 번 더 이동 후 각 기둥의 원판 수를 물음.
    # 정답을 위해 n개 원판 전체가 올바르게 배치되어야 함:
    # P(정답) ≈ p_per_disk^n ≈ 0.91^(12..15) ≈ 0.24–0.32.
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

    def _fmt_board_ko(state: Dict[int, List[int]]) -> str:
        parts = []
        for peg in sorted(state.keys()):
            disks = state[peg]
            label = ", ".join(str(d) for d in disks) if disks else "빈 기둥"
            parts.append(f"  기둥 {peg}: [{label}]  (아래 → 위)")
        return "\n".join(parts)

    board_desc = _fmt_board_ko(state_hide)

    q_count = (
        f"하노이의 탑 퍼즐에 원판이 {n}개 있습니다 (원판 1이 가장 작고, 원판 {n}이 가장 큽니다).\n"
        f"목표는 최적 수열에 따라 모든 원판을 기둥 {src}에서 기둥 {dst}로 이동하는 것이며,\n"
        f"기둥 {aux}를 보조로 사용합니다.\n\n"
        f"최적 해법을 진행하던 중 어느 시점에서 퍼즐 상태가 다음과 같습니다:\n"
        f"{board_desc}\n\n"
        f"이 상태에서 최적 해법의 이동을 {j_more}번 더 수행합니다.\n"
        f"그 {j_more}번의 추가 이동 후,\n"
        f"기둥 0, 기둥 1, 기둥 2에 각각 원판이 몇 개 있습니까?\n"
        f"답은 (기둥0_원판수, 기둥1_원판수, 기둥2_원판수) 형식으로 쓰시오."
    )
    a_count = f"({counts_final[0]}, {counts_final[1]}, {counts_final[2]})"
    sol_count = (
        f"단계 1: 주어진 기둥 배치 확인 (은닉된 단계 k_hide)\n"
        f"단계 2: 기둥 구조로부터 k_hide를 역추론 → k_hide = {k_hide}\n"
        f"단계 3: k_final = {k_hide} + {j_more} = {k_final}\n"
        f"단계 4: {k_final}번 이동 후 상태: {_format_peg_state(state_final)}\n"
        f"단계 5: 기둥 0: {counts_final[0]}개, 기둥 1: {counts_final[1]}개, 기둥 2: {counts_final[2]}개\n"
        f"정답: ({counts_final[0]}, {counts_final[1]}, {counts_final[2]})"
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

    parser = argparse.ArgumentParser(description="하노이의 탑 퍼즐 생성기 v16")
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