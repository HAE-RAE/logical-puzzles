"""
Tower of Hanoi Rule-Based Problem Generator (v2 - Difficulty Rebalance, Korean)
하노이 탑 규칙 기반 문제 생성기 - 한국어 버전

Problem Types:
1. min_moves: 최소 이동 횟수
2. kth_disk: k번째 이동에서 움직이는 디스크
3. kth_from_to: k번째 이동의 출발/도착 기둥
4. kth_full_triplet: k번째 이동의 전체 정보 (디스크, 출발, 도착)
5. largest_disk_move: 가장 큰 디스크의 이동 시점
6. disk_move_count: 특정 디스크의 총 이동 횟수
7. disks_on_peg_after_k: k번 이동 후 특정 기둥의 상태
8. where_is_disk_after_k: k번 이동 후 특정 디스크의 위치
9. inverse_find_n: 이동 정보로부터 디스크 수 역추론
10. disk_k_total_moves: k번째에 움직인 디스크의 총 이동 횟수

Difficulty Levels (gemini-3-flash-preview 기준):
- easy: 2-3개 디스크, 직접 조회 템플릿 → 목표 85-90%
- medium: 4-5개 디스크, 다단계 추론 템플릿 → 목표 65-75%
- hard: 6-8개 디스크, 상태 시뮬레이션/역추론 템플릿 → 목표 40-55%
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
    min_disks: int = 4
    max_disks: int = 5

    def __post_init__(self):
        if self.difficulty == "easy":
            self.min_disks, self.max_disks = 2, 3
        elif self.difficulty == "medium":
            self.min_disks, self.max_disks = 4, 5
        elif self.difficulty == "hard":
            self.min_disks, self.max_disks = 6, 8


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
            parts.append(f"기둥 {peg}: [{', '.join(str(d) for d in disks)}]")
        else:
            parts.append(f"기둥 {peg}: []")
    return ", ".join(parts)


def _build_templates_easy(ctx: Context, rng) -> list:
    n = ctx["n"]
    src, aux, dst = ctx["src"], ctx["aux"], ctx["dst"]
    total = ctx["total_moves"]
    k = ctx["k"]
    disk_k, from_k, to_k = ctx["disk_k"], ctx["from_k"], ctx["to_k"]
    moves = ctx["moves"]

    largest = n
    largest_idx = next(idx for idx, (d, _, _) in enumerate(moves) if d == largest)
    l_disk, l_from, l_to = moves[largest_idx]

    disk_target = rng.randint(1, n)
    disk_count = sum(1 for d, _, _ in moves if d == disk_target)

    return [
        (
            f"하노이 탑 퍼즐에 디스크가 {n}개 있습니다. 모든 디스크는 기둥 {src}에서 시작합니다.\n"
            f"목표는 기둥 {aux}를 보조 기둥으로 사용하여 모든 디스크를 기둥 {dst}로 옮기는 것입니다.\n"
            f"일반적인 규칙을 따릅니다 (한 번에 하나의 디스크만 이동, 큰 디스크를 작은 디스크 위에 놓을 수 없음).\n"
            f"퍼즐을 완성하는 데 필요한 최소 이동 횟수는 몇 번입니까?",
            f"({total}, {total}, {total})",
            3,
            "min_moves",
            f"1단계: n개 디스크의 최소 이동 횟수 = 2^n - 1\n"
            f"2단계: n = {n}이므로 2^{n} - 1 = {total}\n"
            f"최종 답: {total}"
        ),
        (
            f"디스크 {n}개의 하노이 탑 퍼즐의 최적 풀이를 생각해보세요.\n"
            f"모든 디스크는 기둥 {src}에서 시작하여 기둥 {dst}로 옮겨야 합니다 (기둥 {aux}는 보조 기둥).\n"
            f"이 최적 순서에서 {k}번째 이동에서 어떤 디스크가 움직입니까?",
            f"({disk_k}, {from_k}, {to_k})",
            3,
            "kth_disk",
            f"1단계: {n}개 디스크의 최적 이동 순서 생성: 기둥 {src} → 기둥 {dst}\n"
            f"2단계: 총 이동 횟수 = {total}\n"
            f"3단계: {k}번째 이동은 디스크 {disk_k} (기둥 {from_k} → 기둥 {to_k})\n"
            f"최종 답: 디스크 {disk_k}"
        ),
        (
            f"기둥 {src}에서 기둥 {dst}로의 최적 {n}-디스크 하노이 탑 풀이에서\n"
            f"(기둥 {aux}가 보조 기둥), {k}번째 이동에서 디스크는 어느 기둥에서 어느 기둥으로 움직입니까?",
            f"({disk_k}, {from_k}, {to_k})",
            2,
            "kth_from_to",
            f"1단계: {n}개 디스크의 최적 이동 순서 생성\n"
            f"2단계: {k}번째 이동: 디스크 {disk_k}, 기둥 {from_k} → 기둥 {to_k}\n"
            f"최종 답: 기둥 {from_k} → 기둥 {to_k}"
        ),
        (
            f"디스크 {n}개의 하노이 탑 퍼즐의 최적 풀이에서\n"
            f"가장 큰 디스크(디스크 {n})는 몇 번째 이동에서 움직입니까?",
            f"({l_disk}, {l_from}, {l_to})",
            2,
            "largest_disk_move",
            f"1단계: 가장 큰 디스크(디스크 {n})는 최적 풀이에서 정확히 한 번 이동합니다\n"
            f"2단계: {largest_idx + 1}번째 단계에서 이동: 기둥 {l_from} → 기둥 {l_to}\n"
            f"최종 답: {largest_idx + 1}번째 이동"
        ),
        (
            f"디스크 {n}개의 하노이 탑 퍼즐의 최적 풀이에서\n"
            f"디스크 {disk_target}은(는) 총 몇 번 이동합니까?",
            f"({disk_target}, {disk_count}, {disk_count})",
            2,
            "disk_move_count",
            f"1단계: 최적 하노이에서 디스크 d는 2^(n-d)번 이동\n"
            f"2단계: 디스크 {disk_target}, n={n}: 이동 횟수 = 2^({n}-{disk_target}) = {2**(n - disk_target)}\n"
            f"3단계: 카운팅으로 검증: {disk_count}\n"
            f"최종 답: {disk_count}"
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

    return [
        (
            f"디스크 {n}개의 최적 하노이 탑 퍼즐에서, 모든 디스크는 기둥 {src}에서 시작하여\n"
            f"기둥 {aux}를 보조 기둥으로 사용해 기둥 {dst}로 옮겨야 합니다.\n"
            f"{k}번째 이동을 (디스크, 출발_기둥, 도착_기둥) 형태로 설명하세요.",
            f"({disk_k}, {from_k}, {to_k})",
            3,
            "kth_full_triplet",
            f"1단계: {n}개 디스크의 최적 순서 생성: 기둥 {src} → 기둥 {dst}, 보조 기둥 {aux}\n"
            f"2단계: 총 이동 횟수 = 2^{n} - 1 = {total}\n"
            f"3단계: {k}번째 이동은 (디스크 {disk_k}, 기둥 {from_k}, 기둥 {to_k})\n"
            f"최종 답: ({disk_k}, {from_k}, {to_k})"
        ),
        (
            f"디스크 {n}개의 최적 하노이 탑 풀이에서, 모든 디스크는 기둥 {src}에서 시작하여\n"
            f"기둥 {aux}를 보조 기둥으로 사용해 기둥 {dst}로 옮겨야 합니다.\n"
            f"정확히 {k}번 이동한 후, 디스크 {disk_query}은(는) 어느 기둥에 있습니까?",
            f"({disk_query}, {peg_of_disk}, {peg_of_disk})",
            3,
            "where_is_disk_after_k",
            f"1단계: {n}개 디스크의 최적 순서 생성\n"
            f"2단계: 초기 상태에서 {k}번 이동 시뮬레이션\n"
            f"3단계: {k}번 이동 후 상태: {_format_peg_state(pegs_after_k)}\n"
            f"4단계: 디스크 {disk_query}은(는) 기둥 {peg_of_disk}에 위치\n"
            f"최종 답: 기둥 {peg_of_disk}"
        ),
        (
            f"디스크 {n}개의 하노이 탑 퍼즐 (기둥 {src} → 기둥 {dst}, 기둥 {aux} 보조)에서\n"
            f"최적 이동 순서를 따를 때, 정확히 {k}번 이동 후\n"
            f"기둥 {peg_target}에 어떤 디스크들이 있습니까?",
            f"({', '.join(str(d) for d in disks_on_peg) if disks_on_peg else 'none'}, {peg_target}, {peg_target})",
            2,
            "disks_on_peg_after_k",
            f"1단계: {n}개 디스크의 최적 순서 생성\n"
            f"2단계: {k}번 이동 시뮬레이션\n"
            f"3단계: {k}번 이동 후 상태: {_format_peg_state(pegs_after_k)}\n"
            f"4단계: 기둥 {peg_target}: {disks_on_peg if disks_on_peg else '디스크 없음'}\n"
            f"최종 답: {disks_on_peg if disks_on_peg else '없음'}"
        ),
        (
            f"최적 하노이 탑 풀이에서 {k}번째 이동을 보세요.\n"
            f"이 단계에서 이동한 디스크를 디스크 X라 합시다 (여기서 X = 디스크 {disk_k}).\n"
            f"전체 풀이에서 이 디스크 X는 총 몇 번 이동합니까?",
            f"({disk_k}, {disk_count_k}, {disk_count_k})",
            2,
            "disk_k_total_moves",
            f"1단계: {k}번째 이동은 디스크 {disk_k}를 움직임\n"
            f"2단계: 전체 순서에서 디스크 {disk_k}의 출현 횟수 카운팅\n"
            f"3단계: 디스크 {disk_k}은(는) 총 {disk_count_k}번 이동\n"
            f"최종 답: {disk_count_k}"
        ),
        (
            f"디스크 {n}개의 하노이 탑 퍼즐의 최적 풀이에서\n"
            f"디스크 {disk_query}은(는) 총 몇 번 이동합니까?",
            f"({disk_query}, {sum(1 for d, _, _ in moves if d == disk_query)}, {sum(1 for d, _, _ in moves if d == disk_query)})",
            2,
            "disk_move_count",
            f"1단계: 최적 하노이에서 디스크 d는 2^(n-d)번 이동\n"
            f"2단계: 디스크 {disk_query}: 2^({n}-{disk_query}) = {2**(n - disk_query)}\n"
            f"최종 답: {2**(n - disk_query)}"
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

    peg_target = rng.choice([src, aux, dst])
    disks_on_peg = sorted(pegs_after_k[peg_target])

    disk_query = rng.randint(1, n)
    peg_of_disk = None
    for peg, stack in pegs_after_k.items():
        if disk_query in stack:
            peg_of_disk = peg
            break

    disk_count_k = sum(1 for d, _, _ in moves if d == disk_k)

    k2 = rng.randint(1, total)
    disk_k2, from_k2, to_k2 = moves[k2 - 1]
    pegs_after_k2 = simulate_pegs(n, src, aux, dst, moves, k2)

    peg_target2 = rng.choice([src, aux, dst])
    disks_on_peg2 = sorted(pegs_after_k2[peg_target2])

    disk_query2 = rng.randint(1, n)
    peg_of_disk2 = None
    for peg, stack in pegs_after_k2.items():
        if disk_query2 in stack:
            peg_of_disk2 = peg
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

    return [
        (
            f"어떤 최적 하노이 탑 퍼즐에서, 모든 디스크는 기둥 {src}에서 시작하여\n"
            f"기둥 {aux}를 보조 기둥으로 사용해 기둥 {dst}로 옮기는 것이 목표입니다.\n"
            f"{k}번째 이동에서 디스크 {disk_k}이(가) 기둥 {from_k}에서 기둥 {to_k}로 이동한다고 알려져 있습니다.\n"
            f"이 하노이 탑 퍼즐에는 디스크가 몇 개 있습니까?",
            f"({n}, {n}, {n})",
            3,
            "inverse_find_n",
            f"1단계: {k}번째 이동이 디스크 {disk_k}: 기둥 {from_k} → 기둥 {to_k}임을 알고 있음\n"
            f"2단계: 확인된 가장 큰 디스크 번호는 {disk_k}이므로 n >= {disk_k}\n"
            f"3단계: 총 이동 횟수 = 2^n - 1 >= {k}이므로 n을 추론\n"
            f"4단계: 퍼즐에는 {n}개의 디스크가 있음 (검증: {k}번째 이동이 일치)\n"
            f"최종 답: {n}"
        ),
        (
            f"디스크 {n}개의 최적 하노이 탑 풀이 (기둥 {src} → 기둥 {dst}, 기둥 {aux} 보조)에서\n"
            f"정확히 {k2}번 이동 후, 기둥 {peg_target2}에 어떤 디스크들이 있습니까?\n"
            f"모든 디스크 번호를 오름차순으로 나열하세요.",
            f"({', '.join(str(d) for d in disks_on_peg2) if disks_on_peg2 else 'none'}, {peg_target2}, {peg_target2})",
            3,
            "disks_on_peg_after_k",
            f"1단계: {n}개 디스크의 최적 순서 생성: 기둥 {src} → 기둥 {dst}\n"
            f"2단계: {k2}번 이동을 단계별로 시뮬레이션\n"
            f"3단계: {k2}번 이동 후 상태: {_format_peg_state(pegs_after_k2)}\n"
            f"4단계: 기둥 {peg_target2}: {disks_on_peg2 if disks_on_peg2 else '비어있음'}\n"
            f"최종 답: {disks_on_peg2 if disks_on_peg2 else '없음'}"
        ),
        (
            f"디스크 {n}개의 최적 하노이 탑 풀이 (기둥 {src} → 기둥 {dst}, 기둥 {aux} 보조)에서\n"
            f"정확히 {k2}번 이동 후, 디스크 {disk_query2}은(는) 어느 기둥에 있습니까?",
            f"({disk_query2}, {peg_of_disk2}, {peg_of_disk2})",
            3,
            "where_is_disk_after_k",
            f"1단계: {n}개 디스크의 최적 순서 생성\n"
            f"2단계: 초기 상태에서 {k2}번 이동 시뮬레이션\n"
            f"3단계: {k2}번 이동 후 상태: {_format_peg_state(pegs_after_k2)}\n"
            f"4단계: 디스크 {disk_query2}은(는) 기둥 {peg_of_disk2}에 위치\n"
            f"최종 답: 기둥 {peg_of_disk2}"
        ),
        (
            f"디스크 {n}개의 최적 하노이 탑 퍼즐 (기둥 {src} → 기둥 {dst}, 기둥 {aux} 보조)에서\n"
            f"{k2}번째 이동을 (디스크, 출발_기둥, 도착_기둥) 형태로 설명하세요.",
            f"({disk_k2}, {from_k2}, {to_k2})",
            2,
            "kth_full_triplet",
            f"1단계: {n}개 디스크의 최적 순서 생성: 기둥 {src} → 기둥 {dst}\n"
            f"2단계: 총 이동 횟수 = 2^{n} - 1 = {total}\n"
            f"3단계: {k2}번째 이동은 (디스크 {disk_k2}, 기둥 {from_k2}, 기둥 {to_k2})\n"
            f"최종 답: ({disk_k2}, {from_k2}, {to_k2})"
        ),
        (
            f"디스크 {n}개의 최적 하노이 탑 풀이 (기둥 {src} → 기둥 {dst}, 기둥 {aux} 보조)에서\n"
            f"디스크 {target_disk_fl}이(가) 처음 이동하는 것은 몇 번째이고, 마지막으로 이동하는 것은 몇 번째입니까?",
            f"({first_info[0]}, {last_info[0]}, {target_disk_fl})",
            2,
            "first_last_move",
            f"1단계: 전체 순서에서 디스크 {target_disk_fl}을(를) 추적\n"
            f"2단계: 디스크 {target_disk_fl}의 첫 이동: {first_info[0]}번째 (기둥 {first_info[1]} → 기둥 {first_info[2]})\n"
            f"3단계: 디스크 {target_disk_fl}의 마지막 이동: {last_info[0]}번째 (기둥 {last_info[1]} → 기둥 {last_info[2]})\n"
            f"최종 답: 첫 이동 = {first_info[0]}, 마지막 이동 = {last_info[0]}"
        ),
        (
            f"최적 하노이 탑 풀이에서 {k}번째 이동은 디스크 {disk_k}을(를) 움직입니다.\n"
            f"디스크 {n}개의 전체 최적 풀이에서 디스크 {disk_k}은(는) 총 몇 번 이동합니까?",
            f"({disk_k}, {disk_count_k}, {disk_count_k})",
            2,
            "disk_k_total_moves",
            f"1단계: {k}번째 이동은 디스크 {disk_k}을(를) 움직임\n"
            f"2단계: 최적 {n}-디스크 하노이에서 디스크 {disk_k}은(는) 2^({n}-{disk_k}) = {2**(n-disk_k)}번 이동\n"
            f"3단계: 카운팅으로 검증: {disk_count_k}\n"
            f"최종 답: {disk_count_k}"
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
        "solution": solution,
        "difficulty": difficulty,
        "type": qtype,
        "n": n,
        "src": src,
        "aux": aux,
        "dst": dst,
        "seed": seed,
        "id": f"hanoi_kr_{difficulty}_{qtype}_{puzzle_hash}",
    }


def generate_dataset(num_per_difficulty: int = 100, seed: int = 2025) -> List[Dict[str, Any]]:
    puzzles = []
    difficulties = ["easy", "medium", "hard"]

    puzzle_seed = seed
    for difficulty in difficulties:
        for _ in range(num_per_difficulty):
            puzzle = generate_puzzle(difficulty=difficulty, seed=puzzle_seed)
            puzzle["id"] = f"hanoi_korean_{len(puzzles)}"
            puzzles.append(puzzle)
            puzzle_seed += 1

    return puzzles


def save_dataset(puzzles: List[Dict], base_dir: str = "./data"):
    base_path = Path(base_dir)
    csv_dir = base_path / "csv"
    json_dir = base_path / "json"

    csv_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    csv_path = csv_dir / "hanoi_korean.csv"
    jsonl_path = json_dir / "hanoi_korean.jsonl"

    csv_columns = ["id", "question", "answer", "solution", "difficulty"]

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for puzzle in puzzles:
            row = {
                "id": puzzle["id"],
                "question": puzzle["question"],
                "answer": puzzle["answer"],
                "solution": puzzle["solution"],
                "difficulty": puzzle["difficulty"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(puzzles)} puzzles to {jsonl_path}")

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
            })

    print(f"Saved {len(puzzles)} puzzles to {csv_path}")

    stats = {}
    for puzzle in puzzles:
        key = f"{puzzle['difficulty']}_{puzzle['type']}"
        stats[key] = stats.get(key, 0) + 1

    print("\nDataset Statistics:")
    for key, count in sorted(stats.items()):
        print(f"  {key}: {count}")

    return csv_path, jsonl_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hanoi Puzzle Generator (Korean)")
    parser.add_argument("--num", type=int, default=100, help="Number of puzzles per difficulty level")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--output", type=str, default="./data", help="Output base directory")
    parser.add_argument("--demo", action="store_true", help="Print demo puzzles")

    args = parser.parse_args()

    if args.demo:
        print("=" * 60)
        print("하노이 탑 퍼즐 데모 (한국어)")
        print("=" * 60)
        for difficulty in ["easy", "medium", "hard"]:
            puzzle = generate_puzzle(difficulty=difficulty, seed=42)
            print(f"\n[{difficulty} - {puzzle['type']}]")
            print("-" * 40)
            print(puzzle["question"])
            print(f"\n답: {puzzle['answer']}")
            print(f"풀이: {puzzle['solution']}")
            print("=" * 60)
    else:
        puzzles = generate_dataset(num_per_difficulty=args.num, seed=args.seed)
        save_dataset(puzzles, args.output)