#!/usr/bin/env python3
"""
A2: 데이터 staleness 통합 감사 (EN + KO) — 단일 스크립트.

목적
  공개 데이터셋(data/jsonl)의 구조적 feature(힌트 수·변수 수·테이블 행수 등)를
  현재 생성기 config 의 기대 범위와 대조하여, 범위 밖 비율이 >20% 인 (task, tier) 를
  STALE 로 판정한다 — 즉 "구버전 config 로 생성된 공개 데이터" 를 재생성 없이 탐지.

통합 이력
  이 스크립트는 기존 3개 분리 스크립트를 단일화한 것이다:
    - scripts/audit_staleness.py             (EN 일반 task)
    - scripts/audit_staleness_yacht_array.py (EN yacht / array_formula)
    - scripts/audit_staleness_ko.py          (KO 전체)
  분리 스크립트는 helper(jl/pct_out/printer) 가 중복이고 EN yacht/array 가 별도 파일로
  떨어져 있어, 동일 SPECS-구동 루프로 합쳤다.

robust 처리 (중요)
  feature 추출 시 기대 필드/패턴이 없으면 None(=parse 실패) 로 처리하고 크래시하지 않는다.
  예: sat_puzzles_en medium/hard 는 'variables' 필드가 없는 구버전 스키마 →
      해당 (task, tier) 는 "parse 실패" 로 표기하고 나머지 task 감사를 계속한다.
  (기존 EN 스크립트는 이 지점에서 KeyError 로 죽어 sat 이후 task 를 감사하지 못했다.)

사용
    uv run --with numpy python scripts/analysis/audit_staleness.py            # EN + KO
    uv run --with numpy python scripts/analysis/audit_staleness.py --lang en
    uv run --with numpy python scripts/analysis/audit_staleness.py --lang ko
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]   # scripts/analysis/ -> project root
TIERS = ["easy", "medium", "hard"]
INF = 10 ** 9
STALE_PCT = 20.0   # config 범위 밖 비율이 이 값을 초과하면 STALE


def jl(name, tier):
    p = ROOT / f"data/jsonl/{name}_{tier}.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def pct_out(vals, lo, hi):
    return sum(1 for v in vals if not (lo <= v <= hi)) / len(vals) * 100


# --------------------------------------------------------------------------- #
# feature 추출기 — instance(dict) -> 수치 feature 또는 None(=parse 실패).
# 기대 필드/패턴이 없으면 반드시 None 을 반환(크래시 금지).
# --------------------------------------------------------------------------- #

# ---- 공통 (EN/KO 동일) ---- #
def f_sat_vars(r):
    v = r.get("variables")
    return len(v) if v is not None else None


def f_cipher_len(r):
    a = r.get("answer")
    return len(a) if a is not None else None


def f_causal_events(r):
    ms = re.findall(r"\bE(\d+)\b", r.get("question", ""))
    return max(int(x) for x in ms) if ms else None


def f_sudoku_givens(r):
    seg = r.get("question", "").split("Puzzle grid:")[-1]
    g = [l for l in seg.splitlines()
         if len(l.split()) == 9 and all(t == "." or t.isdigit() for t in l.split())][:9]
    return sum(1 for row in g for t in row.split() if t != ".") if len(g) == 9 else None


# ---- EN ---- #
def f_nb_hints_en(r):
    return len(re.findall(r"Guess:", r.get("question", ""))) or None


def f_crypt_wlen_en(r):
    # 2026-07 재작성판: 피연산자 4 고정이라 operands 는 판별력 없음.
    # 난이도축 = 피연산자 단어 길이(=자릿수) op_lens 6/7/8.
    w = [m.group(1) for line in r.get("question", "").splitlines()
         if (m := re.match(r"^[+=]?\s*([A-Z]{2,})\s*$", line.strip()))]
    return len(w[0]) if len(w) >= 3 else None


def _hangul_jcount(word):
    n = 0
    for c in word:
        code = ord(c) - 0xAC00
        if 0 <= code < 11172:
            n += 2 if (code % 28) == 0 else 3
    return n


def f_crypt_wlen_ko(r):
    # KO 재작성판: 피연산자 자모 수(=자릿수) op_jcounts 6/7/(7,8).
    w = []
    for line in r.get("question", "").splitlines():
        m = re.match(r"^[+=]?\s*([가-힣]{2,})\s*$", line.strip())
        if m:
            w.append(m.group(1))
    return _hangul_jcount(w[0]) if len(w) >= 3 else None


def f_lg_people_en(r):
    m = re.search(r"\*\*People:\*\*\s*(.+)", r.get("question", ""))
    return len(m.group(1).split(",")) if m else None


def f_lg_cons_en(r):
    return len(re.findall(r"^\s*\d+\.\s", r.get("question", ""), re.M)) or None


def f_hanoi_n_en(r):
    m = re.search(r"(\d+) disks", r.get("question", ""))
    return int(m.group(1)) if m else None


def f_ferry_dist_en(r):
    m = re.search(r"(\d+)\s*km", r.get("question", ""))
    return int(m.group(1)) if m else None


def f_ineq_grid_en(r):
    m = re.search(r"(\d+)x(\d+) grid", r.get("question", ""))
    return int(m.group(1)) if m else None


def f_yacht_gap_en(r):
    m = re.search(r"Greedy-vs-optimal gap:\s*(\d+)", r.get("solution", ""))
    return int(m.group(1)) if m else None


def f_af_products_en(r):
    q = r.get("question", "")
    pseg = q.split("[Products Table]")[-1].split("[Orders Table]")[0]
    pid = [int(m.group(1)) for m in re.finditer(r"^(\d+)\s*\|", pseg, re.M)]
    return max(pid) if pid else 0


def f_af_orders_en(r):
    return len(set(re.findall(r"ORD-(\d+)", r.get("question", ""))))


def f_af_customers_en(r):
    cust = [int(x) for x in re.findall(r"CUST-(\d+)", r.get("question", ""))]
    return max(cust) if cust else 0


# ---- KO ---- #
def f_nb_dig_ko(r):
    m = re.search(r"(\d+)자리", r.get("question", ""))
    return int(m.group(1)) if m else None


def f_nb_hint_ko(r):
    q = r.get("question", "")
    return (len(re.findall(r"\d{3,}\s*[-→>].*?(?:스트라이크|볼|S\)|B\))", q))
            or len(re.findall(r"스트라이크", q)) - 1) or None


def f_hanoi_ko(r):
    m = re.search(r"(\d+)개의 원판", r.get("question", ""))
    return int(m.group(1)) if m else None


def f_ferry_ko(r):
    m = re.search(r"총 길이 (\d+)\s*km", r.get("question", ""))
    return int(m.group(1)) if m else None


def f_lg_ppl_ko(r):
    m = re.search(r"\*\*사람:\*\*\s*(.+)", r.get("question", ""))
    return len(m.group(1).split(",")) if m else None


def f_ineq_ko(r):
    m = re.search(r"1부터 (\d+)까지", r.get("question", ""))
    return int(m.group(1)) if m else None


def f_ms_grid_ko(r):
    m = re.search(r"(\d+)행", r.get("question", ""))
    return int(m.group(1)) if m else None


def f_yacht_ko(r):
    m = re.search(r"점수 차:\s*(\d+)", r.get("solution", ""))
    return int(m.group(1)) if m else None


def f_af_orders_ko(r):
    return len(set(re.findall(r"ORD-(\d+)", r.get("question", "")))) or None


# --------------------------------------------------------------------------- #
# SPECS: (task, feature_name, extractor, {tier: (lo, hi)})
#
# 범위는 각 task 의 *현재 생성기 코드*(generation/*.py)에서 직접 재도출한 값이다
# (docstring/주석이 아니라 실제 config dict — 주석이 코드보다 stale 한 경우가 있었음.
#  예: array_formula 의 hard orders 는 주석 "280-360" 이지만 코드 _ORDER_COUNTS=(220,300)).
# 각 (task, feature) 의 출처는 README 의 staleness 재도출 표 참조.
# --------------------------------------------------------------------------- #
SPECS_EN = [
    ("number_baseball_en", "hints",       f_nb_hints_en,   {"easy": (4, 8),   "medium": (4, 9),     "hard": (4, 9)}),
    ("sat_puzzles_en",     "num_vars",    f_sat_vars,      {"easy": (9, 9),   "medium": (11, 11),   "hard": (14, 14)}),
    # 2026-07 재작성: operands 4 고정 → 판별 feature = 피연산자 단어 길이(자릿수) op_lens 6/7/8
    ("cryptarithmetic_en", "word_len",    f_crypt_wlen_en, {"easy": (6, 6),   "medium": (7, 7),     "hard": (8, 8)}),
    ("sudoku_en",          "givens",      f_sudoku_givens, {"easy": (40, 43), "medium": (37, 39),   "hard": (31, 35)}),
    ("causal_dag_en",      "events",      f_causal_events, {"easy": (25, 31), "medium": (40, 46),   "hard": (46, 58)}),
    ("logic_grid_en",      "people",      f_lg_people_en,  {"easy": (5, 5),   "medium": (6, 6),     "hard": (8, 8)}),
    ("logic_grid_en",      "constraints", f_lg_cons_en,    {"easy": (18, 20), "medium": (18, 20),   "hard": (45, 48)}),
    # 2026-07 hard 단일유형 개정: hard n 12-15 → 8-11 (k 만이 레버, n_weights easy{5,6,7}/med{7,8,9}/hard{8,9,10,11})
    ("hanoi_en",           "disks",       f_hanoi_n_en,    {"easy": (5, 7),   "medium": (7, 9),     "hard": (8, 11)}),
    ("cipher_en",          "answer_len",  f_cipher_len,    {"easy": (20, 24), "medium": (20, 24),   "hard": (6, 10)}),
    ("ferryman_en",        "distance_km", f_ferry_dist_en, {"easy": (75, 110),"medium": (155, 200), "hard": (380, 460)}),
    # inequality easy = 1D 체인(grid 정규식 미매치 → parse 실패), medium/hard = Futoshiki n∈{5,6}
    ("inequality_en",      "grid(Futoshiki)", f_ineq_grid_en, {"easy": (0, 99), "medium": (5, 6),  "hard": (5, 6)}),
    ("yacht_dice_en",      "greedy_gap",  f_yacht_gap_en,  {"easy": (0, 8),   "medium": (3, 30),    "hard": (20, INF)}),
    ("array_formula_en",   "products",    f_af_products_en, {"easy": (24, 32),"medium": (55, 65),   "hard": (80, 85)}),
    ("array_formula_en",   "orders",      f_af_orders_en,   {"easy": (35, 50),"medium": (120, 170), "hard": (220, 300)}),
    ("array_formula_en",   "customers",   f_af_customers_en,{"easy": (10, 14),"medium": (18, 20),   "hard": (20, 20)}),
]

SPECS_KO = [
    ("number_baseball_ko", "digits",     f_nb_dig_ko,     {"easy": (7, 7),   "medium": (8, 8),     "hard": (8, 8)}),
    ("number_baseball_ko", "hints",      f_nb_hint_ko,    {"easy": (4, 8),   "medium": (4, 9),     "hard": (4, 9)}),
    ("sat_puzzles_ko",     "num_vars",   f_sat_vars,      {"easy": (9, 9),   "medium": (11, 11),   "hard": (14, 14)}),
    # 2026-07 재작성: operands 4 고정 → 판별 feature = 피연산자 자모 수(자릿수) op_jcounts 6/7/(7,8)
    ("cryptarithmetic_ko", "word_jcount", f_crypt_wlen_ko, {"easy": (6, 6),   "medium": (7, 7),     "hard": (7, 8)}),
    ("sudoku_ko",          "givens",     f_sudoku_givens, {"easy": (40, 43), "medium": (37, 39),   "hard": (31, 35)}),
    # 2026-07 hard 단일유형 개정: hard n 12-15 → 8-11 (EN 동형)
    ("hanoi_ko",           "disks",      f_hanoi_ko,      {"easy": (5, 7),   "medium": (7, 9),     "hard": (8, 11)}),
    ("causal_dag_ko",      "events",     f_causal_events, {"easy": (25, 31), "medium": (40, 46),   "hard": (46, 58)}),
    ("ferryman_ko",        "dist_km",    f_ferry_ko,      {"easy": (75, 110),"medium": (155, 200), "hard": (380, 460)}),
    ("cipher_ko",          "ans_len",    f_cipher_len,    {"easy": (8, 8),   "medium": (7, 9),     "hard": (8, 10)}),
    ("logic_grid_ko",      "people",     f_lg_ppl_ko,     {"easy": (5, 5),   "medium": (6, 6),     "hard": (8, 8)}),
    # inequality easy = 1D 체인 size∈{8,9,16}, medium/hard = Futoshiki n∈{5,6} ("1부터 N까지"의 N)
    ("inequality_ko",      "size",       f_ineq_ko,       {"easy": (8, 16),  "medium": (5, 6),     "hard": (5, 6)}),
    ("minesweeper_ko",     "grid",       f_ms_grid_ko,    {"easy": (7, 12),  "medium": (9, 9),     "hard": (9, 9)}),
    ("yacht_dice_ko",      "greedy_gap", f_yacht_ko,      {"easy": (0, 8),   "medium": (3, 30),    "hard": (20, INF)}),
    ("array_formula_ko",   "orders",     f_af_orders_ko,  {"easy": (35, 50), "medium": (120, 170), "hard": (220, 300)}),
]


def run_specs(title, specs):
    """SPECS 한 묶음을 감사하고 STALE (task, tier) 목록을 반환."""
    print("=" * 86)
    print(title)
    print("=" * 86)
    print(f"{'task':22}{'feature':18}{'tier':7}{'data(min/med/max)':>20}{'config':>12}{'밖%':>6}")
    flagged = []
    for task, fname, fn, ranges in specs:
        for t in TIERS:
            rows = jl(task, t)
            if not rows:
                continue
            vals = [v for r in rows if (v := fn(r)) is not None]
            if not vals:
                print(f"{task:22}{fname:18}{t:7}{'parse 실패':>20}")
                continue
            lo, hi = ranges[t]
            pct = pct_out(vals, lo, hi)
            tag = "  <<< STALE" if pct > STALE_PCT else ""
            if pct > STALE_PCT:
                flagged.append((task, fname, t, pct))
            dist = f"{min(vals)}/{int(np.median(vals))}/{max(vals)}"
            cfg = f"[{lo},{'inf' if hi >= INF else hi}]"
            print(f"{task:22}{fname:18}{t:7}{dist:>20}{cfg:>12}{pct:>5.0f}%{tag}")
    print()
    return flagged


def main():
    ap = argparse.ArgumentParser(description="데이터 staleness 통합 감사 (EN + KO)")
    ap.add_argument("--lang", choices=["en", "ko", "all"], default="all",
                    help="감사 언어 (default: all)")
    args = ap.parse_args()

    flagged = []
    if args.lang in ("en", "all"):
        flagged += run_specs("A2 (EN) — 공개 데이터 feature vs 현재 EN config 범위", SPECS_EN)
    if args.lang in ("ko", "all"):
        flagged += run_specs("A2 (KO) — 공개 데이터 feature vs 현재 KO config 범위", SPECS_KO)

    print("=" * 86)
    print(f"STALE 판정 (config 범위 밖 >{STALE_PCT:.0f}%):")
    for task, fname, t, pct in flagged:
        print(f"  - {task} [{t}] {fname}: {pct:.0f}% 범위 밖")
    print(f"\nSTALE task 집합: {sorted({task for task, _, _, _ in flagged})}")
    print("\nNOTE:")
    print("  - sat_puzzles_en medium/hard 는 'variables' 필드 없는 구버전 스키마 → 'parse 실패' 표기.")
    print("  - inequality_en easy = 1D 체인(grid_size 해당없음 → 0,99 통과 처리), medium/hard = Futoshiki 만 검사.")
    print("  - jamo_ko: 별도 확인 — 정상(SINGLE_JONG 0 포함, 비stale). saju_ko: pillar 분류 별도(본 스크립트 미포함).")


if __name__ == "__main__":
    main()
