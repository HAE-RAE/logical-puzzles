"""Cryptarithmetic (재작성판, 3독립테이블) 재측정.

2026-07: cryptarithmetic_en/ko 가 letter-group(EN) / 자모(KO) 3독립치환표로 전면
재작성되고 데이터가 재생성됨. 구 문서(module_algorithm_search_space.md)의 크립트 수치
(10! 상수 공간·operands 2/3/4·backtrack 노드 10.7K/21.6K/31.2K 등)를 갱신하기 위한 실측.

생성기 솔버(find_solutions)와 토큰 분해를 그대로 재사용한다.
측정(공개 jsonl 100건/티어, EN+KO):
  ① 그룹별 distinct 토큰 수 → raw key space ∏ P(10, k_g)  (reveal 전 후보 공간)
  ② 공개키(fixed) 조건부 유일성 (find_solutions(fixed=revealed, max_count=2)==1)
  ③ 공개키 조건부 backtrack 노드 (effective 난이도)  median/mean/max
  ④ solution 메타: 총 자모/글자수·공개·추론·carry
  ⑤ 피연산자 수·단어 길이
실행: uv run python scripts/analysis/measure_cryptarithmetic.py
"""
import sys, json, re
from pathlib import Path
from math import perm, log10
from statistics import median, mean
import importlib.util

ROOT = Path(__file__).resolve().parents[2]
TIERS = ["easy", "medium", "hard"]


def load_mod(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def jl(task, tier):
    p = ROOT / f"data/jsonl/{task}_{tier}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


CR_EN = load_mod("cr_en", "generation/cryptarithmetic_en.py")
CR_KO = load_mod("cr_ko", "generation/cryptarithmetic_ko.py")


# ---------- 파싱 ---------- #
def parse_en(r):
    """EN: 단어 리스트 + 공개키(token->digit)."""
    q = r["question"]
    words = [m.group(1) for line in q.splitlines()
             if (m := re.match(r"^[+=]?\s*([A-Z]{2,})\s*$", line.strip()))]
    revealed = {}
    for m in re.finditer(r"\[group [A-Z]-[A-Z]\]\s*(.+)", q):
        for pair in m.group(1).split(","):
            pair = pair.strip()
            mm = re.match(r"([A-Z])=(\d)", pair)
            if mm:
                ch, d = mm.group(1), int(mm.group(2))
                revealed[f"{CR_EN._group_of(ch)}{ch}"] = d
    return words, revealed


def parse_ko(r):
    """KO: 단어 리스트 + 공개키(token->digit). 토큰 = 타입태그(C/V/F)+자모."""
    q = r["question"]
    # 등식 줄: 한글 단어만 (연산자/화살표 제외)
    words = []
    for line in q.splitlines():
        s = line.strip()
        m = re.match(r"^[+=]?\s*([가-힣]{2,})\s*$", s)
        if m:
            words.append(m.group(1))
    revealed = {}
    type_map = {"초성": "C", "중성": "V", "종성": "F"}
    for label, tag in type_map.items():
        m = re.search(rf"\[공개된 {label}\]\s*(.+)", q)
        if not m:
            continue
        for pair in m.group(1).split(","):
            pair = pair.strip()
            mm = re.match(r"(.+?)=(\d)", pair)
            if mm:
                jamo, d = mm.group(1).strip(), int(mm.group(2))
                revealed[f"{tag}{jamo}"] = d
    return words, revealed


def group_key_space_en(words):
    """3 letter-group 별 distinct 토큰 → ∏ P(10, k_g). (raw, reveal 전)"""
    by_grp = {0: set(), 1: set(), 2: set()}
    for w in words:
        for ch in w:
            by_grp[CR_EN._group_of(ch)].add(ch)
    ks = [len(v) for v in by_grp.values()]
    space = 1
    for k in ks:
        space *= perm(10, k)
    return ks, space


def group_key_space_ko(words):
    by_type = {"C": set(), "V": set(), "F": set()}
    for w in words:
        for t in CR_KO._decompose_word_to_tokens(w):
            by_type[t[0]].add(t)
    ks = [len(v) for v in by_type.values()]
    space = 1
    for k in ks:
        space *= perm(10, k)
    return ks, space


META = {
    "en": (r"(\d+) total \(given (\d+) · deduced (\d+)\)", r"carries:\s*(\d+)"),
    "ko": (r"총 (\d+) \(공개 (\d+) · 추론 (\d+)\)", r"받아올림\(carry\) 수:\s*(\d+)"),
}


def measure(lang):
    task = f"cryptarithmetic_{lang}"
    cr = CR_EN if lang == "en" else CR_KO
    parse = parse_en if lang == "en" else parse_ko
    space_fn = group_key_space_en if lang == "en" else group_key_space_ko
    meta_re, carry_re = META[lang]
    print(f"\n{'='*78}\n{task}  (config: {cr.DIFFICULTY_CONFIGS})\n{'='*78}")
    print(f"{'tier':7} {'N':>4} {'ops':>4} {'wlen':>7} {'k/grp(mean)':>12} "
          f"{'log10 rawKey(med)':>17} {'letters':>8} {'given':>6} {'ded':>5} "
          f"{'carry':>6} {'uniq%':>6} {'nodes med/mean/max':>22}")
    for t in TIERS:
        rows = jl(task, t)
        if not rows:
            print(f"{t:7} (no data)")
            continue
        n_ops, wlens, kmeans, spaces = [], [], [], []
        letters, givens, deds, carries = [], [], [], []
        uniq = 0; nodes = []; measured = 0
        for r in rows:
            words, revealed = parse(r)
            if len(words) < 3:
                continue
            measured += 1
            ops = words[:-1]
            n_ops.append(len(ops))
            wlens.extend(cr._word_letters(w) if lang == "en"
                         else cr._jcount(w) for w in ops)
            ks, space = space_fn(words)
            kmeans.append(mean(ks)); spaces.append(space)
            mm = re.search(meta_re, r["solution"])
            if mm:
                letters.append(int(mm.group(1))); givens.append(int(mm.group(2)))
                deds.append(int(mm.group(3)))
            cc = re.search(carry_re, r["solution"])
            if cc:
                carries.append(int(cc.group(1)))
            stt = {}
            sols = cr.find_solutions(tuple(words), max_count=2,
                                     fixed=dict(revealed), _stats=stt)
            nodes.append(stt.get("nodes", 0))
            ans = str(r["answer"]).strip()
            if len(sols) == 1 and sols[0][0] == ans:
                uniq += 1
        med_space = median(spaces) if spaces else 0
        print(f"{t:7} {measured:>4} "
              f"{(mean(n_ops) if n_ops else 0):>4.1f} "
              f"{(mean(wlens) if wlens else 0):>7.1f} "
              f"{(mean(kmeans) if kmeans else 0):>12.2f} "
              f"{(log10(med_space) if med_space else 0):>17.2f} "
              f"{(mean(letters) if letters else 0):>8.1f} "
              f"{(mean(givens) if givens else 0):>6.1f} "
              f"{(mean(deds) if deds else 0):>5.1f} "
              f"{(mean(carries) if carries else 0):>6.1f} "
              f"{100*uniq/measured if measured else 0:>5.1f}% "
              f"{median(nodes):>7.0f}/{mean(nodes):>7.0f}/{max(nodes):>7.0f}")


if __name__ == "__main__":
    langs = sys.argv[1:] or ("en", "ko")
    for lang in langs:
        measure(lang)
