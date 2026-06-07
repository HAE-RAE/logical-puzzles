"""
Subway shortest-stops puzzle generator (English).

KO 버전(subway_ko.py)의 노선망(EDGES)을 그대로 가져와 역 이름만 로마자로
변환해 영어판을 만든다. 그래프 토폴로지·난이도 대역은 KO와 동일.

- 정답 형식: "N stations" (정수)
- 정답 산출: 무가중 BFS 최단 정거장 수 (= len(path)-1)

[난이도] (최소 정거장 수)
- easy:   4 ~ 11
- medium: 17 ~ 23
- hard:   20 ~ 40
"""

import os
import sys
import json
import random
from collections import deque
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subway_ko import EDGES as KO_EDGES  # noqa: E402

# ---------------------------------------------------------------------------
# 한글 → 로마자 (Revised Romanization, 음절 단위 근사)
# ---------------------------------------------------------------------------
_CHO = ['g', 'kk', 'n', 'd', 'tt', 'r', 'm', 'b', 'pp', 's', 'ss', '',
        'j', 'jj', 'ch', 'k', 't', 'p', 'h']
_JUNG = ['a', 'ae', 'ya', 'yae', 'eo', 'e', 'yeo', 'ye', 'o', 'wa', 'wae',
         'oe', 'yo', 'u', 'wo', 'we', 'wi', 'yu', 'eu', 'ui', 'i']
_JONG = ['', 'k', 'k', 'k', 'n', 'n', 'n', 't', 'l', 'k', 'm', 'l', 'l', 'l',
         'p', 'l', 'm', 'p', 'p', 't', 't', 'ng', 't', 't', 'k', 't', 'p', 't']


def _romanize_syllable(ch):
    o = ord(ch)
    if 0xAC00 <= o <= 0xD7A3:
        s = o - 0xAC00
        return _CHO[s // 588] + _JUNG[(s % 588) // 28] + _JONG[s % 28]
    return ch


def _romanize(name):
    r = ''.join(_romanize_syllable(c) for c in name)
    return r[:1].upper() + r[1:] if r else r


def _build_rom_map(edges):
    """KO 역명 → 고유한 로마자 이름. 충돌 시 숫자 접미사로 분리(토폴로지 보존)."""
    seen = []
    for a, b, _ in edges:
        for s in (a, b):
            if s not in seen:
                seen.append(s)
    rom = {}
    used = set()
    for s in seen:
        base = _romanize(s)
        r, k = base, 2
        while r in used:
            r, k = f"{base}{k}", k + 1
        used.add(r)
        rom[s] = r
    return rom

ROM = _build_rom_map(KO_EDGES)
EDGES = [(ROM[a], ROM[b], line) for a, b, line in KO_EDGES]


# ---------------------------------------------------------------------------
# 그래프 / 노선 시퀀스 / 노선도 텍스트
# ---------------------------------------------------------------------------
def build_graph():
    g = {}
    for a, b, line in EDGES:
        g.setdefault(a, []).append((b, line))
        g.setdefault(b, []).append((a, line))
    return g

GRAPH = build_graph()
ALL_STATIONS = list(GRAPH.keys())


def build_line_sequences():
    seqs = {}
    for a, b, line in EDGES:
        seq = seqs.setdefault(line, [])
        if not seq:
            seq.extend([a, b])
        elif seq[-1] == a:
            seq.append(b)
        elif seq[-1] == b:
            seq.append(a)
        else:
            seq.extend([a, b])
    return seqs

LINE_SEQUENCES = build_line_sequences()


def network_text():
    lines = ["[Line Map]"]
    for line in sorted(LINE_SEQUENCES):
        lines.append(f"Line {line}: " + " - ".join(LINE_SEQUENCES[line]))
    lines.append("(Stations with the same name are transfer stations where different lines meet.)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 최단 정거장 BFS
# ---------------------------------------------------------------------------
def bfs_shortest_stops(start, end):
    if start == end:
        return 0, [(start, None)]
    parent = {start: (None, None)}
    dq = deque([start])
    while dq:
        st = dq.popleft()
        for nb, edge_line in GRAPH.get(st, []):
            if nb in parent:
                continue
            parent[nb] = (st, edge_line)
            if nb == end:
                path = []
                cur = end
                while cur is not None:
                    pst, pln = parent[cur]
                    path.append((cur, pln))
                    cur = pst
                path.reverse()
                return len(path) - 1, path
            dq.append(nb)
    return None


def route_to_text(path):
    if not path:
        return "no route"
    lines = []
    boarding = path[0][0]
    current_line = None
    for i in range(1, len(path)):
        station, line = path[i]
        if current_line is None:
            current_line = line
        elif line != current_line:
            transfer_station = path[i - 1][0]
            lines.append(f"  Line {current_line}: {boarding} -> {transfer_station}")
            lines.append(f"  (transfer at {transfer_station})")
            boarding = transfer_station
            current_line = line
    if current_line is None:
        return f"  (no movement: {path[0][0]})"
    lines.append(f"  Line {current_line}: {boarding} -> {path[-1][0]}")
    return '\n'.join(lines)


STOP_BANDS = {
    'easy':   (4, 11),
    'medium': (17, 23),
    'hard':   (20, 40),
}


def generate_puzzle(difficulty='easy', seed=None):
    rng = random.Random(seed if seed is not None else random.randint(0, 10**9))
    lo, hi = STOP_BANDS[difficulty]

    for _ in range(800):
        start, end = rng.sample(ALL_STATIONS, 2)
        result = bfs_shortest_stops(start, end)
        if result is None:
            continue
        stops, route = result
        if not (lo <= stops <= hi):
            continue

        question = f"""Read the subway line map below and find the minimum number of stops from the start station to the destination.

{network_text()}

[Problem]
Start station: {start}
Destination: {end}

Moving from one station to an adjacent station counts as 1 stop.
What is the minimum number of stops from the start to the destination?
End your solution with a line in the exact format: `Answer: N stations`."""

        answer = f"{stops} stations"
        solution = (f"[Shortest route ({stops} stops)]\n{route_to_text(route)}\n\n"
                    f"Number of stops: {stops}")

        return {
            'question': question,
            'answer': answer,
            'solution': solution,
            'difficulty': difficulty,
            'meta': {'start': start, 'end': end, 'stops': stops},
        }

    return None


TASK = 'subway_en'


def create_dataset_files(num_questions=100):
    from collections import Counter
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    difficulties = ['easy', 'medium', 'hard']
    data_dir = PROJECT_ROOT / 'data' / 'jsonl'
    data_dir.mkdir(parents=True, exist_ok=True)
    answer_cap = max(4, num_questions // 3)

    total = 0
    for diff in difficulties:
        rows = []
        seen = set()
        ans_count = Counter()
        s = 0
        while len(rows) < num_questions and s < num_questions * 400 + 4000:
            item = generate_puzzle(diff, seed=s)
            s += 1
            if item is None:
                continue
            key = item['question']
            if key in seen:
                continue
            if ans_count[item['answer']] >= answer_cap:
                continue
            seen.add(key)
            ans_count[item['answer']] += 1
            idx = len(rows)
            rows.append({
                'id': f'{TASK}_{diff}_{idx}',
                'difficulty': diff,
                'task': TASK,
                'question': item['question'],
                'answer': item['answer'],
                'solution': item['solution'],
            })

        path = data_dir / f'{TASK}_{diff}.jsonl'
        with open(path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        total += len(rows)
        print(f"[{diff}] {len(rows)} generated -> {path}")

    print(f"total {total} generated")
    return total


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Subway shortest-stops puzzle generator (EN)')
    parser.add_argument('--num', type=int, default=100)
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    if args.demo:
        for diff in ['easy', 'medium', 'hard']:
            p = generate_puzzle(diff, seed=42)
            print(f"\n{'='*60}\n[{diff.upper()}]")
            print(p['question'])
            print(f"\nAnswer: {p['answer']}")
            print(f"Solution:\n{p['solution']}")
    else:
        create_dataset_files(args.num)
