"""
Water jug puzzle generator (English).

물 붓기 KO 버전과 동일한 로직(BFS 최소 동작 수)·난이도 대역을 영어 문구로 옮긴 판.

- 정답 형식: "N moves" (정수)
- 정답 산출: 모든 물통 0에서 시작하는 BFS 최소 동작 수 (유일해)

[난이도] (최소 동작 수)
- easy:   2 ~ 4
- medium: 5 ~ 7
- hard:   8 ~ 13
"""

import json
import random
from collections import deque
from pathlib import Path

OP_BANDS = {
    'easy':   (2, 4),
    'medium': (5, 7),
    'hard':   (8, 13),
}


def _moves(state, caps):
    n = len(caps)
    out = []
    L = ['A', 'B', 'C', 'D', 'E'][:n]
    for i in range(n):
        if state[i] < caps[i]:
            s = list(state); s[i] = caps[i]
            out.append((tuple(s), f'Fill {L[i]}'))
        if state[i] > 0:
            s = list(state); s[i] = 0
            out.append((tuple(s), f'Empty {L[i]}'))
        for j in range(n):
            if i == j or state[i] == 0 or state[j] == caps[j]:
                continue
            move = min(state[i], caps[j] - state[j])
            s = list(state); s[i] -= move; s[j] += move
            out.append((tuple(s), f'Pour {L[i]}->{L[j]}'))
    return out


def solve_min_ops(caps, target):
    n = len(caps)
    start = tuple([0] * n)
    if target in start:
        return 0, [(start, 'start')]
    seen = {start}
    dq = deque([(start, [(start, 'start')])])
    while dq:
        state, path = dq.popleft()
        for nxt, desc in _moves(state, caps):
            if nxt in seen:
                continue
            new_path = path + [(nxt, desc)]
            if target in nxt:
                return len(new_path) - 1, new_path
            seen.add(nxt)
            dq.append((nxt, new_path))
    return None


def generate_puzzle(difficulty='easy', seed=None):
    rng = random.Random(seed if seed is not None else random.randint(0, 10**9))
    lo, hi = OP_BANDS[difficulty]

    for _ in range(2000):
        if difficulty == 'easy':
            n = 2
            caps = sorted(rng.sample(range(3, 10), n), reverse=True)
        elif difficulty == 'medium':
            n = rng.choice([2, 3])
            caps = sorted(rng.sample(range(4, 13), n), reverse=True)
        else:
            n = 3
            caps = sorted(rng.sample(range(5, 16), n), reverse=True)

        target = rng.randint(1, caps[0] - 1)
        res = solve_min_ops(caps, target)
        if res is None:
            continue
        ops, path = res
        if not (lo <= ops <= hi):
            continue

        L = ['A', 'B', 'C', 'D', 'E'][:n]
        cap_txt = ', '.join(f'{L[i]}={caps[i]} liters' for i in range(n))

        question = f"""There are {n} jugs. Their capacities are {cap_txt}. All jugs start empty.

One "move" is exactly one of:
- Fill: fill a jug to the top from the tap
- Empty: empty a jug completely
- Pour: pour from one jug into another until the receiving jug is full or the pouring jug is empty

What is the minimum number of moves to get exactly {target} liters in some jug?
End your solution with a line in the exact format: `Answer: N moves`."""

        answer = f"{ops} moves"
        sol_lines = []
        for k, (state, desc) in enumerate(path):
            st = ', '.join(f'{L[i]}={state[i]}' for i in range(n))
            if k == 0:
                sol_lines.append(f"[START] {st}")
            else:
                sol_lines.append(f"[{k}] {desc} -> {st}")
        solution = '\n'.join(sol_lines) + f"\n\nMinimum moves: {ops}"

        return {
            'question': question,
            'answer': answer,
            'solution': solution,
            'difficulty': difficulty,
            'meta': {'caps': caps, 'target': target, 'ops': ops},
        }

    return None


TASK = 'water_jug_en'


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
        while len(rows) < num_questions and s < num_questions * 600 + 6000:
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
    parser = argparse.ArgumentParser(description='Water jug puzzle generator (EN)')
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
