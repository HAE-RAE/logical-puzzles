"""
물 붓기(Water Jug) 퍼즐 생성기 (한국어)

용량이 다른 물통 몇 개로 목표 용량을 만들기 위한 '최소 동작 수'를 구하는 퍼즐.
모든 물통은 비어 있는 상태에서 시작하며, BFS로 최소 동작 수(유일해)를 보장한다.

[동작]
- 채우기: 한 물통을 수도로 가득 채움
- 비우기: 한 물통을 완전히 비움
- 붓기: A→B, 받는 쪽이 가득 차거나 주는 쪽이 빌 때까지

목표: 어느 한 물통에 정확히 T리터를 담는다.

[난이도] (최소 동작 수)
- easy:   2~4번
- medium: 5~7번
- hard:   8~13번
"""

import random
import json
from collections import deque
from pathlib import Path

OP_BANDS = {
    'easy':   (2, 4),
    'medium': (5, 7),
    'hard':   (8, 13),
}


def solve_min_ops(caps, target):
    """모든 물통 0에서 시작, 어느 한 물통이 target이 되는 최소 동작 수 BFS.
    반환: (min_ops, path[(state, desc)]) 또는 None"""
    n = len(caps)
    start = tuple([0] * n)
    if target in start:
        return 0, [(start, '시작')]
    seen = {start}
    dq = deque([(start, [(start, '시작')])])
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


def _moves(state, caps):
    n = len(caps)
    out = []
    L = ['A', 'B', 'C', 'D', 'E'][:n]
    for i in range(n):
        # 채우기
        if state[i] < caps[i]:
            s = list(state); s[i] = caps[i]
            out.append((tuple(s), f'{L[i]} 채우기'))
        # 비우기
        if state[i] > 0:
            s = list(state); s[i] = 0
            out.append((tuple(s), f'{L[i]} 비우기'))
        # 붓기 i->j
        for j in range(n):
            if i == j or state[i] == 0 or state[j] == caps[j]:
                continue
            move = min(state[i], caps[j] - state[j])
            s = list(state); s[i] -= move; s[j] += move
            out.append((tuple(s), f'{L[i]}→{L[j]} 붓기'))
    return out


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
        else:  # hard
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
        cap_txt = ', '.join(f'{L[i]}={caps[i]}리터' for i in range(n))

        question = f"""물통 {n}개가 있습니다. 용량은 각각 {cap_txt}이며, 처음에는 모두 비어 있습니다.

한 번의 '동작'은 다음 중 하나입니다:
- 채우기: 한 물통을 수도로 가득 채운다
- 비우기: 한 물통을 완전히 비운다
- 붓기: 한 물통에서 다른 물통으로, 받는 쪽이 가득 차거나 주는 쪽이 빌 때까지 옮긴다

어느 한 물통에 정확히 {target}리터를 담으려면 최소 몇 번의 동작이 필요합니까?
풀이 마지막 줄에 `정답: N번` 형식으로 답하세요."""

        answer = f"{ops}번"
        sol_lines = []
        for k, (state, desc) in enumerate(path):
            st = ', '.join(f'{L[i]}={state[i]}' for i in range(n))
            if k == 0:
                sol_lines.append(f"[START] {st}")
            else:
                sol_lines.append(f"[{k}] {desc} → {st}")
        solution = '\n'.join(sol_lines) + f"\n\n최소 동작 수: {ops}번"

        return {
            'question': question,
            'answer': answer,
            'solution': solution,
            'difficulty': difficulty,
            'meta': {'caps': caps, 'target': target, 'ops': ops},
        }

    return None


TASK = 'water_jug_ko'


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
        print(f"[{diff}] {len(rows)}개 생성 → {path}")

    print(f"총 {total}개 생성 완료")
    return total


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='물 붓기 퍼즐 생성기')
    parser.add_argument('--num', type=int, default=100)
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    if args.demo:
        for diff in ['easy', 'medium', 'hard']:
            p = generate_puzzle(diff, seed=42)
            print(f"\n{'='*60}\n[{diff.upper()}]")
            print(p['question'])
            print(f"\n정답: {p['answer']}")
            print(f"풀이:\n{p['solution']}")
    else:
        create_dataset_files(args.num)
