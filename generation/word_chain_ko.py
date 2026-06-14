"""
끝말잇기 최장 사슬 퍼즐 생성기 (한국어)

주어진 한국어 단어들로 끝말잇기(앞 단어의 끝 글자 = 다음 단어의 첫 글자)를 할 때
만들 수 있는 **가장 긴 사슬의 길이**를 구하는 퍼즐. 각 단어는 최대 1회 사용.

본질이 최장 단순경로(longest simple path, NP-hard) 탐색이라, frontier 모델도
'어떤 사슬'은 찾지만 *최대*는 잘 놓친다(water_jug의 '최소 찾기'와 같은 결).
단어 목록을 문제에 제시하므로 어휘 암기가 아니라 순수 탐색을 측정한다.

[난이도] (제시 단어 수 / 최장 사슬 길이대역) — Gemini-3-flash 약 90/73/43%
- easy:   8단어,  3~4
- medium: 18단어, 9~12
- hard:   30단어, 16~22
"""

import random
import json
import sys
from pathlib import Path

# 단어 풀: 공개 한국어 명사 목록(generation/korean_nouns.txt)에서 로드.
#   출처: han-dle/pd-korean-noun-list-for-wordles (CC0-1.0, public domain), 2~4글자 명사 ~3091개.
# 파일이 없으면 소형 내장 목록으로 폴백(테스트/오프라인용).
_FALLBACK_POOL = [
    '가지', '감자', '개미', '과자', '구두', '구름', '구슬', '구역', '기둥', '기린',
    '기차', '나라', '나무', '나비', '누나', '다리', '도로', '도시', '도장', '도토리',
    '두부', '두유', '로마', '리본', '리어카', '마차', '마을', '머리', '모자', '무지개',
    '미로', '미소', '미술', '바다', '바람', '바지', '박수', '보리', '부산', '부자',
    '비누', '사과', '사람', '사슬', '사슴', '사자', '사진', '산소', '산수', '소금',
    '소나무', '소리', '수도', '수박', '시계', '시소', '야구', '역사', '오리', '우산',
    '우유', '우표', '유리', '의자', '자두', '자석', '자유', '장미', '전화', '제비',
    '종이', '지구', '지도', '지붕', '차도', '차표', '친구', '카드', '토끼', '포도',
    '표범', '하늘', '학교', '항구', '항아리', '호두', '호수', '화분', '화산',
]


def _load_pool():
    path = Path(__file__).with_name('korean_nouns.txt')
    if path.exists():
        words = []
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line and not line.startswith('#') and all('가' <= ch <= '힣' for ch in line):
                words.append(line)
        if words:
            return sorted(set(words))
    return sorted(set(_FALLBACK_POOL))


POOL = _load_pool()

# 전체 풀 인접: 첫 글자 인덱스로 O(1) 조회(구성용)
_BY_FIRST = {}
for _i, _w in enumerate(POOL):
    _BY_FIRST.setdefault(_w[0], []).append(_i)


def _full_neighbors(i):
    last = POOL[i][-1]
    return [j for j in _BY_FIRST.get(last, ()) if j != i]


def _build_adj(words):
    n = len(words)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and words[i][-1] == words[j][0]:
                adj[i].append(j)
    return adj


def longest_chain(words):
    """최장 단순경로 길이(단어 개수). 메모이즈 DFS(비트마스크)."""
    n = len(words)
    adj = _build_adj(words)
    memo = {}

    def dfs(v, visited):
        key = (v, visited)
        if key in memo:
            return memo[key]
        best = 1
        for u in adj[v]:
            if not (visited >> u) & 1:
                best = max(best, 1 + dfs(u, visited | (1 << u)))
        memo[key] = best
        return best

    return max(dfs(v, 1 << v) for v in range(n)) if n else 0


def _random_path(rng, length, tries=400):
    """전체 풀에서 길이 `length`의 끝말잇기 단순경로(단어 인덱스 리스트)를 구성."""
    n = len(POOL)
    for _ in range(tries):
        start = rng.randrange(n)
        path = [start]
        visited = {start}
        cur = start
        while len(path) < length:
            nbrs = [u for u in _full_neighbors(cur) if u not in visited]
            if not nbrs:
                break
            cur = rng.choice(nbrs)
            path.append(cur)
            visited.add(cur)
        if len(path) == length:
            return path
    return None


PARAMS = {
    'easy':   dict(n=8,  lo=3,  hi=4),
    'medium': dict(n=18, lo=9,  hi=12),
    'hard':   dict(n=30, lo=16, hi=22),
}


def generate_puzzle(difficulty='easy', seed=None):
    rng = random.Random(seed if seed is not None else random.randint(0, 10**9))
    p = PARAMS[difficulty]

    for _ in range(3000):
        # 백본 사슬(길이=lo)을 먼저 구성하고 방해 단어로 채운다
        backbone = _random_path(rng, p['lo'])
        if backbone is None:
            continue
        rest = [i for i in range(len(POOL)) if i not in set(backbone)]
        if len(rest) < p['n'] - p['lo']:
            continue
        distract = rng.sample(rest, p['n'] - p['lo'])
        idxs = backbone + distract
        rng.shuffle(idxs)
        words = [POOL[i] for i in idxs]

        L = longest_chain(words)
        if not (p['lo'] <= L <= p['hi']):
            continue

        wlist = ', '.join(words)
        question = f"""끝말잇기 규칙: 앞 단어의 마지막 글자와 다음 단어의 첫 글자가 같아야 이어집니다.
각 단어는 최대 한 번만 사용할 수 있습니다. (아래 목록의 단어만 사용)

[단어 목록]
{wlist}

위 단어들로 끝말잇기 사슬을 만들 때, 가장 긴 사슬은 몇 개의 단어로 이루어집니까?
풀이 마지막 줄에 `정답: N개` 형식으로 답하세요."""

        return {
            'question': question,
            'answer': f"{L}개",
            'solution': f"제시된 {p['n']}개 단어로 만들 수 있는 최장 끝말잇기 사슬의 길이 = {L}개",
            'difficulty': difficulty,
            'meta': {'n': p['n'], 'longest': L, 'words': words},
        }

    return None


TASK = 'word_chain_ko'


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
        while len(rows) < num_questions and s < num_questions * 800 + 8000:
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
    parser = argparse.ArgumentParser(description='끝말잇기 최장 사슬 퍼즐 생성기')
    parser.add_argument('--num', type=int, default=100)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()

    if args.probe:
        from collections import Counter
        for nn in (7, 9, 11, 13, 15):
            rng = random.Random(1)
            c = Counter()
            for _ in range(2000):
                c[longest_chain(rng.sample(POOL, nn))] += 1
            print(f"n={nn}: {dict(sorted(c.items()))}")
    elif args.demo:
        for diff in ['easy', 'medium', 'hard']:
            pz = generate_puzzle(diff, seed=42)
            print(f"\n{'='*60}\n[{diff.upper()}]")
            print(pz['question'])
            print(f"\n정답: {pz['answer']}")
    else:
        create_dataset_files(args.num)
