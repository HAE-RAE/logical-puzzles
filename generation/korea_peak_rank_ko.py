"""
산 높이 순위 추론 퍼즐 (한국어) — 추론 + 한국 지식 하이브리드

"A는 B보다 높다" 형식의 단서 몇 개와 산 목록을 주고, 가장 높은 산을 묻는다.
- 추론: 단서에서 '다른 산보다 낮다'고 적힌 산은 정답일 수 없으므로 후보에서 제외한다.
- 지식: 그렇게 남은 후보 중 실제로 가장 높은 산은, 단서로는 결정되지 않으므로
        대한민국 산의 실제 높이를 알아야 고를 수 있다.

난이도 = 후보로 남는 산들의 '외짐 정도'(별칭 수 nalt = 문서화/유명도 proxy). 후보 수(c)만
늘리면 ~67%에서 막히고(1등만 맞히면 되니까), 후보 산을 더 외지게 할수록 Gemini가 높이를
몰라 급격히 어려워진다.
- easy   잘 알려진 산(nalt≥6 또는 유명산), c=2  → ~93%
- medium 중간(nalt 4~5), c=4                  → ~70%
- hard   가장 외짐(nalt≤3), c=5               → ~40%

고도(GeoNames KR, CC-BY)는 정답/단서 생성·검증에만 쓰고 프롬프트엔 넣지 않는다.
dem 근사라 결정적 비교(후보 1위 vs 2위)는 ≥150m 차로 둬 신뢰한다.
정답 = 가장 높은 산 이름.
[Gemini-3-flash 기준선] 약 93/70/40%.
"""

import csv
import json
import random
from pathlib import Path

FAMOUS = {
    '한라산', '지리산', '설악산', '덕유산', '태백산', '소백산', '오대산', '북한산',
    '관악산', '무등산', '속리산', '가야산', '월악산', '치악산', '계룡산', '팔공산',
    '주왕산', '내장산', '도봉산', '금정산', '월출산', '두타산', '명성산', '화악산',
    '운악산', '백운산', '마이산', '청계산', '수락산', '불암산', '감악산', '주흘산',
}


def _load():
    path = Path(__file__).with_name('korea_mountains.csv')
    out = []
    with open(path, encoding='utf-8') as f:
        for row in csv.reader(f):
            if not row or row[0].startswith('#') or row[0] == 'name':
                continue
            out.append({'name': row[0], 'elev': int(row[1]), 'nalt': int(row[2]) if len(row)>2 and row[2].isdigit() else 3})
    return out

MTS = _load()
KNOWN   = [m for m in MTS if m['name'] in FAMOUS or m['nalt'] >= 6]   # 잘 알려짐/문서화 많음
MID     = [m for m in MTS if m['name'] not in FAMOUS and m['nalt'] in (4, 5)]
OBSCURE = [m for m in MTS if m['name'] not in FAMOUS and m['nalt'] <= 3]  # 별칭 최소=가장 외짐
POOLS = {'KNOWN': KNOWN, 'MID': MID, 'OBSCURE': OBSCURE}

PARAMS = {
    'easy':   dict(n=5,  c=2, pool='KNOWN'),
    'medium': dict(n=8,  c=4, pool='MID'),
    'hard':   dict(n=10, c=5, pool='OBSCURE'),
}


def generate_puzzle(difficulty='easy', seed=None):
    rng = random.Random(seed if seed is not None else random.randint(0, 10**9))
    p = PARAMS[difficulty]
    n, c = p['n'], p['c']

    for _ in range(4000):
        cand = rng.sample(POOLS[p['pool']], n)
        r = sorted(cand, key=lambda m: m['elev'], reverse=True)  # r[0]=최고
        # 신뢰성: 결정적 1·2위 차 ≥150m, 전체 연속 차 ≥50m
        if r[0]['elev'] - r[1]['elev'] < 150:
            continue
        if r[c - 1]['elev'] - r[c]['elev'] < 50:
            continue

        clues = []
        if c == 1:
            # 사슬: r0>r1>...>r(n-1) → 추론만으로 최고가 r0로 확정
            for i in range(n - 1):
                clues.append((r[i]['name'], r[i + 1]['name']))
        else:
            # r[c:](탈락)을 후보 r[1](비-최고)로 눌러 제거 → 미지배 후보 = r[0..c-1]
            for i in range(c, n):
                clues.append((r[1]['name'], r[i]['name']))

        answer = r[0]['name']
        clue_lines = [f"- {a}은(는) {b}보다 높다." for a, b in clues]
        rng.shuffle(clue_lines)
        opts = [m['name'] for m in cand]
        rng.shuffle(opts)
        opt_txt = '\n'.join('- ' + o for o in opts)

        question = f"""아래 '단서'와 실제 산 높이 지식을 함께 사용해, 보기 중 **가장 높은 산**을 고르세요.

[단서]
{chr(10).join(clue_lines)}

[보기]
{opt_txt}

힌트: 단서에서 '다른 산보다 낮다'고 나온 산은 가장 높은 산이 될 수 없습니다. 그렇게 후보를
좁힌 뒤, 남은 후보 중 실제로 가장 높은 산을 고르세요.
풀이 마지막 줄에 `정답: <산 이름>` 형식으로 답하세요."""

        return {
            'question': question,
            'answer': answer,
            'solution': f"가장 높은 산 = {answer} ({r[0]['elev']}m). 후보 {c}개 중 실제 높이로 결정.",
            'difficulty': difficulty,
            'meta': {'answer': answer, 'c': c, 'options': opts},
        }

    return None


TASK = 'korea_peak_rank_ko'


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
        print(f"[{diff}] {len(rows)}개 생성 → {path}")

    print(f"총 {total}개 생성 완료")
    return total


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='산 높이 순위 추론 퍼즐 생성기')
    parser.add_argument('--num', type=int, default=100)
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    if args.demo:
        for diff in ['easy', 'medium', 'hard']:
            pz = generate_puzzle(diff, seed=42)
            print(f"\n{'='*60}\n[{diff.upper()}]"); print(pz['question'])
            print("정답:", pz['answer'], "|", pz['solution'])
    else:
        create_dataset_files(args.num)
