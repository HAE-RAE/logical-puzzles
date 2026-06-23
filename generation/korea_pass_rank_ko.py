"""
고개(령/재/치) 높이 순위 추론 퍼즐 (한국어) — 추론 + 한국 지식 하이브리드

korea_peak_rank(산)과 같은 설계를, 한국 고유 지형인 '고개(령·재·치·고개)'에 적용한 것.
"A는 B보다 높다" 단서로 진 고개를 후보에서 제거(추론) → 남은 외진 고개 중 실제로 가장 높은
것을 고도 지식으로 결정. 단서에서 많이 이긴 고개가 최고처럼 보이는 함정 포함.

난이도 = 후보 고개의 '외짐 정도'(별칭 수 nalt tier). 고도(GeoNames KR PASS, CC-BY)는 정답/단서
생성·검증에만 쓰고 프롬프트엔 미제공. dem 근사라 결정적 비교(1·2위)는 ≥120m 차로 둬 신뢰.
- easy   잘 알려진 고개(대관령 등/nalt≥6), c=2  → ~83%
- medium 중간(nalt 4~5), c=4                  → ~50%
- hard   가장 외짐(nalt≤3), c=6                → ~30%
정답 = 가장 높은 고개 이름.
[Gemini-3-flash 기준선] 약 83/50/30%. (고개 높이는 산보다도 덜 알려져 전반적으로 더 어려움)
"""

import csv
import json
import random
from pathlib import Path

FAMOUS = {
    '대관령', '한계령', '미시령', '진부령', '죽령', '추풍령', '이화령', '박달재',
    '만항재', '정령치', '성삼재', '곰배령', '구룡령', '운두령', '두문동재', '백복령',
    '댓재', '화방재', '조령', '벽소령', '한티재', '말티재', '저구령', '구절양장',
}


def _load():
    path = Path(__file__).with_name('korea_passes.csv')
    out = []
    with open(path, encoding='utf-8') as f:
        for row in csv.reader(f):
            if not row or row[0].startswith('#') or row[0] == 'name':
                continue
            out.append({'name': row[0], 'elev': int(row[1]),
                        'nalt': int(row[2]) if len(row) > 2 and row[2].isdigit() else 3})
    return out

PASSES = _load()
KNOWN   = [m for m in PASSES if m['name'] in FAMOUS or m['nalt'] >= 6]
MID     = [m for m in PASSES if m['name'] not in FAMOUS and m['nalt'] in (4, 5)]
OBSCURE = [m for m in PASSES if m['name'] not in FAMOUS and m['nalt'] <= 3]
POOLS = {'KNOWN': KNOWN, 'MID': MID, 'OBSCURE': OBSCURE}

PARAMS = {
    'easy':   dict(n=5,  c=2, pool='KNOWN'),
    'medium': dict(n=8,  c=4, pool='MID'),
    'hard':   dict(n=12, c=6, pool='OBSCURE'),
}


def generate_puzzle(difficulty='easy', seed=None):
    rng = random.Random(seed if seed is not None else random.randint(0, 10**9))
    p = PARAMS[difficulty]
    n, c = p['n'], p['c']
    pool = POOLS[p['pool']]

    for _ in range(4000):
        cand = rng.sample(pool, n)
        r = sorted(cand, key=lambda m: m['elev'], reverse=True)
        if r[0]['elev'] - r[1]['elev'] < 120:      # 결정적 1·2위 차(고개는 표고차가 작아 120m)
            continue
        if r[c - 1]['elev'] - r[c]['elev'] < 50:   # 후보/탈락 경계
            continue

        clues = []
        if c == 1:
            for i in range(n - 1):
                clues.append((r[i]['name'], r[i + 1]['name']))
        else:
            for i in range(c, n):
                clues.append((r[1]['name'], r[i]['name']))

        answer = r[0]['name']
        clue_lines = [f"- {a}은(는) {b}보다 높다." for a, b in clues]
        rng.shuffle(clue_lines)
        opts = [m['name'] for m in cand]
        rng.shuffle(opts)
        opt_txt = '\n'.join('- ' + o for o in opts)

        question = f"""아래 '단서'와 실제 고개 높이 지식을 함께 사용해, 보기 중 **가장 높은 고개**를 고르세요.

[단서]
{chr(10).join(clue_lines)}

[보기]
{opt_txt}

힌트: 단서에서 '다른 고개보다 낮다'고 나온 고개는 가장 높은 고개가 될 수 없습니다. 그렇게 후보를
좁힌 뒤, 남은 후보 중 실제로 가장 높은 고개를 고르세요.
풀이 마지막 줄에 `정답: <고개 이름>` 형식으로 답하세요."""

        return {
            'question': question,
            'answer': answer,
            'solution': f"가장 높은 고개 = {answer} ({r[0]['elev']}m). 후보 {c}개 중 실제 높이로 결정.",
            'difficulty': difficulty,
            'meta': {'answer': answer, 'c': c, 'options': opts},
        }

    return None


TASK = 'korea_pass_rank_ko'


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
    parser = argparse.ArgumentParser(description='고개 높이 순위 추론 퍼즐 생성기')
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
