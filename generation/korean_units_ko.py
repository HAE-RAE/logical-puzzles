"""
한국 전통 단위 환산 퍼즐 생성기 (한국어)

되·말·섬(부피), 푼·치·자·장(길이), 돈·냥·근·관(무게), 평·단·정보(넓이) 등
한국 전통 도량형을 **문제에 함께 제시된 환산표만 보고** 기준 단위로 환산/합산하는 퍼즐.

실제 단위 지식이 아니라 주어진 표로만 풀게 하는 자기완결형. 난이도는 단위 항 수,
어색한 환산비(근=16냥, 단=300평), 수 크기, 합산할 수량 개수로 스케일한다.

[난이도]
- easy:   1수량·2~3항, ×10 환산만 (부피/길이/무게 냥·돈) — 1~2단계 산술
- medium: 1수량·2~4항, 비-십진 비율(무게 근=16냥, 넓이 단=300평) 포함
- hard:   3수량 합산·2~4항, 관·근 등 큰 단위 + 합산

주의: 본질이 다단계 산술이라 frontier 모델(예: Gemini-3-flash)은 cranked hard까지도
거의 100%로 포화된다(난이도 분리 안 됨). 반면 작은 모델(Qwen 0.6B~8B 등)은 다단계
산술에서 빠르게 무너져 난이도가 잘 분리되는, '작은 모델용' 퍼즐이다.
"""

import random
import json
from pathlib import Path

# 각 도량형: 기준(base) 단위 + (단위명, 기준단위 환산값) 오름차순 + 프롬프트용 환산표
SYSTEMS = {
    '부피': {
        'base': '홉',
        'units': [('홉', 1), ('되', 10), ('말', 100), ('섬', 1000)],
        'table': ['1되 = 10홉', '1말 = 10되', '1섬 = 10말'],
        'awkward': 3,   # 섬(×1000) — 큰 단위
    },
    '길이': {
        'base': '푼',
        'units': [('푼', 1), ('치', 10), ('자', 100), ('장', 1000)],
        'table': ['1치 = 10푼', '1자 = 10치', '1장 = 10자'],
        'awkward': 3,
    },
    '무게': {
        'base': '돈',
        'units': [('돈', 1), ('냥', 10), ('근', 160), ('관', 1000)],
        'table': ['1냥 = 10돈', '1근 = 16냥', '1관 = 1000돈'],
        'awkward': 2,   # 근 = 16냥 (비-십진 비율)
    },
    '넓이': {
        'base': '평',
        'units': [('평', 1), ('단', 300), ('정보', 3000)],
        'table': ['1단 = 300평', '1정보 = 10단'],
        'awkward': 1,   # 단 = 300평
    },
}


def _make_quantity(rng, system, levels, cmax):
    """levels(허용 단위 인덱스)에서 항을 골라 수량 dict{level:count} 생성."""
    k = min(len(levels), rng.randint(2, max(2, len(levels))))
    chosen = rng.sample(levels, k)
    return {lvl: rng.randint(1, cmax) for lvl in chosen}


def _base_total(system, q):
    units = SYSTEMS[system]['units']
    return sum(cnt * units[lvl][1] for lvl, cnt in q.items())


def _render_quantity(system, q):
    units = SYSTEMS[system]['units']
    parts = [f"{q[lvl]}{units[lvl][0]}" for lvl in sorted(q, reverse=True)]
    return ' '.join(parts)


def generate_puzzle(difficulty='easy', seed=None):
    rng = random.Random(seed if seed is not None else random.randint(0, 10**9))

    for _ in range(400):
        if difficulty == 'easy':
            system = rng.choice(['부피', '길이', '무게'])
            # ×10 환산만: 무게는 돈·냥(0,1), 부피·길이는 홉·되·말 / 푼·치·자(0,1,2)
            levels = [0, 1] if system == '무게' else [0, 1, 2]
            n_quant, cmax = 1, 9

        elif difficulty == 'medium':
            # 비-십진 비율 시스템(무게=근16, 넓이=단300)으로 한정해 실제 산술 부하
            system = rng.choice(['무게', '넓이'])
            nlev = len(SYSTEMS[system]['units'])
            levels = list(range(nlev))
            n_quant, cmax = 1, 20

        else:  # hard
            # 근=16냥 비율 + 여러 수량 합산. 작은 모델 기준 난이도가 잘 벌어짐.
            system = '무게'
            levels = [0, 1, 2, 3]
            n_quant, cmax = 3, 30

        quants = [_make_quantity(rng, system, levels, cmax) for _ in range(n_quant)]
        totals = [_base_total(system, q) for q in quants]

        # 난이도별 수용 조건(너무 시시하거나 단위가 1개뿐인 경우 배제)
        if any(len(q) < 2 for q in quants):
            continue
        if difficulty == 'medium':
            # 어색한 단위(근/단/×1000)가 최소 한 번은 등장하도록
            aw = SYSTEMS[system]['awkward']
            if not any(aw in q for q in quants):
                continue

        base = SYSTEMS[system]['base']
        table = '\n'.join('- ' + t for t in SYSTEMS[system]['table'])
        total = sum(totals)
        answer = f"{total}{base}"

        if n_quant == 1:
            body = (f"다음 양은 모두 몇 {base}입니까?\n\n"
                    f"  {_render_quantity(system, quants[0])}")
            sol = (f"{_render_quantity(system, quants[0])}\n"
                   f"= {total}{base}")
        else:
            labels = ['갑', '을', '병', '정', '무', '기', '경', '신'][:n_quant]
            lines = [f"  {labels[i]}: {_render_quantity(system, quants[i])}"
                     for i in range(n_quant)]
            body = ("다음 사람들이 가진 양을 모두 합하면 몇 " + base + "입니까?\n\n"
                    + '\n'.join(lines))
            sol_lines = [f"{labels[i]}: {_render_quantity(system, quants[i])} = {totals[i]}{base}"
                         for i in range(n_quant)]
            sol = '\n'.join(sol_lines) + f"\n합계 = {total}{base}"

        question = f"""아래 환산표를 보고 물음에 답하세요. (제시된 표만 사용하세요)

[{system} 환산표]
{table}

[문제]
{body}

풀이 마지막 줄에 `정답: N{base}` 형식으로 답하세요."""

        return {
            'question': question,
            'answer': answer,
            'solution': f"[환산]\n{sol}",
            'difficulty': difficulty,
            'meta': {'system': system, 'total': total, 'base': base},
        }

    return None


TASK = 'korean_units_ko'


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
    parser = argparse.ArgumentParser(description='한국 전통 단위 환산 퍼즐 생성기')
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
