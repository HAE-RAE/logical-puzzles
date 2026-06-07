"""
서울 지하철 경로 추론 퍼즐 생성기 (한국어)

노선도(각 호선의 역 순서)를 프롬프트에 함께 제시하고, 출발역→도착역의
'최소 정거장 수(이동 구간 수)'를 추론하게 하는 자기완결형 퍼즐.

참고: 이 그래프는 환승 지름(diameter)이 최대 2회라 '최소 환승'으로는
난이도 3단계가 분리되지 않는다(실험 확인). 그래서 거리에 따라 자연히
다양·증가하는 '최소 정거장 수'를 정답 지표로 사용한다.

[난이도] (최소 정거장 수 대역) — Gemini-3-flash 기준 약 85/72/43%로 캘리브레이션
- easy:   4~11 정거장
- medium: 14~19 정거장
- hard:   20~40 정거장
"""

import random
import json
from pathlib import Path
from collections import deque

# ============================================================================
# 서울 지하철 그래프 (간소화)
# ============================================================================
# 형식: (역A, 역B, 호선번호)

EDGES = [
    # 1호선
    ('소요산', '동두천', 1), ('동두천', '보산', 1), ('보산', '동두천중앙', 1),
    ('동두천중앙', '지행', 1), ('지행', '덕정', 1), ('덕정', '덕계', 1),
    ('덕계', '양주', 1), ('양주', '녹양', 1), ('녹양', '가능', 1),
    ('가능', '의정부', 1), ('의정부', '회룡', 1), ('회룡', '망월사', 1),
    ('망월사', '도봉산', 1), ('도봉산', '도봉', 1), ('도봉', '방학', 1),
    ('방학', '창동', 1), ('창동', '녹천', 1), ('녹천', '월계', 1),
    ('월계', '광운대', 1), ('광운대', '석계', 1), ('석계', '신이문', 1),
    ('신이문', '외대앞', 1), ('외대앞', '회기', 1), ('회기', '청량리', 1),
    ('청량리', '제기동', 1), ('제기동', '신설동', 1), ('신설동', '동묘앞', 1),
    ('동묘앞', '동대문', 1), ('동대문', '종로5가', 1), ('종로5가', '종로3가', 1),
    ('종로3가', '종각', 1), ('종각', '시청', 1), ('시청', '서울역', 1),
    ('서울역', '남영', 1), ('남영', '용산', 1),

    # 2호선 (순환선 일부)
    ('시청', '을지로입구', 2), ('을지로입구', '을지로3가', 2),
    ('을지로3가', '동대문역사문화공원', 2), ('동대문역사문화공원', '신당', 2),
    ('신당', '상왕십리', 2), ('상왕십리', '왕십리', 2), ('왕십리', '한양대', 2),
    ('한양대', '뚝섬', 2), ('뚝섬', '성수', 2), ('성수', '건대입구', 2),
    ('건대입구', '구의', 2), ('구의', '강변', 2), ('강변', '잠실나루', 2),
    ('잠실나루', '잠실', 2), ('잠실', '종합운동장', 2),
    ('종합운동장', '삼성', 2), ('삼성', '선릉', 2), ('선릉', '역삼', 2),
    ('역삼', '강남', 2), ('강남', '교대', 2), ('교대', '서초', 2),
    ('서초', '방배', 2), ('방배', '사당', 2), ('사당', '낙성대', 2),
    ('낙성대', '서울대입구', 2), ('서울대입구', '봉천', 2), ('봉천', '신림', 2),
    ('신림', '신대방', 2), ('신대방', '구로디지털단지', 2),
    ('구로디지털단지', '대림', 2), ('대림', '신도림', 2), ('신도림', '문래', 2),
    ('문래', '영등포구청', 2), ('영등포구청', '당산', 2), ('당산', '합정', 2),
    ('합정', '홍대입구', 2), ('홍대입구', '신촌', 2), ('신촌', '이대', 2),
    ('이대', '아현', 2), ('아현', '충정로', 2), ('충정로', '시청', 2),

    # 3호선
    ('대화', '주엽', 3), ('주엽', '정발산', 3), ('정발산', '마두', 3),
    ('마두', '백석', 3), ('백석', '대곡', 3), ('대곡', '화정', 3),
    ('화정', '원당', 3), ('원당', '원흥', 3), ('원흥', '삼송', 3),
    ('삼송', '지축', 3), ('지축', '구파발', 3), ('구파발', '연신내', 3),
    ('연신내', '불광', 3), ('불광', '녹번', 3), ('녹번', '홍제', 3),
    ('홍제', '무악재', 3), ('무악재', '독립문', 3), ('독립문', '경복궁', 3),
    ('경복궁', '안국', 3), ('안국', '종로3가', 3), ('종로3가', '을지로3가', 3),
    ('을지로3가', '충무로', 3), ('충무로', '동대입구', 3),
    ('동대입구', '약수', 3), ('약수', '금호', 3), ('금호', '옥수', 3),
    ('옥수', '압구정', 3), ('압구정', '신사', 3), ('신사', '잠원', 3),
    ('잠원', '고속터미널', 3), ('고속터미널', '교대', 3), ('교대', '남부터미널', 3),
    ('남부터미널', '양재', 3), ('양재', '매봉', 3), ('매봉', '도곡', 3),
    ('도곡', '대치', 3), ('대치', '학여울', 3), ('학여울', '대청', 3),
    ('대청', '일원', 3), ('일원', '수서', 3), ('수서', '가락시장', 3),
    ('가락시장', '경찰병원', 3), ('경찰병원', '오금', 3),

    # 4호선
    ('당고개', '상계', 4), ('상계', '노원', 4), ('노원', '창동', 4),
    ('창동', '쌍문', 4), ('쌍문', '수유', 4), ('수유', '미아', 4),
    ('미아', '미아사거리', 4), ('미아사거리', '길음', 4), ('길음', '성신여대입구', 4),
    ('성신여대입구', '한성대입구', 4), ('한성대입구', '혜화', 4),
    ('혜화', '동대문', 4), ('동대문', '동대문역사문화공원', 4),
    ('동대문역사문화공원', '충무로', 4), ('충무로', '명동', 4),
    ('명동', '회현', 4), ('회현', '서울역', 4), ('서울역', '숙대입구', 4),
    ('숙대입구', '삼각지', 4), ('삼각지', '신용산', 4), ('신용산', '이촌', 4),
    ('이촌', '동작', 4), ('동작', '총신대입구', 4), ('총신대입구', '사당', 4),
    ('사당', '남태령', 4),

    # 5호선
    ('방화', '개화산', 5), ('개화산', '김포공항', 5), ('김포공항', '송정', 5),
    ('송정', '마곡', 5), ('마곡', '발산', 5), ('발산', '우장산', 5),
    ('우장산', '화곡', 5), ('화곡', '까치산', 5), ('까치산', '신정', 5),
    ('신정', '목동', 5), ('목동', '오목교', 5), ('오목교', '양평', 5),
    ('양평', '영등포구청', 5), ('영등포구청', '영등포시장', 5),
    ('영등포시장', '신길', 5), ('신길', '여의도', 5), ('여의도', '여의나루', 5),
    ('여의나루', '마포', 5), ('마포', '공덕', 5), ('공덕', '애오개', 5),
    ('애오개', '충정로', 5), ('충정로', '서대문', 5), ('서대문', '광화문', 5),
    ('광화문', '종로3가', 5), ('종로3가', '을지로4가', 5),
    ('을지로4가', '동대문역사문화공원', 5), ('동대문역사문화공원', '청구', 5),
    ('청구', '신금호', 5), ('신금호', '행당', 5), ('행당', '왕십리', 5),
    ('왕십리', '마장', 5), ('마장', '답십리', 5), ('답십리', '장한평', 5),
    ('장한평', '군자', 5), ('군자', '아차산', 5), ('아차산', '광나루', 5),
    ('광나루', '천호', 5), ('천호', '강동', 5),

    # 9호선 (주요 구간)
    ('개화', '김포공항', 9), ('김포공항', '공항시장', 9), ('공항시장', '신방화', 9),
    ('신방화', '마곡나루', 9), ('마곡나루', '양천향교', 9), ('양천향교', '가양', 9),
    ('가양', '증미', 9), ('증미', '등촌', 9), ('등촌', '염창', 9),
    ('염창', '신목동', 9), ('신목동', '선유도', 9), ('선유도', '당산', 9),
    ('당산', '국회의사당', 9), ('국회의사당', '여의도', 9), ('여의도', '샛강', 9),
    ('샛강', '노량진', 9), ('노량진', '노들', 9), ('노들', '흑석', 9),
    ('흑석', '동작', 9), ('동작', '구반포', 9), ('구반포', '신반포', 9),
    ('신반포', '고속터미널', 9), ('고속터미널', '사평', 9), ('사평', '신논현', 9),
    ('신논현', '언주', 9), ('언주', '선정릉', 9), ('선정릉', '삼성중앙', 9),
    ('삼성중앙', '봉은사', 9), ('봉은사', '종합운동장', 9),
    ('종합운동장', '삼전', 9), ('삼전', '석촌고분', 9), ('석촌고분', '석촌', 9),
    ('석촌', '송파나루', 9), ('송파나루', '한성백제', 9),
    ('한성백제', '올림픽공원', 9), ('올림픽공원', '둔촌오륜', 9),
    ('둔촌오륜', '중앙보훈병원', 9),
]

# ============================================================================
# 그래프 빌드
# ============================================================================

def build_graph():
    g = {}
    for a, b, line in EDGES:
        g.setdefault(a, []).append((b, line))
        g.setdefault(b, []).append((a, line))
    return g

GRAPH = build_graph()
ALL_STATIONS = list(GRAPH.keys())


def build_line_sequences():
    """EDGES(연속 쌍)로부터 노선별 정렬된 역 시퀀스를 복원."""
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

# 같은 이름의 역이 여러 노선에 나타나면 그 역이 환승역
def _station_lines():
    m = {}
    for line, seq in LINE_SEQUENCES.items():
        for s in seq:
            m.setdefault(s, set()).add(line)
    return m

STATION_LINES = _station_lines()


def network_text():
    """프롬프트에 넣을 노선도 텍스트(자기완결형 문제용)."""
    lines = ["[지하철 노선도]"]
    for line in sorted(LINE_SEQUENCES):
        seq = LINE_SEQUENCES[line]
        lines.append(f"{line}호선: " + " - ".join(seq))
    lines.append("※ 같은 이름의 역은 서로 다른 노선이 만나는 환승역입니다.")
    return "\n".join(lines)

# 주요 환승역
TRANSFER_STATIONS = {
    '종로3가': [1, 3, 5],
    '동대문역사문화공원': [2, 4, 5],
    '시청': [1, 2],
    '서울역': [1, 4],
    '충무로': [3, 4],
    '충정로': [2, 5],
    '동대문': [1, 4],
    '사당': [2, 4],
    '교대': [2, 3],
    '고속터미널': [3, 9],
    '노원': [4, 7],
    '왕십리': [2, 5],
    '영등포구청': [2, 5],
    '여의도': [5, 9],
    '당산': [2, 9],
    '합정': [2, 6],
    '김포공항': [5, 9],
    '동작': [4, 9],
    '종합운동장': [2, 9],
    '건대입구': [2, 7],
    '군자': [5, 7],
    '창동': [1, 4],
    '신설동': [1, 2],
    '을지로3가': [2, 3],
    '을지로입구': [2],
    '석계': [1, 6],
}


# ============================================================================
# BFS 최단 경로 (최소 정거장 수)
# ============================================================================

def bfs_shortest_stops(start, end):
    """
    최소 정거장(이동 구간) 경로 탐색 (무가중 BFS).

    한 역에서 인접 역으로 가는 것을 1정거장으로 센다. 정답 = 이동 구간 수
    (= len(path) - 1). 최단이므로 같은 노선 왕복 같은 퇴화 경로가 없다.

    반환: (stops, path[(station, arrival_line)]) 또는 None
    """
    if start == end:
        return 0, [(start, None)]

    parent = {start: (None, None)}  # station -> (prev_station, line_used)
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
    """경로를 텍스트로 변환.

    path[0]의 노선은 None(출발역, 미탑승)이며, 각 역의 노선은 그 역에
    '도착한' 노선을 의미한다. 노선이 바뀌는 지점의 직전 역이 곧 환승역이고
    새 노선의 탑승역이 된다. (출발역 누락/환승역 오프바이원 버그 수정)
    """
    if not path:
        return "경로 없음"
    lines = []
    boarding = path[0][0]      # 현재 노선의 탑승역
    current_line = None
    for i in range(1, len(path)):
        station, line = path[i]
        if current_line is None:
            current_line = line
        elif line != current_line:
            transfer_station = path[i - 1][0]
            lines.append(f"  {current_line}호선: {boarding} → {transfer_station}")
            lines.append(f"  ↔ {transfer_station}역 환승")
            boarding = transfer_station
            current_line = line
    if current_line is None:
        return f"  (이동 없음: {path[0][0]})"
    lines.append(f"  {current_line}호선: {boarding} → {path[-1][0]}")
    return '\n'.join(lines)


# ============================================================================
# 퍼즐 생성
# ============================================================================

# 난이도별 최소 정거장 수(이동 구간 수) 목표 대역.
# 코어 환승 지름이 2회로 붕괴해 '최소 환승'은 3단계로 못 벌어지므로,
# 거리에 따라 자연히 다양·증가하는 '최소 정거장 수'를 정답으로 쓴다.
STOP_BANDS = {
    'easy':   (4, 11),
    'medium': (14, 19),
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

        question = f"""아래 지하철 노선도를 보고, 출발역에서 도착역까지 최소 몇 정거장을 가야 하는지 구하세요.

{network_text()}

[문제]
출발역: {start}
도착역: {end}

한 역에서 바로 옆 역으로 이동하는 것을 1정거장으로 셉니다.
출발역에서 도착역까지 가는 최소 정거장 수는 몇 개입니까?
풀이 마지막 줄에 `정답: N정거장` 형식으로 답하세요."""

        answer = f"{stops}정거장"

        solution = (f"[최단 경로 ({stops}정거장)]\n{route_to_text(route)}\n\n"
                    f"정거장 수: {stops}")

        return {
            'question': question,
            'answer': answer,
            'solution': solution,
            'difficulty': difficulty,
            'meta': {
                'start': start,
                'end': end,
                'stops': stops,
            }
        }

    return None


# ============================================================================
# 데이터셋 생성
# ============================================================================

TASK = 'subway_ko'


def create_dataset_files(num_questions=100):
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    difficulties = ['easy', 'medium', 'hard']

    data_dir = PROJECT_ROOT / 'data' / 'jsonl'
    data_dir.mkdir(parents=True, exist_ok=True)

    from collections import Counter
    # 한 정답값이 파일을 지배해 '최빈값 찍기'로 풀리는 것을 막는 상한
    answer_cap = max(4, num_questions // 3)

    total = 0
    for diff in difficulties:
        rows = []
        seen = set()
        ans_count = Counter()
        s = 0
        # 무한 루프 방지: 충분히 큰 시드 상한
        while len(rows) < num_questions and s < num_questions * 400 + 4000:
            item = generate_puzzle(diff, seed=s)
            s += 1
            if item is None:
                continue
            key = item['question']  # 노선도 prefix가 동일하므로 전체로 중복 판정
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
    parser = argparse.ArgumentParser(description='서울 지하철 경로 퍼즐 생성기')
    parser.add_argument('--num', type=int, default=100)
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    if args.demo:
        for diff in ['easy', 'medium', 'hard']:
            p = generate_puzzle(diff, seed=42)
            if p:
                print(f"\n{'='*60}\n[{diff.upper()}]")
                print(p['question'])
                print(f"\n정답: {p['answer']}")
                print(f"풀이:\n{p['solution']}")
    else:
        create_dataset_files(args.num)
