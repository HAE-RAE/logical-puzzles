"""Causal DAG (Directed Acyclic Graph) Reasoning Puzzle Generator - Korean Version
[진행도] ☑ 완료
[파일명] causal_dag_korean.py
[목적] 한국어 기반 인과관계 DAG 추론 퍼즐 생성

한국어로 사건-원인 관계를 표현하고 시간 전파를 추론하는 퍼즐을 생성합니다.

주요 기능:
1. DAG 기반 사건 그래프 (순환 없음)
2. 시간 지연 인과 관계
3. 최단 경로 추론 (Dijkstra)
4. 유일 해 보장 (결정적 그래프)
"""

import random
import heapq
import json
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class EventType(Enum):
    """사건 카테고리"""
    TECHNICAL = "기술"
    BUSINESS = "비즈니스"
    ENVIRONMENTAL = "환경"
    OPERATIONAL = "운영"


@dataclass
class CausalEdge:
    """사건 간 인과 관계"""
    from_event: str
    to_event: str
    delay: int  # 분 단위 시간 지연
    from_events: Optional[List[str]] = None  # 다중 전제 조건 (AND용)
    condition: str = 'OR'  # 'OR' 또는 'AND'
    
    def __repr__(self):
        if self.from_events and len(self.from_events) > 1:
            cond = ' 그리고 ' if self.condition == 'AND' else ' 또는 '
            return f"[{cond.join(self.from_events)}] → {self.to_event} (+{self.delay}분)"
        return f"{self.from_event} → {self.to_event} (+{self.delay}분)"


@dataclass
class Event:
    """사건 노드"""
    id: str
    name: str
    description: str
    event_type: EventType
    
    def __repr__(self):
        return f"{self.id}: {self.name}"


@dataclass
class CausalPuzzle:
    """완전한 인과관계 추론 퍼즐"""
    events: Dict[str, Event]
    edges: List[CausalEdge]
    trigger: str
    trigger_time: int
    target_event: str
    answer: int
    difficulty: str
    
    def to_dict(self):
        """JSON 직렬화를 위한 딕셔너리 변환"""
        return {
            'events': {k: {
                'id': v.id,
                'name': v.name,
                'description': v.description,
                'event_type': v.event_type.value
            } for k, v in self.events.items()},
            'edges': [{
                'from_event': e.from_event,
                'to_event': e.to_event,
                'delay': e.delay,
                'from_events': e.from_events,
                'condition': e.condition
            } for e in self.edges],
            'trigger': self.trigger,
            'trigger_time': self.trigger_time,
            'target': self.target_event,
            'answer': self.answer,
            'difficulty': self.difficulty
        }


class CausalPuzzleGenerator:
    """인과 DAG 추론 퍼즐 생성기"""
    
    def __init__(self):
        """사건 템플릿 초기화"""
        self.event_templates = {
            EventType.TECHNICAL: [
                ('전력차단', '주 전력망 장애 발생'),
                ('서버다운', '애플리케이션 서버 중단'),
                ('데이터베이스장애', '데이터베이스 서비스 응답 중지'),
                ('네트워크장애', '네트워크 연결 끊김'),
                ('디스크포화', '저장 공간 100% 도달'),
                ('메모리누수', '애플리케이션 메모리 사용량 급증'),
                ('백업실패', '자동 백업 프로세스 실패'),
                ('보안침해', '무단 접근 탐지'),
                ('API타임아웃', '외부 API 응답 중지'),
                ('캐시만료', '캐시 무효화 발생'),
            ],
            EventType.BUSINESS: [
                ('주문접수', '고객 신규 주문 발생'),
                ('결제완료', '결제 트랜잭션 완료'),
                ('재고부족', '재고 수준이 임계치 이하로 하락'),
                ('배송지연', '배송 일정 연기'),
                ('고객불만', '고객 지원 티켓 생성'),
                ('환불처리', '고객에게 금액 반환'),
                ('가격변경', '제품 가격 업데이트'),
                ('프로모션시작', '마케팅 캠페인 시작'),
                ('계약체결', '법적 계약 최종 확정'),
                ('청구서발송', '청구 문서 생성'),
            ],
            EventType.ENVIRONMENTAL: [
                ('폭우', '시간당 50mm 이상 강수량'),
                ('도로침수', '수위 상승으로 교통 차단'),
                ('교통체증', '차량 정체 발생'),
                ('전력급등', '전력망 전압 급등'),
                ('지진', '지진 활동 감지'),
                ('폭염', '기온 35도 이상'),
                ('폭풍경보', '기상 경보 발령'),
                ('폭설', '적설량 누적 시작'),
                ('강풍피해', '강풍으로 인한 기반 시설 손상'),
                ('가뭄', '수자원 부족'),
            ],
            EventType.OPERATIONAL: [
                ('정기점검', '계획된 시스템 유지보수 시작'),
                ('인력부족', '가용 인력이 최소치 이하'),
                ('설비고장', '핵심 설비 작동 중단'),
                ('품질문제', '제품 결함 발견'),
                ('공급망중단', '협력사 납품 지연'),
                ('용량초과', '최대 처리량 초과'),
                ('정책변경', '신규 운영 규정 시행'),
                ('검사불합격', '규정 준수 검사 미통과'),
                ('교육완료', '직원 자격 인증 취득'),
                ('시스템업그레이드', '소프트웨어 버전 업데이트'),
            ]
        }
    
    def generate_puzzle(self, difficulty: str, seed: Optional[int] = None) -> CausalPuzzle:
        """
        인과관계 추론 퍼즐 생성
        
        Args:
            difficulty: 'Easy', 'Medium', 또는 'Hard'
            seed: 재현성을 위한 난수 시드
        
        Returns:
            유일 해를 갖는 CausalPuzzle
        """
        if seed is not None:
            random.seed(seed)
        
        # 난이도 설정
        config = {
            'Easy': {
                'num_events': random.randint(18, 22),
                'edge_density': 0.50,
                'delay_range': (15, 80),
                'max_out_degree': 3,
                'and_probability': 0.4,
            },
            'Medium': {
                'num_events': random.randint(45, 55),
                'edge_density': 0.70,
                'delay_range': (30, 150),
                'max_out_degree': 5,
                'and_probability': 0.92,
            },
            'Hard': {
                'num_events': random.randint(80, 100),
                'edge_density': 0.80,
                'delay_range': (50, 250),
                'max_out_degree': 7,
                'and_probability': 0.99,
            }
        }[difficulty]
        
        max_attempts = 100
        for attempt in range(max_attempts):
            try:
                # 사건 생성
                events = self._generate_events(config['num_events'])
                
                # 인과 그래프 생성 (DAG)
                edges = self._generate_causal_graph(events, config)
                
                if not edges:
                    continue
                
                # 트리거 선택 (진입 차수 0인 노드)
                in_degree = self._calculate_in_degree(events, edges)
                possible_triggers = [e_id for e_id, degree in in_degree.items() 
                                    if degree == 0]
                
                if not possible_triggers:
                    continue
                
                trigger = random.choice(possible_triggers)
                trigger_time = random.randint(0, 60)
                
                # 도달 시간 계산
                reach_times = self._calculate_reach_times(events, edges, trigger, trigger_time)
                
                # 타겟 선택 (도달 가능한 사건, 트리거 제외)
                reachable = [e_id for e_id, time in reach_times.items() 
                           if time < float('inf') and e_id != trigger]
                
                if not reachable:
                    continue
                
                # 너무 가깝거나 너무 먼 사건 제외
                target_candidates = sorted(reachable, 
                                         key=lambda e: reach_times[e])
                
                # 중간 범위에서 선택
                mid_start = len(target_candidates) // 3
                mid_end = 2 * len(target_candidates) // 3
                if mid_end > mid_start:
                    target_event = random.choice(target_candidates[mid_start:mid_end])
                else:
                    target_event = random.choice(target_candidates)
                
                answer = reach_times[target_event]
                
                return CausalPuzzle(
                    events=events,
                    edges=edges,
                    trigger=trigger,
                    trigger_time=trigger_time,
                    target_event=target_event,
                    answer=answer,
                    difficulty=difficulty
                )
            
            except Exception as e:
                continue
        
        # 모든 시도 실패 시 단순 선형 체인 생성
        return self._generate_simple_puzzle(difficulty)
    
    def _generate_events(self, num_events: int) -> Dict[str, Event]:
        """사건 노드 생성"""
        events = {}
        
        # 사건 타입별 분배
        event_types = list(EventType)
        selected = []
        
        for i in range(num_events):
            event_type = event_types[i % len(event_types)]
            available = [t for t in self.event_templates[event_type] 
                        if t[0] not in [s[0] for s in selected]]
            
            if not available:
                available = self.event_templates[event_type]
            
            name, description = random.choice(available)
            selected.append((name, description))
            
            events[f"E{i+1}"] = Event(
                id=f"E{i+1}",
                name=name,
                description=description,
                event_type=event_type
            )
        
        return events
    
    def _generate_causal_graph(self, events: Dict[str, Event], 
                               config: Dict) -> List[CausalEdge]:
        """AND/OR 조건을 갖는 DAG 인과 관계 생성"""
        edges = []
        event_ids = sorted(events.keys())
        and_prob = config.get('and_probability', 0.0)
        
        # 각 타겟 노드에 대해 간선 생성
        for i, to_id in enumerate(event_ids):
            if i == 0:
                continue  # 첫 노드는 트리거 후보
            
            # 전제 조건 수 결정
            max_prereqs = min(config['max_out_degree'], i)
            num_prereqs = random.randint(1, max(1, max_prereqs))
            
            # 전제 조건 사건 선택 (이전 노드들)
            possible_from = event_ids[:i]
            if not possible_from:
                continue
            
            from_events = random.sample(possible_from, min(num_prereqs, len(possible_from)))
            
            # 조건 타입 결정
            condition = 'OR'
            if len(from_events) > 1 and random.random() < and_prob:
                condition = 'AND'
            
            delay = random.randint(*config['delay_range'])
            
            # 간선 생성
            if len(from_events) == 1:
                edges.append(CausalEdge(
                    from_event=from_events[0],
                    to_event=to_id,
                    delay=delay,
                    from_events=from_events,
                    condition='OR'
                ))
            else:
                edges.append(CausalEdge(
                    from_event=from_events[0],
                    to_event=to_id,
                    delay=delay,
                    from_events=from_events,
                    condition=condition
                ))
        
        return edges
    
    def _calculate_in_degree(self, events: Dict[str, Event], 
                            edges: List[CausalEdge]) -> Dict[str, int]:
        """각 노드의 진입 차수 계산"""
        in_degree = {e_id: 0 for e_id in events}
        for edge in edges:
            in_degree[edge.to_event] += 1
        return in_degree
    
    def _calculate_reach_times(self, events: Dict[str, Event],
                               edges: List[CausalEdge],
                               trigger: str,
                               trigger_time: int) -> Dict[str, int]:
        """
        AND/OR 조건으로 각 사건의 최초 발생 시간 계산
        
        Returns:
            사건 ID -> 최초 발생 시간 매핑 딕셔너리
        """
        # 전제 조건 추적 구조 구축
        prereqs_for_event = {}
        edges_for_event = {}
        
        for edge in edges:
            from_events = edge.from_events if edge.from_events else [edge.from_event]
            prereqs_for_event[edge.to_event] = from_events
            edges_for_event[edge.to_event] = edge
        
        # 각 사건의 최초 발생 시간 추적
        earliest_time = {e_id: float('inf') for e_id in events}
        earliest_time[trigger] = trigger_time
        
        # 각 전제 조건이 사건에 도달하는 시간 추적
        prereq_arrival_times = {e_id: {} for e_id in events}
        
        # 우선순위 큐: (시간, 사건 ID)
        pq = [(trigger_time, trigger)]
        processed = set()
        
        while pq:
            current_time, current_event = heapq.heappop(pq)
            
            if current_event in processed:
                continue
            processed.add(current_event)
            
            # current_event에 의존하는 모든 사건 찾기
            for edge in edges:
                from_events = edge.from_events if edge.from_events else [edge.from_event]
                if current_event not in from_events:
                    continue
                
                to_event = edge.to_event
                arrival_time = current_time + edge.delay
                
                # 이 전제 조건의 도달 시간 기록
                if current_event not in prereq_arrival_times[to_event]:
                    prereq_arrival_times[to_event][current_event] = arrival_time
                else:
                    # 최초 도달 시간 유지
                    prereq_arrival_times[to_event][current_event] = min(
                        prereq_arrival_times[to_event][current_event],
                        arrival_time
                    )
                
                # 모든 전제 조건이 도달했는지 확인
                all_prereqs_arrived = all(
                    prereq in prereq_arrival_times[to_event]
                    for prereq in from_events
                )
                
                if all_prereqs_arrived:
                    # 조건 타입에 따라 트리거 시간 계산
                    if edge.condition == 'AND':
                        # 모든 전제 조건 대기
                        trigger_time_for_event = max(
                            prereq_arrival_times[to_event][prereq]
                            for prereq in from_events
                        )
                    else:  # OR
                        # 첫 번째 전제 조건에서 트리거
                        trigger_time_for_event = min(
                            prereq_arrival_times[to_event][prereq]
                            for prereq in from_events
                        )
                    
                    # 현재 최선보다 빠르면 업데이트
                    if trigger_time_for_event < earliest_time[to_event]:
                        earliest_time[to_event] = trigger_time_for_event
                        heapq.heappush(pq, (trigger_time_for_event, to_event))
        
        return earliest_time
    
    def _generate_simple_puzzle(self, difficulty: str) -> CausalPuzzle:
        """폴백용 단순 선형 체인 생성"""
        num_events = 4
        events = self._generate_events(num_events)
        event_ids = sorted(events.keys())
        
        edges = []
        for i in range(len(event_ids) - 1):
            edges.append(CausalEdge(
                from_event=event_ids[i],
                to_event=event_ids[i+1],
                delay=random.randint(10, 30)
            ))
        
        trigger = event_ids[0]
        trigger_time = 0
        target_event = event_ids[-1]
        
        reach_times = self._calculate_reach_times(events, edges, trigger, trigger_time)
        answer = reach_times[target_event]
        
        return CausalPuzzle(
            events=events,
            edges=edges,
            trigger=trigger,
            trigger_time=trigger_time,
            target_event=target_event,
            answer=answer,
            difficulty=difficulty
        )
    
    def has_unique_solution(self, puzzle: CausalPuzzle) -> bool:
        """
        퍼즐의 유일 해 검증
        
        Dijkstra는 결정적이므로 주어진 그래프에 대해 항상 유일 해를 가집니다.
        그래프가 유효한지만 검증 (DAG, 연결됨).
        """
        # 타겟이 도달 가능한지 확인
        reach_times = self._calculate_reach_times(
            puzzle.events,
            puzzle.edges,
            puzzle.trigger,
            puzzle.trigger_time
        )
        
        return reach_times[puzzle.target_event] < float('inf')


def create_question(puzzle: CausalPuzzle) -> str:
    """퍼즐에 대한 한국어 질문 텍스트 생성"""
    
    # 사건 형식화
    event_lines = []
    for event_id in sorted(puzzle.events.keys()):
        event = puzzle.events[event_id]
        event_lines.append(f"  {event_id}: {event.name}")
        event_lines.append(f"      ({event.description})")
    
    events_description = '\n'.join(event_lines)
    
    # 인과 관계 형식화
    causal_lines = []
    sorted_edges = sorted(puzzle.edges, key=lambda e: e.to_event)
    
    for edge in sorted_edges:
        from_events = edge.from_events if edge.from_events else [edge.from_event]
        to_name = puzzle.events[edge.to_event].name
        
        if len(from_events) == 1:
            from_name = puzzle.events[from_events[0]].name
            causal_lines.append(
                f"  {from_events[0]} ({from_name}) → "
                f"{edge.to_event} ({to_name}): {edge.delay}분"
            )
        else:
            if edge.condition == 'AND':
                prereq_str = ' 그리고 '.join(f"{e} ({puzzle.events[e].name})" 
                                          for e in from_events)
                line = f"  [{prereq_str}] → {edge.to_event} ({to_name}): {edge.delay}분"
                line += "\n      (모든 전제 조건 필요)"
            else:
                prereq_str = ' 또는 '.join(f"{e} ({puzzle.events[e].name})" 
                                         for e in from_events)
                line = f"  [{prereq_str}] → {edge.to_event} ({to_name}): {edge.delay}분"
                line += "\n      (첫 번째 전제 조건에서 트리거)"
            causal_lines.append(line)
    
    causality_description = '\n'.join(causal_lines)
    
    trigger_name = puzzle.events[puzzle.trigger].name
    target_name = puzzle.events[puzzle.target_event].name
    
    question = f"""인과 관계 사건 시스템과 시간에 따른 전파를 분석하고 있습니다.

사건 목록:
{events_description}

인과 관계 (시간 지연 표시):
{causality_description}

규칙:
- 사건이 발생하면 지정된 지연 시간 후에 영향을 미칩니다
- OR 조건: 첫 번째 전제 조건이 도달하면 사건이 발생합니다
- AND 조건: 모든 전제 조건이 발생한 후에만 사건이 발생합니다
- 모든 시간은 기준점(0분)에서부터 측정됩니다

초기 조건:
- 사건 {puzzle.trigger} ({trigger_name})이(가) {puzzle.trigger_time}분에 발생합니다

질문:
사건 {puzzle.target_event} ({target_name})은(는) 몇 분에 처음 발생합니까?

답변은 분 단위 정수로 제공하세요.
예를 들어, 사건이 시작 후 45분에 발생하면 답변: 45
"""
    
    return question


def generate_dataset(puzzles_per_difficulty: int = 3, verbose: bool = True) -> List[Dict]:
    """
    인과 DAG 퍼즐 전체 데이터셋 생성
    
    Args:
        puzzles_per_difficulty: 난이도별 퍼즐 수
        verbose: 생성 진행 상황 출력
    
    Returns:
        평가 준비된 퍼즐 딕셔너리 리스트
    """
    generator = CausalPuzzleGenerator()
    difficulties = ['Easy', 'Medium', 'Hard']
    dataset = []
    
    for difficulty in difficulties:
        if verbose:
            print(f"\n=== {difficulty} 퍼즐 생성 중 ===")
        
        for i in range(puzzles_per_difficulty):
            puzzle = generator.generate_puzzle(difficulty)
            question = create_question(puzzle)
            
            puzzle_data = {
                'question': question,
                'answer': str(puzzle.answer),
                'difficulty': difficulty,
                'metadata': puzzle.to_dict()
            }
            
            dataset.append(puzzle_data)
            
            if verbose:
                print(f"  [{i+1}/{puzzles_per_difficulty}] "
                      f"{puzzle.trigger} → {puzzle.target_event}: "
                      f"{puzzle.answer}분")
    
    return dataset


def create_dataset_files(num_questions: int):
    """
    인과 DAG 퍼즐 데이터셋 파일 생성
    
    Args:
        num_questions: 생성할 질문 수
        version: 파일명 버전 문자열
    
    Returns:
        Tuple[pd.DataFrame, List[Dict]]: (데이터프레임, JSON 리스트)
    """
    import pandas as pd
    
    print(f"{num_questions}개의 인과 DAG 퍼즐 생성 중...")
    
    generator = CausalPuzzleGenerator()
    
    # 난이도별 퍼즐 계산
    puzzles_per_diff = num_questions // 3
    remainder = num_questions % 3
    
    difficulties = ['Easy', 'Medium', 'Hard']
    all_puzzles = []
    
    for i, difficulty in enumerate(difficulties):
        count = puzzles_per_diff + (1 if i < remainder else 0)
        
        for j in range(count):
            puzzle = generator.generate_puzzle(difficulty, seed=i*1000+j)
            puzzle_data = {
                'id': f'causal_dag_korean_{len(all_puzzles)}',
                'question': create_question(puzzle),
                'answer': str(puzzle.answer),
                'difficulty': difficulty,
                'trigger': puzzle.trigger,
                'target': puzzle.target_event,
                'num_events': len(puzzle.events),
                'num_edges': len(puzzle.edges)
            }
            all_puzzles.append(puzzle_data)
    
    print(f"\n{len(all_puzzles)}개의 퍼즐 생성 완료")
    
    df = pd.DataFrame(all_puzzles)
    
    # 파일 저장
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # CSV
    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"causal_dag_korean.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV 파일 생성: {csv_path}")
    
    # JSONL
    json_dir = PROJECT_ROOT / "data" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = json_dir / f"causal_dag_korean.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_puzzles:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL 파일 생성: {jsonl_path}")
    
    return df, all_puzzles


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Causal DAG Korean Puzzle Generator")
    parser.add_argument("--num", type=int, default=300, help="Number of questions to generate")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("인과 DAG 추론 퍼즐 생성기 (한국어)")
    print("=" * 70)
    
    create_dataset_files(num_questions=args.num)
    
    # 샘플 데이터셋 생성
    # dataset = generate_dataset(puzzles_per_difficulty=2, verbose=True)
    
    # print("\n" + "=" * 70)
    # print("샘플 퍼즐")
    # print("=" * 70)
    
    # sample = dataset[0]
    # print(sample['question'])
    # print(f"\n✅ 정답: {sample['answer']}분")
    
    # # 모든 퍼즐 검증
    # print("\n" + "=" * 70)
    # print("검증")
    # print("=" * 70)
    
    # generator = CausalPuzzleGenerator()
    # for i, puzzle_data in enumerate(dataset):
    #     metadata = puzzle_data['metadata']
        
    #     # EventType enum으로 Event 객체 재구성
    #     events = {}
    #     for k, v in metadata['events'].items():
    #         events[k] = Event(
    #             id=v['id'],
    #             name=v['name'],
    #             description=v['description'],
    #             event_type=EventType(v['event_type'])
    #         )
        
    #     puzzle = CausalPuzzle(
    #         events=events,
    #         edges=[CausalEdge(
    #             from_event=e['from_event'],
    #             to_event=e['to_event'],
    #             delay=e['delay'],
    #             from_events=e.get('from_events'),
    #             condition=e.get('condition', 'OR')
    #         ) for e in metadata['edges']],
    #         trigger=metadata['trigger'],
    #         trigger_time=metadata['trigger_time'],
    #         target_event=metadata['target'],
    #         answer=metadata['answer'],
    #         difficulty=metadata['difficulty']
    #     )
        
    #     is_valid = generator.has_unique_solution(puzzle)
    #     status = "✓" if is_valid else "✗"
    #     print(f"  퍼즐 {i+1}: {status} {'유효' if is_valid else '무효'}")
    
    # print("\n✓ 모든 퍼즐 생성 성공!")
