import random
from pathlib import Path

class JourneyState:
    """뱃사공의 여행 상태를 추적하는 클래스"""
    def __init__(self):
        self.total_moving_time_hours = 0.0
        self.total_rest_time_hours = 0.0
        self.continuous_moving_time_hours = 0.0
        self.current_position_km = 0.0

    @property
    def total_journey_time_hours(self):
        """전체 여행 시간 = 이동 시간 + 휴식 시간"""
        return self.total_moving_time_hours + self.total_rest_time_hours

def generate_puzzle_question(difficulty="medium"):
    max_retries = 100
    
    for attempt in range(max_retries):
        try:
            # 난이도별 파라미터 설정
            if difficulty == "easy":
                params = {
                    "distance_range": (50, 80),
                    "zone_a_range": (30, 50),
                    "zone_a_limit_range": (40, 60),  # 높은 제한 속도
                    "zone_b_limit_range": (35, 55),
                    "rest_trigger_range": (5, 8),  # 휴식 거의 없도록
                    "rest_duration_range": (1, 2),  # 10-20분
                    "heavy_threshold_range": (800, 1000),  # 높은 기준 (규정 미적용 가능성)
                    "speed_reduction_range": (5, 10),
                    "base_speed_bonus_range": (20, 30),
                }
            elif difficulty == "medium":
                params = {
                    "distance_range": (100, 150),
                    "zone_a_range": (40, 70),
                    "zone_a_limit_range": (25, 45),
                    "zone_b_limit_range": (20, 40),
                    "rest_trigger_range": (3, 4),  # 1-2회 휴식
                    "rest_duration_range": (2, 4),  # 20-40분
                    "heavy_threshold_range": (400, 600),  # 중간 기준
                    "speed_reduction_range": (10, 15),
                    "base_speed_bonus_range": (15, 25),
                }
            else:  # hard
                params = {
                    "distance_range": (180, 250),
                    "zone_a_range": (30, 60),  # 작은 zone_a (빠른 전환)
                    "zone_a_limit_range": (20, 35),  # 낮은 제한
                    "zone_b_limit_range": (15, 30),
                    "rest_trigger_range": (2, 3),  # 2-4회 휴식
                    "rest_duration_range": (3, 5),  # 30-50분
                    "heavy_threshold_range": (300, 500),  # 낮은 기준 (쉽게 초과)
                    "speed_reduction_range": (15, 20),
                    "base_speed_bonus_range": (10, 20),
                }
            
            # --- 1. 시스템 규칙 및 초기 변수 랜덤 설정 ---
            regulations = {
                "speed_limit": {
                    "zone_A_km": random.randint(*params["zone_a_range"]),
                    "zone_A_limit_kph": random.randint(*params["zone_a_limit_range"]),
                    "zone_B_limit_kph": random.randint(*params["zone_b_limit_range"]),
                },
                "mandatory_rest": {
                    "trigger_hours": random.randint(*params["rest_trigger_range"]),
                    "duration_minutes": random.randint(*params["rest_duration_range"]) * 10,
                },
                "cargo_effect": {
                    "heavy_load_kg": random.randint(params["heavy_threshold_range"][0]//100, params["heavy_threshold_range"][1]//100) * 100,
                    "speed_reduction_percent": random.randint(*params["speed_reduction_range"]),
                }
            }

            total_distance = regulations["speed_limit"]["zone_A_km"] + random.randint(*params["distance_range"])
            base_boat_speed_kph = regulations["speed_limit"]["zone_A_limit_kph"] + random.randint(*params["base_speed_bonus_range"])

            cargo_items = [
                {"name": "구호품 상자", "weight_range": (30, 70)},
                {"name": "의료품 키트", "weight_range": (5, 15)},
                {"name": "식수통", "weight_range": (15, 25)},
                {"name": "건축 자재", "weight_range": (50, 80)}
            ]
            item1_spec, item2_spec = random.sample(cargo_items, 2)
            item1_weight = random.randint(*item1_spec["weight_range"])
            item1_qty = random.randint(5, 15)
            item2_weight = random.randint(*item2_spec["weight_range"])
            item2_qty = random.randint(1, 5)
            cargo_weight_kg = (item1_weight * item1_qty) + (item2_weight * item2_qty)

            noise = {
                "departure_time": f"오전 {random.randint(7, 10)}시",
                "career_years": random.randint(5, 15),
                "river_depth_meters": random.randint(5, 15)
            }

            # --- 2. 시뮬레이션 및 풀이 과정 생성 ---
            journey = JourneyState()
            solution = ["[STEP 0] 초기 조건 및 규정 확인"]
            solution.append(f"  - 기본 조건: 총 거리={total_distance}km, 배 속력={base_boat_speed_kph}km/h")
            solution.append(f"  - 화물 구성: {item1_weight}kg {item1_spec['name']} x {item1_qty}, {item2_weight}kg {item2_spec['name']} x {item2_qty}")
            solution.append(f"  - 운항 규정: {regulations}")
            step_cnt = 1
            
            solution.append(f"[STEP {step_cnt}] 총 화물 무게 계산: ({item1_weight}kg * {item1_qty}) + ({item2_weight}kg * {item2_qty}) = {cargo_weight_kg}kg.")
            step_cnt += 1
            
            adjusted_zone_A_limit_kph = regulations["speed_limit"]["zone_A_limit_kph"]
            adjusted_zone_B_limit_kph = regulations["speed_limit"]["zone_B_limit_kph"]

            if cargo_weight_kg > regulations["cargo_effect"]["heavy_load_kg"]:
                reduction = regulations["cargo_effect"]["speed_reduction_percent"]
                adjusted_zone_A_limit_kph *= (1 - reduction / 100)
                adjusted_zone_B_limit_kph *= (1 - reduction / 100)
                solution.append(f"[STEP {step_cnt}] 화물 규정 적용: 계산된 총 무게({cargo_weight_kg}kg)가 기준({regulations['cargo_effect']['heavy_load_kg']}kg)을 초과하여 모든 구역의 제한 속력이 {reduction}% 감소합니다. (A구역: {adjusted_zone_A_limit_kph:.2f}km/h, B구역: {adjusted_zone_B_limit_kph:.2f}km/h)")
                step_cnt += 1

            distance_to_go = total_distance
            while distance_to_go > 0.001:
                current_pos = journey.current_position_km
                zone_a_boundary = regulations["speed_limit"]["zone_A_km"]
                
                if current_pos < zone_a_boundary:
                    speed_limit = adjusted_zone_A_limit_kph
                    distance_in_this_zone = zone_a_boundary - current_pos
                    solution.append(f"[STEP {step_cnt}] A구역 진입 (현재위치: {current_pos:.1f}km): 제한속도 {speed_limit:.2f}km/h 적용.")
                else:
                    speed_limit = adjusted_zone_B_limit_kph
                    distance_in_this_zone = total_distance - current_pos
                    solution.append(f"[STEP {step_cnt}] B구역 진입 (현재위치: {current_pos:.1f}km): 제한속도 {speed_limit:.2f}km/h 적용.")
                step_cnt += 1

                actual_speed = min(base_boat_speed_kph, speed_limit)
                
                # 속도가 0 이하면 재시도
                if actual_speed <= 0:
                    raise ValueError("생성된 문제가 유효하지 않습니다.") 

                rest_trigger_time = regulations["mandatory_rest"]["trigger_hours"]
                time_to_reach_rest_trigger = rest_trigger_time - journey.continuous_moving_time_hours
                distance_to_rest_trigger = time_to_reach_rest_trigger * actual_speed

                distance_this_segment = min(distance_to_go, distance_in_this_zone, distance_to_rest_trigger)
                time_this_segment = distance_this_segment / actual_speed

                journey.current_position_km += distance_this_segment
                journey.total_moving_time_hours += time_this_segment
                journey.continuous_moving_time_hours += time_this_segment
                distance_to_go -= distance_this_segment

                solution.append(f"  - 운항: 실제 속력 {actual_speed:.2f}km/h로 {distance_this_segment:.2f}km 이동. ({time_this_segment:.2f}시간 소요)")

                if journey.continuous_moving_time_hours >= rest_trigger_time - 0.0001:
                    rest_minutes = regulations["mandatory_rest"]["duration_minutes"]
                    rest_hours = rest_minutes / 60.0
                    journey.total_rest_time_hours += rest_hours
                    journey.continuous_moving_time_hours = 0
                    solution.append(f"  - 의무 휴식: {rest_trigger_time}시간 연속 운항하여 {rest_minutes}분 휴식. (연속 운항 시간 초기화)")

            # --- 3. 최종 질문 및 정답 생성 (서사 중심) ---
            protagonist = random.choice(["뱃사공 김씨", "물품 운송원 박씨", "하천 탐사대원 이씨"])
            
            question = (
                f"{protagonist}는 이 강에서만 {noise['career_years']}년을 일한 베테랑으로, 총 길이 {total_distance}km의 상류 지역에 물품을 운송하는 임무를 맡았다. "
                f"그는 {noise['departure_time']}에 {item1_weight}kg짜리 {item1_spec['name']} {item1_qty}개와 {item2_weight}kg짜리 {item2_spec['name']} {item2_qty}개를 싣고 출발했다. "
                f"그의 배는 시속 {base_boat_speed_kph}km/h로 이동 가능하다.\n\n"
                f"출발 전, 그는 항해 지도를 보며 첫 {regulations['speed_limit']['zone_A_km']}km 구간은 A구역으로 지정되어 실제 운항 속력을 {regulations['speed_limit']['zone_A_limit_kph']}km/h 이하로 유지해야 하고, "
                f"그 이후 B구역부터는 {regulations['speed_limit']['zone_B_limit_kph']}km/h로 더 서행해야 한다는 점을 상기했다. "
                f"또한, 회사 내규에 따라 '안전 중량 기준({regulations['cargo_effect']['heavy_load_kg']}kg) 초과 화물 적재 시 안전 운항을 위해 모든 구역의 제한 속력이 추가로 {regulations['cargo_effect']['speed_reduction_percent']}% 감소'한다는 규정도 있었다. "
                f"이번 임무는 장거리라, 동료로부터 '휴식 없이 {regulations['mandatory_rest']['trigger_hours']}시간을 계속 운항하면 규정상 즉시 {regulations['mandatory_rest']['duration_minutes']}분간 쉬어야 한다'는 조언도 들었다.\n\n"
                f"이 모든 조건과 규정을 준수하여 최종 목적지까지 도착했을 때, 의무 휴식을 포함한 총 소요 시간은 몇 시간 몇 분입니까? (분은 소숫점 첫째 자리에서 반올림)"
            )

            total_hours = journey.total_journey_time_hours
            hours = int(total_hours)
            minutes = round((total_hours - hours) * 60)

            if minutes == 60:
                hours += 1
                minutes = 0
                
            answer = f"{hours}시간 {minutes}분"
            solution.append(f"[STEP {step_cnt}] 총 소요 시간은 {total_hours:.2f}시간이며, 이를 '시간'과 '분'으로 변환하면 '{answer}'입니다.")

            return question, answer, solution
            
        except ValueError:
            continue
    
    # 최대 재시도 횟수 초과 시 예외 발생
    raise RuntimeError(f"최대 재시도 횟수({max_retries})를 초과했습니다. 문제를 생성할 수 없습니다.")

def create_dataset_files(num_questions, difficulty=None):
    import pandas as pd
    import json
    
    if difficulty is None:
        difficulties = ["easy", "medium", "hard"]
        total_questions = num_questions * len(difficulties)
        print(f"뱃사공 문제를 생성 중... (각 난이도별 {num_questions}개, 총 {total_questions}개)")
    else:
        difficulties = [difficulty]
        total_questions = num_questions
        print(f"뱃사공 문제 {num_questions}개를 생성 중... (난이도: {difficulty})")
    
    output = []
    seen_questions = set()  # 중복 체크용
    
    # 각 난이도별로 문제 생성
    for diff in difficulties:
        print(f"\n[{diff.upper()}] 난이도 {num_questions}개 생성 중...")
        diff_count = 0
        attempt_count = 0
        max_attempts = num_questions * 200
        
        while diff_count < num_questions and attempt_count < max_attempts:
            attempt_count += 1
            
            try:
                q, answer, expl = generate_puzzle_question(difficulty=diff)
                
                if q not in seen_questions:
                    output.append([q, answer, "\n".join(expl), diff])
                    seen_questions.add(q)
                    diff_count += 1
                    
                    if diff_count % 10 == 0:
                        print(f"  진행: {diff_count}/{num_questions}")
            except Exception as e:
                continue
        
        if diff_count < num_questions:
            print(f"  경고: [{diff}] 목표 {num_questions}개 중 {diff_count}개만 생성되었습니다.")
    
    ferryman_df = pd.DataFrame(output, columns=['question', 'answer', 'solution', 'difficulty'])
    
    print(f"\n생성 통계:")
    print(f"  생성된 문제 수: {len(ferryman_df)}")
    print(f"  고유한 문제 수: {ferryman_df['question'].nunique()}")
    print(f"  고유한 정답 수: {ferryman_df['answer'].nunique()}")
    print(f"\n난이도별 분포:")
    print(ferryman_df['difficulty'].value_counts().sort_index())
    
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # CSV
    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = csv_dir / "ferryman.csv"
    ferryman_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV 파일이 생성: {csv_path}")
    
    # JSONL
    json_dir = PROJECT_ROOT / "data" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    
    ferryman_json = []
    for idx, row in ferryman_df.iterrows():
        question_data = {
            "question": row['question'],
            "answer": row['answer'],
            "solution": row['solution'],
            "difficulty": row['difficulty'],
        }
        ferryman_json.append(question_data)
    
    jsonl_path = json_dir / "ferryman.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in ferryman_json:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"JSONL 파일이 생성: {jsonl_path}")
    # print(json.dumps(ferryman_json[0], ensure_ascii=False, indent=2)) # sample 출력
    
    return ferryman_df, ferryman_json

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Ferryman Puzzle Generator")
    parser.add_argument("--num", type=int, default=100, 
                        help="Number of questions to generate per difficulty level. If --difficulty is not specified, generates this many questions for EACH difficulty (easy, medium, hard).")
    parser.add_argument("--difficulty", type=str, default=None, 
                        choices=["easy", "medium", "hard"], 
                        help="Difficulty level (easy/medium/hard). If not specified, generates questions for all three difficulty levels.")
    
    args = parser.parse_args()
    
    create_dataset_files(num_questions=args.num, difficulty=args.difficulty)

    # 샘플 출력
    # print("\n" + "="*80)
    # print("샘플 문제 (각 난이도별 1개씩)")
    # print("="*80)
    # for diff in ["easy", "medium", "hard"]:
    #     question, answer, solution = generate_puzzle_question(difficulty=diff)
    #     print(f"\n========== [{diff.upper()}] 문제 샘플 ==========")
    #     print("- question -\n", question)
    #     print("\n- answer -\n", answer)
    #     print("\n- solution -")
    #     for step in solution:
    #         print(step)
        # print("\n")