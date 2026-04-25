import random
from pathlib import Path

SFT_SOLUTION_RUBRIC_KO = (
    "STEP0=문제 메타 · STEP1=주어진 조건 · STEP2=풀이 전개 · STEP3=답·검산"
)


class JourneyState:
    def __init__(self):
        self.total_moving_time_hours = 0.0
        self.total_rest_time_hours = 0.0
        self.continuous_moving_time_hours = 0.0
        self.current_position_km = 0.0

    @property
    def continuous_drive_time_min(self):
        """휴식 없이 누적된 연속 운항 시간(분). 휴식 시 continuous_moving_time_hours와 함께 0으로 리셋."""
        return self.continuous_moving_time_hours * 60.0

    @property
    def total_journey_time_hours(self):
        return self.total_moving_time_hours + self.total_rest_time_hours


def _fmt_hour(h):
    if h < 12:
        return f"오전 {h}시"
    elif h == 12:
        return "낮 12시"
    else:
        return f"오후 {h - 12}시"


def _fmt_hhmm_from_decimal_hour(dec_h):
    """시뮬레이션 절대시각(소수 시간)을 하루 주기의 HH:MM으로 표시."""
    minutes = int(round((float(dec_h) % 24.0) * 60.0)) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _fmt_minutes_from_hours(hours):
    """시간(h)을 분 단위 실수 문자열로."""
    return f"{hours * 60.0:.2f}"


REST_COUNT_RANGES = {
    "easy":   (8, 11),  # interval(7-9)+trigger(40-60) 달성 가능 / gap=2 to med min(14)
    "medium": (14, 17), # gap=2 from easy max(11), gap=5 to hard min(22) / interval(10-13)+dist(150-200) 달성 가능
    "hard":   (26, 33), # gap=9 from med max(17) / hard만 소폭 상향
}

def generate_puzzle_question(difficulty="easy", rest_count_target=None):
    max_retries = 3000

    for attempt in range(max_retries):
        try:
            if difficulty == "easy":
                params = {
                    "distance_range": (60, 100),             # w=40, gap=50 to med(150)
                    "zone_a_range": (15, 25),
                    "zone_a_limit_range": (25, 38),
                    "zone_b_limit_range": (23, 26),          # w=3, gap=1 to med max(21)
                    "rest_trigger_minutes_range": (40, 60),   # gap=15 to med min(75) — 하향으로 휴식 더 자주 발생
                    "rest_duration_minutes_range": (25, 50),
                    "rest_stop_interval_km_range": (7, 9),    # gap=1 to med min(10) — 더 짧은 구간으로 최대 11회 휴식 가능
                    "heavy_threshold_range": (1500, 2500),   # gap=500 to med min(1000)
                    "zone_a_reduction_range": (10, 16),
                    "zone_b_reduction_range": (10, 18),      # w=8, gap=2 to med min(21)
                    "uniform_reduction": False,
                    "base_speed_bonus_range": (8, 18),
                    "current_speed_range": (2, 3),           # w=1, gap=1 to med min(5)
                    "has_delivery": True,
                    "rest_increment_range": (5, 8),          # w=3, gap=6 to med min(14)
                    "congestion_start_offset_range": (2, 4),
                    "congestion_duration_range": (1, 2),     # w=1, gap=0 to med min(3)
                    "congestion_reduction_range": (10, 18),  # w=8, gap=2 to med min(21)
                    "congestion_affected_zone": "all",
                    "max_segments": 45,
                    "rest_count_range": (8, 11),             # w=3, gap=2 to med min(14) — interval(7-9)+trigger(40-60)로 최대 11회 달성 가능
                }
            elif difficulty == "medium":
                params = {
                    "distance_range": (150, 200),            # w=50, gap=50 from easy(100), gap=40 to hard(240)
                    "zone_a_range": (18, 35),
                    "zone_a_limit_range": (20, 32),
                    "zone_b_limit_range": (18, 21),          # w=3, gap=1 from easy(23), gap=1 to hard max(16)
                    "rest_trigger_minutes_range": (75, 93),   # gap=1 to hard min(95)
                    "rest_duration_minutes_range": (35, 65),
                    "rest_stop_interval_km_range": (10, 13),  # gap=1 from easy max(9) — 물리적 한계 해소
                    "heavy_threshold_range": (600, 1000),    # gap=500 from easy, gap=150 to hard max(450)
                    "zone_a_reduction_range": (15, 25),
                    "zone_b_reduction_range": (21, 29),      # w=8, gap=2 from easy(18), gap=2 to hard min(32)
                    "uniform_reduction": False,
                    "base_speed_bonus_range": (4, 11),
                    "current_speed_range": (5, 6),           # w=1, gap=1 from easy(3), gap=0 to hard min(7)
                    "has_delivery": True,
                    "rest_increment_range": (14, 17),        # w=3, gap=6 from easy max(8), gap=3 to hard min(20)
                    "congestion_start_offset_range": (1, 2),
                    "congestion_duration_range": (3, 4),     # w=1, gap=0 from easy(2), gap=0 to hard min(5)
                    "congestion_reduction_range": (21, 29),  # w=8, gap=2 from easy(18), gap=2 to hard min(32)
                    "congestion_affected_zone": random.choice(["all", "B"]),
                    "max_segments": 80,
                    "rest_count_range": (14, 17),            # w=3, gap=2 from easy max(11), gap=5 to hard min(22)
                }
            else:  # hard
                params = {
                    "distance_range": (270, 330),            # w=60, gap=70 from med(200) — 더 긴 다일차 여정
                    "zone_a_range": (20, 45),
                    "zone_a_limit_range": (17, 29),
                    "zone_b_limit_range": (13, 16),          # w=3, gap=1 from med min(18)
                    "rest_trigger_minutes_range": (95, 120),  # gap=1 from med max(93) — 물리적 모순 방지
                    "rest_duration_minutes_range": (45, 85),
                    "rest_stop_interval_km_range": (8, 11),   # gap=2 from med max(13) — 최악속도 기준 간격 단축
                    "heavy_threshold_range": (250, 450),     # gap=150 from med min(600)
                    "zone_a_reduction_range": (15, 25),
                    "zone_b_reduction_range": (32, 40),      # w=8, gap=2 from med max(29)
                    "uniform_reduction": False,
                    "base_speed_bonus_range": (3, 9),
                    "current_speed_range": (7, 8),           # w=1, gap=0 from med max(6)
                    "has_delivery": True,
                    "rest_increment_range": (27, 38),        # w=11, gap=10 from med max(17)
                    "congestion_start_offset_range": (1, 2),
                    "congestion_duration_range": (5, 6),     # w=1, gap=0 from med max(4)
                    "congestion_reduction_range": (32, 40),  # w=8, gap=2 from med max(29)
                    "congestion_affected_zone": "B",
                    "max_segments": 180,
                    "rest_count_range": (26, 33),            # w=7, gap=9 from med max(17) / hard만 소폭 상향
                }

            if rest_count_target is not None:
                params["rest_count_range"] = (rest_count_target, rest_count_target)

            # --- 1. 시스템 규칙 및 초기 변수 랜덤 설정 ---
            zone_a_reduction = random.randint(*params["zone_a_reduction_range"])
            if params.get("uniform_reduction"):
                zone_b_reduction = zone_a_reduction
            else:
                zone_b_reduction = random.randint(*params["zone_b_reduction_range"])
                if zone_a_reduction == zone_b_reduction:
                    zone_b_reduction = min(
                        zone_b_reduction + random.randint(2, 6),
                        params["zone_b_reduction_range"][1])

            regulations = {
                "speed_limit": {
                    "zone_A_km": random.randint(*params["zone_a_range"]),
                    "zone_A_limit_kph": random.randint(*params["zone_a_limit_range"]),
                    "zone_B_limit_kph": random.randint(*params["zone_b_limit_range"]),
                },
                "mandatory_rest": {
                    "trigger_minutes": random.randint(
                        *params["rest_trigger_minutes_range"]),
                    "duration_minutes": random.randint(
                        *params["rest_duration_minutes_range"]),
                    "rest_stop_interval_km": random.randint(
                        *params["rest_stop_interval_km_range"]),
                },
                "cargo_effect": {
                    "heavy_load_kg": random.randint(
                        params["heavy_threshold_range"][0] // 100,
                        params["heavy_threshold_range"][1] // 100) * 100,
                    "zone_A_reduction_percent": zone_a_reduction,
                    "zone_B_reduction_percent": zone_b_reduction,
                }
            }

            total_distance = (regulations["speed_limit"]["zone_A_km"]
                              + random.randint(*params["distance_range"]))
            base_boat_speed_kph = (regulations["speed_limit"]["zone_A_limit_kph"]
                                   + random.randint(*params["base_speed_bonus_range"]))
            current_speed_kph = random.randint(*params["current_speed_range"])
            current_favorable_zone = random.choice(["A", "B"])

            cargo_items = [
                {"name": "구호품 상자", "weight_range": (30, 70)},
                {"name": "의료품 키트", "weight_range": (5, 15)},
                {"name": "식수통", "weight_range": (15, 25)},
                {"name": "건축 자재", "weight_range": (50, 80)}
            ]
            item1_spec, item2_spec = random.sample(cargo_items, 2)
            item1_weight = random.randint(*item1_spec["weight_range"])
            item2_weight = random.randint(*item2_spec["weight_range"])

            if difficulty == "easy":
                item1_qty = random.randint(5, 15)
                item2_qty = random.randint(2, 6)
            elif difficulty == "medium":
                item1_qty = random.randint(8, 18)
                item2_qty = random.randint(3, 8)
            else:
                item1_qty = random.randint(10, 22)
                item2_qty = random.randint(4, 10)

            cargo_weight_kg = (item1_weight * item1_qty) + (item2_weight * item2_qty)

            # 중간 하역 설정
            delivery_km = None
            delivery_spec = None
            delivery_qty = 0
            delivery_weight_per_unit = 0
            if params["has_delivery"]:
                zone_a_km = regulations["speed_limit"]["zone_A_km"]
                rest_interval = regulations["mandatory_rest"]["rest_stop_interval_km"]
                possible_stops = [
                    rest_interval * k
                    for k in range(1, total_distance // rest_interval + 1)
                    if zone_a_km + 5 < rest_interval * k < total_distance - 10
                ]
                if possible_stops:
                    delivery_km = random.choice(possible_stops)
                    if random.random() < 0.5:
                        delivery_spec = item1_spec
                        delivery_weight_per_unit = item1_weight
                        delivery_qty = random.randint(
                            max(1, item1_qty // 3), max(2, item1_qty * 2 // 3))
                    else:
                        delivery_spec = item2_spec
                        delivery_weight_per_unit = item2_weight
                        delivery_qty = random.randint(
                            max(1, item2_qty // 3), max(2, item2_qty * 2 // 3))

            rest_increment = random.randint(*params["rest_increment_range"])

            # 혼잡시간 설정
            departure_hour = random.randint(7, 9)
            cong_offset = random.randint(*params["congestion_start_offset_range"])
            congestion_start_hour = departure_hour + cong_offset
            cong_dur = random.randint(*params["congestion_duration_range"])
            congestion_end_hour = congestion_start_hour + cong_dur
            congestion_reduction_pct = random.randint(
                *params["congestion_reduction_range"])
            congestion_affected_zone = params["congestion_affected_zone"]

            career_years = random.randint(5, 15)

            # --- 2. 시뮬레이션 ---
            journey = JourneyState()
            solution = [
                SFT_SOLUTION_RUBRIC_KO,
                "[STEP 0] 문제 메타",
                "  - 운항·휴식·화물·혼잡 시뮬레이션; 정답은 [STEP 3] '총 소요'에만.",
                "[STEP 1] 주어진 조건 (초기·규정)",
            ]
            solution.append(
                f"  - 총 거리={total_distance}km, 배 정수속력="
                f"{base_boat_speed_kph}km/h, 출발={_fmt_hour(departure_hour)}")
            solution.append(
                f"  - 유속: {current_speed_kph}km/h "
                f"({current_favorable_zone}구역 순류)")
            solution.append(
                f"  - 화물: {item1_weight}kg {item1_spec['name']}×{item1_qty}"
                f" + {item2_weight}kg {item2_spec['name']}×{item2_qty}")
            solution.append(
                f"  - 혼잡: {_fmt_hour(congestion_start_hour)}~"
                f"{_fmt_hour(congestion_end_hour)}, "
                f"{'전 구역' if congestion_affected_zone == 'all' else congestion_affected_zone + '구역'}"
                f" -{congestion_reduction_pct}%")
            if delivery_km:
                solution.append(
                    f"  - 중간 하역: {delivery_km}km에서 "
                    f"{delivery_spec['name']} {delivery_qty}개")
            if rest_increment > 0:
                solution.append(f"  - 누적 휴식: 매 휴식마다 +{rest_increment}분")
            solution.append(f"  - 규정: {regulations}")
            solution.append("[STEP 2] 풀이 전개 (구간별 시뮬 로그)")
            step2_header_idx = len(solution)
            summary_rests: list[tuple[float, int]] = []
            summary_unload: tuple[float, str, int] | None = None
            summary_congest_cnt = 0
            step_cnt = 1

            solution.append(
                f"[SEG {step_cnt}] 화물 무게: "
                f"({item1_weight}×{item1_qty})+({item2_weight}×{item2_qty})"
                f"={cargo_weight_kg}kg")
            step_cnt += 1

            def _apply_cargo_effect(weight_kg):
                base_a = regulations["speed_limit"]["zone_A_limit_kph"]
                base_b = regulations["speed_limit"]["zone_B_limit_kph"]
                heavy = weight_kg > regulations["cargo_effect"]["heavy_load_kg"]
                if heavy:
                    ra = regulations["cargo_effect"]["zone_A_reduction_percent"]
                    rb = regulations["cargo_effect"]["zone_B_reduction_percent"]
                    return (base_a * (1 - ra / 100),
                            base_b * (1 - rb / 100), True)
                return float(base_a), float(base_b), False

            adj_A, adj_B, is_heavy = _apply_cargo_effect(cargo_weight_kg)

            if is_heavy:
                ra = regulations["cargo_effect"]["zone_A_reduction_percent"]
                rb = regulations["cargo_effect"]["zone_B_reduction_percent"]
                solution.append(
                    f"[SEG {step_cnt}] 화물규정: {cargo_weight_kg}kg > "
                    f"{regulations['cargo_effect']['heavy_load_kg']}kg → "
                    f"A구역 -{ra}%({adj_A:.1f}), B구역 -{rb}%({adj_B:.1f})")
                step_cnt += 1

            distance_to_go = total_distance
            rest_due = False
            trigger_hours = (regulations["mandatory_rest"]["trigger_minutes"]
                             / 60.0)
            rest_stop_interval_km = (
                regulations["mandatory_rest"]["rest_stop_interval_km"])
            rest_count = 0
            delivered = delivery_km is None

            # --- 연속운항 제약 모순 사전 검증 ---
            for _zone in ("A", "B"):
                if _zone == current_favorable_zone:
                    _eff = base_boat_speed_kph + current_speed_kph
                else:
                    _eff = base_boat_speed_kph - current_speed_kph
                _lim = adj_A if _zone == "A" else adj_B
                if (congestion_affected_zone == "all"
                        or congestion_affected_zone == _zone):
                    _lim *= (1 - congestion_reduction_pct / 100)
                _worst_speed = min(_eff, _lim)
                if _worst_speed <= 0:
                    raise ValueError("최악 조건 속도 0 이하")
                if rest_stop_interval_km / _worst_speed >= trigger_hours:
                    raise ValueError(
                        f"연속운항 모순: {_zone}구역 최악속도 "
                        f"{_worst_speed:.2f}km/h, "
                        f"{rest_stop_interval_km}km 이동 "
                        f"{rest_stop_interval_km/_worst_speed*60:.0f}분 "
                        f">= 한계 {trigger_hours*60:.0f}분")

            def _next_rest_stop_dist(current_km):
                eps = 1e-9
                k = int((current_km + eps) // rest_stop_interval_km)
                next_stop = k * rest_stop_interval_km
                if abs(current_km - next_stop) < 1e-6:
                    return 0.0
                return (k + 1) * rest_stop_interval_km - current_km

            def _compute_time_to(from_km, to_km,
                                 _adj_A=None, _adj_B=None, _t_offset=0.0):
                eff_adj_A = adj_A if _adj_A is None else _adj_A
                eff_adj_B = adj_B if _adj_B is None else _adj_B
                pos = from_km
                remaining = to_km - from_km
                t = 0.0
                zone_a_km = regulations["speed_limit"]["zone_A_km"]
                base_abs = (departure_hour
                            + journey.total_moving_time_hours
                            + journey.total_rest_time_hours
                            + _t_offset)
                max_iter = 10000
                for _ in range(max_iter):
                    if remaining <= 0.001:
                        break
                    if pos < zone_a_km - 1e-9:
                        lim = eff_adj_A
                        dz = min(zone_a_km - pos, remaining)
                        z = "A"
                    else:
                        lim = eff_adj_B
                        dz = remaining
                        z = "B"

                    cur_abs = base_abs + t
                    if abs(cur_abs - round(cur_abs)) < 1e-9:
                        cur_abs = round(cur_abs)
                    abs_hr = cur_abs % 24  # 24시간 주기 보정
                    in_cong = (congestion_start_hour <= abs_hr
                               < congestion_end_hour)
                    if (in_cong
                            and (congestion_affected_zone == "all"
                                 or z == congestion_affected_zone)):
                        lim *= (1 - congestion_reduction_pct / 100)

                    if z == current_favorable_zone:
                        eff = base_boat_speed_kph + current_speed_kph
                    else:
                        eff = base_boat_speed_kph - current_speed_kph
                    spd = min(eff, lim)
                    if spd <= 0:
                        raise ValueError("_compute_time_to: 속도 0 이하")

                    if not in_cong:
                        if abs_hr < congestion_start_hour:
                            t_to_s = congestion_start_hour - abs_hr
                        else:  # 오늘 혼잡 이미 지남 → 다음 날 혼잡까지 남은 시간
                            t_to_s = (24 - abs_hr) + congestion_start_hour
                        d_to_s = spd * t_to_s
                        if d_to_s < dz:
                            dz = d_to_s
                    else:
                        t_to_e = congestion_end_hour - abs_hr
                        if t_to_e > 1e-9:
                            d_to_e = spd * t_to_e
                            if d_to_e < dz:
                                dz = d_to_e

                    if dz < 1e-12:
                        raise ValueError("_compute_time_to: 이동 거리 0")

                    t += dz / spd
                    pos += dz
                    remaining -= dz
                else:
                    raise ValueError("_compute_time_to: 최대 반복 초과")
                return t

            max_favorable_speed = base_boat_speed_kph + current_speed_kph
            zone_a_boundary = regulations["speed_limit"]["zone_A_km"]

            # --- 메인 시뮬레이션 루프 ---
            while distance_to_go > 0.001:
                current_pos = journey.current_position_km

                if current_pos < zone_a_boundary:
                    speed_limit = adj_A
                    distance_in_zone = zone_a_boundary - current_pos
                    in_zone = "A"
                else:
                    speed_limit = adj_B
                    distance_in_zone = total_distance - current_pos
                    in_zone = "B"

                # 혼잡시간 체크
                abs_time = (departure_hour
                            + journey.total_moving_time_hours
                            + journey.total_rest_time_hours)
                if abs(abs_time - round(abs_time)) < 1e-9:
                    abs_time = round(abs_time)
                abs_hour = abs_time % 24  # 24시간 주기 보정
                in_congestion = (congestion_start_hour <= abs_hour
                                 < congestion_end_hour)
                cong_applies = (
                    in_congestion
                    and (congestion_affected_zone == "all"
                         or in_zone == congestion_affected_zone))
                limit_zone_kph = speed_limit
                if cong_applies:
                    speed_limit *= (1 - congestion_reduction_pct / 100)
                limit_after_congest_kph = speed_limit

                if in_zone == current_favorable_zone:
                    eff_speed = base_boat_speed_kph + current_speed_kph
                    cur_label = "순류"
                else:
                    eff_speed = base_boat_speed_kph - current_speed_kph
                    cur_label = "역류"

                actual_speed = min(eff_speed, speed_limit)
                if actual_speed <= 0:
                    raise ValueError("유효하지 않은 속도")

                dist_to_stop = _next_rest_stop_dist(current_pos)
                if dist_to_stop < 1e-6:
                    dist_to_stop = rest_stop_interval_km

                boundaries = [distance_to_go, distance_in_zone, dist_to_stop]
                if not delivered:
                    dist_to_delivery = delivery_km - current_pos
                    if dist_to_delivery > 1e-6:
                        boundaries.append(dist_to_delivery)

                # 혼잡시간 경계 (시간 기반 → 거리 변환)
                if not in_congestion:
                    if abs_hour < congestion_start_hour:
                        t_to_cong = congestion_start_hour - abs_hour
                    else:  # 오늘 혼잡 이미 지남 → 다음 날 혼잡까지 남은 시간
                        t_to_cong = (24 - abs_hour) + congestion_start_hour
                    d_to_cong = actual_speed * t_to_cong
                    if d_to_cong > 0.001:
                        boundaries.append(d_to_cong)
                else:
                    t_to_cong_end = congestion_end_hour - abs_hour
                    if t_to_cong_end > 1e-9:
                        d_to_cong_end = actual_speed * t_to_cong_end
                        if d_to_cong_end > 0.001:
                            boundaries.append(d_to_cong_end)

                seg_dist = min(boundaries)
                seg_time = seg_dist / actual_speed

                seg_start_km = current_pos
                drive_cont_before_min = journey.continuous_drive_time_min
                hits_delivery = (
                    not delivered
                    and delivery_km is not None
                    and abs((current_pos + seg_dist) - delivery_km) < 1e-6)

                journey.current_position_km += seg_dist
                journey.total_moving_time_hours += seg_time
                journey.continuous_moving_time_hours += seg_time
                distance_to_go -= seg_dist

                seg_end_km = journey.current_position_km
                drive_cont_after_min = journey.continuous_drive_time_min
                seg_minutes = seg_time * 60.0
                thr_min = regulations["mandatory_rest"]["trigger_minutes"]

                seg_lines: list[str] = []
                seg_has_event = False

                tags = []
                if cong_applies:
                    tags.append("[CONGESTED TIME]")
                    summary_congest_cnt += 1
                    seg_has_event = True
                if hits_delivery:
                    tags.append("[UNLOAD]")
                    seg_has_event = True
                tag_str = (" ".join(tags) + " ") if tags else ""

                seg_lines.append(
                    f"[SEG {step_cnt}] {tag_str}{in_zone}구역 "
                    f"({seg_start_km:.1f}km ~ {seg_end_km:.1f}km)")
                seg_lines.append(
                    f"  - 현재 시각: {_fmt_hhmm_from_decimal_hour(abs_time)} "
                    f"({abs_time:.2f}h)")
                cong_note = (
                    f", 혼잡으로 제한 {limit_after_congest_kph:.1f}km/h"
                    if cong_applies else "")
                seg_lines.append(
                    f"  - 속도 검산: {cur_label} 실효 "
                    f"{base_boat_speed_kph}{'+' if cur_label == '순류' else '-'}"
                    f"{current_speed_kph}={eff_speed:.1f}km/h, "
                    f"구역 제한 {limit_zone_kph:.1f}km/h{cong_note} "
                    f"→ 적용 속력 {actual_speed:.1f}km/h")
                seg_lines.append(
                    f"  - 이동: {seg_dist:.1f}km / {actual_speed:.1f}km/h = "
                    f"{seg_time:.3f}h ({_fmt_minutes_from_hours(seg_time)}분)")
                seg_lines.append(
                    f"  - 연속 운행 누적: "
                    f"{drive_cont_before_min:.1f}분 + {seg_minutes:.1f}분 = "
                    f"{drive_cont_after_min:.1f}분")

                # 중간 하역
                if (not delivered
                        and abs(journey.current_position_km - delivery_km)
                        < 1e-6):
                    delivered = True
                    unloaded_kg = delivery_weight_per_unit * delivery_qty
                    cargo_weight_kg -= unloaded_kg
                    old_heavy = is_heavy
                    adj_A, adj_B, is_heavy = _apply_cargo_effect(
                        cargo_weight_kg)
                    summary_unload = (
                        float(delivery_km),
                        delivery_spec['name'],
                        int(delivery_qty))
                    seg_has_event = True
                    seg_lines.append(
                        f"  - 하역: {delivery_spec['name']} "
                        f"{delivery_qty}개({unloaded_kg}kg) → "
                        f"잔여 화물 {cargo_weight_kg}kg")
                    if old_heavy and not is_heavy:
                        seg_lines.append(
                            f"    → 화물규정 해제! "
                            f"A:{adj_A:.1f}, B:{adj_B:.1f}")
                    elif is_heavy:
                        seg_lines.append("    → 여전히 초과, 규정 유지")

                # 연속 운항 트리거 (fallback)
                if ((not rest_due)
                        and journey.continuous_moving_time_hours
                        >= trigger_hours - 1e-9):
                    rest_due = True
                    seg_has_event = True
                    seg_lines.append(
                        f"  - 연속 운항 한계: "
                        f"{drive_cont_after_min:.1f}분 ≥ 임계 {thr_min}분 "
                        f"(rest_due=True)")

                # 휴게 지점 도착 → 휴식 판단
                at_stop = abs(
                    (journey.current_position_km / rest_stop_interval_km)
                    - round(journey.current_position_km
                            / rest_stop_interval_km)
                ) < 1e-6
                at_destination = distance_to_go <= 0.001

                need_rest = False
                rest_check_lines = []
                if rest_due and at_stop and not at_destination:
                    need_rest = True
                    rest_check_lines.append(
                        f"  - 휴식 검사: 휴게소 도착 시점에 의무 휴식 대기(rest_due). "
                        f"연속 운행 {drive_cont_after_min:.1f}분 "
                        f"(임계 {thr_min}분).")
                elif (at_stop and distance_to_go > 0.001
                      and journey.continuous_moving_time_hours > 1e-6):
                    cur_pos = journey.current_position_km
                    k = round(cur_pos / rest_stop_interval_km)
                    next_rest_km = (k + 1) * rest_stop_interval_km
                    target_km = min(next_rest_km, total_distance)

                    # 빠른 판정: 물리적으로 가능한 최고 속도를 가정해도
                    # 연속 운항 한계를 넘는다면 정밀 예측 없이 즉시 휴식.
                    distance_to_target = target_km - cur_pos
                    max_possible_speed = min(max_favorable_speed, max(adj_A, adj_B))
                    if max_possible_speed > 1e-9:
                        optimistic_time = distance_to_target / max_possible_speed
                    else:
                        optimistic_time = float("inf")

                    if (journey.continuous_moving_time_hours + optimistic_time
                            >= trigger_hours - 1e-9):
                        need_rest = True
                        opt_min = optimistic_time * 60.0
                        sum_pred = drive_cont_after_min + opt_min
                        rest_check_lines.append(
                            f"  - 휴식 검사: 현재 연속 운행 "
                            f"{drive_cont_after_min:.1f}분 + "
                            f"다음 휴게소({target_km:.1f}km)까지 "
                            f"{distance_to_target:.1f}km, "
                            f"낙관 최단(v={max_possible_speed:.1f}km/h) "
                            f"{opt_min:.1f}분 → 합계 {sum_pred:.1f}분 "
                            f"> 임계 {thr_min}분")
                    else:
                        used_delivery_split = False
                        if (not delivered
                                and delivery_km is not None
                                and cur_pos < delivery_km < target_km):
                            t1 = _compute_time_to(cur_pos, delivery_km)
                            post_weight = (cargo_weight_kg
                                           - delivery_weight_per_unit
                                           * delivery_qty)
                            post_A, post_B, _ = _apply_cargo_effect(
                                post_weight)
                            t2 = _compute_time_to(
                                delivery_km, target_km,
                                _adj_A=post_A, _adj_B=post_B,
                                _t_offset=t1)
                            time_to_target = t1 + t2
                            used_delivery_split = True
                        else:
                            time_to_target = _compute_time_to(
                                cur_pos, target_km)
                        if (journey.continuous_moving_time_hours + time_to_target
                                >= trigger_hours - 1e-9):
                            need_rest = True
                            t_tgt_min = time_to_target * 60.0
                            sum_pred = drive_cont_after_min + t_tgt_min
                            if used_delivery_split:
                                rest_check_lines.append(
                                    f"  - 휴식 검사: 현재 연속 운행 "
                                    f"{drive_cont_after_min:.1f}분 + "
                                    f"다음 휴게소까지(중간 하역 포함) "
                                    f"{t1 * 60:.1f}+{t2 * 60:.1f} = "
                                    f"{t_tgt_min:.1f}분 → 합계 {sum_pred:.1f}분 "
                                    f"> 임계 {thr_min}분")
                            else:
                                rest_check_lines.append(
                                    f"  - 휴식 검사: 현재 연속 운행 "
                                    f"{drive_cont_after_min:.1f}분 + "
                                    f"다음 휴게소({target_km:.1f}km)까지 "
                                    f"{distance_to_target:.1f}km 예상 "
                                    f"{t_tgt_min:.1f}분 → 합계 {sum_pred:.1f}분 "
                                    f"> 임계 {thr_min}분")

                if rest_check_lines:
                    seg_has_event = True
                    seg_lines.extend(rest_check_lines)

                if need_rest:
                    base_rest = (
                        regulations["mandatory_rest"]["duration_minutes"])
                    extra = rest_count * rest_increment
                    this_rest = base_rest + extra
                    journey.total_rest_time_hours += this_rest / 60.0
                    journey.continuous_moving_time_hours = 0
                    rest_due = False
                    rest_count += 1
                    seg_has_event = True
                    summary_rests.append(
                        (float(journey.current_position_km), int(this_rest)))
                    if extra > 0:
                        seg_lines.append(
                            f"  - 조치 [휴식 #{rest_count}]: "
                            f"{journey.current_position_km:.1f}km에서 "
                            f"{this_rest}분 휴식 "
                            f"({base_rest}+{extra}). "
                            f"연속 운행 시간 초기화")
                    else:
                        seg_lines.append(
                            f"  - 조치 [휴식 #{rest_count}]: "
                            f"{journey.current_position_km:.1f}km에서 "
                            f"{this_rest}분 휴식. "
                            f"연속 운행 시간 초기화")

                if seg_has_event:
                    solution.extend(seg_lines)
                else:
                    solution.append(
                        f"[SEG {step_cnt}] {in_zone}구역 "
                        f"{seg_start_km:.1f}→{seg_end_km:.1f}km · "
                        f"{cur_label} {actual_speed:.1f}km/h · "
                        f"{seg_dist:.1f}km/{seg_minutes:.1f}분 · "
                        f"누적 {drive_cont_after_min:.1f}분")

                step_cnt += 1

            if rest_due:
                raise ValueError("마지막 구간 연속 운항 한계 초과 (휴게소 없음)")

            max_segments = params.get("max_segments", 20)
            if step_cnt > max_segments:
                raise ValueError(f"구간 수({step_cnt}) 초과")

            summary_bits: list[str] = []
            if summary_rests:
                pts = ", ".join(
                    f"{km:.0f}km({m}분)" for km, m in summary_rests)
                summary_bits.append(
                    f"휴식 {len(summary_rests)}회 @ {pts}")
            if summary_unload is not None:
                uk, un, uq = summary_unload
                summary_bits.append(f"하역 1회 @ {uk:.0f}km({un} {uq}개)")
            if summary_congest_cnt:
                summary_bits.append(
                    f"혼잡 영향 {summary_congest_cnt}구간")
            if summary_bits:
                solution.insert(
                    step2_header_idx,
                    "  · 요약: " + " · ".join(summary_bits))

            rc_range = params.get("rest_count_range")
            if rc_range:
                rc_min, rc_max = rc_range
                if rc_min is not None and rest_count < rc_min:
                    raise ValueError(
                        f"휴식 횟수({rest_count}) < 최소({rc_min})")
                if rc_max is not None and rest_count > rc_max:
                    raise ValueError(
                        f"휴식 횟수({rest_count}) > 최대({rc_max})")

            # --- 3. 최종 질문 및 정답 ---
            protagonist = random.choice([
                "뱃사공 김씨", "물품 운송원 박씨", "하천 탐사대원 이씨",
                "수상 운송기사 최씨", "화물선 선장 한씨", "내수면 조종사 정씨",
                "강변 물류기사 윤씨", "수운 담당관 장씨", "선박 운항원 조씨",
                "하천 배달원 서씨", "수로 안내원 오씨", "강운 기관사 황씨"])
            opp_zone = "B" if current_favorable_zone == "A" else "A"

            if congestion_affected_zone == "all":
                cong_zone_desc = "모든 구역의 제한속도가"
            else:
                cong_zone_desc = (f"{congestion_affected_zone}구역의 "
                                  f"제한속도가")

            q_parts = [
                (f"{protagonist}는 이 강에서만 {career_years}년을 일한 "
                 f"베테랑으로, 총 길이 {total_distance}km의 상류 지역에 "
                 f"물품을 운송하는 임무를 맡았다. "
                 f"그는 {_fmt_hour(departure_hour)}에 "
                 f"{item1_weight}kg짜리 {item1_spec['name']} {item1_qty}개와 "
                 f"{item2_weight}kg짜리 {item2_spec['name']} {item2_qty}개를 "
                 f"싣고 출발했다. "
                 f"배는 정수(靜水)에서 시속 {base_boat_speed_kph}km로 "
                 f"이동 가능하다."),

                (f"이 강에는 시속 {current_speed_kph}km의 물살이 있는데, "
                 f"{current_favorable_zone}구역에서는 순류"
                 f"(실효 속력 = 배 속력 + 유속), "
                 f"{opp_zone}구역에서는 역류"
                 f"(실효 속력 = 배 속력 - 유속)이다."),

                (f"첫 {regulations['speed_limit']['zone_A_km']}km는 A구역"
                 f"(제한속도 "
                 f"{regulations['speed_limit']['zone_A_limit_kph']}km/h), "
                 f"이후 B구역"
                 f"(제한속도 "
                 f"{regulations['speed_limit']['zone_B_limit_kph']}km/h)이다."
                 f" 제한 속도는 유속 반영 후 실효 속력에 적용된다."),

                (f"안전 중량 기준"
                 f"({regulations['cargo_effect']['heavy_load_kg']}kg) "
                 f"초과 시, "
                 + (f"모든 구역의 제한속도가 "
                    f"{regulations['cargo_effect']['zone_A_reduction_percent']}"
                    f"% 감소한다."
                    if zone_a_reduction == zone_b_reduction
                    else
                    f"A구역 제한속도는 "
                    f"{regulations['cargo_effect']['zone_A_reduction_percent']}"
                    f"% 감소하고 "
                    f"B구역 제한속도는 "
                    f"{regulations['cargo_effect']['zone_B_reduction_percent']}"
                    f"% 감소한다.")),

                (f"{_fmt_hour(congestion_start_hour)}부터 "
                 f"{_fmt_hour(congestion_end_hour)}까지는 혼잡시간대로, "
                 f"{cong_zone_desc} 추가로 "
                 f"{congestion_reduction_pct}% 감소한다. "
                 f"이 감속은 화물 규정 적용 후 제한속도에 추가 적용된다."),
            ]

            if delivery_km:
                q_parts.append(
                    f"{delivery_km}km 지점의 중간 기착지에서 "
                    f"{delivery_spec['name']} {delivery_qty}개를 하역한다. "
                    f"하역 후 잔여 화물 무게에 따라 화물 규정이 재적용된다.")

            rest_desc = (
                f"연속 {regulations['mandatory_rest']['trigger_minutes']}분 "
                f"이상 운항할 수 없으며, 휴게 지점"
                f"(매 {rest_stop_interval_km}km)에서만 쉴 수 있다. "
                f"기본 휴식 시간은 "
                f"{regulations['mandatory_rest']['duration_minutes']}분이다.")
            if rest_increment > 0:
                base_r = regulations["mandatory_rest"]["duration_minutes"]
                rest_desc += (
                    f" 단, 누적 피로 규정에 따라 매 휴식마다 "
                    f"{rest_increment}분씩 추가된다"
                    f"(첫째 {base_r}분, 둘째 {base_r + rest_increment}분, "
                    f"셋째 {base_r + rest_increment * 2}분, …).")
            q_parts.append(rest_desc)

            q_parts.append(
                "이 모든 조건을 준수하여 최종 목적지까지 도착했을 때, "
                "의무 휴식을 포함한 총 소요 시간은 몇 시간 몇 분입니까? "
                "총 소요 시간을 먼저 분(分) 단위 정수로 계산한 후 "
                "시간과 분으로 변환하여 답하시오. "
                "(예: 총 1450분 → 24시간 10분)")

            question = "\n".join(q_parts)

            total_hours = journey.total_journey_time_hours
            total_minutes = round(total_hours * 60)
            hours = total_minutes // 60
            minutes = total_minutes % 60

            answer = f"{hours}시간 {minutes}분"
            solution.append(
                "[STEP 3] 답·검산\n"
                f"  - 총 소요: {total_hours:.4f}시간 = {total_minutes}분 = {answer}\n"
                "  - 위 '[SEG n]' 로그는 STEP2(풀이 전개)에 해당하며, "
                "합계/단위 변환(분→시·분)과 규정(무거운 화물 감속·의무 휴식)이 "
                "빠짐 없이 반영됐는지 끝에서 한번 더 확인한다.")

            return question, answer, solution

        except ValueError:
            continue

    raise RuntimeError(
        f"최대 재시도 횟수({max_retries})를 초과했습니다.")


def _build_rest_count_targets(num_questions, rest_count_range):
    """rest_count_range를 num_questions에 맞게 균등 배분한 타겟 리스트 반환."""
    if rest_count_range is None:
        return [None] * num_questions
    rc_min, rc_max = rest_count_range
    values = list(range(rc_min, rc_max + 1))
    random.shuffle(values)
    targets = []
    per_val = num_questions // len(values)
    remainder = num_questions % len(values)
    for i, v in enumerate(values):
        count = per_val + (1 if i < remainder else 0)
        targets.extend([v] * count)
    random.shuffle(targets)
    return targets

def create_dataset_files(num_questions, difficulty=None):
    import csv
    import json
    import pandas as pd
    from collections import Counter

    if difficulty is None:
        difficulties = ["easy", "medium", "hard"]
        total_questions = num_questions * len(difficulties)
        print(f"뱃사공 문제를 생성 중... (각 난이도별 {num_questions}개, "
              f"총 {total_questions}개)")
    else:
        difficulties = [difficulty]
        total_questions = num_questions
        print(f"뱃사공 문제 {num_questions}개를 생성 중... "
              f"(난이도: {difficulty})")

    output = []
    seen_questions = set()
    unique_answers = set()
    difficulty_counts = {diff: 0 for diff in difficulties}

    for diff in difficulties:
        targets = _build_rest_count_targets(
            num_questions, REST_COUNT_RANGES.get(diff))
        if REST_COUNT_RANGES.get(diff):
            dist = Counter(targets)
            print(f"\n[{diff.upper()}] 난이도 {num_questions}개 생성 중... "
                  f"(휴식 횟수 배분: {dict(sorted(dist.items()))})")
        else:
            print(f"\n[{diff.upper()}] 난이도 {num_questions}개 생성 중...")

        diff_count = 0
        attempt_count = 0
        max_attempts = num_questions * 200

        while diff_count < num_questions and attempt_count < max_attempts:
            attempt_count += 1
            try:
                target = targets[diff_count]
                q, answer, expl = generate_puzzle_question(
                    difficulty=diff, rest_count_target=target)
                if q not in seen_questions:
                    output.append([q, answer, "\n".join(expl), diff])
                    seen_questions.add(q)
                    unique_answers.add(answer)
                    difficulty_counts[diff] += 1
                    diff_count += 1
                    if diff_count % 10 == 0:
                        print(f"  진행: {diff_count}/{num_questions}")
            except Exception:
                continue

        if diff_count < num_questions:
            print(f"  경고: [{diff}] 목표 {num_questions}개 중 "
                  f"{diff_count}개만 생성되었습니다.")

    print(f"\n생성 통계:")
    print(f"  생성된 문제 수: {len(output)}")
    print(f"  고유한 문제 수: {len(seen_questions)}")
    print(f"  고유한 정답 수: {len(unique_answers)}")
    print(f"\n난이도별 분포:")
    for diff in sorted(difficulty_counts):
        print(f"{diff:<6} {difficulty_counts[diff]}")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "ferryman_ko.csv"
    ferryman_json = []
    diff_counters = {}
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "answer", "solution", "difficulty"])
        for i, (question, answer, solution, diff) in enumerate(output):
            diff_idx = diff_counters.get(diff, 0)
            diff_counters[diff] = diff_idx + 1
            qid = f"ferryman_ko_{diff}_{diff_idx:04d}"
            row = {
                "id": qid,
                "question": question,
                "answer": answer,
                "solution": solution,
                "difficulty": diff,
            }
            ferryman_json.append(row)
            writer.writerow([qid, question, answer, solution, diff])
    print(f"\nCSV 파일이 생성: {csv_path}")

    json_dir = PROJECT_ROOT / "data" / "jsonl"
    json_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = json_dir / "ferryman_ko.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in ferryman_json:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL 파일이 생성: {jsonl_path}")

    ferryman_df = pd.DataFrame(
        ferryman_json, columns=["id", "question", "answer", "solution", "difficulty"])
    return ferryman_df, ferryman_json


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Ferryman Puzzle Generator")
    parser.add_argument(
        "--num", type=int, default=100,
        help="Number of questions per difficulty level.")
    parser.add_argument(
        "--difficulty", type=str, default=None,
        choices=["easy", "medium", "hard"],
        help="Difficulty level. If not specified, all three.")

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
    #     print("\n")