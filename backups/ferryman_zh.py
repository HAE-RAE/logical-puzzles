import random
from pathlib import Path

SFT_SOLUTION_RUBRIC_ZH = (
    "STEP0=题述 · STEP1=已知条件 · STEP2=分步推演 · "
    "STEP3=答案与验算"
)


class JourneyState:
    def __init__(self):
        self.total_moving_time_hours = 0.0
        self.total_rest_time_hours = 0.0
        self.continuous_moving_time_hours = 0.0
        self.current_position_km = 0.0

    @property
    def continuous_drive_time_min(self):
        """连续航行时间（分钟）；休息后与 continuous_moving_time_hours 一并清零。"""
        return self.continuous_moving_time_hours * 60.0

    @property
    def total_journey_time_hours(self):
        return self.total_moving_time_hours + self.total_rest_time_hours


def _fmt_hour(h):
    if h < 12:
        return f"上午{h}时"
    elif h == 12:
        return "中午12时"
    else:
        return f"下午{h - 12}时"


def _fmt_hhmm_from_decimal_hour(dec_h):
    """将模拟绝对时刻（小数小时）格式化为一日内的 HH:MM。"""
    minutes = int(round((float(dec_h) % 24.0) * 60.0)) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _fmt_minutes_from_hours(hours):
    """将小时数格式化为分钟（小数）字符串。"""
    return f"{hours * 60.0:.2f}"


REST_COUNT_RANGES = {
    "easy":   (10, 11), # target 75%; current 82% needs one more rest tier
    "medium": (14, 16), # target 50%; current 49% is nearly calibrated
    "hard":   (30, 30), # target 25%; keep stable generation, tune burden via parameters
}


def generate_puzzle_question(difficulty="easy", rest_count_target=None):
    max_retries = 3000

    for attempt in range(max_retries):
        try:
            if difficulty == "easy":
                params = {
                    "distance_range": (75, 110),             # slightly harder easy, still below medium
                    "zone_a_range": (15, 25),
                    "zone_a_limit_range": (25, 38),
                    "zone_b_limit_range": (23, 26),
                    "rest_trigger_minutes_range": (40, 56),
                    "rest_duration_minutes_range": (27, 50),
                    "rest_stop_interval_km_range": (7, 9),    # gap=1 to med min(10) — shorter intervals allow up to 11 rests
                    "heavy_threshold_range": (1500, 2500),   # gap=500 to med min(1000)
                    "zone_a_reduction_range": (10, 16),
                    "zone_b_reduction_range": (10, 18),
                    "uniform_reduction": False,
                    "base_speed_bonus_range": (8, 18),
                    "current_speed_range": (2, 3),
                    "has_delivery": True,
                    "rest_increment_range": (7, 9),
                    "congestion_start_offset_range": (2, 4),
                    "congestion_duration_range": (1, 2),     # w=1, gap=0 to med min(3)
                    "congestion_reduction_range": (10, 18),
                    "congestion_affected_zone": "all",
                    "max_segments": 45,
                    "rest_count_range": (10, 11),
                }
            elif difficulty == "medium":
                params = {
                    "distance_range": (155, 200),            # tiny ease toward exact 50% while staying above easy
                    "zone_a_range": (18, 35),
                    "zone_a_limit_range": (21, 33),
                    "zone_b_limit_range": (19, 22),
                    "rest_trigger_minutes_range": (84, 100),
                    "rest_duration_minutes_range": (30, 53),
                    "rest_stop_interval_km_range": (11, 13),  # supports 14-16 rests with fewer stop decisions
                    "heavy_threshold_range": (650, 1050),
                    "zone_a_reduction_range": (13, 23),
                    "zone_b_reduction_range": (19, 27),
                    "uniform_reduction": False,
                    "base_speed_bonus_range": (6, 12),
                    "current_speed_range": (5, 6),           # w=1, gap=1 from easy(3), gap=0 to hard min(7)
                    "has_delivery": True,
                    "rest_increment_range": (11, 15),
                    "congestion_start_offset_range": (1, 2),
                    "congestion_duration_range": (2, 4),
                    "congestion_reduction_range": (18, 26),
                    "congestion_affected_zone": random.choice(["all", "B"]),
                    "max_segments": 100,
                    "rest_count_range": (14, 16),
                }
            else:  # hard
                params = {
                    "distance_range": (380, 460),            # slightly harder, longer multi-day journey
                    "zone_a_range": (20, 45),
                    "zone_a_limit_range": (14, 25),
                    "zone_b_limit_range": (10, 14),
                    "rest_trigger_minutes_range": (84, 104),
                    "rest_duration_minutes_range": (68, 112),
                    "rest_stop_interval_km_range": (8, 9),
                    "heavy_threshold_range": (180, 350),
                    "zone_a_reduction_range": (24, 36),
                    "zone_b_reduction_range": (42, 56),
                    "uniform_reduction": False,
                    "base_speed_bonus_range": (2, 7),
                    "current_speed_range": (7, 9),
                    "has_delivery": True,
                    "rest_increment_range": (46, 62),
                    "congestion_start_offset_range": (1, 2),
                    "congestion_duration_range": (6, 8),
                    "congestion_reduction_range": (48, 62),
                    "congestion_affected_zone": "B",
                    "max_segments": 240,
                    "rest_count_range": (30, 30),
                }

            if rest_count_target is not None:
                params["rest_count_range"] = (rest_count_target, rest_count_target)

            # --- 1. System rules and random variable initialization ---
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
                {"name": "救援物资箱", "weight_range": (30, 70)},
                {"name": "医疗包", "weight_range": (5, 15)},
                {"name": "储水桶", "weight_range": (15, 25)},
                {"name": "建材", "weight_range": (50, 80)}
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

            # Mid-journey delivery setup
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

            # Congestion time setup
            departure_hour = random.randint(7, 9)
            cong_offset = random.randint(*params["congestion_start_offset_range"])
            congestion_start_hour = departure_hour + cong_offset
            cong_dur = random.randint(*params["congestion_duration_range"])
            congestion_end_hour = congestion_start_hour + cong_dur
            congestion_reduction_pct = random.randint(
                *params["congestion_reduction_range"])
            congestion_affected_zone = params["congestion_affected_zone"]

            career_years = random.randint(5, 15)

            # --- 2. Simulation ---
            journey = JourneyState()
            solution = [
                SFT_SOLUTION_RUBRIC_ZH,
                "[STEP 0] 题述",
                "  - 航行、休息、货物与拥堵的数值模拟；"
                "最终数值答案仅写在 [STEP 3]。",
                "[STEP 1] 已知条件（初值与规则）",
            ]
            solution.append(
                f"  - 总航程={total_distance}km, "
                f"静水船速={base_boat_speed_kph}km/h, "
                f"出发={_fmt_hour(departure_hour)}")
            solution.append(
                f"  - 流速: {current_speed_kph}km/h "
                f"（区域{current_favorable_zone}为顺流）")
            solution.append(
                f"  - 货物: {item1_weight}kg {item1_spec['name']}×{item1_qty}"
                f" + {item2_weight}kg {item2_spec['name']}×{item2_qty}")
            solution.append(
                f"  - 拥堵: {_fmt_hour(congestion_start_hour)}至"
                f"{_fmt_hour(congestion_end_hour)}, "
                f"{'全区域' if congestion_affected_zone == 'all' else '区域' + congestion_affected_zone}"
                f" -{congestion_reduction_pct}%")
            if delivery_km:
                solution.append(
                    f"  - 中途卸货: {delivery_km}km处, "
                    f"{delivery_spec['name']} ×{delivery_qty}")
            if rest_increment > 0:
                solution.append(
                    f"  - 累计疲劳: 每次休息额外 +{rest_increment}分钟")
            solution.append(f"  - 规则: {regulations}")
            solution.append("[STEP 2] 分步推演（按航段日志）")
            step2_header_idx = len(solution)
            summary_rests: list[tuple[float, int]] = []
            summary_unload: tuple[float, str, int] | None = None
            summary_congest_cnt = 0
            step_cnt = 1

            solution.append(
                f"[SEG {step_cnt}] 货物重量: "
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
                    f"[SEG {step_cnt}] 货物规则: {cargo_weight_kg}kg > "
                    f"{regulations['cargo_effect']['heavy_load_kg']}kg → "
                    f"A区 -{ra}%({adj_A:.1f}), B区 -{rb}%({adj_B:.1f})")
                step_cnt += 1

            distance_to_go = total_distance
            rest_due = False
            trigger_hours = (regulations["mandatory_rest"]["trigger_minutes"]
                             / 60.0)
            rest_stop_interval_km = (
                regulations["mandatory_rest"]["rest_stop_interval_km"])
            rest_count = 0
            delivered = delivery_km is None

            # --- Continuous operation constraint pre-validation ---
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
                    raise ValueError("最坏情况速度≤0")
                if rest_stop_interval_km / _worst_speed >= trigger_hours:
                    raise ValueError(
                        f"连续航行矛盾: 区域{_zone}")

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
                    abs_hr = cur_abs % 24  # 24-hour cycle correction
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
                        raise ValueError("_compute_time_to: 速度≤0")

                    if not in_cong:
                        if abs_hr < congestion_start_hour:
                            t_to_s = congestion_start_hour - abs_hr
                        else:  # today's congestion already passed → time to next day's congestion
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
                        raise ValueError("_compute_time_to: 位移为0")

                    t += dz / spd
                    pos += dz
                    remaining -= dz
                else:
                    raise ValueError("_compute_time_to: 超过最大迭代")
                return t

            max_favorable_speed = base_boat_speed_kph + current_speed_kph
            zone_a_boundary = regulations["speed_limit"]["zone_A_km"]

            # --- Main simulation loop ---
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

                abs_time = (departure_hour
                            + journey.total_moving_time_hours
                            + journey.total_rest_time_hours)
                if abs(abs_time - round(abs_time)) < 1e-9:
                    abs_time = round(abs_time)
                abs_hour = abs_time % 24  # 24-hour cycle correction
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
                    cur_label = "顺流"
                else:
                    eff_speed = base_boat_speed_kph - current_speed_kph
                    cur_label = "逆流"

                actual_speed = min(eff_speed, speed_limit)
                if actual_speed <= 0:
                    raise ValueError("无效速度")

                dist_to_stop = _next_rest_stop_dist(current_pos)
                if dist_to_stop < 1e-6:
                    dist_to_stop = rest_stop_interval_km

                boundaries = [distance_to_go, distance_in_zone, dist_to_stop]
                if not delivered:
                    dist_to_delivery = delivery_km - current_pos
                    if dist_to_delivery > 1e-6:
                        boundaries.append(dist_to_delivery)

                if not in_congestion:
                    if abs_hour < congestion_start_hour:
                        t_to_cong = congestion_start_hour - abs_hour
                    else:  # today's congestion already passed → time to next day's congestion
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
                    f"[SEG {step_cnt}] {tag_str}区域{in_zone} "
                    f"({seg_start_km:.1f}km 至 {seg_end_km:.1f}km)")
                seg_lines.append(
                    f"  - 当前时刻: {_fmt_hhmm_from_decimal_hour(abs_time)} "
                    f"({abs_time:.2f}h)")
                cong_note = (
                    f"，拥堵上限 {limit_after_congest_kph:.1f}km/h"
                    if cong_applies else "")
                seg_lines.append(
                    f"  - 速度核算: {cur_label} 实效 "
                    f"{base_boat_speed_kph}{'+' if cur_label == '顺流' else '-'}"
                    f"{current_speed_kph}={eff_speed:.1f}km/h, "
                    f"区域上限 {limit_zone_kph:.1f}km/h{cong_note} "
                    f"→ 采用 {actual_speed:.1f}km/h")
                seg_lines.append(
                    f"  - 航行: {seg_dist:.1f}km / {actual_speed:.1f}km/h = "
                    f"{seg_time:.3f}h ({_fmt_minutes_from_hours(seg_time)} 分钟)")
                seg_lines.append(
                    f"  - 连续航行累计: "
                    f"{drive_cont_before_min:.1f} 分钟 + {seg_minutes:.1f} 分钟 = "
                    f"{drive_cont_after_min:.1f} 分钟")

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
                        f"  - 卸货: {delivery_spec['name']} ×{delivery_qty} "
                        f"（{unloaded_kg}kg）→ 剩余货物 {cargo_weight_kg}kg")
                    if old_heavy and not is_heavy:
                        seg_lines.append(
                            f"    → 货物规则解除！ "
                            f"A:{adj_A:.1f}, B:{adj_B:.1f}")
                    elif is_heavy:
                        seg_lines.append(
                            "    → 仍超重，规则继续")

                if ((not rest_due)
                        and journey.continuous_moving_time_hours
                        >= trigger_hours - 1e-9):
                    rest_due = True
                    seg_has_event = True
                    seg_lines.append(
                        f"  - 连续航行上限: "
                        f"{drive_cont_after_min:.1f} 分钟 ≥ 阈值 "
                        f"{thr_min} 分钟（rest_due=True）")

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
                        f"  - 休息判定: 在休息点待强制休息（rest_due）。"
                        f"连续航行 {drive_cont_after_min:.1f} 分钟 "
                        f"（阈值 {thr_min} 分钟）。")
                elif (at_stop and distance_to_go > 0.001
                      and journey.continuous_moving_time_hours > 1e-6):
                    cur_pos = journey.current_position_km
                    k = round(cur_pos / rest_stop_interval_km)
                    next_rest_km = (k + 1) * rest_stop_interval_km
                    target_km = min(next_rest_km, total_distance)

                    distance_to_target = target_km - cur_pos
                    max_possible_speed = min(
                        max_favorable_speed, max(adj_A, adj_B))
                    if max_possible_speed > 1e-9:
                        optimistic_time = (
                            distance_to_target / max_possible_speed)
                    else:
                        optimistic_time = float("inf")

                    if (journey.continuous_moving_time_hours + optimistic_time
                            >= trigger_hours - 1e-9):
                        need_rest = True
                        opt_min = optimistic_time * 60.0
                        sum_pred = drive_cont_after_min + opt_min
                        rest_check_lines.append(
                            f"  - 休息判定: 当前连续航行 "
                            f"{drive_cont_after_min:.1f} 分钟 + "
                            f"到下一休息点（{target_km:.1f}km）"
                            f"{distance_to_target:.1f} km, "
                            f"乐观最短（v={max_possible_speed:.1f}km/h） "
                            f"{opt_min:.1f} 分钟 → 合计 {sum_pred:.1f} 分钟 "
                            f"> 阈值 {thr_min} 分钟")
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
                        if (journey.continuous_moving_time_hours
                                + time_to_target
                                >= trigger_hours - 1e-9):
                            need_rest = True
                            t_tgt_min = time_to_target * 60.0
                            sum_pred = drive_cont_after_min + t_tgt_min
                            if used_delivery_split:
                                rest_check_lines.append(
                                    f"  - 休息判定: 当前连续航行 "
                                    f"{drive_cont_after_min:.1f} 分钟 + "
                                    f"到下一休息点（含中途卸货） "
                                    f"{t1 * 60:.1f}+{t2 * 60:.1f} = "
                                    f"{t_tgt_min:.1f} 分钟 → 合计 "
                                    f"{sum_pred:.1f} 分钟 > 阈值 "
                                    f"{thr_min} 分钟")
                            else:
                                rest_check_lines.append(
                                    f"  - 休息判定: 当前连续航行 "
                                    f"{drive_cont_after_min:.1f} 分钟 + "
                                    f"到下一休息点（{target_km:.1f}km）"
                                    f"{distance_to_target:.1f} km 预计 "
                                    f"{t_tgt_min:.1f} 分钟 → 合计 "
                                    f"{sum_pred:.1f} 分钟 > 阈值 "
                                    f"{thr_min} 分钟")

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
                            f"  - 措施 [休息 #{rest_count}]: "
                            f"在 {journey.current_position_km:.1f}km 休息 "
                            f"{this_rest} 分钟 "
                            f"（{base_rest}+{extra}）。"
                            f"连续航行时间清零")
                    else:
                        seg_lines.append(
                            f"  - 措施 [休息 #{rest_count}]: "
                            f"在 {journey.current_position_km:.1f}km 休息 "
                            f"{this_rest} 分钟。"
                            f"连续航行时间清零")

                if seg_has_event:
                    solution.extend(seg_lines)
                else:
                    solution.append(
                        f"[SEG {step_cnt}] 区域{in_zone} "
                        f"{seg_start_km:.1f}→{seg_end_km:.1f}km | "
                        f"{cur_label} {actual_speed:.1f}km/h | "
                        f"{seg_dist:.1f}km/{seg_minutes:.1f}分钟 | "
                        f"累计 {drive_cont_after_min:.1f}分钟")

                step_cnt += 1

            if rest_due:
                raise ValueError(
                    "末段超过连续上限（无休息点）")

            max_segments = params.get("max_segments", 20)
            if step_cnt > max_segments:
                raise ValueError(f"航段数({step_cnt})超过上限")

            summary_bits: list[str] = []
            if summary_rests:
                pts = ", ".join(
                    f"{km:.0f}km（{m}分钟）" for km, m in summary_rests)
                summary_bits.append(
                    f"休息 {len(summary_rests)}次 @ {pts}")
            if summary_unload is not None:
                uk, un, uq = summary_unload
                summary_bits.append(f"卸货 1次 @ {uk:.0f}km（{un} ×{uq}）")
            if summary_congest_cnt:
                summary_bits.append(
                    f"拥堵影响 {summary_congest_cnt} 段")
            if summary_bits:
                solution.insert(
                    step2_header_idx,
                    "  · 摘要: " + " | ".join(summary_bits))

            rc_range = params.get("rest_count_range")
            if rc_range:
                rc_min, rc_max = rc_range
                if rc_min is not None and rest_count < rc_min:
                    raise ValueError(
                        f"休息次数({rest_count}) < 最小({rc_min})")
                if rc_max is not None and rest_count > rc_max:
                    raise ValueError(
                        f"休息次数({rest_count}) > 最大({rc_max})")

            # --- 3. 最终题干与答案 ---
            protagonist = random.choice([
                "摆渡人老王", "货运员小张", "河道巡查员小刘",
                "水运调度员小陈", "货船船长老赵", "内河驾驶员小吴",
                "河岸物流司机小周", "水运调度老郑", "船舶操作员小孙",
                "河道快递员小马", "航道引导员小朱", "河运工程师小胡"])
            opp_zone = "B" if current_favorable_zone == "A" else "A"

            if congestion_affected_zone == "all":
                cong_zone_desc = "各区域限速均"
            else:
                cong_zone_desc = (
                    f"区域{congestion_affected_zone}的"
                    f"限速")

            q_parts = [
                (f"{protagonist}在这条河上工作了{career_years}年，"
                 f"奉命将货物运往总长{total_distance}km的上游地区。"
                 f"{_fmt_hour(departure_hour)}出发，"
                 f"装载{item1_spec['name']}{item1_weight}kg×{item1_qty}件与"
                 f"{item2_spec['name']}{item2_weight}kg×{item2_qty}件。"
                 f"船在静水中的航速为{base_boat_speed_kph}km/h。"),

                (f"河流流速为{current_speed_kph}km/h。"
                 f"区域{current_favorable_zone}为顺流"
                 f"（实效航速＝船速＋流速），"
                 f"区域{opp_zone}为逆流"
                 f"（实效航速＝船速−流速）。"),

                (f"前{regulations['speed_limit']['zone_A_km']}km为A区"
                 f"（限速"
                 f"{regulations['speed_limit']['zone_A_limit_kph']}km/h），"
                 f"之后为B区"
                 f"（限速"
                 f"{regulations['speed_limit']['zone_B_limit_kph']}km/h）。"
                 f"限速作用于考虑流速后的实效航速。"),

                (f"若货物超过安全重量阈值"
                 f"（{regulations['cargo_effect']['heavy_load_kg']}kg），"
                 + (f"则所有区域限速均降低"
                    f"{regulations['cargo_effect']['zone_A_reduction_percent']}"
                    f"%。"
                    if zone_a_reduction == zone_b_reduction
                    else
                    f"A区限速降低"
                    f"{regulations['cargo_effect']['zone_A_reduction_percent']}"
                    f"%，B区限速降低"
                    f"{regulations['cargo_effect']['zone_B_reduction_percent']}"
                    f"%。")),

                (f"{_fmt_hour(congestion_start_hour)}至"
                 f"{_fmt_hour(congestion_end_hour)}为拥堵时段，"
                 f"期间{cong_zone_desc}再降低"
                 f"{congestion_reduction_pct}%。"
                 f"该减速在货物规则生效后的限速之上叠加。"),
            ]

            if delivery_km:
                q_parts.append(
                    f"在{delivery_km}km处卸下"
                    f"{delivery_spec['name']}{delivery_qty}件。"
                    f"卸货后按剩余货重重新适用货物规则。")

            rest_desc = (
                f"连续航行不得超过{regulations['mandatory_rest']['trigger_minutes']}分钟。"
                f"仅可在指定休息点"
                f"（每{rest_stop_interval_km}km）休息。"
                f"基础休息时长为"
                f"{regulations['mandatory_rest']['duration_minutes']}分钟。")
            if rest_increment > 0:
                base_r = regulations["mandatory_rest"]["duration_minutes"]
                rest_desc += (
                    f"但依累计疲劳规定，"
                    f"每次休息递增{rest_increment}分钟"
                    f"（第1次:{base_r}分钟，第2次:{base_r + rest_increment}分钟，"
                    f"第3次:{base_r + rest_increment * 2}分钟，…）。")
            q_parts.append(rest_desc)

            q_parts.append(
                "在严格遵守以上规则抵达终点时，"
                "含强制休息在内的总耗时是多少小时多少分钟？"
                "请先将总耗时算为整分钟数，"
                "再换算为小时与分钟作答。"
                "（例：合计1450分钟 → 24小时10分钟）")

            question = "\n".join(q_parts)

            total_hours = journey.total_journey_time_hours
            total_minutes = round(total_hours * 60)
            hours = total_minutes // 60
            minutes = total_minutes % 60

            answer = f"{hours}小时{minutes}分钟"
            solution.append(
                "[STEP 3] 答案与验算\n"
                f"  - 总耗时: {total_hours:.4f}h = {total_minutes}分钟 = {answer}\n"
                "  - 上文 '[SEG n]' 为 STEP2 分步日志；"
                "最后核对分钟→时分的换算及重载限速、强制休息是否全部落实。")

            return question, answer, solution

        except ValueError:
            continue

    raise RuntimeError(
        f"超过最大重试次数（{max_retries}）。")


def _build_rest_count_targets(num_questions, rest_count_range):
    """将 rest_count_range 均匀分配到 num_questions 个目标。"""
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
        print(f"正在生成 ferryman（简体中文）… "
              f"（各难度 {num_questions} 条，共 {total_questions} 条）")
    else:
        difficulties = [difficulty]
        total_questions = num_questions
        print(f"正在生成 ferryman（简体中文）{num_questions} 条… "
              f"（难度: {difficulty}）")

    output = []
    seen_questions = set()
    unique_answers = set()
    difficulty_counts = {diff: 0 for diff in difficulties}

    for diff in difficulties:
        targets = _build_rest_count_targets(
            num_questions, REST_COUNT_RANGES.get(diff))
        if REST_COUNT_RANGES.get(diff):
            dist = Counter(targets)
            print(f"\n[{diff.upper()}] 正在生成 {num_questions} 条… "
                  f"（休息次数分布: {dict(sorted(dist.items()))}）")
        else:
            print(f"\n[{diff.upper()}] 正在生成 {num_questions} 条…")

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
                        print(f"  进度: {diff_count}/{num_questions}")
            except Exception:
                continue

        if diff_count < num_questions:
            print(f"  警告: [{diff}] 仅生成 {diff_count}/{num_questions} 条。")

    print(f"\n生成统计:")
    print(f"  题目数: {len(output)}")
    print(f"  唯一题目: {len(seen_questions)}")
    print(f"  唯一答案: {len(unique_answers)}")
    print(f"\n难度分布:")
    for diff in sorted(difficulty_counts):
        print(f"{diff:<6} {difficulty_counts[diff]}")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "ferryman_zh.csv"
    ferryman_json = []
    diff_counters = {}
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "answer", "solution", "difficulty"])
        for i, (question, answer, solution, diff) in enumerate(output):
            diff_idx = diff_counters.get(diff, 0)
            diff_counters[diff] = diff_idx + 1
            qid = f"ferryman_zh_{diff}_{diff_idx:04d}"
            row = {
                "id": qid,
                "question": question,
                "answer": answer,
                "solution": solution,
                "difficulty": diff,
            }
            ferryman_json.append(row)
            writer.writerow([qid, question, answer, solution, diff])
    print(f"\n已写入 CSV: {csv_path}")

    json_dir = PROJECT_ROOT / "data" / "jsonl"
    json_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = json_dir / "ferryman_zh.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in ferryman_json:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"已写入 JSONL: {jsonl_path}")

    ferryman_df = pd.DataFrame(
        ferryman_json,
        columns=["id", "question", "answer", "solution", "difficulty"])
    return ferryman_df, ferryman_json


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Ferryman 题目生成（简体中文）")
    parser.add_argument(
        "--num", type=int, default=100,
        help="Number of questions per difficulty level.")
    parser.add_argument(
        "--difficulty", type=str, default=None,
        choices=["easy", "medium", "hard"],
        help="Difficulty level. If not specified, all three.")

    args = parser.parse_args()
    create_dataset_files(num_questions=args.num, difficulty=args.difficulty)

    print("\n" + "="*80)
    print("Sample puzzles (one per difficulty)")
    print("="*80)
    for diff in ["easy", "medium", "hard"]:
        question, answer, solution = generate_puzzle_question(difficulty=diff)
        print(f"\n========== [{diff.upper()}] Sample ==========")
        print("- question -\n", question)
        print("\n- answer -\n", answer)
        print("\n- solution -")
        for step in solution:
            print(step)
        print("\n")
