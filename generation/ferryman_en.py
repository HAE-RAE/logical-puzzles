import random
from pathlib import Path

SFT_SOLUTION_RUBRIC_EN = (
    "STEP0=meta · STEP1=given · STEP2=worked solution · "
    "STEP3=answer and verification"
)


class JourneyState:
    def __init__(self):
        self.total_moving_time_hours = 0.0
        self.total_rest_time_hours = 0.0
        self.continuous_moving_time_hours = 0.0
        self.current_position_km = 0.0

    @property
    def continuous_drive_time_min(self):
        """Continuous driving time in minutes; resets with continuous_moving_time_hours after rest."""
        return self.continuous_moving_time_hours * 60.0

    @property
    def total_journey_time_hours(self):
        return self.total_moving_time_hours + self.total_rest_time_hours


def _fmt_hour(h):
    if h < 12:
        return f"{h}:00 AM"
    elif h == 12:
        return "12:00 PM"
    else:
        return f"{h - 12}:00 PM"


def _fmt_hhmm_from_decimal_hour(dec_h):
    """Format simulation absolute time (decimal hours) as HH:MM within a 24h cycle."""
    minutes = int(round((float(dec_h) % 24.0) * 60.0)) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _fmt_minutes_from_hours(hours):
    """Decimal minutes as a string from duration in hours."""
    return f"{hours * 60.0:.2f}"


REST_COUNT_RANGES = {
    "easy":   (8, 11),  # interval(7-9)+trigger(40-60) achievable / gap=2 to med min(14)
    "medium": (14, 17), # gap=2 from easy max(11), gap=5 to hard min(22) / interval(10-13)+dist(150-200) achievable
    "hard":   (26, 33), # gap=9 from med max(17) / slight hard-only uplift
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
                    "rest_trigger_minutes_range": (40, 60),   # gap=15 to med min(75) — lowered for more frequent rests
                    "rest_duration_minutes_range": (25, 50),
                    "rest_stop_interval_km_range": (7, 9),    # gap=1 to med min(10) — shorter intervals allow up to 11 rests
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
                    "rest_count_range": (8, 11),             # w=3, gap=2 to med min(14) — interval(7-9)+trigger(40-60) allows up to 11
                }
            elif difficulty == "medium":
                params = {
                    "distance_range": (150, 200),            # w=50, gap=50 from easy(100), gap=40 to hard(240)
                    "zone_a_range": (18, 35),
                    "zone_a_limit_range": (20, 32),
                    "zone_b_limit_range": (18, 21),          # w=3, gap=1 from easy(23), gap=1 to hard max(16)
                    "rest_trigger_minutes_range": (75, 93),   # gap=1 to hard min(95)
                    "rest_duration_minutes_range": (35, 65),
                    "rest_stop_interval_km_range": (10, 13),  # gap=1 from easy max(9) — physical limit resolved
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
                    "distance_range": (270, 330),            # w=60, gap=70 from med(200) — longer multi-day journey
                    "zone_a_range": (20, 45),
                    "zone_a_limit_range": (17, 29),
                    "zone_b_limit_range": (13, 16),          # w=3, gap=1 from med min(18)
                    "rest_trigger_minutes_range": (95, 120),  # gap=1 from med max(93) — prevent physical contradiction
                    "rest_duration_minutes_range": (45, 85),
                    "rest_stop_interval_km_range": (8, 11),   # gap=2 from med max(13) — shortened for worst-case speed
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
                    "rest_count_range": (26, 33),            # w=7, gap=9 from med max(17) / slight hard-only uplift
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
                {"name": "relief supply box", "weight_range": (30, 70)},
                {"name": "medical kit", "weight_range": (5, 15)},
                {"name": "water barrel", "weight_range": (15, 25)},
                {"name": "construction material", "weight_range": (50, 80)}
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
                SFT_SOLUTION_RUBRIC_EN,
                "[STEP 0] Problem meta",
                "  - River-trip simulation with cargo, congestion, and rests; "
                "final numeric answer lives only in [STEP 3].",
                "[STEP 1] Given (initial conditions & rules)",
            ]
            solution.append(
                f"  - Total distance={total_distance}km, "
                f"boat still-water speed={base_boat_speed_kph}km/h, "
                f"departure={_fmt_hour(departure_hour)}")
            solution.append(
                f"  - Current: {current_speed_kph}km/h "
                f"(downstream in Zone {current_favorable_zone})")
            solution.append(
                f"  - Cargo: {item1_weight}kg {item1_spec['name']}x{item1_qty}"
                f" + {item2_weight}kg {item2_spec['name']}x{item2_qty}")
            solution.append(
                f"  - Congestion: {_fmt_hour(congestion_start_hour)}~"
                f"{_fmt_hour(congestion_end_hour)}, "
                f"{'all zones' if congestion_affected_zone == 'all' else 'Zone ' + congestion_affected_zone}"
                f" -{congestion_reduction_pct}%")
            if delivery_km:
                solution.append(
                    f"  - Mid-journey delivery: {delivery_km}km, "
                    f"{delivery_spec['name']} x{delivery_qty}")
            if rest_increment > 0:
                solution.append(
                    f"  - Cumulative fatigue: +{rest_increment}min per rest")
            solution.append(f"  - Regulations: {regulations}")
            solution.append("[STEP 2] Worked solution (per-segment simulation log)")
            step2_header_idx = len(solution)
            summary_rests: list[tuple[float, int]] = []
            summary_unload: tuple[float, str, int] | None = None
            summary_congest_cnt = 0
            step_cnt = 1

            solution.append(
                f"[SEG {step_cnt}] Cargo weight: "
                f"({item1_weight}x{item1_qty})+({item2_weight}x{item2_qty})"
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
                    f"[SEG {step_cnt}] Cargo regulation: {cargo_weight_kg}kg > "
                    f"{regulations['cargo_effect']['heavy_load_kg']}kg -> "
                    f"Zone A -{ra}%({adj_A:.1f}), Zone B -{rb}%({adj_B:.1f})")
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
                    raise ValueError("Worst-case speed <= 0")
                if rest_stop_interval_km / _worst_speed >= trigger_hours:
                    raise ValueError(
                        f"Continuous operation contradiction: Zone {_zone}")

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
                        raise ValueError("_compute_time_to: speed <= 0")

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
                        raise ValueError("_compute_time_to: zero distance")

                    t += dz / spd
                    pos += dz
                    remaining -= dz
                else:
                    raise ValueError("_compute_time_to: max iterations")
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
                    cur_label = "downstream"
                else:
                    eff_speed = base_boat_speed_kph - current_speed_kph
                    cur_label = "upstream"

                actual_speed = min(eff_speed, speed_limit)
                if actual_speed <= 0:
                    raise ValueError("Invalid speed")

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
                    f"[SEG {step_cnt}] {tag_str}Zone {in_zone} "
                    f"({seg_start_km:.1f}km ~ {seg_end_km:.1f}km)")
                seg_lines.append(
                    f"  - Current time: {_fmt_hhmm_from_decimal_hour(abs_time)} "
                    f"({abs_time:.2f}h)")
                cong_note = (
                    f", congestion limit {limit_after_congest_kph:.1f}km/h"
                    if cong_applies else "")
                seg_lines.append(
                    f"  - Speed check: {cur_label} effective "
                    f"{base_boat_speed_kph}{'+' if cur_label == 'downstream' else '-'}"
                    f"{current_speed_kph}={eff_speed:.1f}km/h, "
                    f"zone cap {limit_zone_kph:.1f}km/h{cong_note} "
                    f"-> applied {actual_speed:.1f}km/h")
                seg_lines.append(
                    f"  - Travel: {seg_dist:.1f}km / {actual_speed:.1f}km/h = "
                    f"{seg_time:.3f}h ({_fmt_minutes_from_hours(seg_time)} min)")
                seg_lines.append(
                    f"  - Continuous drive accrued: "
                    f"{drive_cont_before_min:.1f} min + {seg_minutes:.1f} min = "
                    f"{drive_cont_after_min:.1f} min")

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
                        f"  - Unload: {delivery_spec['name']} x{delivery_qty} "
                        f"({unloaded_kg}kg) -> remaining cargo {cargo_weight_kg}kg")
                    if old_heavy and not is_heavy:
                        seg_lines.append(
                            f"    -> Cargo regulation lifted! "
                            f"A:{adj_A:.1f}, B:{adj_B:.1f}")
                    elif is_heavy:
                        seg_lines.append(
                            "    -> Still overweight, regulation maintained")

                if ((not rest_due)
                        and journey.continuous_moving_time_hours
                        >= trigger_hours - 1e-9):
                    rest_due = True
                    seg_has_event = True
                    seg_lines.append(
                        f"  - Continuous drive limit: "
                        f"{drive_cont_after_min:.1f} min >= threshold "
                        f"{thr_min} min (rest_due=True)")

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
                        f"  - Rest check: mandatory rest pending (rest_due) "
                        f"at rest stop. Continuous drive "
                        f"{drive_cont_after_min:.1f} min "
                        f"(threshold {thr_min} min).")
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
                            f"  - Rest check: current continuous drive "
                            f"{drive_cont_after_min:.1f} min + "
                            f"to next rest ({target_km:.1f}km) "
                            f"{distance_to_target:.1f} km, "
                            f"optimistic min (v={max_possible_speed:.1f}km/h) "
                            f"{opt_min:.1f} min -> total {sum_pred:.1f} min "
                            f"> threshold {thr_min} min")
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
                                    f"  - Rest check: current continuous drive "
                                    f"{drive_cont_after_min:.1f} min + "
                                    f"to next rest (with mid-route unload) "
                                    f"{t1 * 60:.1f}+{t2 * 60:.1f} = "
                                    f"{t_tgt_min:.1f} min -> total "
                                    f"{sum_pred:.1f} min > threshold "
                                    f"{thr_min} min")
                            else:
                                rest_check_lines.append(
                                    f"  - Rest check: current continuous drive "
                                    f"{drive_cont_after_min:.1f} min + "
                                    f"to next rest ({target_km:.1f}km) "
                                    f"{distance_to_target:.1f} km expected "
                                    f"{t_tgt_min:.1f} min -> total "
                                    f"{sum_pred:.1f} min > threshold "
                                    f"{thr_min} min")

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
                            f"  - Action [REST #{rest_count}]: "
                            f"rest at {journey.current_position_km:.1f}km for "
                            f"{this_rest} min ({base_rest}+{extra}). "
                            f"Continuous drive time reset")
                    else:
                        seg_lines.append(
                            f"  - Action [REST #{rest_count}]: "
                            f"rest at {journey.current_position_km:.1f}km for "
                            f"{this_rest} min. "
                            f"Continuous drive time reset")

                if seg_has_event:
                    solution.extend(seg_lines)
                else:
                    solution.append(
                        f"[SEG {step_cnt}] Zone {in_zone} "
                        f"{seg_start_km:.1f}->{seg_end_km:.1f}km | "
                        f"{cur_label} {actual_speed:.1f}km/h | "
                        f"{seg_dist:.1f}km/{seg_minutes:.1f}min | "
                        f"accrued {drive_cont_after_min:.1f}min")

                step_cnt += 1

            if rest_due:
                raise ValueError(
                    "Last segment exceeds continuous limit (no rest stop)")

            max_segments = params.get("max_segments", 20)
            if step_cnt > max_segments:
                raise ValueError(f"Segment count({step_cnt}) exceeded")

            summary_bits: list[str] = []
            if summary_rests:
                pts = ", ".join(
                    f"{km:.0f}km({m}min)" for km, m in summary_rests)
                summary_bits.append(
                    f"Rests {len(summary_rests)}x @ {pts}")
            if summary_unload is not None:
                uk, un, uq = summary_unload
                summary_bits.append(f"Unload 1x @ {uk:.0f}km({un} x{uq})")
            if summary_congest_cnt:
                summary_bits.append(
                    f"Congestion {summary_congest_cnt} segs")
            if summary_bits:
                solution.insert(
                    step2_header_idx,
                    "  . Summary: " + " | ".join(summary_bits))

            rc_range = params.get("rest_count_range")
            if rc_range:
                rc_min, rc_max = rc_range
                if rc_min is not None and rest_count < rc_min:
                    raise ValueError(
                        f"Rest count({rest_count}) < min({rc_min})")
                if rc_max is not None and rest_count > rc_max:
                    raise ValueError(
                        f"Rest count({rest_count}) > max({rc_max})")

            # --- 3. Final question and answer ---
            protagonist = random.choice([
                "Ferryman Kim", "Cargo transporter Park", "River scout Lee",
                "Marine operator Choi", "Freight captain Han",
                "Inland navigator Jung", "Riverside logistics driver Yoon",
                "Waterway officer Jang", "Vessel operator Cho",
                "River courier Seo", "Channel guide Oh",
                "River engineer Hwang"])
            opp_zone = "B" if current_favorable_zone == "A" else "A"

            if congestion_affected_zone == "all":
                cong_zone_desc = "the speed limits in all zones are"
            else:
                cong_zone_desc = (
                    f"the speed limit in Zone "
                    f"{congestion_affected_zone} is")

            q_parts = [
                (f"{protagonist} is a veteran who has worked on this river "
                 f"for {career_years} years and has been assigned to "
                 f"transport goods to an upstream region spanning a total "
                 f"of {total_distance}km. "
                 f"He departs at {_fmt_hour(departure_hour)} carrying "
                 f"{item1_qty} units of {item1_spec['name']} "
                 f"({item1_weight}kg each) and "
                 f"{item2_qty} units of {item2_spec['name']} "
                 f"({item2_weight}kg each). "
                 f"The boat can travel at {base_boat_speed_kph}km/h "
                 f"in still water."),

                (f"The river has a current of {current_speed_kph}km/h. "
                 f"In Zone {current_favorable_zone}, the current is "
                 f"downstream (effective speed = boat speed + current speed), "
                 f"while in Zone {opp_zone}, the current is upstream "
                 f"(effective speed = boat speed - current speed)."),

                (f"The first {regulations['speed_limit']['zone_A_km']}km "
                 f"is Zone A (speed limit: "
                 f"{regulations['speed_limit']['zone_A_limit_kph']}km/h), "
                 f"followed by Zone B (speed limit: "
                 f"{regulations['speed_limit']['zone_B_limit_kph']}km/h). "
                 f"Speed limits apply to the effective speed after "
                 f"accounting for the current."),

                (f"If the cargo exceeds the safety weight threshold "
                 f"({regulations['cargo_effect']['heavy_load_kg']}kg), "
                 + (f"the speed limits in all zones are reduced by "
                    f"{regulations['cargo_effect']['zone_A_reduction_percent']}"
                    f"%."
                    if zone_a_reduction == zone_b_reduction
                    else
                    f"the Zone A speed limit is reduced by "
                    f"{regulations['cargo_effect']['zone_A_reduction_percent']}"
                    f"% and the Zone B speed limit is reduced by "
                    f"{regulations['cargo_effect']['zone_B_reduction_percent']}"
                    f"%.")),

                (f"From {_fmt_hour(congestion_start_hour)} to "
                 f"{_fmt_hour(congestion_end_hour)} is a congestion period, "
                 f"during which {cong_zone_desc} additionally reduced by "
                 f"{congestion_reduction_pct}%. "
                 f"This reduction is applied on top of the speed limit "
                 f"after the cargo regulation has been applied."),
            ]

            if delivery_km:
                q_parts.append(
                    f"At the {delivery_km}km waypoint, "
                    f"{delivery_qty} units of {delivery_spec['name']} "
                    f"are to be unloaded. After unloading, the cargo "
                    f"regulation is reapplied based on the remaining "
                    f"cargo weight.")

            rest_desc = (
                f"The boat cannot operate continuously for more than "
                f"{regulations['mandatory_rest']['trigger_minutes']} minutes. "
                f"Rest is only permitted at designated rest points "
                f"(every {rest_stop_interval_km}km). "
                f"The base rest duration is "
                f"{regulations['mandatory_rest']['duration_minutes']} minutes.")
            if rest_increment > 0:
                base_r = regulations["mandatory_rest"]["duration_minutes"]
                rest_desc += (
                    f" However, due to cumulative fatigue regulations, "
                    f"each rest period increases by {rest_increment} minutes "
                    f"(1st: {base_r}min, 2nd: {base_r + rest_increment}min, "
                    f"3rd: {base_r + rest_increment * 2}min, ...).")
            q_parts.append(rest_desc)

            q_parts.append(
                "Following all of the above rules, what is the total "
                "travel time including mandatory rest stops to reach the "
                "final destination? First calculate the total travel time "
                "as an integer number of minutes, then convert to hours "
                "and minutes. (e.g., 1450 minutes total → 24 hours 10 minutes)")

            question = "\n\n".join(q_parts)

            total_hours = journey.total_journey_time_hours
            total_minutes = round(total_hours * 60)
            hours = total_minutes // 60
            minutes = total_minutes % 60

            answer = f"{hours} hours {minutes} minutes"
            solution.append(
                "[STEP 3] Answer and verification\n"
                f"  - Total time: {total_hours:.4f}h = {total_minutes}min = {answer}\n"
                "  - The '[SEG n]' lines above are the STEP2 worked solution; "
                "double-check min→h+min conversion and that heavy-cargo "
                "speed caps / mandatory rests are applied.")

            return question, answer, solution

        except ValueError:
            continue

    raise RuntimeError(
        f"Maximum retries ({max_retries}) exceeded.")


def _build_rest_count_targets(num_questions, rest_count_range):
    """Build evenly distributed rest count targets from rest_count_range."""
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
        print(f"Generating ferryman puzzles... "
              f"({num_questions} per difficulty, "
              f"{total_questions} total)")
    else:
        difficulties = [difficulty]
        total_questions = num_questions
        print(f"Generating {num_questions} ferryman puzzles... "
              f"(difficulty: {difficulty})")

    output = []
    seen_questions = set()
    unique_answers = set()
    difficulty_counts = {diff: 0 for diff in difficulties}

    for diff in difficulties:
        targets = _build_rest_count_targets(
            num_questions, REST_COUNT_RANGES.get(diff))
        if REST_COUNT_RANGES.get(diff):
            dist = Counter(targets)
            print(f"\n[{diff.upper()}] Generating {num_questions}... "
                  f"(rest count distribution: {dict(sorted(dist.items()))})")
        else:
            print(f"\n[{diff.upper()}] Generating {num_questions}...")

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
                        print(f"  Progress: {diff_count}/{num_questions}")
            except Exception:
                continue

        if diff_count < num_questions:
            print(f"  Warning: [{diff}] Only {diff_count}/{num_questions} "
                  f"generated.")

    print(f"\nGeneration stats:")
    print(f"  Total puzzles: {len(output)}")
    print(f"  Unique puzzles: {len(seen_questions)}")
    print(f"  Unique answers: {len(unique_answers)}")
    print(f"\nDifficulty distribution:")
    for diff in sorted(difficulty_counts):
        print(f"{diff:<6} {difficulty_counts[diff]}")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "ferryman_en.csv"
    ferryman_json = []
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "answer", "solution", "difficulty"])
        for i, (question, answer, solution, diff) in enumerate(output):
            qid = f"ferryman_en_{i}"
            row = {
                "id": qid,
                "question": question,
                "answer": answer,
                "solution": solution,
                "difficulty": diff,
            }
            ferryman_json.append(row)
            writer.writerow([qid, question, answer, solution, diff])
    print(f"\nCSV file created: {csv_path}")

    json_dir = PROJECT_ROOT / "data" / "jsonl"
    json_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = json_dir / "ferryman_en.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in ferryman_json:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL file created: {jsonl_path}")

    ferryman_df = pd.DataFrame(
        ferryman_json,
        columns=["id", "question", "answer", "solution", "difficulty"])
    return ferryman_df, ferryman_json


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Ferryman Puzzle Generator (English)")
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
