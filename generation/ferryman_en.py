import random
from pathlib import Path

class JourneyState:
    def __init__(self):
        self.total_moving_time_hours = 0.0
        self.total_rest_time_hours = 0.0
        self.continuous_moving_time_hours = 0.0
        self.current_position_km = 0.0

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


def generate_puzzle_question(difficulty="easy", rest_count_target=None):
    max_retries = 500

    for attempt in range(max_retries):
        try:
            if difficulty == "easy":
                params = {
                    "distance_range": (60, 110),
                    "zone_a_range": (15, 30),
                    "zone_a_limit_range": (25, 38),
                    "zone_b_limit_range": (18, 28),
                    "rest_trigger_minutes_range": (80, 130),
                    "rest_duration_minutes_range": (20, 50),
                    "rest_stop_interval_km_range": (15, 25),
                    "heavy_threshold_range": (1500, 2500),
                    "zone_a_reduction_range": (10, 16),
                    "zone_b_reduction_range": (10, 16),
                    "uniform_reduction": True,
                    "base_speed_bonus_range": (8, 18),
                    "current_speed_range": (2, 4),
                    "has_delivery": False,
                    "rest_increment_range": (0, 0),
                    "congestion_start_offset_range": (3, 5),
                    "congestion_duration_range": (1, 2),
                    "congestion_reduction_range": (5, 10),
                    "congestion_affected_zone": "all",
                    "max_segments": 15,
                    "rest_count_range": (1, 4),
                }
            elif difficulty == "medium":
                params = {
                    "distance_range": (90, 160),
                    "zone_a_range": (18, 40),
                    "zone_a_limit_range": (20, 32),
                    "zone_b_limit_range": (14, 24),
                    "rest_trigger_minutes_range": (70, 110),
                    "rest_duration_minutes_range": (25, 60),
                    "rest_stop_interval_km_range": (14, 22),
                    "heavy_threshold_range": (300, 550),
                    "zone_a_reduction_range": (15, 25),
                    "zone_b_reduction_range": (20, 30),
                    "uniform_reduction": False,
                    "base_speed_bonus_range": (4, 11),
                    "current_speed_range": (4, 8),
                    "has_delivery": True,
                    "rest_increment_range": (8, 15),
                    "congestion_start_offset_range": (1, 3),
                    "congestion_duration_range": (2, 4),
                    "congestion_reduction_range": (20, 30),
                    "congestion_affected_zone": "all",
                    "max_segments": 25,
                    "rest_count_range": (8, 10),
                }
            else:  # hard
                params = {
                    "distance_range": (110, 180),
                    "zone_a_range": (20, 50),
                    "zone_a_limit_range": (17, 29),
                    "zone_b_limit_range": (14, 22),
                    "rest_trigger_minutes_range": (70, 110),
                    "rest_duration_minutes_range": (25, 55),
                    "rest_stop_interval_km_range": (12, 18),
                    "heavy_threshold_range": (200, 450),
                    "zone_a_reduction_range": (15, 25),
                    "zone_b_reduction_range": (25, 40),
                    "uniform_reduction": False,
                    "base_speed_bonus_range": (3, 9),
                    "current_speed_range": (4, 8),
                    "has_delivery": True,
                    "rest_increment_range": (5, 10),
                    "congestion_start_offset_range": (1, 2),
                    "congestion_duration_range": (3, 5),
                    "congestion_reduction_range": (22, 32),
                    "congestion_affected_zone": "all",
                    "max_segments": 30,
                    "rest_count_range": (9, 12),
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
            solution = ["[STEP 0] Initial conditions"]
            solution.append(
                f"  Total distance={total_distance}km, "
                f"boat still-water speed={base_boat_speed_kph}km/h, "
                f"departure={_fmt_hour(departure_hour)}")
            solution.append(
                f"  Current: {current_speed_kph}km/h "
                f"(downstream in Zone {current_favorable_zone})")
            solution.append(
                f"  Cargo: {item1_weight}kg {item1_spec['name']}x{item1_qty}"
                f" + {item2_weight}kg {item2_spec['name']}x{item2_qty}")
            solution.append(
                f"  Congestion: {_fmt_hour(congestion_start_hour)}~"
                f"{_fmt_hour(congestion_end_hour)}, "
                f"{'all zones' if congestion_affected_zone == 'all' else 'Zone ' + congestion_affected_zone}"
                f" -{congestion_reduction_pct}%")
            if delivery_km:
                solution.append(
                    f"  Mid-journey delivery: {delivery_km}km, "
                    f"{delivery_spec['name']} x{delivery_qty}")
            if rest_increment > 0:
                solution.append(
                    f"  Cumulative fatigue: +{rest_increment}min per rest")
            solution.append(f"  Regulations: {regulations}")
            step_cnt = 1

            solution.append(
                f"[STEP {step_cnt}] Cargo weight: "
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
                    f"[STEP {step_cnt}] Cargo regulation: {cargo_weight_kg}kg > "
                    f"{regulations['cargo_effect']['heavy_load_kg']}kg -> "
                    f"Zone A -{ra}%({adj_A:.2f}), Zone B -{rb}%({adj_B:.2f})")
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
                    in_cong = (congestion_start_hour <= cur_abs
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
                        t_to_s = congestion_start_hour - cur_abs
                        if t_to_s > 1e-9:
                            d_to_s = spd * t_to_s
                            if d_to_s < dz:
                                dz = d_to_s
                    else:
                        t_to_e = congestion_end_hour - cur_abs
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
                in_congestion = (congestion_start_hour <= abs_time
                                 < congestion_end_hour)
                cong_applies = (
                    in_congestion
                    and (congestion_affected_zone == "all"
                         or in_zone == congestion_affected_zone))
                if cong_applies:
                    speed_limit *= (1 - congestion_reduction_pct / 100)

                if in_zone == current_favorable_zone:
                    eff_speed = base_boat_speed_kph + current_speed_kph
                    cur_label = "downstream"
                else:
                    eff_speed = base_boat_speed_kph - current_speed_kph
                    cur_label = "upstream"

                actual_speed = min(eff_speed, speed_limit)
                if actual_speed <= 0:
                    raise ValueError("Invalid speed")

                cong_tag = " [CONGESTED]" if cong_applies else ""
                solution.append(
                    f"[STEP {step_cnt}] Zone {in_zone} "
                    f"({current_pos:.1f}km, {abs_time:.2f}h){cong_tag}: "
                    f"{base_boat_speed_kph}"
                    f"{'+' if cur_label == 'downstream' else '-'}"
                    f"{current_speed_kph}={eff_speed:.1f}, "
                    f"limit {speed_limit:.2f} -> {actual_speed:.2f}km/h")
                step_cnt += 1

                dist_to_stop = _next_rest_stop_dist(current_pos)
                if dist_to_stop < 1e-6:
                    dist_to_stop = rest_stop_interval_km

                boundaries = [distance_to_go, distance_in_zone, dist_to_stop]
                if not delivered:
                    dist_to_delivery = delivery_km - current_pos
                    if dist_to_delivery > 1e-6:
                        boundaries.append(dist_to_delivery)

                if not in_congestion:
                    t_to_cong = congestion_start_hour - abs_time
                    if t_to_cong > 1e-9:
                        d_to_cong = actual_speed * t_to_cong
                        if d_to_cong > 0.001:
                            boundaries.append(d_to_cong)
                else:
                    t_to_cong_end = congestion_end_hour - abs_time
                    if t_to_cong_end > 1e-9:
                        d_to_cong_end = actual_speed * t_to_cong_end
                        if d_to_cong_end > 0.001:
                            boundaries.append(d_to_cong_end)

                seg_dist = min(boundaries)
                seg_time = seg_dist / actual_speed

                journey.current_position_km += seg_dist
                journey.total_moving_time_hours += seg_time
                journey.continuous_moving_time_hours += seg_time
                distance_to_go -= seg_dist

                solution.append(
                    f"  -> {seg_dist:.2f}km / {actual_speed:.2f}km/h "
                    f"= {seg_time:.4f}h")

                if (not delivered
                        and abs(journey.current_position_km - delivery_km)
                        < 1e-6):
                    delivered = True
                    unloaded_kg = delivery_weight_per_unit * delivery_qty
                    cargo_weight_kg -= unloaded_kg
                    old_heavy = is_heavy
                    adj_A, adj_B, is_heavy = _apply_cargo_effect(
                        cargo_weight_kg)
                    solution.append(
                        f"  * Unload: {delivery_spec['name']} x{delivery_qty}"
                        f"({unloaded_kg}kg) -> remaining {cargo_weight_kg}kg")
                    if old_heavy and not is_heavy:
                        solution.append(
                            f"    -> Cargo regulation lifted! "
                            f"A:{adj_A:.2f}, B:{adj_B:.2f}")
                    elif is_heavy:
                        solution.append(
                            "    -> Still overweight, regulation maintained")

                if ((not rest_due)
                        and journey.continuous_moving_time_hours
                        >= trigger_hours - 1e-9):
                    rest_due = True
                    solution.append(
                        f"  >> Continuous "
                        f"{regulations['mandatory_rest']['trigger_minutes']}"
                        f"min reached")

                at_stop = abs(
                    (journey.current_position_km / rest_stop_interval_km)
                    - round(journey.current_position_km
                            / rest_stop_interval_km)
                ) < 1e-6
                at_destination = distance_to_go <= 0.001

                need_rest = False
                if rest_due and at_stop and not at_destination:
                    need_rest = True
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
                        solution.append(
                            f"  >> Next segment expected to exceed "
                            f"{regulations['mandatory_rest']['trigger_minutes']}"
                            f"min continuous")
                    else:
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
                        else:
                            time_to_target = _compute_time_to(
                                cur_pos, target_km)
                        if (journey.continuous_moving_time_hours
                                + time_to_target
                                >= trigger_hours - 1e-9):
                            need_rest = True
                            solution.append(
                                f"  >> Next segment expected to exceed "
                                f"{regulations['mandatory_rest']['trigger_minutes']}"
                                f"min continuous")

                if need_rest:
                    base_rest = (
                        regulations["mandatory_rest"]["duration_minutes"])
                    extra = rest_count * rest_increment
                    this_rest = base_rest + extra
                    journey.total_rest_time_hours += this_rest / 60.0
                    journey.continuous_moving_time_hours = 0
                    rest_due = False
                    rest_count += 1
                    if extra > 0:
                        solution.append(
                            f"  # Rest#{rest_count}: "
                            f"at {journey.current_position_km:.0f}km, "
                            f"{base_rest}+{extra}={this_rest}min")
                    else:
                        solution.append(
                            f"  # Rest#{rest_count}: "
                            f"at {journey.current_position_km:.0f}km, "
                            f"{this_rest}min")

            if rest_due:
                raise ValueError(
                    "Last segment exceeds continuous limit (no rest stop)")

            max_segments = params.get("max_segments", 20)
            if step_cnt > max_segments:
                raise ValueError(f"Segment count({step_cnt}) exceeded")

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
                "final destination? Express in hours and minutes "
                "(round minutes to the nearest whole number).")

            question = "\n\n".join(q_parts)

            total_hours = journey.total_journey_time_hours
            hours = int(total_hours)
            minutes = round((total_hours - hours) * 60)
            if minutes == 60:
                hours += 1
                minutes = 0

            answer = f"{hours} hours {minutes} minutes"
            solution.append(
                f"[STEP {step_cnt}] Total {total_hours:.4f}h "
                f"= {answer}")

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


REST_COUNT_RANGES = {
    "easy": (1, 4),
    "medium": (8, 10),
    "hard": (9, 12),
}


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
            qid = f"ferryman_{i}"
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

    json_dir = PROJECT_ROOT / "data" / "json"
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
