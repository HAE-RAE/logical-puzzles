#!/usr/bin/env python3
"""
Final Dataset Generation - Korean logical puzzles.

Generates 100 puzzles × 3 difficulties × 3 puzzle types = 900 total.

Uses the same calibrated parameters as the English version.
Calibrated for gemini-3-flash-preview (temperature=0.0, max_tokens=2000):
  Easy: ~75% (65-85%)   Medium: ~50% (40-60%)   Hard: ~25% (15-35%)

Puzzle types:
  - Causal DAG: 인과 그래프를 통한 신호 전파 추적
  - Logic Grid: 제약 조건으로부터 배정 추론
  - SAT Puzzle: 모든 절을 만족시키는 불리언 변수 할당 찾기
"""

import sys, os, json, csv, random, logging, time
from pathlib import Path
from typing import Dict, List, Optional, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
env_path = PROJECT_ROOT / '.env'
if env_path.exists(): load_dotenv(env_path)
else: load_dotenv()

from generation.causal_dag_ko import CausalPuzzleGenerator, create_question as create_causal_q
from generation.logic_grid_ko import LogicGridGenerator, LogicGridPuzzle, Difficulty as LGDifficulty
from generation.sat_puzzle_ko import SATPuzzleGenerator, Difficulty as SATDifficulty

PUZZLES_PER_DIFFICULTY = 100
OUTPUT_DIR = PROJECT_ROOT / "data" / "final_calibrated_ko"

# =============================================================================
# Calibrated Parameters (same structure as English version)
# =============================================================================

# 영어 generate_calibrated.py와 동일 파라미터. 한국어는 create_question(..., shuffle_edges=True)로 제시 순서 난이도 보정.
CAUSAL_DAG_PARAMS = {
    'easy': {
        'num_events': 3, 'delay_range': (5, 20),
        'max_out_degree': 2, 'and_probability': 0.0, 'edge_density': 0.5,
    },
    'medium': {
        'num_events': 4, 'delay_range': (5, 20),
        'max_out_degree': 2, 'and_probability': 0.0, 'edge_density': 0.4,
    },
    'hard': {
        'num_events': 5, 'delay_range': (10, 50),
        'max_out_degree': 3, 'and_probability': 0.0, 'edge_density': 0.5,
    },
}

LOGIC_GRID_PARAMS = {
    'easy': {
        'num_people': 4, 'num_categories': 4,
        'categories': ['집색깔', '애완동물', '음료', '직업'],
        'min_constraints': 10, 'max_constraints': 14, 'direct_ratio': 0.4,
    },
    'medium': {
        'num_people': 5, 'num_categories': 4,
        'categories': ['집색깔', '애완동물', '음료', '직업'],
        'min_constraints': 12, 'max_constraints': 16, 'direct_ratio': 0.42,
    },
    'hard': {
        'num_people': 6, 'num_categories': 4,
        'categories': ['집색깔', '애완동물', '음료', '직업'],
        'min_constraints': 15, 'max_constraints': 18, 'direct_ratio': 0.15,
    },
}

SAT_PARAMS = {
    'easy': {
        'num_vars_range': (3, 4), 'clauses_per_var': 1.5,
        'clause_length': (2, 2), 'negation_ratio': 0.25, 'min_clauses': 5,
    },
    'medium': {
        'num_vars_range': (3, 4), 'clauses_per_var': 2.0,
        'clause_length': (2, 3), 'negation_ratio': 0.35, 'min_clauses': 6,
    },
    'hard': {
        'num_vars_range': (4, 4), 'clauses_per_var': 1.5,
        'clause_length': (2, 3), 'negation_ratio': 0.30, 'min_clauses': 6,
    },
}


# =============================================================================
# Generators
# =============================================================================

_causal_gen = CausalPuzzleGenerator()

def gen_causal_dag(difficulty: str, idx: int, seed: int) -> Optional[dict]:
    from generation.causal_dag_ko import CausalPuzzle
    params = CAUSAL_DAG_PARAMS[difficulty]
    ne = params['num_events']
    cfg = {k: params[k] for k in ['delay_range', 'max_out_degree', 'and_probability', 'edge_density']}

    random.seed(seed)
    events = _causal_gen._generate_events(ne)
    edges = _causal_gen._generate_causal_graph(events, cfg)
    if not edges: return None
    in_deg = _causal_gen._calculate_in_degree(events, edges)
    triggers = [e for e, d in in_deg.items() if d == 0]
    if not triggers: return None
    trigger = random.choice(triggers)
    tt = random.randint(0, 10)
    times = _causal_gen._calculate_reach_times(events, edges, trigger, tt)
    reachable = [e for e in events if e != trigger and times.get(e, float('inf')) < float('inf')]
    if not reachable: return None
    target = random.choice(reachable)
    p = CausalPuzzle(events=events, edges=edges, trigger=trigger, trigger_time=tt,
                     target_event=target, answer=times[target], difficulty=difficulty)
    q = create_causal_q(p, shuffle_edges=True)
    return {
        'id': f"causal_dag_ko_{difficulty}_{idx:04d}",
        'question': q, 'answer': str(p.answer),
        'difficulty': difficulty, 'puzzle_type': 'causal_dag',
        'num_events': ne,
    }


class ConfigurableLogicGrid(LogicGridGenerator):
    def __init__(self, cfg, seed=None):
        super().__init__(seed); self._cfg = cfg; self._retry = 0
    def _get_difficulty_config(self, d): return self._cfg
    def generate(self, d=LGDifficulty.EASY):
        c = self._get_difficulty_config(d)
        p, a, s = self._generate_solution(c)
        cs = self._generate_constraints(p, a, s, c)
        if not self._verify_unique_solution(p, a, cs, s):
            self._retry += 1
            if self._retry >= 50: self._retry = 0; return None
            return self.generate(d)
        pz = LogicGridPuzzle(id="t", difficulty=str(d), people=p, attributes=a,
                             constraints=cs, question="", answer=s)
        pz.question = pz.to_prompt()
        self._retry = 0; return pz


def gen_logic_grid(difficulty: str, idx: int, seed: int) -> Optional[dict]:
    params = LOGIC_GRID_PARAMS[difficulty]
    g = ConfigurableLogicGrid(params, seed=seed)
    p = g.generate()
    if not p: return None
    p.id = f"logic_grid_ko_{difficulty}_{idx:04d}"
    p.difficulty = difficulty
    d = p.to_dict()
    d['puzzle_type'] = 'logic_grid'
    return d


class ConfigurableSAT(SATPuzzleGenerator):
    def __init__(self, cfg, seed=None):
        super().__init__(seed); self._cfg = cfg
    def _get_difficulty_config(self, d):
        c = dict(self._cfg)
        if 'num_vars_range' in c:
            lo, hi = c.pop('num_vars_range')
            c['num_vars'] = random.randint(lo, hi)
        return c


def gen_sat(difficulty: str, idx: int, seed: int) -> Optional[dict]:
    params = SAT_PARAMS[difficulty]
    g = ConfigurableSAT(params, seed=seed)
    try:
        p = g.generate(SATDifficulty.EASY)
    except Exception:
        return None
    if not p: return None
    d = p.to_dict()
    d['id'] = f"sat_puzzle_ko_{difficulty}_{idx:04d}"
    d['difficulty'] = difficulty
    d['puzzle_type'] = 'sat_puzzle'
    return d


# =============================================================================
# Main
# =============================================================================

def generate_batch(task_name, gen_fn, n, difficulties, base_seed):
    all_puzzles = {}
    for diff in difficulties:
        logger.info(f"  Generating {n} {task_name} [{diff}]...")
        puzzles = []
        attempts = 0
        while len(puzzles) < n and attempts < n * 10:
            seed = base_seed + hash(diff) % 10000 + attempts
            p = gen_fn(diff, len(puzzles), seed)
            if p: puzzles.append(p)
            attempts += 1
        all_puzzles[diff] = puzzles
        logger.info(f"    → {len(puzzles)}/{n}")
    return all_puzzles


def save_jsonl(puzzles, path):
    with open(path, 'w') as f:
        for p in puzzles:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')


def save_csv(puzzles, path):
    if not puzzles: return
    keys = puzzles[0].keys()
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for p in puzzles:
            row = {}
            for k, v in p.items():
                row[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
            w.writerow(row)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_dir = OUTPUT_DIR / "json"
    csv_dir = OUTPUT_DIR / "csv"
    json_dir.mkdir(exist_ok=True)
    csv_dir.mkdir(exist_ok=True)

    tasks = {
        'causal_dag_ko': (gen_causal_dag, 1000),
        'logic_grid_ko': (gen_logic_grid, 2000),
        'sat_puzzles_ko': (gen_sat, 3000),
    }
    difficulties = ['easy', 'medium', 'hard']

    summary = {}
    for task_name, (gen_fn, base_seed) in tasks.items():
        logger.info(f"\n{'='*60}\n  {task_name.upper()}\n{'='*60}")
        all_puzzles = generate_batch(task_name, gen_fn, PUZZLES_PER_DIFFICULTY, difficulties, base_seed)
        for diff, puzzles in all_puzzles.items():
            fname = f"{task_name}_{diff}"
            save_jsonl(puzzles, json_dir / f"{fname}.jsonl")
            save_csv(puzzles, csv_dir / f"{fname}.csv")
            summary[f"{task_name}_{diff}"] = len(puzzles)

        all_flat = [p for ps in all_puzzles.values() for p in ps]
        save_jsonl(all_flat, json_dir / f"{task_name}_all.jsonl")
        save_csv(all_flat, csv_dir / f"{task_name}_all.csv")

    print(f"\n{'='*60}")
    print(f"  FINAL DATASET SUMMARY (Korean)")
    print(f"{'='*60}")
    for k, v in sorted(summary.items()):
        print(f"  {k:<35} {v:>4} puzzles")
    total = sum(summary.values())
    print(f"  {'TOTAL':<35} {total:>4} puzzles")
    print(f"{'='*60}")
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
