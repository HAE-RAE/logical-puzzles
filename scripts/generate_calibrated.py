#!/usr/bin/env python3
"""
Final Dataset Generation — calibrated logical puzzles (EN/KO).

Generates 100 puzzles × 3 difficulties × 3 puzzle types = 900 total per language.

Calibrated for gemini-3-flash-preview (temperature=0.0, max_tokens=2000):
  Easy: ~75% (65-85%)   Medium: ~50% (40-60%)   Hard: ~25% (15-35%)

Puzzle types: Causal DAG, Logic Grid, SAT Puzzle.

Usage:
  python scripts/generate_calibrated.py --lang en
  python scripts/generate_calibrated.py --lang ko

KO note: causal_dag uses `shuffle_edges=True` to randomise edge presentation order
(adds difficulty equivalence to the EN version's text patterns).
"""

import sys, os, json, csv, random, importlib
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import PROJECT_ROOT, ensure_dotenv, setup_logger

logger = setup_logger(__name__)

sys.path.insert(0, str(PROJECT_ROOT))
ensure_dotenv(PROJECT_ROOT / '.env' if (PROJECT_ROOT / '.env').exists() else None)

PUZZLES_PER_DIFFICULTY = 100

# =============================================================================
# Calibrated Parameters
# =============================================================================

CAUSAL_DAG_PARAMS = {
    'easy':   {'num_events': 3, 'delay_range': (5, 20),  'max_out_degree': 2, 'and_probability': 0.0, 'edge_density': 0.5},
    'medium': {'num_events': 4, 'delay_range': (5, 20),  'max_out_degree': 2, 'and_probability': 0.0, 'edge_density': 0.4},
    'hard':   {'num_events': 5, 'delay_range': (10, 50), 'max_out_degree': 3, 'and_probability': 0.0, 'edge_density': 0.5},
}

# logic_grid categories differ per language
LOGIC_GRID_CATEGORIES = {
    'en': ['HouseColor', 'Pet', 'Drink', 'Job'],
    'ko': ['집색깔', '애완동물', '음료', '직업'],
}
# medium direct_ratio differs (KO 0.42 vs EN 0.30) — KO needs more direct hints
# because positional cue text is harder to chain in Korean.
LOGIC_GRID_MEDIUM_DIRECT_RATIO = {'en': 0.3, 'ko': 0.42}


def make_logic_grid_params(lang: str):
    cats = LOGIC_GRID_CATEGORIES[lang]
    return {
        'easy':   {'num_people': 4, 'num_categories': 4, 'categories': cats,
                   'min_constraints': 10, 'max_constraints': 14, 'direct_ratio': 0.4},
        'medium': {'num_people': 5, 'num_categories': 4, 'categories': cats,
                   'min_constraints': 12, 'max_constraints': 16, 'direct_ratio': LOGIC_GRID_MEDIUM_DIRECT_RATIO[lang]},
        'hard':   {'num_people': 6, 'num_categories': 4, 'categories': cats,
                   'min_constraints': 15, 'max_constraints': 18, 'direct_ratio': 0.15},
    }


SAT_PARAMS = {
    'easy':   {'num_vars_range': (3, 4), 'clauses_per_var': 1.5, 'clause_length': (2, 2), 'negation_ratio': 0.25, 'min_clauses': 5},
    'medium': {'num_vars_range': (3, 4), 'clauses_per_var': 2.0, 'clause_length': (2, 3), 'negation_ratio': 0.35, 'min_clauses': 6},
    'hard':   {'num_vars_range': (4, 4), 'clauses_per_var': 1.5, 'clause_length': (2, 3), 'negation_ratio': 0.30, 'min_clauses': 6},
}


# =============================================================================
# Generators (lang-aware factory)
# =============================================================================

class CalibratedGenerator:
    """Bundles the per-language modules + params used to produce one suite."""

    def __init__(self, lang: str):
        if lang not in ("en", "ko"):
            raise ValueError(f"lang must be 'en' or 'ko', got {lang!r}")
        self.lang = lang
        self.id_suffix = "" if lang == "en" else "_ko"
        self.output_dir = PROJECT_ROOT / f"data/final_calibrated{self.id_suffix}"

        causal_mod = importlib.import_module(f"generation.causal_dag_{lang}")
        logic_mod = importlib.import_module(f"generation.logic_grid_{lang}")
        sat_mod = importlib.import_module(f"generation.sat_puzzle_{lang}")

        self._causal_gen = causal_mod.CausalPuzzleGenerator()
        self._CausalPuzzle = causal_mod.CausalPuzzle
        self._create_causal_q = causal_mod.create_question

        self._LogicGridGenerator = logic_mod.LogicGridGenerator
        self._LogicGridPuzzle = logic_mod.LogicGridPuzzle
        self._LGDifficulty = logic_mod.Difficulty

        self._SATPuzzleGenerator = sat_mod.SATPuzzleGenerator
        self._SATDifficulty = sat_mod.Difficulty

        self.logic_grid_params = make_logic_grid_params(lang)

    def _causal_id(self, difficulty: str, idx: int) -> str:
        return f"causal_dag{self.id_suffix}_{difficulty}_{idx:04d}"

    def _logic_grid_id(self, difficulty: str, idx: int) -> str:
        return f"logic_grid{self.id_suffix}_{difficulty}_{idx:04d}"

    def _sat_id(self, difficulty: str, idx: int) -> str:
        return f"sat_puzzle{self.id_suffix}_{difficulty}_{idx:04d}"

    def gen_causal_dag(self, difficulty: str, idx: int, seed: int) -> Optional[dict]:
        params = CAUSAL_DAG_PARAMS[difficulty]
        ne = params['num_events']
        cfg = {k: params[k] for k in ('delay_range', 'max_out_degree', 'and_probability', 'edge_density')}
        random.seed(seed)
        events = self._causal_gen._generate_events(ne)
        edges = self._causal_gen._generate_causal_graph(events, cfg)
        if not edges:
            return None
        in_deg = self._causal_gen._calculate_in_degree(events, edges)
        triggers = [e for e, d in in_deg.items() if d == 0]
        if not triggers:
            return None
        trigger = random.choice(triggers)
        tt = random.randint(0, 10)
        times = self._causal_gen._calculate_reach_times(events, edges, trigger, tt)
        reachable = [e for e in events if e != trigger and times.get(e, float('inf')) < float('inf')]
        if not reachable:
            return None
        target = random.choice(reachable)
        p = self._CausalPuzzle(events=events, edges=edges, trigger=trigger, trigger_time=tt,
                               target_event=target, answer=times[target], difficulty=difficulty)
        # KO: shuffle_edges=True; EN: default False
        q = self._create_causal_q(p, shuffle_edges=True) if self.lang == "ko" else self._create_causal_q(p)
        return {
            'id': self._causal_id(difficulty, idx),
            'question': q, 'answer': str(p.answer),
            'difficulty': difficulty, 'puzzle_type': 'causal_dag',
            'num_events': ne,
        }

    def _build_configurable_logic_grid(self):
        LG_BASE = self._LogicGridGenerator
        LGPuzzle = self._LogicGridPuzzle
        LGDiff = self._LGDifficulty

        class ConfigurableLogicGrid(LG_BASE):
            def __init__(self, cfg, seed=None):
                super().__init__(seed)
                self._cfg = cfg
                self._retry = 0

            def _get_difficulty_config(self, d):
                return self._cfg

            def generate(self, d=LGDiff.EASY):
                c = self._get_difficulty_config(d)
                p, a, s = self._generate_solution(c)
                cs = self._generate_constraints(p, a, s, c)
                if not self._verify_unique_solution(p, a, cs, s):
                    self._retry += 1
                    if self._retry >= 50:
                        self._retry = 0
                        return None
                    return self.generate(d)
                pz = LGPuzzle(id="t", difficulty=str(d), people=p, attributes=a,
                              constraints=cs, question="", answer=s)
                pz.question = pz.to_prompt()
                self._retry = 0
                return pz

        return ConfigurableLogicGrid

    def gen_logic_grid(self, difficulty: str, idx: int, seed: int) -> Optional[dict]:
        params = self.logic_grid_params[difficulty]
        ConfigurableLG = self._build_configurable_logic_grid()
        g = ConfigurableLG(params, seed=seed)
        p = g.generate()
        if not p:
            return None
        p.id = self._logic_grid_id(difficulty, idx)
        p.difficulty = difficulty
        d = p.to_dict()
        d['puzzle_type'] = 'logic_grid'
        return d

    def _build_configurable_sat(self):
        SAT_BASE = self._SATPuzzleGenerator

        class ConfigurableSAT(SAT_BASE):
            def __init__(self, cfg, seed=None):
                super().__init__(seed)
                self._cfg = cfg

            def _get_difficulty_config(self, d):
                c = dict(self._cfg)
                if 'num_vars_range' in c:
                    lo, hi = c.pop('num_vars_range')
                    c['num_vars'] = random.randint(lo, hi)
                return c

        return ConfigurableSAT

    def gen_sat(self, difficulty: str, idx: int, seed: int) -> Optional[dict]:
        params = SAT_PARAMS[difficulty]
        ConfigurableSAT = self._build_configurable_sat()
        g = ConfigurableSAT(params, seed=seed)
        try:
            p = g.generate(self._SATDifficulty.EASY)
        except Exception:
            return None
        if not p:
            return None
        d = p.to_dict()
        d['id'] = self._sat_id(difficulty, idx)
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
            if p:
                puzzles.append(p)
            attempts += 1
        all_puzzles[diff] = puzzles
        logger.info(f"    → {len(puzzles)}/{n}")
    return all_puzzles


def save_jsonl(puzzles, path):
    with open(path, 'w') as f:
        for p in puzzles:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')


def save_csv(puzzles, path):
    if not puzzles:
        return
    keys = puzzles[0].keys()
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for p in puzzles:
            row = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in p.items()}
            w.writerow(row)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", choices=["en", "ko"], default="en", help="Language to generate")
    args = parser.parse_args()

    cg = CalibratedGenerator(args.lang)
    output_dir = cg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dir = output_dir / "json"
    csv_dir = output_dir / "csv"
    json_dir.mkdir(exist_ok=True)
    csv_dir.mkdir(exist_ok=True)

    suffix = "" if args.lang == "en" else "_ko"
    tasks = {
        f'causal_dag{suffix}':   (cg.gen_causal_dag, 1000),
        f'logic_grid{suffix}':   (cg.gen_logic_grid, 2000),
        f'sat_puzzles{suffix}':  (cg.gen_sat, 3000),
    }
    difficulties = ['easy', 'medium', 'hard']

    summary = {}
    for task_name, (gen_fn, base_seed) in tasks.items():
        logger.info(f"\n{'=' * 60}\n  {task_name.upper()}\n{'=' * 60}")
        all_puzzles = generate_batch(task_name, gen_fn, PUZZLES_PER_DIFFICULTY, difficulties, base_seed)
        for diff, puzzles in all_puzzles.items():
            fname = f"{task_name}_{diff}"
            save_jsonl(puzzles, json_dir / f"{fname}.jsonl")
            save_csv(puzzles, csv_dir / f"{fname}.csv")
            summary[f"{task_name}_{diff}"] = len(puzzles)
        all_flat = [p for ps in all_puzzles.values() for p in ps]
        save_jsonl(all_flat, json_dir / f"{task_name}_all.jsonl")
        save_csv(all_flat, csv_dir / f"{task_name}_all.csv")

    label = "English" if args.lang == "en" else "Korean"
    print(f"\n{'=' * 60}")
    print(f"  FINAL DATASET SUMMARY ({label})")
    print(f"{'=' * 60}")
    for k, v in sorted(summary.items()):
        print(f"  {k:<35} {v:>4} puzzles")
    print(f"  {'TOTAL':<35} {sum(summary.values()):>4} puzzles")
    print(f"{'=' * 60}")
    print(f"\nSaved to: {output_dir}")


if __name__ == '__main__':
    main()
