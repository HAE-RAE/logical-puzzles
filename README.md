# Logical Puzzles
Comprehensive Logical Puzzle Dataset with LLM Evaluation Framework


## Puzzle Types

### 1. Array Formula (EN / KO)
Apply formulas to 2D arrays to calculate final results. Sequentially apply row/column operations and aggregate functions to derive answers.

- **Multi-step Operations**: Apply functions like SUM, MEAN, MAX, MIN sequentially
- **Row/Column Aggregation**: Perform aggregate functions on row or column units
- **Complex Reasoning**: Track intermediate calculation results to derive final values
- **Difficulty Levels**: Adjusted based on array size and number of operation steps

### 2. Causal DAG (EN / KO)
Infer event propagation time in causal relationship graphs with time delays.

- **DAG-based Generation**: Represent causal relationships as directed acyclic graphs
- **Shortest Path Algorithm**: Calculate answers using Dijkstra's algorithm
- **Unique Solution**: Automatically guaranteed by deterministic graph structure
- **Difficulty Levels**: Based on number of events (4-12) and connection density (Easy/Medium/Hard)
- **Realistic Scenarios**: Real events from technology, business, environment, operations domains

### 3. Cipher (EN / KO)
Decode multi-layer cipher algorithms. Evaluates LLM's pure algorithmic reasoning ability.

- **Multi-layer Algorithms**: Stack of Substitution, Vigenere, Reverse, Playfair, Transposition
- **Meaningless Answers**: Use random strings to prevent linguistic guessing
- **Variable Hints**: Adjust number of examples by difficulty (2-20 examples)
- **Difficulty Levels**: Easy, Medium, Hard (cipher stack depth and hint count increase with tier)

### 4. Cryptarithmetic (EN / KO)
Find the digit corresponding to each letter in arithmetic expressions represented by letters.

- **Scripts**: `generation/cryptarithmetic_en.py`, `generation/cryptarithmetic_ko.py`
- **Constraints**: No leading zeros, each letter is a unique digit
- **Backtracking Solver**: Verify unique solution
- **Various Operations**: Support addition, subtraction, multiplication
- **Difficulty Levels**: Adjusted based on number of letters and digits

### 5. Ferryman (EN / KO)
Calculate total time required for a ferryman to transport goods considering various navigation rules and constraints.

- **Scripts**: `generation/ferryman_en.py`, `generation/ferryman_ko.py`
- **Speed-limited Zones**: Speed changes in Zone A and Zone B
- **Cargo Weight**: Speed reduction based on weight
- **Mandatory Rest Time**: Compliance with navigation regulations
- **Complex Condition Reasoning**: Consider multiple constraints simultaneously

### 6. Hanoi (EN / KO)
Classic puzzle of moving disks from one peg to another. Calculate the minimum number of moves.

- **Scripts**: `generation/hanoi_en.py`, `generation/hanoi_ko.py`
- **Recursive Structure**: Requires divide-and-conquer thinking
- **Constraints**: Only smaller disks can be placed on larger disks
- **Minimum Moves**: 2^n - 1 moves (n is number of disks)
- **Difficulty Levels**: Adjusted based on number of disks

### 7. Inequality (EN / KO)
Find number placements that satisfy given inequality constraints.

- **Scripts**: `generation/inequality_en.py`, `generation/inequality_ko.py`
- **Constraint Satisfaction Problem**: Satisfy multiple inequalities simultaneously
- **Backtracking Solver**: Verify unique solution
- **Logical Reasoning**: Narrow value ranges from constraints
- **Difficulty Levels**: Adjusted based on number of variables and constraints

### 8. Kinship (KO)
Infer family relationship clues presented through dialogue to derive final Korean kinship terms.

- **Korean Kinship Terms**: Support various family relationships (paternal/maternal/in-laws)
- **Dialogue-based Clues**: Provide relationship information in natural conversation format
- **Multiple Answer Support**: Handle synonyms like "큰아버지, 백부"
- **Relationship Chain Reasoning**: Derive final kinship term through stepwise relationship connections
- **Language+Logic Fusion**: Simultaneously evaluate Korean honorific system understanding and logical reasoning

### 9. Kinship Vision (KO)
Multimodal problem combining family photo images and dialogue clues to identify people by visual features and infer relationships.

- **Vision+Language Fusion**: Simultaneously perform person identification in images and relationship reasoning
- **17-person Actor DB**: Consider gender (male/female) × age group (SENIOR/ADULT/YOUNG_ADULT/CHILD)
- **Visual Feature Mapping**: Distinguish people using 3 features: clothing color, position, appearance
- **Multiple Choice Format**: 1 correct answer + 3 distractors (inducing visual confusion)
- **Multimodal Reasoning**: Understand dialogue context → Search image → Connect relationship chain
- **Data Status**: generation script only (`generation/kinship_vision.py`); no dataset in `data/` yet (sample assets in `evaluation/eval_data/kinship_vision/`)

### 10. Logic Grid (EN / KO)
Constraint-based logical reasoning problem famous as Einstein's Riddle. Infer relationships among multiple people and attributes from constraint conditions.

- **CSP (Constraint Satisfaction Problem)**: Verify unique solution with backtracking
- **Natural Language Constraints**: Simultaneously require linguistic understanding and logical reasoning
- **Unique Solution Guaranteed**: Verified with Constraint Propagation algorithm
- **Difficulty Levels**: Easy (3×3), Medium (4×4), Hard (5×5)
- **Backward Generation**: Generate valid solution first, then derive constraint conditions

### 11. Minesweeper (EN / KO)
Minesweeper puzzle designed as a Constraint Satisfaction Problem (CSP). Evaluates LLM's logical reasoning ability.

- **Scripts**: `generation/minesweeper_en.py`, `generation/minesweeper_ko.py`
- **Unique Solution Guaranteed**: Verify unique solution with backtracking solver
- **Minimal Hints**: Minimize hints while maintaining unique solution
- **Difficulty Levels**: Easy (6×6), Medium (8×8), Hard (10×10)
- **Coordinate-based Evaluation**: Output mine locations in (r,c) format

### 12. Number Baseball (EN / KO)
Infer hidden N-digit numbers through hints (Strike/Ball).

- **Scripts**: `generation/number_baseball_en.py`, `generation/number_baseball_ko.py`
- **Constraint Reasoning**: Narrow possible number ranges with Strike/Ball hints
- **Variable Digits**: Adjust difficulty with 3-digit, 4-digit, 5-digit numbers
- **No Duplicate Digits**: Each position has a different digit
- **Stepwise Reasoning**: Derive answer by combining multiple hints

### 13. SAT Puzzle (EN / KO)
NP-complete problem of finding value combinations that satisfy given logical expressions (CNF) for Boolean variables.

- **NP-Complete Problem**: Theoretically difficult problem
- **CNF (Conjunctive Normal Form)**: Express logical expressions in standard form
- **Natural Language Translation**: Represent as real situations like crime, meetings, task assignments
- **Unique Solution Guaranteed**: Backward generation ensures answer satisfies clauses
- **Difficulty Levels**: Easy (3-4 variables), Medium (5-7 variables), Hard (10-12 variables)
- **Pure Logical Reasoning**: Focus on Boolean logic

### 14. Sudoku (EN / KO)
9×9 Sudoku puzzle generation and difficulty-level evaluation dataset.

- **Scripts**: `generation/sudoku_en.py`, `generation/sudoku_ko.py`
- **Unique Solution Guaranteed**: All puzzles have exactly one solution
- **Difficulty Evaluation**: Automatic classification into Easy, Medium, Hard
- **Spot-check Evaluation**: HMAC-based K-cell selection for LLM evaluation support
- **Symmetry Support**: Improve aesthetic quality with rotation/reflection symmetry
- **Reproducible**: Regenerate identical puzzles with fixed seeds

### 15. Water Jug (EN / KO)
Classic water pouring puzzle. Measure exactly the target amount using jugs of fixed capacities, and answer the minimum number of operations.

- **Scripts**: `generation/water_jug_en.py`, `generation/water_jug_ko.py`
- **Three Operations**: Fill a jug, empty a jug, pour until the source empties or the target fills
- **BFS Ground Truth**: Minimum operation count guaranteed by breadth-first search
- **Planning & Search**: Requires state-space search, not pattern matching
- **Data Status**: 213 items per language in `data/csv/` only; JSONL splits currently live in `backups/data/`
- **Details**: see `docs/kor/readme_for_water_jug_ko.md`

### 16. Yacht Dice (EN / KO)
Combinatorial optimization problem of optimally assigning 12 dice results to 12 categories to maximize total score.

- **Scripts**: `generation/yacht_dice_en.py`, `generation/yacht_dice_ko.py`
- **Combinatorial Optimization**: 12! = 479,001,600 possible assignments
- **Hungarian Algorithm**: Calculate optimal solution
- **Bonus Score Calculation**: Bonus based on upper section total
- **Various Rule Modifications**: Support changes to bonus, scores, optimization objectives
- **Complex Scoring Rules**: Evaluate LLM's rule understanding and optimization ability

### 17. Saju — Four Pillars / Manseryeok (KO)
Korea-specific 사주(四柱) almanac task (nation-specific, KO-only like Kinship). Given a birth date/time, compute the sexagenary (60갑자) pillars: 연주(year), 월주(month), 일주(day), 시주(hour).

- **Scripts**: `generation/saju_ko.py`
- **Cultural Knowledge + Algorithm**: 입춘(立春) year boundary, 절기(節)-based month branch + 월두법(五虎遁), continuous 60갑자 day count (일진), 시두법(五鼠遁) hour stem
- **Difficulty from Knowledge the Model Lacks**: 일주/일진(day pillar) and 시주(hour pillar) require memorized almanac data that frontier models cannot reconstruct by reasoning → naturally lands in the hard band (~24% for gemini-3-flash-preview, thinking medium)
- **Deterministic Ground Truth**: 절기 via solar longitude (`ephem`, verified against KASI), 일주 cross-checked with `korean_lunar_calendar`

### 18. Jamo Composition (KO)
Korean-script structure task (nation-specific, KO-only). Decompose each 한글 syllable into 초성/중성/종성, shift the 초성 by a fixed number of positions, and recompose. Difficulty comes from 한글's syllable-block composition (including 겹받침), which has no English equivalent — a translated version collapses to a trivial Caesar shift.

- **Scripts**: `generation/jamo_ko.py`
- **Korean-script-dependent (not cultural)**: rules are given in the prompt; the challenge is correct decomposition/recomposition of 한글 blocks (초성·중성·종성, 겹받침)
- **Difficulty from script structure (not length)**: the 받침(종성) handling is the knob. easy = 2 syllables, light 받침; medium = 2 syllables, single 받침; hard = 3 syllables with 겹받침. All three tiers land in-band for gemini-3-flash-preview (thinking medium): easy ~80% / medium ~45% / hard ~20%
- **Deterministic Ground Truth**: pure Unicode composition (`0xAC00 + (초성×21+중성)×28+종성`); no lexical ambiguity, no external dependency

### 19. Time — Korean Calendar Reasoning (KO)
Korea-specific date reasoning task (KO-only). Starting from a Korean holiday anchor (새해 첫날, 어린이날, 식목일, ...) and a relative-day expression (금일/명일/모레, ...), compute an offset date.

- **Scripts**: `generation/time_ko.py`
- **Language + Calendar Fusion**: Korean relative-day vocabulary and holiday knowledge combined with date arithmetic
- **Answer Formats**: Gregorian date (`YYYY.M.D`) or the day's 일진(60갑자) depending on the variant
- **Deterministic Ground Truth**: pure calendar computation

### 20. Korean Units (KO)
Traditional Korean measurement-unit conversion task (KO-only). Convert mixed-unit quantities (e.g. 결·정보·마지기·단·평) to a base unit using a conversion table given in the prompt, then compute a weighted signed sum.

- **Scripts**: `generation/korean_units_ko.py`
- **Unit Families**: area (평/단/마지기/정보/결), volume (되/말/섬), length (푼/치/자/장), weight (돈/냥/근/관)
- **Randomized Rates**: conversion values are generated per problem — the table in the prompt is the only valid source, blocking memorized-knowledge shortcuts
- **Multi-step Arithmetic**: unit conversion → per-item multiplier → signed aggregation

### 21. Subway (KO)
Seoul metro route inference. Answer the minimum number of stations between two stations (KO dataset only in `data/`; an EN generator with romanized station names exists).

- **Scripts**: `generation/subway_ko.py`, `generation/subway_en.py`
- **Two Variants**: the current dataset in `data/` is the **knowledge-based** variant (only lines and terminal stations are given — station-order/transfer knowledge of the 2023 Seoul network is required); the generator script produces the **self-contained** variant (full line map included in the prompt)
- **BFS Ground Truth**: minimum station count via unweighted shortest path on the internal network graph
- **Details**: see `docs/kor/readme_for_subway_ko.md`


## Installation

```bash
# Clone the repository
git clone https://github.com/HAE-RAE/logical-puzzles.git
cd logical-puzzles

# Install dependencies
pip install -r requirements.txt
```

## Environment Setup

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

## Usage

### Puzzle Generation

```bash
# Generate all puzzles
bash run/generate/gen_data.sh

# Generate specific puzzle type
python generation/kinship.py --num 100
python generation/cipher_en.py --num 100
```

See [docs/eng/generation.md](docs/eng/generation.md) for detailed usage.

### Evaluation

The evaluation system supports two model routers:

- **liteLLM**: Call cloud APIs (Gemini, OpenAI, Anthropic, etc.) via liteLLM library
- **remote**: Call a self-hosted server (e.g. vLLM on Colab) via OpenAI-compatible API

**liteLLM mode (cloud APIs):**

```bash
python evaluation/run.py \
    --model gemini/gemini-3-flash-preview \
    --model_router litellm \
    --gen-kwargs "temperature=1.0,max_tokens=65536,top_p=0.95,top_k=64" \
    --tasks kinship --async
```

**Remote mode (self-hosted vLLM, etc.):**

```bash
python evaluation/run.py \
    --model Qwen/Qwen3-0.6B \
    --model_router remote \
    --remote_url "https://xxxx.ngrok-free.app" \
    --gen-kwargs "temperature=0.6,max_tokens=16384,top_p=0.95,top_k=20,reasoning=on" \
    --tasks kinship --async --max-concurrent 30
```

**Shell Scripts (Batch Evaluation):**

| Script | Mode | Execution |
|--------|------|-----------|
| `run/eval/eval_litellm.sh` | liteLLM | Sequential |
| `run/eval/eval_litellm_parallel.sh` | liteLLM | Parallel (5 concurrent) |
| `run/eval/eval_remote.sh` | Remote | Sequential |
| `run/eval/eval_remote_parallel.sh` | Remote | Parallel (5 concurrent) |

```bash
bash run/eval/eval_litellm_parallel.sh   # liteLLM (Gemini, etc.)
bash run/eval/eval_remote_parallel.sh    # Remote (Colab vLLM, etc.)
```

**Monitoring & Visualization:**

```bash
bash run/monitor/monitor.sh              # Monitor running evaluations
bash run/monitor/monitor.sh detailed     # Detailed view

jupyter notebook scripts/plot/viz_results.ipynb   # Visualize results
```

See [docs/eng/evaluation.md](docs/eng/evaluation.md) for detailed usage.

## Data Format
All puzzles are stored in two formats:

- **CSV**: `data/csv/{puzzle_name}.csv` - All difficulties in one file, easy to view in spreadsheets
- **JSONL**: `data/jsonl/{puzzle_name}_{difficulty}.jsonl` - Split by difficulty (`easy` / `medium` / `hard`), easy to process programmatically

**Dataset Size:** 100 items per difficulty → 300 items per task. (Exception: `water_jug` has 213 items per language, CSV only — its JSONL splits are in `backups/data/`.)

**Common Fields:**
- `id`: Unique identifier
- `question`: Problem description
- `answer`: Correct answer
- `solution`: Step-by-step reasoning process
- `difficulty`: Difficulty level (`easy` / `medium` / `hard`)
- Additional puzzle-specific metadata (optional)

## Project Structure

```
logical-puzzles/
├── backups/                    # Old generator versions & data snapshots
│
├── data/                       # Generated datasets (tracked in git)
│   ├── accuracy/               # accuracy_per_task.json (per-task/difficulty accuracy)
│   ├── csv/                    # {puzzle_name}.csv
│   └── jsonl/                  # {puzzle_name}_{difficulty}.jsonl
│
├── docs/                       # Documentation
│   ├── eng/                    # English docs (generation.md, evaluation.md)
│   ├── kor/                    # Korean docs incl. per-task design docs
│   └── methodology/            # Difficulty calibration & analysis docs
│
├── evaluation/                 # Unified evaluation system
│   ├── core/
│   │   ├── base.py
│   │   └── result_handler.py
│   ├── eval_data/              # Static eval assets (e.g. kinship_vision image)
│   ├── evaluators/
│   │   ├── cipher.py
│   │   ├── ferryman.py
│   │   ├── hanoi.py
│   │   ├── kinship.py
│   │   └── ... (more evaluators)
│   ├── legacy/                 # Legacy evaluation scripts (deprecated)
│   ├── model/                  # LLM client package
│   │   ├── __init__.py         # create_client() factory
│   │   ├── base.py             # BaseLLMClient (ABC)
│   │   ├── litellm.py          # LiteLLMClient
│   │   └── remote.py           # RemoteLLMClient (OpenAI-compatible)
│   ├── run.py
│   └── task_names.py
│
├── generation/                 # Puzzle generation scripts
│   ├── array_formula_en.py
│   ├── array_formula_ko.py
│   ├── causal_dag_en.py
│   ├── causal_dag_ko.py
│   └── ...
│
├── reports/                    # Aggregated evaluation reports
│
├── results/                    # Evaluation results (tracked; run logs gitignored)
│   └── {model_name}/
│       └── {task_name}_{difficulty}/
│           ├── {model}_{task}_{timestamp}__{accuracy}.csv
│           └── {model}_{task}_{timestamp}__{accuracy}.json
│
├── run/                        # Executable shell scripts
│   ├── generate/               # gen_data.sh, gen_data_by_difficulty.sh, ...
│   ├── eval/                   # eval_litellm.sh, eval_remote.sh, ...
│   ├── monitor/                # monitor.sh
│   └── pipeline/               # Distillation / training pipelines
│
├── scripts/                    # Python utilities
│   ├── _lib/                   # Shared helpers (io, parsing, paths, ...)
│   ├── analysis/               # Accuracy / search-space analysis
│   ├── calltest/               # API connection smoke tests
│   ├── distill/                # Distillation data pipeline
│   ├── eval/                   # Ad-hoc eval utilities
│   ├── gen/                    # Data post-processing (split, csv convert, ...)
│   ├── plot/                   # viz_results.ipynb, qwen_baseline.ipynb, plots
│   └── train/                  # SFT / GRPO training scripts
│
├── validators/
│   ├── audit_uniqueness.py
│   ├── check_logic_grid_uniqueness.py
│   ├── verify_logic_grid.py
│   └── verify_sat.py
│
├── .env                        # API keys (gitignored)
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Evaluation Results

Results are saved in `results/` directory with the following structure:

```
results/
└── {model_name}/
    └── {task_name}_{difficulty}/
        ├── {model}_{task}_{timestamp}__{accuracy}.csv  # Detailed results
        └── {model}_{task}_{timestamp}__{accuracy}.json  # Summary
```

Aggregated cross-model reports live in `reports/`, and a per-task accuracy summary is kept at `data/accuracy/accuracy_per_task.json`.

## Adding New Puzzles

When adding a new puzzle, please follow this structure:

### Required Files
```
generation/{puzzle_name}.py           # Puzzle generation logic
evaluation/evaluators/{puzzle_name}.py # Evaluator (for unified system)
```

### Recommendations
- **Difficulty Levels**: Minimum 3 levels (easy/medium/hard)
- **Validation Tools**: Consider adding validation scripts to `validators/` folder
- **Evaluator Integration**: Add to `evaluation/evaluators/__init__.py` registry

## Notes

- The `data/` and `results/` directories are tracked in git; only run logs (`results/**/log/`) are gitignored
- Do not commit API keys or sensitive information (use `.env` file)
- Generated data is automatically saved to `data/csv/` and `data/jsonl/`
- Evaluation results are saved in `results/{model}/{task}_{difficulty}/` directory

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.