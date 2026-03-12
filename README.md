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
- **Difficulty Levels**: EASY (Sub+Rev), MEDIUM (Vig), HARD (Vig+Rev), VERY_HARD (Playfair+Vig), EXTREME (Play+Trans+Vig)

### 4. Cryptarithmetic
Find the digit corresponding to each letter in arithmetic expressions represented by letters.

- **Constraints**: No leading zeros, each letter is a unique digit
- **Backtracking Solver**: Verify unique solution
- **Various Operations**: Support addition, subtraction, multiplication
- **Difficulty Levels**: Adjusted based on number of letters and digits

### 5. Ferryman (EN / KO)
Calculate total time required for a ferryman to transport goods considering various navigation rules and constraints.

- **Speed-limited Zones**: Speed changes in Zone A and Zone B
- **Cargo Weight**: Speed reduction based on weight
- **Mandatory Rest Time**: Compliance with navigation regulations
- **Complex Condition Reasoning**: Consider multiple constraints simultaneously

### 6. Hanoi (EN / KO)
Classic puzzle of moving disks from one peg to another. Calculate the minimum number of moves.

- **Recursive Structure**: Requires divide-and-conquer thinking
- **Constraints**: Only smaller disks can be placed on larger disks
- **Minimum Moves**: 2^n - 1 moves (n is number of disks)
- **Difficulty Levels**: Adjusted based on number of disks

### 7. Inequality
Find number placements that satisfy given inequality constraints.

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

### 10. Logic Grid (EN / KO)
Constraint-based logical reasoning problem famous as Einstein's Riddle. Infer relationships among multiple people and attributes from constraint conditions.

- **CSP (Constraint Satisfaction Problem)**: Verify unique solution with backtracking
- **Natural Language Constraints**: Simultaneously require linguistic understanding and logical reasoning
- **Unique Solution Guaranteed**: Verified with Constraint Propagation algorithm
- **Difficulty Levels**: Easy (3×3), Medium (4×4), Hard (5×5)
- **Backward Generation**: Generate valid solution first, then derive constraint conditions

### 11. Minesweeper
Minesweeper puzzle designed as a Constraint Satisfaction Problem (CSP). Evaluates LLM's logical reasoning ability.

- **Unique Solution Guaranteed**: Verify unique solution with backtracking solver
- **Minimal Hints**: Minimize hints while maintaining unique solution
- **Difficulty Levels**: Easy (6×6), Medium (8×8), Hard (10×10)
- **Coordinate-based Evaluation**: Output mine locations in (r,c) format

### 12. Number Baseball
Infer hidden N-digit numbers through hints (Strike/Ball).

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

### 14. Sudoku
9×9 Sudoku puzzle generation and difficulty-level evaluation dataset.

- **Unique Solution Guaranteed**: All puzzles have exactly one solution
- **Difficulty Evaluation**: Automatic classification into Easy, Medium, Hard, Expert, Extreme
- **Spot-check Evaluation**: HMAC-based K-cell selection for LLM evaluation support
- **Symmetry Support**: Improve aesthetic quality with rotation/reflection symmetry
- **Reproducible**: Regenerate identical puzzles with fixed seeds

### 15. Yacht Dice
Combinatorial optimization problem of optimally assigning 12 dice results to 12 categories to maximize total score.

- **Combinatorial Optimization**: 12! = 479,001,600 possible assignments
- **Hungarian Algorithm**: Calculate optimal solution
- **Bonus Score Calculation**: Bonus based on upper section total
- **Various Rule Modifications**: Support changes to bonus, scores, optimization objectives
- **Complex Scoring Rules**: Evaluate LLM's rule understanding and optimization ability


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
bash scripts/generate_all.sh

# Generate specific puzzle type
python generation/kinship.py --num 100
python generation/cipher_en.py --num 100
```

See [docs/eng/generation.md](docs/eng/generation.md) for detailed usage.

### Evaluation

#### Unified Evaluation System (Recommended)

**Basic Usage:**

```bash
# Evaluate all tasks (uses config.yaml settings)
python evaluation/run.py

# Evaluate specific tasks
python evaluation/run.py --tasks kinship cipher hanoi

# Use different models
python evaluation/run.py --model gemini/gemini-3-flash-preview
python evaluation/run.py --model gpt-4o
python evaluation/run.py --model claude-3-5-sonnet-20241022

# Filter by difficulty and limit
python evaluation/run.py --difficulty easy --limit 10
```

**Async Mode:**

The async mode is controlled by `evaluation/config.yaml` (default: `use_async: true`):

```bash
# Async mode evaluation (default from config.yaml)
python evaluation/run.py

# Explicitly enable async mode (same as default if config.yaml has use_async: true)
python evaluation/run.py --async

# To disable async mode, modify evaluation/config.yaml: use_async: false
# Then run without --async flag for sync mode

# Adjust concurrent execution count (default: 30 from config.yaml)
python evaluation/run.py --max-concurrent 50
```

**Configuration File (`evaluation/config.yaml`):**

The evaluation system uses `evaluation/config.yaml` for default settings:
- **LLM Configuration**: model, temperature, max_tokens (65536), timeout (600s)
- **Evaluation Settings**: use_async (true), max_concurrent (30)
- **Task List**: 17 tasks (excluding sudoku and minesweeper)
- **Difficulty Levels**: easy, medium, hard

You can modify this file to change default behavior, or override with command-line arguments.

**Note:** Currently, `--async` flag has no effect because `config.yaml` already sets `use_async: true` as default. The flag is useful when you want to override a `false` setting in config.yaml.

**Advanced Options:**

```bash
python evaluation/run.py \
    --model gemini/gemini-3-flash-preview \
    --tasks kinship cipher \
    --difficulty medium \
    --limit 20 \
    --output-dir results/my_test \
    --async \
    --max-concurrent 50 \
    --quiet
```

**Shell Scripts (Batch Evaluation of 17 Tasks):**

Two scripts are available for batch evaluation:

1. **Sequential Execution** (`evaluate_all.sh`):
   ```bash
   # Evaluate 17 tasks one by one (stable, slower)
   bash scripts/evaluate_all.sh
   ```
   - Executes tasks sequentially (one at a time)
   - More stable and easier to debug
   - Lower resource usage
   - Clearer log output

2. **Parallel Execution** (`evaluate_all_parallel.sh`):
   ```bash
   # Evaluate 17 tasks in parallel (5 at a time, faster)
   bash scripts/evaluate_all_parallel.sh
   ```
   - Executes up to 5 tasks simultaneously
   - Significantly faster (approximately 3-5x speedup)
   - Higher resource usage
   - Both scripts evaluate all 17 tasks (excluding sudoku and minesweeper)

**Monitoring Running Evaluations:**

```bash
# Simple table view
bash scripts/monitor_eval.sh

# Detailed view with full information
bash scripts/monitor_eval.sh detailed

# Show help
bash scripts/monitor_eval.sh help
```

The monitoring script shows:
- Running evaluation processes (PID, model, task)
- Progress information from log files
- Accuracy (when available)
- Log file locations

**Result Visualization:**

```bash
# Visualize results with Jupyter notebook
jupyter notebook scripts/visualize_results.ipynb
# or
jupyter lab scripts/visualize_results.ipynb
```

See [docs/eng/evaluation.md](docs/eng/evaluation.md) for detailed usage.

## Data Format
All puzzles are stored in two formats:

- **CSV**: `data/csv/{puzzle_name}.csv` - Easy to view in spreadsheets
- **JSONL**: `data/json/{puzzle_name}.jsonl` - Easy to process programmatically

**Common Fields:**
- `id`: Unique identifier
- `question`: Problem description
- `answer`: Correct answer
- `solution`: Step-by-step reasoning process
- `difficulty`: Difficulty level (easy/medium/hard, etc.)
- Additional puzzle-specific metadata (optional)

## Project Structure

```
logical-puzzles/
├── data/                       # Generated datasets (gitignored)
│   ├── csv/
│   └── json/
│
├── docs/                       # Documentation
│   ├── README.md
│   ├── evaluation.md
│   ├── generation.md
│   └── puzzles/
│       ├── array_formula.md
│       └── YACHT_DICE_USAGE.md
│
├── evaluation/                 # Unified evaluation system
│   ├── core/
│   │   ├── base.py
│   │   ├── llm_client.py
│   │   └── result_handler.py
│   ├── eval_data/              # Static evaluation data
│   │   ├── kinship_vision/
│   │   │   └── kinship.jpg
│   │   └── minesweeper/
│   │       ├── eval_metadata.jsonl
│   │       ├── eval_puzzles.jsonl
│   │       ├── eval_solutions.jsonl
│   │       └── solution.md
│   ├── evaluators/
│   │   ├── cipher.py
│   │   ├── ferryman.py
│   │   ├── hanoi.py
│   │   ├── kinship.py
│   │   └── ... (more evaluators)
│   ├── legacy/                 # Legacy evaluation scripts (deprecated)
│   ├── __init__.py
│   ├── config.yaml
│   └── run.py
│
├── generation/                 # Puzzle generation scripts
│   ├── array_formula_en.py
│   ├── array_formula_ko.py
│   ├── causal_dag_en.py
│   ├── cipher_en.py
│   ├── cryptarithmetic.py
│   ├── kinship.py
│   └── ... (15 puzzle types, _en/_ko for bilingual)
│
├── results/                    # Evaluation results (gitignored)
│   └── {model_name}/
│       └── {task_name}/
│           ├── {model}_{task}_{timestamp}__{accuracy}.csv
│           └── {model}_{task}_{timestamp}__{accuracy}.json
│
├── scripts/
│   ├── generate_all.sh         # Generate all puzzles
│   ├── evaluate_all.sh          # Sequential evaluation of 17 tasks
│   ├── evaluate_all_parallel.sh # Parallel evaluation of 17 tasks (5 concurrent)
│   ├── monitor_eval.sh          # Monitor running evaluations
│   └── visualize_results.ipynb  # Result visualization notebook
│
├── validators/
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
    └── {task_name}/
        ├── {model}_{task}_{timestamp}__{accuracy}.csv  # Detailed results
        └── {model}_{task}_{timestamp}__{accuracy}.json  # Summary by difficulty
```

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

- The `data/` and `results/` directories are gitignored and stored locally only
- Do not commit API keys or sensitive information (use `.env` file)
- Generated data is automatically saved to `data/csv/` and `data/json/`
- Evaluation results are saved in `results/{model}/{task}/` directory

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.