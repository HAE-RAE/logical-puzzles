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

### 15. Yacht Dice (EN / KO)
Combinatorial optimization problem of optimally assigning 12 dice results to 12 categories to maximize total score.

- **Scripts**: `generation/yacht_dice_en.py`, `generation/yacht_dice_ko.py`
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
bash scripts/gen_data.sh

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
| `eval_litellm.sh` | liteLLM | Sequential |
| `eval_litellm_parallel.sh` | liteLLM | Parallel (5 concurrent) |
| `eval_remote.sh` | Remote | Sequential |
| `eval_remote_parallel.sh` | Remote | Parallel (5 concurrent) |

```bash
bash scripts/eval_litellm_parallel.sh   # liteLLM (Gemini, etc.)
bash scripts/eval_remote_parallel.sh    # Remote (Colab vLLM, etc.)
```

**Monitoring & Visualization:**

```bash
bash scripts/monitor.sh              # Monitor running evaluations
bash scripts/monitor.sh detailed     # Detailed view

jupyter notebook scripts/viz_results.ipynb   # Visualize results
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
- `difficulty`: Difficulty level (`easy` / `medium` / `hard`)
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
│   │   └── result_handler.py
│   ├── model/                  # LLM client package
│   │   ├── __init__.py         # create_client() factory
│   │   ├── base.py             # BaseLLMClient (ABC)
│   │   ├── litellm.py          # LiteLLMClient
│   │   └── remote.py           # RemoteLLMClient (OpenAI-compatible)
│   ├── evaluators/
│   │   ├── cipher.py
│   │   ├── ferryman.py
│   │   ├── hanoi.py
│   │   ├── kinship.py
│   │   └── ... (more evaluators)
│   ├── legacy/                 # Legacy evaluation scripts (deprecated)
│   └── run.py
│
├── generation/                 # Puzzle generation scripts
│   ├── array_formula_en.py
│   ├── array_formula_ko.py
│   ├── causal_dag_en.py
│   ├── causal_dag_ko.py
│   ├── cryptarithmetic_en.py
│   ├── cryptarithmetic_ko.py
│   ├── ferryman_en.py
│   ├── ferryman_ko.py
│   ├── hanoi_en.py
│   ├── hanoi_ko.py
│   └── ...
│
├── results/                    # Evaluation results (gitignored)
│   └── {model_name}/
│       └── {task_name}/
│           ├── {model}_{task}_{timestamp}__{accuracy}.csv
│           └── {model}_{task}_{timestamp}__{accuracy}.json
│
├── scripts/
│   ├── gen_data.sh              # Generate all puzzles
│   ├── eval_litellm.sh          # liteLLM evaluation (sequential)
│   ├── eval_litellm_parallel.sh # liteLLM evaluation (parallel)
│   ├── eval_remote.sh           # Remote evaluation (sequential)
│   ├── eval_remote_parallel.sh  # Remote evaluation (parallel)
│   ├── calltest.py              # API connection smoke test
│   ├── qwen_baseline.ipynb      # Colab vLLM server for Qwen baseline
│   ├── monitor.sh               # Monitor running evaluations
│   └── viz_results.ipynb        # Result visualization notebook
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