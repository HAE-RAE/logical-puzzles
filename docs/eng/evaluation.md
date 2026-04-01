# Logical-Puzzles Unified Evaluation System

A unified evaluation system for **28** logical puzzle tasks.

## Supported Tasks

The evaluator registry defines **28** task keys. English and Korean datasets use `*_en` and `*_ko` suffixes; many pairs share the same evaluator class.

### Multilingual subset (en/ko) — 13 pairs, 26 tasks

- **Array Formula** — `array_formula_en` / `array_formula_ko` — Spreadsheet-style array formulas (lookup tables, conditional aggregation, multi-condition arithmetic)
- **Causal DAG** — `causal_dag_en` / `causal_dag_ko` — Causal graphs (infer time-lagged propagation paths between events)
- **Cipher** — `cipher_en` / `cipher_ko` — Cipher decryption (reverse a stack of layered ciphers)
- **Cryptarithmetic** — `cryptarithmetic_en` / `cryptarithmetic_ko` — Cryptarithms (digit substitution to satisfy arithmetic equalities)
- **Ferryman** — `ferryman_en` / `ferryman_ko` — Journey planning (speed limits, rest rules, congestion, and other leg constraints)
- **Hanoi** — `hanoi_en` / `hanoi_ko` — Tower of Hanoi (disk move sequences and state tracking)
- **Inequality** — `inequality_en` / `inequality_ko` — Inequality grids (place 1…N subject to inequality clues between cells)
- **Logic Grid** — `logic_grid_en` / `logic_grid_ko` — Einstein / Zebra-style logic (multi-attribute deduction)
- **Minesweeper** — `minesweeper_en` / `minesweeper_ko` — Minesweeper (infer mine locations from adjacent number hints)
- **Number Baseball** — `number_baseball_en` / `number_baseball_ko` — Number baseball (infer a secret code from strike/ball feedback)
- **SAT Puzzle** — `sat_puzzles_en` / `sat_puzzles_ko` — Boolean satisfiability (evaluate or satisfy CNF formulas)
- **Sudoku** — `sudoku_en` / `sudoku_ko` — Sudoku (fill a grid under row, column, and box constraints)
- **Yacht Dice** — `yacht_dice_en` / `yacht_dice_ko` — Yacht dice (assign twelve dice rolls to twelve categories for maximum score)

### Korean-only subset — 2 tasks

- **Kinship** — `kinship` — Korean kinship titles from text (multiple choice)
- **Kinship Vision** — `kinship_vision` — Korean kinship titles with images (multimodal: photo + dialogue)

## Installation

```bash
# Install required packages
pip install litellm python-dotenv
```

## Environment Setup

Create a `.env` file in the project root and set your API keys:

```bash
# .env file example
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

The system automatically loads the `.env` file to use API keys.

## Usage

The evaluation system supports two model routers:

- **liteLLM**: Call cloud APIs (Gemini, OpenAI, Anthropic, etc.) via liteLLM library
- **remote**: Call a self-hosted server (e.g. vLLM on Colab) via OpenAI-compatible API

All configuration is done through CLI arguments (no config file needed).

### liteLLM Mode (Cloud APIs)

```bash
# Basic usage
python evaluation/run.py \
    --model gemini/gemini-3-flash-preview \
    --model_router litellm \
    --gen-kwargs "temperature=1.0,max_tokens=65536,top_p=0.95,top_k=64" \
    --tasks kinship --async

# Evaluate specific tasks with difficulty filter
python evaluation/run.py \
    --model gemini/gemini-3-flash-preview \
    --model_router litellm \
    --gen-kwargs "temperature=1.0,max_tokens=65536" \
    --tasks kinship cipher_en hanoi_en \
    --difficulty medium --limit 20 \
    --async --max-concurrent 50
```

### Remote Mode (Self-hosted vLLM, etc.)

```bash
python evaluation/run.py \
    --model Qwen/Qwen3-0.6B \
    --model_router remote \
    --remote_url "https://xxxx.ngrok-free.app" \
    --gen-kwargs "temperature=0.6,max_tokens=16384,top_p=0.95,top_k=20,reasoning=on" \
    --tasks kinship --async --max-concurrent 30
```

### CLI Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--model` | Yes | Model name (e.g. `gemini/gemini-3-flash-preview`, `Qwen/Qwen3-0.6B`) |
| `--model_router` | Yes | `litellm` or `remote` |
| `--remote_url` | For remote | Remote server URL |
| `--gen-kwargs` | No | Generation params as `key=value,key=value` |
| `--timeout` | No | Request timeout in seconds (default: 600) |
| `--tasks` | No | Tasks to evaluate (all if omitted) |
| `--async` | No | Enable async mode |
| `--max-concurrent` | No | Max concurrent requests (default: 30) |
| `--difficulty` | No | Filter by difficulty |
| `--limit` | No | Max puzzles per task |
| `--quiet` | No | Minimize output |

### Shell Scripts (Batch Evaluation)

| Script | Mode | Execution |
|--------|------|-----------|
| `eval_litellm.sh` | liteLLM | Sequential |
| `eval_litellm_parallel.sh` | liteLLM | Parallel (5 concurrent) |
| `eval_remote.sh` | Remote | Sequential |
| `eval_remote_parallel.sh` | Remote | Parallel (5 concurrent) |

```bash
# liteLLM (Gemini, etc.)
bash scripts/eval_litellm.sh
bash scripts/eval_litellm_parallel.sh     # Recommended

# Remote (Colab vLLM, etc.)
bash scripts/eval_remote.sh
bash scripts/eval_remote_parallel.sh      # Recommended
```

Parallel scripts create per-task log files at `results/{model_name}/log/{task}.log`.

### Monitoring Running Evaluations

```bash
bash scripts/monitor.sh              # Simple table view
bash scripts/monitor.sh detailed     # Detailed view
bash scripts/monitor.sh help         # Show help
```

### Result Visualization

```bash
jupyter notebook scripts/viz_results.ipynb
# or
jupyter lab scripts/viz_results.ipynb
```

## API Key Priority (liteLLM mode)

The system searches for API keys in the following order:

1. `.env` file (project root)
2. Environment variables (`export GEMINI_API_KEY=...`)
3. LiteLLM default settings

**Recommended**: Store all API keys in the `.env` file.

> **Note**: Remote mode does not require API keys in `.env`. The server URL is passed via `--remote_url`.

## Output Results

Results are saved in the `results/` directory with a **model/task folder structure**.

```
results/
├── gemini_gemini-3-flash-preview/
│   ├── kinship/
│   │   ├── gemini_gemini-3-flash-preview_kinship_2026-02-02T13-46-22__0.50.csv
│   │   └── gemini_gemini-3-flash-preview_kinship_2026-02-02T13-46-22__0.50.json
│   └── cipher/
│       ├── gemini_gemini-3-flash-preview_cipher_2026-02-02T14-00-00__0.60.csv
│       └── gemini_gemini-3-flash-preview_cipher_2026-02-02T14-00-00__0.60.json
└── gpt-4o/
    └── kinship/
        └── ...
```

**File Formats:**
- **CSV**: Detailed results (`{model}_{task}_{timestamp}__{accuracy}.csv`)
  - Columns: `id`, `question`, `answer`, `resps` (raw LLM response), `filtered_resps` (parsed answer), `exact_match` (0 or 1), `difficulty` (lowercase)
  - Each row represents one puzzle evaluation
- **JSON**: Summary by difficulty (`{model}_{task}_{timestamp}__{accuracy}.json`)
  - `summary.overall`: Overall accuracy, correct count, total count, average latency (ms)
  - `summary.by_difficulty`: Statistics by difficulty level (keys are lowercase: `easy`, `medium`, `hard`)

**Advantages:**
- Organized by model for easy comparison
- Separated by task for easy management
- CSV for detailed analysis, JSON for summary

### CSV File Structure

```csv
id,question,answer,resps,filtered_resps,exact_match,difficulty
kinship_0,"Question content...",A,"Model raw response",A,1,easy
kinship_1,"Question content...",B,"Model raw response",C,0,medium
```

**Note:** Difficulty values are stored in lowercase (`easy`, `medium`, `hard`) for consistency.

### JSON File Structure (Summary by Difficulty)

```json
{
  "metadata": {
    "task": "kinship",
    "model": "gemini/gemini-3-flash-preview",
    "timestamp": "2026-02-02T13-46-22",
    "total_puzzles": 300
  },
  "summary": {
    "overall": {
      "accuracy": 0.5566666666666666,
      "correct_count": 167,
      "total_count": 300,
      "avg_latency_ms": 95893.96
    },
    "by_difficulty": {
      "easy": {
        "total": 100,
        "correct": 60,
        "accuracy": 0.60
      },
      "medium": {
        "total": 100,
        "correct": 50,
        "accuracy": 0.50
      },
      "hard": {
        "total": 100,
        "correct": 40,
        "accuracy": 0.40
      }
    }
  }
}
```

## Generation Parameters (`--gen-kwargs`)

Pass generation parameters as comma-separated `key=value` pairs:

```bash
# liteLLM (Gemini)
--gen-kwargs "temperature=1.0,max_tokens=65536,top_p=0.95,top_k=64"

# Remote (Qwen3 thinking mode)
--gen-kwargs "temperature=0.6,max_tokens=16384,top_p=0.95,top_k=20,reasoning=on"
```

Special keys:
- `reasoning=on`: Enable thinking mode (adds `enable_thinking: true` for remote mode)
- Numeric values are auto-converted to `int` or `float`

## Structure

```
evaluation/
├── core/                     # Core components
│   ├── base.py               # Base data structures
│   └── result_handler.py     # Result saving
├── model/                    # LLM client package
│   ├── __init__.py           # create_client() factory
│   ├── base.py               # BaseLLMClient (ABC)
│   ├── litellm.py            # LiteLLMClient
│   └── remote.py             # RemoteLLMClient (OpenAI-compatible)
├── evaluators/               # Task-specific evaluators
│   ├── __init__.py           # Registry
│   ├── array_formula.py
│   ├── causal_dag.py
│   ├── cipher.py
│   ├── cryptarithmetic.py
│   ├── ferryman.py
│   ├── hanoi.py
│   ├── inequality.py
│   ├── kinship.py
│   ├── logic_grid.py
│   ├── minesweeper.py
│   ├── number_baseball.py
│   ├── sat_puzzle.py
│   ├── sudoku.py
│   └── yacht_dice.py
├── legacy/                 # Legacy evaluation scripts (reference)
│   ├── README.md
│   └── eval_*.py
├── eval_data/              # Static evaluation data
│   ├── kinship_vision/
│   │   └── kinship.jpg
│   └── minesweeper/
│       └── ...
└── run.py                   # Main execution script
```

## Adding New Tasks

1. Create a new evaluator file in `evaluators/`
2. Implement `_parse_answer()` and `_check_answer()` methods
3. Register in `evaluators/__init__.py`'s `EVALUATOR_REGISTRY`

Example:
```python
# evaluators/my_task.py
from ..core.base import BaseEvaluator

class MyTaskEvaluator(BaseEvaluator):
    SYSTEM_PROMPT = "..."
    
    def _parse_answer(self, response: str) -> Any:
        # Parse answer from response
        pass
        
    def _check_answer(self, expected: Any, predicted: Any) -> Tuple[bool, float]:
        # Check if answer is correct
        # Returns: (is_correct, score)
        # Note: All evaluators use binary scoring (1.0 for correct, 0.0 for incorrect)
        # Partial scores have been removed for consistency
        return correct, 1.0 if correct else 0.0
```

## Supported Models

### liteLLM Mode

Various models are available through LiteLLM:

**Google Gemini** (requires GEMINI_API_KEY):
- `gemini/gemini-3-flash-preview` (latest, powerful)
- `gemini/gemini-2.5-flash` (stable, fast)

**OpenAI** (requires OPENAI_API_KEY):
- `gpt-4o`
- `gpt-4o-mini`

**Anthropic** (requires ANTHROPIC_API_KEY):
- `claude-3-5-sonnet-20241022`
- `claude-3-opus-20240229`

### Remote Mode

Any model served via an OpenAI-compatible API (e.g. vLLM):
- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen3-1.7B`
- Any HuggingFace model supported by vLLM
