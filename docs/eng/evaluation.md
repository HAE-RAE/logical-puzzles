# Logical-Puzzles Unified Evaluation System

A unified evaluation system for 17 logical puzzle tasks.

## Supported Tasks

Currently supports 17 tasks (excluding sudoku and minesweeper from batch evaluation):

- `kinship`: Korean kinship relationships (multiple choice A-E)
- `kinship_vision`: Image-based kinship relationships (uses same evaluator)
- `cipher`: English cipher decryption
- `cipher_korean`: Korean cipher decryption (uses same evaluator)
- `hanoi`: Tower of Hanoi (disk, from, to)
- `ferryman`: Ferryman navigation (X hours Y minutes)
- `array_formula`: Array formula calculations
- `causal_dag`: Causal DAG inference (English)
- `causal_dag_korean`: Causal DAG inference (Korean, uses same evaluator)
- `cryptarithmetic`: Cryptarithmetic puzzles
- `inequality`: Inequality constraint satisfaction
- `logic_grid`: Logic grid puzzles (English)
- `logic_grid_korean`: Logic grid puzzles (Korean, uses same evaluator)
- `number_baseball`: Number baseball (Strike/Ball)
- `sat_puzzles`: SAT puzzle solving (English)
- `sat_puzzles_korean`: SAT puzzle solving (Korean, uses same evaluator)
- `yacht_dice`: Yacht dice optimization

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

### Basic Usage

```bash
# Run from project root
cd logical-puzzles

# Evaluate all tasks (uses config.yaml settings)
python evaluation/run.py
python -m evaluation.run

# Evaluate specific tasks
python evaluation/run.py --tasks kinship cipher hanoi

# Use different models
python evaluation/run.py --model gemini/gemini-3-flash-preview
python evaluation/run.py --model gemini/gemini-2.5-flash
python evaluation/run.py --model gpt-4o
python evaluation/run.py --model claude-3-5-sonnet-20241022

# Filter by difficulty and limit
python evaluation/run.py --difficulty easy --limit 10
```

### Async Mode

The async mode is controlled by `evaluation/config.yaml` (default: `use_async: true`):

```bash
# Async mode evaluation (default from config.yaml)
python evaluation/run.py

# Explicitly enable async mode (same as default if config.yaml has use_async: true)
python evaluation/run.py --async

# Adjust concurrent execution count (default: 30 from config.yaml)
python evaluation/run.py --max-concurrent 50
```

**Note:** Currently, `--async` flag has no effect because `config.yaml` already sets `use_async: true` as default. The flag is useful when you want to override a `false` setting in config.yaml.

### Advanced Options

```bash
python evaluation/run.py \
    --model gemini/gemini-3-flash-preview \
    --tasks kinship cipher hanoi \
    --difficulty medium \
    --limit 20 \
    --output-dir results/my_test \
    --async \
    --max-concurrent 50 \
    --quiet
```

### Shell Scripts (Batch Evaluation of 17 Tasks)

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

2. **Parallel Execution** (`evaluate_all_parallel.sh`) ⭐ **Recommended**:
   ```bash
   # Evaluate 17 tasks in parallel (5 at a time, faster)
   bash scripts/evaluate_all_parallel.sh
   ```
   - Executes up to 5 tasks simultaneously
   - Significantly faster (approximately 3-5x speedup)
   - Higher resource usage
   - Both scripts evaluate all 17 tasks (excluding sudoku and minesweeper)
   - **Due to parallel processing, separate log files are created for each task**
     - Log file location: `results/log/{task_name}.log`
     - Records start/end time and status (SUCCESS/FAILED) for each task
     - You can monitor progress in real-time by checking the logs

### Monitoring Running Evaluations

```bash
# Simple table view (default)
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

### Result Visualization

```bash
# Visualize results with Jupyter notebook
jupyter notebook scripts/visualize_results.ipynb
# or
jupyter lab scripts/visualize_results.ipynb
```

The visualization notebook provides:
- Overall accuracy by task
- Accuracy by task and difficulty (grouped bar chart)
- Accuracy heatmap by task and difficulty
- Average latency by task
- Accuracy vs latency scatter plot

## API Key Priority

The system searches for API keys in the following order:

1. `.env` file (project root)
2. Environment variables (`export GEMINI_API_KEY=...`)
3. LiteLLM default settings

**Recommended**: Store all API keys in the `.env` file.

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
  - `summary.by_difficulty`: Statistics by difficulty level (keys are lowercase: `easy`, `medium`, `hard`, `expert`)

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

**Note:** Difficulty values are stored in lowercase (`easy`, `medium`, `hard`, `expert`) for consistency.

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

## Configuration File (config.yaml)

You can manage default settings in `evaluation/config.yaml`:

```yaml
llm:
  model: gemini/gemini-3-flash-preview
  temperature: 1.0
  max_tokens: 65536  # Increased for tasks requiring long responses (e.g., yacht_dice)
  top_p: 0.95
  top_k: 64
  # reasoning_effort: medium  # Optional, currently disabled
  timeout: 600.0     # Timeout in seconds

data_dir: data/json
output_dir: results

evaluation:
  use_async: true      # Async mode enabled by default
  max_concurrent: 30    # Maximum concurrent executions

tasks:
  - kinship
  - kinship_vision
  - cipher
  - cipher_korean
  - hanoi
  - ferryman
  - array_formula
  - causal_dag
  - causal_dag_korean
  - cryptarithmetic
  - inequality
  - logic_grid
  - logic_grid_korean
  - number_baseball
  - sat_puzzles
  - sat_puzzles_korean
  - yacht_dice

difficulties:
  - easy
  - medium
  - hard
```

**Configuration Priority:**
1. Command-line arguments (highest priority)
2. `config.yaml` settings
3. Default values (lowest priority)

## Structure

```
evaluation/
├── core/                     # Core components
│   ├── base.py               # Base data structures
│   ├── llm_client.py         # LiteLLM wrapper (auto-loads .env)
│   └── result_handler.py     # Result saving
├── evaluators/               # Task-specific evaluators
│   ├── __init__.py           # Registry
│   ├── kinship.py
│   ├── cipher.py
│   ├── hanoi.py
│   ├── ferryman.py
│   ├── array_formula.py
│   ├── causal_dag.py
│   ├── cryptarithmetic.py
│   ├── inequality.py
│   ├── logic_grid.py
│   ├── number_baseball.py
│   ├── sat_puzzle.py
│   ├── yacht_dice.py
│   └── ... (more evaluators)
├── legacy/                 # Legacy evaluation scripts (reference)
│   ├── README.md
│   └── eval_*.py
├── eval_data/              # Static evaluation data
│   ├── kinship_vision/
│   │   └── kinship.jpg
│   └── minesweeper/
│       ├── eval_metadata.jsonl
│       ├── eval_puzzles.jsonl
│       ├── eval_solutions.jsonl
│       └── solution.md
├── run.py                   # Main execution script (auto-loads .env)
├── config.yaml              # Configuration file
└── README.md                # This document
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

Various models are available through LiteLLM:

**Google Gemini** (requires GEMINI_API_KEY, as of February 2026):
- `gemini/gemini-3-flash-preview` ⭐ (default, latest, powerful)
- `gemini/gemini-2.5-flash` (stable, fast)

> **Note**: LiteLLM automatically uses `GEMINI_API_KEY` from `.env`.

**OpenAI** (requires OPENAI_API_KEY):
- `gpt-4o`
- `gpt-4o-mini`
- `gpt-4-turbo`

**Anthropic** (requires ANTHROPIC_API_KEY):
- `claude-3-5-sonnet-20241022`
- `claude-3-opus-20240229`
