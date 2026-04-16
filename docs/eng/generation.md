# Logical-Puzzles Dataset Generation Guide

A collection of scripts for generating datasets for 28 logical puzzle generators (including English/Korean variants).

## 📋 Supported Tasks

| Task | Script | Difficulty | Default Count |
|------|--------|------------|---------------|
| Array Formula (EN) | `array_formula_en.py` | Easy, Medium, Hard | 100 |
| Array Formula (KO) | `array_formula_ko.py` | Easy, Medium, Hard | 100 |
| Causal DAG (KO) | `causal_dag_ko.py` | Easy, Medium, Hard | 300 |
| Causal DAG (EN) | `causal_dag_en.py` | Easy, Medium, Hard | 300 |
| Cipher (KO) | `cipher_ko.py` | Easy, Medium, Hard | 100 |
| Cipher (EN) | `cipher_en.py` | Easy, Medium, Hard | 100 |
| Cryptarithmetic (EN) | `cryptarithmetic_en.py` | Easy, Medium, Hard | 400 |
| Cryptarithmetic (KO) | `cryptarithmetic_ko.py` | Easy, Medium, Hard | 400 |
| Ferryman (EN) | `ferryman_en.py` | Easy, Medium, Hard | 100 |
| Ferryman (KO) | `ferryman_ko.py` | Easy, Medium, Hard | 100 |
| Hanoi (EN) | `hanoi_en.py` | - | 100 |
| Hanoi (KO) | `hanoi_ko.py` | - | 100 |
| Inequality (EN) | `inequality_en.py` | Easy, Medium, Hard | 400 |
| Inequality (KO) | `inequality_ko.py` | Easy, Medium, Hard | 400 |
| Kinship | `kinship.py` | - | 100 |
| Kinship Vision | `kinship_vision.py` | - | 100 |
| Logic Grid (KO) | `logic_grid_ko.py` | Easy, Medium, Hard | 300 |
| Logic Grid (EN) | `logic_grid_en.py` | Easy, Medium, Hard | 300 |
| Minesweeper (EN) | `minesweeper_en.py` | Easy, Medium, Hard | - |
| Minesweeper (KO) | `minesweeper_ko.py` | Easy, Medium, Hard | - |
| Number Baseball (EN) | `number_baseball_en.py` | Easy, Medium, Hard | 400 |
| Number Baseball (KO) | `number_baseball_ko.py` | Easy, Medium, Hard | 400 |
| SAT Puzzle (KO) | `sat_puzzle_ko.py` | Easy, Medium, Hard | 300 |
| SAT Puzzle (EN) | `sat_puzzle_en.py` | Easy, Medium, Hard | 300 |
| Sudoku (EN) | `sudoku_en.py` | Easy, Medium, Hard | - |
| Sudoku (KO) | `sudoku_ko.py` | Easy, Medium, Hard | - |
| Yacht Dice (EN) | `yacht_dice_en.py` | - | 100 |
| Yacht Dice (KO) | `yacht_dice_ko.py` | - | 100 |

## 🚀 Usage

### Individual Task Generation

```bash
# Run from project root
cd logical-puzzles

# Generate with default count
python generation/kinship.py

# Specify count
python generation/kinship.py --num 200

# Other task examples
python generation/cipher_en.py --num 100
python generation/logic_grid_en.py --num-samples 300
```

### Batch Generation

```bash
# Generate all tasks at once
bash scripts/gen_data.sh
```

**Note**: Some tasks in `gen_data.sh` may be commented out. Modify as needed.

## 📁 Output Format

Generated data is saved in two formats:

### 1. CSV Format (`data/csv/`)
- Simple format for evaluation
- Columns: `id`, `question`, `answer`, `difficulty`, `type`, etc.

### 2. JSONL Format (`data/jsonl/`)
- Used by the evaluation system
- Each line is a JSON object
- Includes additional metadata (e.g., `choices`, `solution`, etc.)

**Example:**
```json
{
  "id": "kinship_0",
  "question": "나의 아버지의 형의 아내는?",
  "answer": "A",
  "difficulty": "easy",
  "type": "kinship",
  "choices": ["큰어머니", "작은어머니", "고모", "이모", "할머니"]
}
```

**Note:** Difficulty levels are stored in lowercase (`easy`, `medium`, `hard`) for consistency with the evaluation system.

## 🔧 Script-Specific Options

### Common Options

Most scripts support the following options:

- `--num`: Number of problems to generate (by difficulty or total)
- `--num-samples`: Number of samples (some scripts)

### Special Options

Some scripts may support additional options. Check each script's `--help` option:

```bash
python generation/kinship.py --help
python generation/cipher_en.py --help
```

## 📊 Generation Statistics

After generation, each script outputs the following information:

- Total number of problems generated
- Distribution by difficulty
- Saved file paths

**Example Output:**
```
Generated 100 questions
Difficulty breakdown:
easy      34
medium    33
hard      33

CSV file created! -> data/csv/kinship.csv
JSONL file created! -> data/jsonl/kinship.jsonl
```

## ⚙️ Configuration and Customization

### Difficulty Adjustment

Each script can internally set the generation ratio by difficulty. Modify the difficulty settings within the script.

### Generation Count Adjustment

You can adjust the generation count for each task in the `scripts/gen_data.sh` file:

```bash
# Example: Change kinship problem count
python generation/kinship.py --num 200  # 100 → 200
```

## 🔍 Data Validation

Generated data can be checked at the following locations:

```bash
# Check CSV file
head data/csv/kinship.csv

# Check JSONL file
head data/jsonl/kinship.jsonl | python -m json.tool
```

## 📝 Notes

1. **Repeated Execution**: Running the same script multiple times will overwrite existing files.
2. **Generation Time**: Some complex tasks may take time to generate.
3. **Memory**: Check memory usage when generating large quantities.

## 📚 Additional Information

- Evaluation system usage: [evaluation.md](evaluation.md)
- Task-specific details: [puzzles/](puzzles/)
- Project structure: [../README.md](../README.md)

## 🔗 Related Files

- Batch generation script: `../scripts/gen_data.sh`
- Data storage location: `../data/`
- Evaluation data: `../eval_data/`
