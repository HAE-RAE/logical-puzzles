# Logical Puzzles
Logical Puzzle Evaluation Dataset and LLM Assessment Pipeline


## Puzzle Types

### 1. Array Formula
Apply formulas to 2D arrays to calculate final results. Sequentially apply row/column operations and aggregate functions to derive answers.

- **Multi-step Operations**: Apply functions like SUM, MEAN, MAX, MIN sequentially
- **Row/Column Aggregation**: Perform aggregate functions on row or column units
- **Complex Reasoning**: Track intermediate calculation results to derive final values
- **Difficulty Levels**: Adjusted based on array size and number of operation steps

### 2. Causal DAG Korean
Infer event propagation time in causal relationship graphs with time delays. Events and causal relationships are described in Korean.

- **DAG-based Generation**: Represent causal relationships as directed acyclic graphs
- **Shortest Path Algorithm**: Calculate answers using Dijkstra's algorithm
- **Unique Solution**: Automatically guaranteed by deterministic graph structure
- **Difficulty Levels**: Based on number of events (4-12) and connection density (Easy/Medium/Hard)
- **Realistic Scenarios**: Real events from technology, business, environment, operations domains

### 3. Causal DAG
Same as Causal DAG Korean, but events and causal relationships are described in English.

### 4. Cipher Korean
Decode multi-layer cipher algorithms. Evaluates LLM's pure algorithmic reasoning ability. Encrypts Korean strings.

- **Multi-layer Algorithms**: Stack of Substitution, Vigenere, Reverse, Playfair, Transposition
- **Meaningless Answers**: Use random strings to prevent linguistic guessing
- **Variable Hints**: Adjust number of examples by difficulty (2-20 examples)
- **Difficulty Levels**: EASY (Sub+Rev), MEDIUM (Vig), HARD (Vig+Rev), VERY_HARD (Playfair+Vig), EXTREME (Play+Trans+Vig)

### 5. Cipher
Same as Cipher Korean, but encrypts English strings.

### 6. Cryptarithmetic
Find the digit corresponding to each letter in arithmetic expressions represented by letters.

- **Constraints**: No leading zeros, each letter is a unique digit
- **Backtracking Solver**: Verify unique solution
- **Various Operations**: Support addition, subtraction, multiplication
- **Difficulty Levels**: Adjusted based on number of letters and digits

### 7. Ferryman
Calculate total time required for a ferryman to transport goods considering various navigation rules and constraints.

- **Speed-limited Zones**: Speed changes in Zone A and Zone B
- **Cargo Weight**: Speed reduction based on weight
- **Mandatory Rest Time**: Compliance with navigation regulations
- **Complex Condition Reasoning**: Consider multiple constraints simultaneously

### 8. Hanoi
Classic puzzle of moving disks from one peg to another. Calculate the minimum number of moves.

- **Recursive Structure**: Requires divide-and-conquer thinking
- **Constraints**: Only smaller disks can be placed on larger disks
- **Minimum Moves**: 2^n - 1 moves (n is number of disks)
- **Difficulty Levels**: Adjusted based on number of disks

### 9. Inequality
Find number placements that satisfy given inequality constraints.

- **Constraint Satisfaction Problem**: Satisfy multiple inequalities simultaneously
- **Backtracking Solver**: Verify unique solution
- **Logical Reasoning**: Narrow value ranges from constraints
- **Difficulty Levels**: Adjusted based on number of variables and constraints

### 10. Kinship
Infer family relationship clues presented through dialogue to derive final Korean kinship terms.

- **Korean Kinship Terms**: Support various family relationships (paternal/maternal/in-laws)
- **Dialogue-based Clues**: Provide relationship information in natural conversation format
- **Multiple Answer Support**: Handle synonyms like "큰아버지, 백부"
- **Relationship Chain Reasoning**: Derive final kinship term through stepwise relationship connections
- **Language+Logic Fusion**: Simultaneously evaluate Korean honorific system understanding and logical reasoning

### 11. Kinship Vision
Multimodal problem combining family photo images and dialogue clues to identify people by visual features and infer relationships.

- **Vision+Language Fusion**: Simultaneously perform person identification in images and relationship reasoning
- **17-person Actor DB**: Consider gender (male/female) × age group (SENIOR/ADULT/YOUNG_ADULT/CHILD)
- **Visual Feature Mapping**: Distinguish people using 3 features: clothing color, position, appearance
- **Multiple Choice Format**: 1 correct answer + 3 distractors (inducing visual confusion)
- **Multimodal Reasoning**: Understand dialogue context → Search image → Connect relationship chain

### 12. Logic Grid Korean
Constraint-based logical reasoning problem famous as Einstein's Riddle. Infer relationships among multiple people and attributes from Korean constraint conditions.

- **CSP (Constraint Satisfaction Problem)**: Verify unique solution with backtracking
- **Natural Language Constraints**: Simultaneously require linguistic understanding and logical reasoning
- **Unique Solution Guaranteed**: Verified with Constraint Propagation algorithm
- **Difficulty Levels**: Easy (3×3), Medium (4×4), Hard (5×5)
- **Backward Generation**: Generate valid solution first, then derive constraint conditions

### 13. Logic Grid
Same as Logic Grid Korean, but names, attributes, and constraints are expressed in English.

### 14. Minesweeper
Minesweeper puzzle designed as a Constraint Satisfaction Problem (CSP). Evaluates LLM's logical reasoning ability.

- **Unique Solution Guaranteed**: Verify unique solution with backtracking solver
- **Minimal Hints**: Minimize hints while maintaining unique solution
- **Difficulty Levels**: Easy (6×6), Medium (8×8), Hard (10×10)
- **Coordinate-based Evaluation**: Output mine locations in (r,c) format
- **Partial Scores**: Exact Match, Precision, Recall, F1 Score

### 15. Number Baseball
Infer hidden N-digit numbers through hints (Strike/Ball).

- **Constraint Reasoning**: Narrow possible number ranges with Strike/Ball hints
- **Variable Digits**: Adjust difficulty with 3-digit, 4-digit, 5-digit numbers
- **No Duplicate Digits**: Each position has a different digit
- **Stepwise Reasoning**: Derive answer by combining multiple hints

### 16. SAT Puzzle Korean
NP-complete problem of finding value combinations that satisfy given logical expressions (CNF) for Boolean variables. Problem scenarios and constraints are described in Korean.

- **NP-Complete Problem**: Theoretically difficult problem
- **CNF (Conjunctive Normal Form)**: Express logical expressions in standard form
- **Natural Language Translation**: Represent as real situations like crime, meetings, task assignments
- **Unique Solution Guaranteed**: Backward generation ensures answer satisfies clauses
- **Difficulty Levels**: Easy (3-4 variables), Medium (5-7 variables), Hard (10-12 variables)
- **Pure Logical Reasoning**: Focus on Boolean logic

### 17. SAT Puzzle
Same as SAT Puzzle Korean, but variable names and constraints are expressed in English.

### 18. Sudoku
9×9 Sudoku puzzle generation and difficulty-level evaluation dataset.

- **Unique Solution Guaranteed**: All puzzles have exactly one solution
- **Difficulty Evaluation**: Automatic classification into Easy, Medium, Hard, Expert, Extreme
- **Spot-check Evaluation**: HMAC-based K-cell selection for LLM evaluation support
- **Symmetry Support**: Improve aesthetic quality with rotation/reflection symmetry
- **Reproducible**: Regenerate identical puzzles with fixed seeds

### 19. Yacht Dice
Combinatorial optimization problem of optimally assigning 12 dice results to 12 categories to maximize total score.

- **Combinatorial Optimization**: 12! = 479,001,600 possible assignments
- **Hungarian Algorithm**: Calculate optimal solution
- **Bonus Score Calculation**: Bonus based on upper section total
- **Various Rule Modifications**: Support changes to bonus, scores, optimization objectives
- **Complex Scoring Rules**: Evaluate LLM's rule understanding and optimization ability


## Usage

### Puzzle Generation
```bash
# Generate all puzzles
bash scripts/generate_all.sh
```

### Evaluation

```bash
# Ferryman evaluation
cd evaluation
python eval_ferryman.py

# Yacht Dice evaluation (default rules)
python eval_yacht_dice.py --model gpt-4o

# Yacht Dice evaluation (custom rules)
python eval_yacht_dice.py \
  --model gpt-4o \
  --bonus-threshold 70 \
  --bonus-points 50 \
  --yacht-points 100 \
  --recalculate

# Sudoku evaluation
python eval_sudoku.py --model gpt-4o
python eval_sudoku.py --model gpt-4o-mini
python eval_sudoku.py --model o1

# Minesweeper evaluation
python eval_minesweeper.py --model gpt-4o
python eval_minesweeper.py --model o1
```

## Data Format
All puzzles are stored in two formats:

- **CSV**: `data/csv/{puzzle_name}.csv` - Easy to view in spreadsheets
- **JSONL**: `data/json/{puzzle_name}.jsonl` - Easy to process programmatically

**Common Fields:**
- `id`: Unique identifier
- `question`: Problem description
- `answer`: Correct answer
- `solution`: Step-by-step reasoning process
- `difficulty`: Difficulty level (Easy/Medium/Hard, etc.)
- Additional puzzle-specific metadata (optional)

## Project Structure
```
logical-puzzles/
├── data/                       # Generated datasets (local only, gitignored)
│   ├── csv/                    # CSV format
│   └── json/                   # JSONL format
├── description/                # Puzzle documentation
│   ├── array_formula.md
│   └── YACHT_DICE_USAGE.md
├── evaluation/                 # Evaluation scripts
│   ├── eval_array_formula.py
│   ├── eval_causal_dag.py
│   ├── eval_cipher.py
│   ├── eval_cryptarithmetic.py
│   └── ... (19 evaluation scripts)
├── evaluation_data/            # Static evaluation data
│   ├── kinshop_vision/
│   │   └── kinship.jpg         # Family photo (17 people)
│   └── minesweeper/
│       ├── eval_metadata.jsonl
│       ├── eval_puzzles.jsonl
│       ├── eval_solutions.jsonl
│       └── solution.md
├── guess/                      # Puzzle generators
│   ├── array_formula.py
│   ├── causal_dag.py
│   ├── cipher.py
│   ├── cryptarithmetic.py
│   └── ... (19 puzzle types)
├── scripts/                    # Automation scripts
│   └── generate_all.sh
├── validators/                 # Data validation tools
│   ├── verify_logic_grid.py
│   └── verify_sat.py
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

## Adding New Puzzles
When adding a new puzzle, please follow this structure:

### Required Files
```
guess/{puzzle_name}.py           # Puzzle generation logic
evaluation/eval_{puzzle_name}.py # Evaluation script
```

### Recommendations
- **Difficulty Levels**: Minimum 3 levels (Easy/Medium/Hard)
- **Validation Tools**: Consider adding validation scripts to `validators/` folder

## Notes
- The `data/` directory is included in `.gitignore` and will not be uploaded to the repository
- Evaluation results are stored locally only
- Do not commit API keys or sensitive information
- Generated data is automatically saved to `data/csv/` and `data/json/`

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.