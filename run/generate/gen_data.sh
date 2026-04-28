#!/usr/bin/env bash

echo "============================================"
echo "Logical Puzzles Dataset Generation"
echo "============================================"
echo ""

# Array Formula (EN)
echo ""; echo "[1/28] array_formula_en..."
python generation/array_formula_en.py --num 100

# Array Formula (KO)
echo ""; echo "[2/28] array_formula_ko..."
python generation/array_formula_ko.py --num 100

# Causal DAG (EN)
echo ""; echo "[3/28] causal_dag_en..."
python generation/causal_dag_en.py --num 300

# Causal DAG (KO)
echo ""; echo "[4/28] causal_dag_ko..."
python generation/causal_dag_ko.py --num 300

# Cipher (EN)
echo ""; echo "[5/28] cipher_en..."
python generation/cipher_en.py --num 100

# Cipher (KO)
echo ""; echo "[6/28] cipher_ko..."
python generation/cipher_ko.py --num 100

# Cryptarithmetic (EN)
echo ""; echo "[7/28] cryptarithmetic_en..."
python generation/cryptarithmetic_en.py --num 300

# Cryptarithmetic (KO)
echo ""; echo "[8/28] cryptarithmetic_ko..."
python generation/cryptarithmetic_ko.py --num 300

# Ferryman (EN)
echo ""; echo "[9/28] ferryman_en..."
python generation/ferryman_en.py --num 100

# Ferryman (KO)
echo ""; echo "[10/28] ferryman_ko..."
python generation/ferryman_ko.py --num 100

# Hanoi (EN)
echo ""; echo "[11/28] hanoi_en..."
python generation/hanoi_en.py --num 100

# Hanoi (KO)
echo ""; echo "[12/28] hanoi_ko..."
python generation/hanoi_ko.py --num 100

# Inequality (EN)
echo ""; echo "[13/28] inequality_en..."
python generation/inequality_en.py --num 300

# Inequality (KO)
echo ""; echo "[14/28] inequality_ko..."
python generation/inequality_ko.py --num 300

# Kinship
echo ""; echo "[15/28] kinship..."
python generation/kinship.py --num 100

# Kinship Vision
echo ""; echo "[16/28] kinship_vision..."
python generation/kinship_vision.py --num 100

# Logic Grid (EN)
echo ""; echo "[17/28] logic_grid_en..."
python generation/logic_grid_en.py --num-samples 300

# Logic Grid (KO)
echo ""; echo "[18/28] logic_grid_ko..."
python generation/logic_grid_ko.py --num-samples 300

# Minesweeper (EN)
echo ""; echo "[19/28] minesweeper_en..."
python generation/minesweeper_en.py --num 300

# Minesweeper (KO)
echo ""; echo "[20/28] minesweeper_ko..."
python generation/minesweeper_ko.py --num 300

# Number Baseball (EN)
echo ""; echo "[21/28] number_baseball_en..."
python generation/number_baseball_en.py --num 300

# Number Baseball (KO)
echo ""; echo "[22/28] number_baseball_ko..."
python generation/number_baseball_ko.py --num 300

# SAT Puzzle (EN)
echo ""; echo "[23/28] sat_puzzle_en..."
python generation/sat_puzzle_en.py --num-samples 300

# SAT Puzzle (KO)
echo ""; echo "[24/28] sat_puzzle_ko..."
python generation/sat_puzzle_ko.py --num-samples 300

# Sudoku (EN)
echo ""; echo "[25/28] sudoku_en..."
python generation/sudoku_en.py --num 300

# Sudoku (KO)
echo ""; echo "[26/28] sudoku_ko..."
python generation/sudoku_ko.py --num 300

# Yacht Dice (EN)
echo ""; echo "[27/28] yacht_dice_en..."
python generation/yacht_dice_en.py --num 100

# Yacht Dice (KO)
echo ""; echo "[28/28] yacht_dice_ko..."
python generation/yacht_dice_ko.py --num 100


# bash scripts/gen_data.sh
