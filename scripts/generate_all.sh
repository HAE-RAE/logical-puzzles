#!/usr/bin/env bash

echo "============================================"
echo "Logical Puzzles Dataset Generation"
echo "============================================"
echo ""

# Array Formula (EN)
echo ""; echo "[1/20] array_formula_en..."
python generation/array_formula_en.py --num 100

# Array Formula (KO)
echo ""; echo "[2/20] array_formula_ko..."
python generation/array_formula_ko.py --num 100

# Causal DAG (KO)
echo ""; echo "[3/20] causal_dag_ko..."
python generation/causal_dag_ko.py --num 300

# Causal DAG (EN)
echo ""; echo "[4/20] causal_dag_en..."
python generation/causal_dag_en.py --num 300

# Cipher (KO)
echo ""; echo "[5/20] cipher_ko..."
python generation/cipher_ko.py --num 100

# Cipher (EN)
echo ""; echo "[6/20] cipher_en..."
python generation/cipher_en.py --num 100

# Cryptarithmetic
echo ""; echo "[7/20] cryptarithmetic..."
python generation/cryptarithmetic.py --num 400

# Ferryman (EN)
echo ""; echo "[8/20] ferryman_en..."
python generation/ferryman_en.py --num 100

# Ferryman (KO)
echo ""; echo "[9/20] ferryman_ko..."
python generation/ferryman_ko.py --num 100

# Hanoi (EN)
echo ""; echo "[10/20] hanoi_en..."
python generation/hanoi_en.py --num 100

# Hanoi (KO)
echo ""; echo "[11/20] hanoi_ko..."
python generation/hanoi_ko.py --num 100

# Inequality
echo ""; echo "[12/20] inequality..."
python generation/inequality.py --num 400

# Kinship
echo ""; echo "[13/20] kinship..."
python generation/kinship.py --num 100

# Kinship Vision
echo ""; echo "[14/20] kinship_vision..."
python generation/kinship_vision.py --num 100

# Logic Grid (KO)
echo ""; echo "[15/20] logic_grid_ko..."
python generation/logic_grid_ko.py --num-samples 300

# Logic Grid (EN)
echo ""; echo "[16/20] logic_grid_en..."
python generation/logic_grid_en.py --num-samples 300

# Minesweeper
# echo ""; echo "[17/20] minesweeper..."
python generation/minesweeper.py

# Number Baseball
echo ""; echo "[18/20] number_baseball..."
python generation/number_baseball.py --num 400

# SAT Puzzle (KO)
echo ""; echo "[19/20] sat_puzzle_ko..."
python generation/sat_puzzle_ko.py --num-samples 300

# SAT Puzzle (EN)
echo ""; echo "[20/20] sat_puzzle_en..."
python generation/sat_puzzle_en.py --num-samples 300

# Sudoku
# echo ""; echo "[21/20] sudoku..."
python generation/sudoku.py

# Yacht Dice
echo ""; echo "[22/20] yacht_dice..."
python generation/yacht_dice.py --num 100


# bash scripts/generate_all.sh