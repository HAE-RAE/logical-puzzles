#!/usr/bin/env bash

echo "============================================"
echo "Logical Puzzles Dataset Generation"
echo "============================================"
echo ""

# Array Formula
echo ""; echo "[1/20] array_formula..."
python generation/array_formula.py --num 100

# Array Formula Korean
echo ""; echo "[2/20] array_formula_korean..."
python generation/array_formula_korean.py --num 100

# Causal DAG Korean
echo ""; echo "[3/20] causal_dag_korean..."
python generation/causal_dag_korean.py --num 300

# Causal DAG
echo ""; echo "[4/20] causal_dag..."
python generation/causal_dag.py --num 300

# Cipher Korean
echo ""; echo "[5/20] cipher_korean..."
python generation/cipher_korean.py --num 100

# Cipher
echo ""; echo "[6/20] cipher..."
python generation/cipher.py --num 100

# Cryptarithmetic
echo ""; echo "[7/20] cryptarithmetic..."
python generation/cryptarithmetic.py --num 400

# Ferryman
echo ""; echo "[8/20] ferryman..."
python generation/ferryman.py --num 100

# Ferryman Korean
echo ""; echo "[9/20] ferryman_korean..."
python generation/ferryman_korean.py --num 100

# Hanoi
echo ""; echo "[10/20] hanoi..."
python generation/hanoi.py --num 100

# Hanoi Korean
echo ""; echo "[11/20] hanoi_korean..."
python generation/hanoi_korean.py --num 100

# Inequality
echo ""; echo "[12/20] inequality..."
python generation/inequality.py --num 400

# Kinship
echo ""; echo "[13/20] kinship..."
python generation/kinship.py --num 100

# Kinship Vision
echo ""; echo "[14/20] kinship_vision..."
python generation/kinship_vision.py --num 100

# Logic Grid Korean
echo ""; echo "[15/20] logic_grid_korean..."
python generation/logic_grid_korean.py --num-samples 300

# Logic Grid
echo ""; echo "[16/20] logic_grid..."
python generation/logic_grid.py --num-samples 300

# Minesweeper
# echo ""; echo "[17/20] minesweeper..."
python generation/minesweeper.py

# Number Baseball
echo ""; echo "[18/20] number_baseball..."
python generation/number_baseball.py --num 400

# SAT Puzzle Korean
echo ""; echo "[19/20] sat_puzzle_korean..."
python generation/sat_puzzle_korean.py --num-samples 300

# SAT Puzzle
echo ""; echo "[20/20] sat_puzzle..."
python generation/sat_puzzle.py --num-samples 300

# Sudoku
# echo ""; echo "[21/20] sudoku..."
python generation/sudoku.py

# Yacht Dice
echo ""; echo "[22/20] yacht_dice..."
python generation/yacht_dice.py --num 100


# bash scripts/generate_all.sh