#!/usr/bin/env bash

echo "============================================"
echo "Logical Puzzles Dataset Generation"
echo "============================================"
echo ""

# Array Formula (Easy, Medium, Hard)
echo ""; echo "[1/19] array_formula..."
python generation/array_formula.py --num 25

# Causal DAG Korean (Easy, Medium, Hard)
echo ""; echo "[2/19] causal_dag_korean..."
python generation/causal_dag_korean.py --num 300

# Causal DAG (Easy, Medium, Hard)
echo ""; echo "[3/19] causal_dag..."
python generation/causal_dag.py --num 300

# Cipher Korean (Easy, Medium, Hard, Very Hard, Extreme)
echo ""; echo "[4/19] cipher_korean..."
python generation/cipher_korean.py --num 100      

# Cipher (Easy, Medium, Hard, Expert)
echo ""; echo "[5/19] cipher..."
python generation/cipher.py --num 100

# Ferryman (Easy, Medium, Hard)
echo ""; echo "[6/19] ferryman..."
python generation/ferryman.py --num 100

# Hanoi (No difficulty)
echo ""; echo "[7/19] hanoi..."
python generation/hanoi.py --num 100

# Inequality (Easy, Medium, Hard, Expert)
echo ""; echo "[8/19] inequality..."
python generation/inequality.py --num 400

# Kinship (Easy, Medium, Hard)
echo ""; echo "[9/19] kinship..."
python generation/kinship.py --num 100

# Kinship Vision  (Easy, Medium, Hard)
echo ""; echo "[10/19] kinship_vision..."
python generation/kinship_vision.py --num 100

# Logic Grid Korean (Easy, Medium, Hard)
echo ""; echo "[11/19] logic_grid_korean..."
python generation/logic_grid_korean.py --num-samples 300

# Logic Grid (Easy, Medium, Hard)
echo ""; echo "[12/19] logic_grid..."
python generation/logic_grid.py --num-samples 300

# Minesweeper (Easy, Medium, Hard)
# echo ""; echo "[13/19] minesweeper..."
# python generation/minesweeper.py

# Number Baseball (Easy, Medium, Hard, Expert)
echo ""; echo "[14/19] number_baseball..."
python generation/number_baseball.py --num 400

# SAT Puzzle Korean (Easy, Medium, Hard)
echo ""; echo "[15/19] sat_puzzle_korean..."
python generation/sat_puzzle_korean.py --num-samples 300

# SAT Puzzle (Easy, Medium, Hard)
echo ""; echo "[16/19] sat_puzzle..."
python generation/sat_puzzle.py --num-samples 300

# Sudoku (Medium, Hard, Expert, Extreme)
# echo ""; echo "[17/19] sudoku..."
# python generation/sudoku.py

# Yacht Dice (No difficulty)
echo ""; echo "[18/19] yacht_dice..."
python generation/yacht_dice.py --num 100

# Cryptarithmetic (Easy, Medium, Hard, Expert)
echo ""; echo "[19/19] cryptarithmetic..."
python generation/cryptarithmetic.py --num 400


# bash scripts/generate_all.sh