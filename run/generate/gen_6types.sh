#!/bin/bash
set -e

NUM=100

echo "============================================"
echo "6-Type Puzzle Generation (num=$NUM)"
echo "============================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate

echo ""; echo "[1/12] cryptarithmetic_en..."
python generation/cryptarithmetic_en.py --num $NUM

echo ""; echo "[2/12] cryptarithmetic_ko..."
python generation/cryptarithmetic_ko.py --num $NUM

echo ""; echo "[3/12] inequality_en..."
python generation/inequality_en.py --num $NUM

echo ""; echo "[4/12] inequality_ko..."
python generation/inequality_ko.py --num $NUM

echo ""; echo "[5/12] minesweeper_en..."
python generation/minesweeper_en.py --num $NUM

echo ""; echo "[6/12] minesweeper_ko..."
python generation/minesweeper_ko.py --num $NUM

echo ""; echo "[7/12] number_baseball_en..."
python generation/number_baseball_en.py --num $NUM

echo ""; echo "[8/12] number_baseball_ko..."
python generation/number_baseball_ko.py --num $NUM

echo ""; echo "[9/12] sudoku_en..."
python generation/sudoku_en.py --num $NUM

echo ""; echo "[10/12] sudoku_ko..."
python generation/sudoku_ko.py --num $NUM

echo ""; echo "[11/12] yacht_dice_en..."
python generation/yacht_dice.py --num $NUM

echo ""; echo "[12/12] yacht_dice_ko..."
python generation/yacht_dice_ko.py --num $NUM

echo ""
echo "============================================"
echo "Generation Complete!"
echo "============================================"
ls -la data/json/
