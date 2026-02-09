#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Evaluation Started${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# sudoku, minesweeper
TASKS=(
    "array_formula"
    "array_formula_korean"
    # "causal_dag_korean"
    # "causal_dag"
    # "cipher_korean"
    # "cipher"
    # "cryptarithmetic"
    # "ferryman"
    # "hanoi"
    # "inequality"
    # "kinship_vision"
    # "kinship"
    # "logic_grid_korean"
    # "logic_grid"
    # "number_baseball"
    # "sat_puzzles_korean"
    # "sat_puzzles"
    # "yacht_dice"
)

START_TIME=$(date +%s)

TOTAL_TASKS=${#TASKS[@]}
CURRENT_TASK=0
SUCCESS_COUNT=0
FAIL_COUNT=0

for task in "${TASKS[@]}"; do
    CURRENT_TASK=$((CURRENT_TASK + 1))
    
    echo -e "${YELLOW}[$CURRENT_TASK/$TOTAL_TASKS] Evaluating: $task${NC}"
    echo "----------------------------------------"
    
    set +e
    if python evaluation/run.py \
        --tasks "$task" \
        --async \
        --max-concurrent 30; then
        echo -e "${GREEN}✓ $task Completed${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo -e "${RED}✗ $task Failed${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    set -e
    
    echo ""
done

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Evaluation Completed${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total Tasks: ${TOTAL_TASKS}개"
echo -e "${GREEN}Success: ${SUCCESS_COUNT}개${NC}"
echo -e "${RED}Fail: ${FAIL_COUNT}개${NC}"

if [ $HOURS -gt 0 ]; then
    echo -e "Elapsed Time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
else
    echo -e "Elapsed Time: ${MINUTES}m ${SECONDS}s"
fi
echo -e "${BLUE}========================================${NC}"

if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
fi

exit 0

# bash scripts/evaluate_all.sh