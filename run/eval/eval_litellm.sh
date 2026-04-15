#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# export LITELLM_DEBUG=true

# ============ Gemini 설정 ============
MODEL="gemini/gemini-3-flash-preview"
GEN_KWARGS="temperature=1.0,max_tokens=65536,top_p=0.95,top_k=64,reasoning_effort=high"
# =====================================

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Evaluation Started (Gemini)${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Model: ${MODEL}"
echo -e "Mode: liteLLM"
echo -e "Gen kwargs: ${GEN_KWARGS}"
echo ""


TASKS=(
    # "array_formula_en"
    # "array_formula_ko"
    # "causal_dag_en"
    # "causal_dag_ko"
    # "cipher_en"
    # "cipher_ko"
    # "cryptarithmetic_en"
    # "cryptarithmetic_ko"
    # "ferryman_en"
    # "ferryman_ko"
    # "hanoi_en"
    # "hanoi_ko"
    # "inequality_en"
    # "inequality_ko"
    # "kinship"
    # "kinship_vision"
    # "logic_grid_en"
    # "logic_grid_ko"
    # "minesweeper_en"
    # "minesweeper_ko"
    # "number_baseball_en"
    # "number_baseball_ko"
    # "sat_puzzles_en"
    # "sat_puzzles_ko"
    # "sudoku_en"
    # "sudoku_ko"
    # "yacht_dice_en"
    # "yacht_dice_ko"
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
        --model "$MODEL" \
        --model_router litellm \
        --gen-kwargs "$GEN_KWARGS" \
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
echo -e "${BLUE}Evaluation Completed (Gemini)${NC}"
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

# bash scripts/eval_litellm.sh
