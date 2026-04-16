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

# ============ Qwen 설정 (Remote 서버) ============
MODEL="Qwen/Qwen3-0.6B"
# MODEL="Qwen/Qwen3-1.7B"
REMOTE_URL="https://tremendously-bureaucratic-alda.ngrok-free.dev"
GEN_KWARGS="temperature=0.6,max_tokens=16384,top_p=0.95,top_k=20,reasoning=on,stream=on"
# =================================================

MODEL_DIR_NAME="${MODEL//\//_}"
LOG_DIR="$PROJECT_ROOT/results/$MODEL_DIR_NAME/log"
DATA_JSONL_DIR="$PROJECT_ROOT/data/jsonl"
mkdir -p "$LOG_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Evaluation Started (Qwen)${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Model: ${MODEL}"
echo -e "Remote URL: ${REMOTE_URL}"
echo -e "Gen kwargs: ${GEN_KWARGS}"
echo -e "Log saved location: ${LOG_DIR}"
echo ""


TASKS=(
    # "array_formula_en_easy"
    # "array_formula_en_hard"
    # "array_formula_en_medium"
    # "array_formula_ko_easy"
    # "array_formula_ko_hard"
    # "array_formula_ko_medium"
    # "causal_dag_en_easy"
    # "causal_dag_en_hard"
    # "causal_dag_en_medium"
    # "causal_dag_ko_easy"
    # "causal_dag_ko_hard"
    # "causal_dag_ko_medium"
    # "cipher_en_easy"
    # "cipher_en_hard"
    # "cipher_en_medium"
    # "cipher_ko_easy"
    # "cipher_ko_hard"
    # "cipher_ko_medium"
    # "ferryman_en_easy"
    # "ferryman_en_hard"
    # "ferryman_en_medium"
    # "ferryman_ko_easy"
    # "ferryman_ko_hard"
    # "ferryman_ko_medium"
    # "inequality_en_easy"
    # "inequality_en_hard"
    # "inequality_en_medium"
    # "logic_grid_en_easy"
    # "logic_grid_en_hard"
    # "logic_grid_en_medium"
    # "logic_grid_ko_easy"
    # "logic_grid_ko_hard"
    # "logic_grid_ko_medium"
    # "minesweeper_en_easy"
    # "minesweeper_en_hard"
    # "minesweeper_en_medium"
    # "sat_puzzles_en_easy"
    # "sat_puzzles_en_hard"
    # "sat_puzzles_en_medium"
    # "sat_puzzles_ko_easy"
    # "sat_puzzles_ko_hard"
    # "sat_puzzles_ko_medium"
    # "sudoku_en_easy"
    # "sudoku_en_hard"
    # "sudoku_en_medium"
)

START_TIME=$(date +%s)

TOTAL_TASKS=${#TASKS[@]}
CURRENT_TASK=0
SUCCESS_COUNT=0
FAIL_COUNT=0

for task in "${TASKS[@]}"; do
    CURRENT_TASK=$((CURRENT_TASK + 1))
    log_file="$LOG_DIR/${task}.log"

    echo "========================================" >> "$log_file"
    echo "Task: $task" >> "$log_file"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')" >> "$log_file"
    echo "========================================" >> "$log_file"
    echo "" >> "$log_file"

    echo -e "${YELLOW}[$CURRENT_TASK/$TOTAL_TASKS] Evaluating: $task${NC}"
    echo -e "  Log: ${log_file}"
    echo "----------------------------------------"

    set +e
    if python evaluation/run.py \
        --model "$MODEL" \
        --model_router remote \
        --remote_url "$REMOTE_URL" \
        --gen-kwargs "$GEN_KWARGS" \
        --data-dir "$DATA_JSONL_DIR" \
        --tasks "$task" \
        --async \
        --max-concurrent 5 2>&1 | tee -a "$log_file"; then
        echo -e "${GREEN}✓ $task Completed${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "" >> "$log_file"
        echo "========================================" >> "$log_file"
        echo "Status: SUCCESS" >> "$log_file"
        echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')" >> "$log_file"
        echo "========================================" >> "$log_file"
    else
        echo -e "${RED}✗ $task Failed${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "" >> "$log_file"
        echo "========================================" >> "$log_file"
        echo "Status: FAILED" >> "$log_file"
        echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')" >> "$log_file"
        echo "========================================" >> "$log_file"
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
echo -e "${BLUE}Evaluation Completed (Qwen)${NC}"
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

# bash run/eval/eval_remote.sh
