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

# ============ Qwen 설정 (Remote 서버) ============
# MODEL="Qwen/Qwen3-0.6B"
MODEL="Qwen/Qwen3-1.7B"
REMOTE_URL="https://tremendously-bureaucratic-alda.ngrok-free.dev"
GEN_KWARGS="temperature=0.6,max_tokens=16384,top_p=0.95,top_k=20,reasoning=on"
# =================================================

MODEL_DIR_NAME="${MODEL//\//_}"
LOG_DIR="$PROJECT_ROOT/results/$MODEL_DIR_NAME/log"
mkdir -p "$LOG_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}All Tasks Evaluation Started (Qwen)${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Model: ${MODEL}"
echo -e "Remote URL: ${REMOTE_URL}"
echo -e "Gen kwargs: ${GEN_KWARGS}"
echo -e "Log saved location: ${LOG_DIR}"
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
MAX_PARALLEL=2

run_task() {
    local task=$1
    local task_num=$2
    local log_file="$LOG_DIR/${task}.log"
    
    echo "========================================" >> "$log_file"
    echo "Task: $task" >> "$log_file"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')" >> "$log_file"
    echo "========================================" >> "$log_file"
    echo "" >> "$log_file"
    
    echo -e "${YELLOW}[$task_num/$TOTAL_TASKS] Evaluating: $task${NC}"
    echo -e "  Log: ${log_file}"
    echo "----------------------------------------"
    
    set +e
    if python evaluation/run.py \
        --model "$MODEL" \
        --model_router remote \
        --remote_url "$REMOTE_URL" \
        --gen-kwargs "$GEN_KWARGS" \
        --tasks "$task" \
        --async \
        --max-concurrent 5 2>&1 | tee -a "$log_file"; then
        echo -e "${GREEN}✓ $task Completed${NC}"
        echo "$task:SUCCESS" >> /tmp/eval_results_$$
        echo "" >> "$log_file"
        echo "========================================" >> "$log_file"
        echo "Status: SUCCESS" >> "$log_file"
        echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')" >> "$log_file"
        echo "========================================" >> "$log_file"
    else
        echo -e "${RED}✗ $task Failed${NC}"
        echo "$task:FAIL" >> /tmp/eval_results_$$
        echo "" >> "$log_file"
        echo "========================================" >> "$log_file"
        echo "Status: FAILED" >> "$log_file"
        echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')" >> "$log_file"
        echo "========================================" >> "$log_file"
    fi
    set -e
    echo ""
}

rm -f /tmp/eval_results_$$

for task in "${TASKS[@]}"; do
    CURRENT_TASK=$((CURRENT_TASK + 1))
    
    run_task "$task" "$CURRENT_TASK" &
    
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 1
    done
done

wait

if [ -f /tmp/eval_results_$$ ]; then
    while IFS=: read -r task result; do
        if [ "$result" = "SUCCESS" ]; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done < /tmp/eval_results_$$
    rm -f /tmp/eval_results_$$
fi

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

# bash scripts/eval_remote_parallel.sh
