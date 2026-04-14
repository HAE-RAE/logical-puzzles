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

# ============ 설정 ============
GPU_ID=3
VLLM_PORT=8000
VLLM_HOST="0.0.0.0"
REMOTE_URL="http://localhost:${VLLM_PORT}"
GEN_KWARGS="temperature=0.6,max_tokens=16384,top_p=0.95,top_k=20,reasoning=on"

# 평가할 모델 목록 (순서대로 실행)
MODELS=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
)

# 평가할 태스크 목록
TASKS=(
    "cryptarithmetic"
    "cryptarithmetic_ko"
    "inequality"
    "inequality_ko"
    "minesweeper"
    "minesweeper_ko"
    "number_baseball"
    "number_baseball_ko"
    "sudoku"
    "sudoku_ko"
    "yacht_dice"
    "yacht_dice_ko"
)
# ==============================

VENV_ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo -e "${RED}Error: .venv not found at $PROJECT_ROOT/.venv${NC}"
    exit 1
fi
source "$VENV_ACTIVATE"

start_vllm_server() {
    local model=$1
    echo -e "${BLUE}Starting vLLM server: ${model} on GPU ${GPU_ID}...${NC}"

    CUDA_VISIBLE_DEVICES=$GPU_ID python -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --host "$VLLM_HOST" \
        --port "$VLLM_PORT" \
        --trust-remote-code 2>&1 &
    VLLM_PID=$!

    echo -e "  vLLM PID: ${VLLM_PID}"

    # 서버 준비 대기
    local max_wait=300
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo -e "${GREEN}  vLLM server ready (${waited}s)${NC}"
            return 0
        fi
        # 프로세스가 죽었는지 확인
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo -e "${RED}  vLLM server process died${NC}"
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
    done

    echo -e "${RED}  vLLM server failed to start within ${max_wait}s${NC}"
    kill $VLLM_PID 2>/dev/null
    return 1
}

stop_vllm_server() {
    if [ -n "$VLLM_PID" ] && kill -0 $VLLM_PID 2>/dev/null; then
        echo -e "${YELLOW}Stopping vLLM server (PID: ${VLLM_PID})...${NC}"
        kill $VLLM_PID 2>/dev/null
        wait $VLLM_PID 2>/dev/null || true
        echo -e "${GREEN}  vLLM server stopped${NC}"
    fi
    VLLM_PID=""
}

# Ctrl+C 시 vLLM 서버 정리
trap stop_vllm_server EXIT INT TERM

OVERALL_START=$(date +%s)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Local vLLM Evaluation${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "GPU: ${GPU_ID}"
echo -e "Models: ${MODELS[*]}"
echo -e "Tasks: ${TASKS[*]}"
echo -e "Gen kwargs: ${GEN_KWARGS}"
echo ""

for MODEL in "${MODELS[@]}"; do
    MODEL_DIR_NAME="${MODEL//\//_}"
    LOG_DIR="$PROJECT_ROOT/results/$MODEL_DIR_NAME/log"
    mkdir -p "$LOG_DIR"

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Model: ${MODEL}${NC}"
    echo -e "${BLUE}========================================${NC}"

    # vLLM 서버 시작
    start_vllm_server "$MODEL"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to start vLLM for ${MODEL}. Skipping...${NC}"
        continue
    fi

    MODEL_START=$(date +%s)
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

        echo -e "${YELLOW}[$CURRENT_TASK/$TOTAL_TASKS] Evaluating: $task${NC}"
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
            echo -e "${GREEN}  $task Completed${NC}"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            echo "Status: SUCCESS" >> "$log_file"
        else
            echo -e "${RED}  $task Failed${NC}"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            echo "Status: FAILED" >> "$log_file"
        fi
        set -e

        echo ""
    done

    MODEL_END=$(date +%s)
    MODEL_ELAPSED=$((MODEL_END - MODEL_START))

    echo -e "${BLUE}--- ${MODEL} Summary ---${NC}"
    echo -e "  ${GREEN}Success: ${SUCCESS_COUNT}${NC} / ${RED}Fail: ${FAIL_COUNT}${NC} (Total: ${TOTAL_TASKS})"
    echo -e "  Elapsed: $((MODEL_ELAPSED / 60))m $((MODEL_ELAPSED % 60))s"
    echo ""

    # 다음 모델을 위해 서버 종료
    stop_vllm_server
done

OVERALL_END=$(date +%s)
OVERALL_ELAPSED=$((OVERALL_END - OVERALL_START))

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}All Evaluations Completed${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total elapsed: $((OVERALL_ELAPSED / 60))m $((OVERALL_ELAPSED % 60))s"

# bash scripts/eval_local_vllm.sh
