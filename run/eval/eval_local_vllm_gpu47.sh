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
# 사용할 GPU 목록 (쉼표로 구분). 예: "0,1,2,3" → 4-way 텐서 병렬
GPUS="4,5,6,7"
# 텐서 병렬 크기 (비우면 GPUS 개수로 자동 설정). 모든 모델에 동일 적용.
TENSOR_PARALLEL_SIZE=""
# GPU 메모리 사용률 (0~1). 대형 모델 OOM 시 조정.
GPU_MEM_UTIL=0.90
VLLM_PORT=8001
VLLM_HOST="0.0.0.0"
REMOTE_URL="http://localhost:${VLLM_PORT}"
GEN_KWARGS="temperature=0.6,max_tokens=16384,top_p=0.95,top_k=20,reasoning=on"
# 재개(resume): 이미 결과(json)가 있는 (모델,task)는 건너뜀. 중단 후 재실행하면 이어서 진행.
SKIP_EXISTING=true
# 서버 기동 대기 최대 시간(초). 대형 모델 첫 다운로드(예: 100B+ 200GB)까지 커버.
SERVER_START_TIMEOUT=7200

# 텐서 병렬 크기 자동 계산 (GPUS 개수)
if [ -z "$TENSOR_PARALLEL_SIZE" ]; then
    TENSOR_PARALLEL_SIZE=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
fi

# 평가할 모델 목록 - 그룹 B (GPU 4-7). 그룹 A는 eval_local_vllm.sh
MODELS=(
    "LGAI-EXAONE/EXAONE-4.0-32B"   # 32B dense (텍스트 전용, 4.5는 vision이라 제외)
    "openai/gpt-oss-120b"          # 117B-A5B MoE (MXFP4)
)

# 평가할 태스크 목록
TASKS=(
    "array_formula_en_easy"
    "array_formula_en_hard"
    "array_formula_en_medium"
    "array_formula_ko_easy"
    "array_formula_ko_hard"
    "array_formula_ko_medium"
    "causal_dag_en_easy"
    "causal_dag_en_hard"
    "causal_dag_en_medium"
    "causal_dag_ko_easy"
    "causal_dag_ko_hard"
    "causal_dag_ko_medium"
    "cipher_en_easy"
    "cipher_en_hard"
    "cipher_en_medium"
    "cipher_ko_easy"
    "cipher_ko_hard"
    "cipher_ko_medium"
    "cryptarithmetic_en_easy"
    "cryptarithmetic_en_hard"
    "cryptarithmetic_en_medium"
    "cryptarithmetic_ko_easy"
    "cryptarithmetic_ko_hard"
    "cryptarithmetic_ko_medium"
    "ferryman_en_easy"
    "ferryman_en_hard"
    "ferryman_en_medium"
    "ferryman_ko_easy"
    "ferryman_ko_hard"
    "ferryman_ko_medium"
    "hanoi_en_easy"
    "hanoi_en_hard"
    "hanoi_en_medium"
    "hanoi_ko_easy"
    "hanoi_ko_hard"
    "hanoi_ko_medium"
    "inequality_en_easy"
    "inequality_en_hard"
    "inequality_en_medium"
    "inequality_ko_easy"
    "inequality_ko_hard"
    "inequality_ko_medium"
    "jamo_ko_easy"
    "jamo_ko_hard"
    "jamo_ko_medium"
    "kinship_ko_easy"
    "kinship_ko_hard"
    "kinship_ko_medium"
    "korean_units_ko_easy"
    "korean_units_ko_hard"
    "korean_units_ko_medium"
    "logic_grid_en_easy"
    "logic_grid_en_hard"
    "logic_grid_en_medium"
    "logic_grid_ko_easy"
    "logic_grid_ko_hard"
    "logic_grid_ko_medium"
    "minesweeper_en_easy"
    "minesweeper_en_hard"
    "minesweeper_en_medium"
    "minesweeper_ko_easy"
    "minesweeper_ko_hard"
    "minesweeper_ko_medium"
    "number_baseball_en_easy"
    "number_baseball_en_hard"
    "number_baseball_en_medium"
    "number_baseball_ko_easy"
    "number_baseball_ko_hard"
    "number_baseball_ko_medium"
    "saju_ko_easy"
    "saju_ko_hard"
    "saju_ko_medium"
    "sat_puzzles_en_easy"
    "sat_puzzles_en_hard"
    "sat_puzzles_en_medium"
    "sat_puzzles_ko_easy"
    "sat_puzzles_ko_hard"
    "sat_puzzles_ko_medium"
    "sudoku_en_easy"
    "sudoku_en_hard"
    "sudoku_en_medium"
    "sudoku_ko_easy"
    "sudoku_ko_hard"
    "sudoku_ko_medium"
    "time_ko_easy"
    "time_ko_hard"
    "time_ko_medium"
    "yacht_dice_en_easy"
    "yacht_dice_en_hard"
    "yacht_dice_en_medium"
    "yacht_dice_ko_easy"
    "yacht_dice_ko_hard"
    "yacht_dice_ko_medium"
)
# ==============================

VENV_ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo -e "${RED}Error: .venv not found at $PROJECT_ROOT/.venv${NC}"
    exit 1
fi
source "$VENV_ACTIVATE"

# Blackwell(sm_120) flashinfer JIT 컴파일용 CUDA 툴킷 경로 (nvcc)
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"

start_vllm_server() {
    local model=$1
    echo -e "${BLUE}Starting vLLM server: ${model} on GPUs ${GPUS} (TP=${TENSOR_PARALLEL_SIZE})...${NC}"

    # setsid 로 새 프로세스 그룹 생성 → 종료 시 워커까지 그룹 단위로 확실히 정리 (PGID == VLLM_PID)
    setsid env CUDA_VISIBLE_DEVICES=$GPUS python -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --host "$VLLM_HOST" \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --trust-remote-code 2>&1 &
    VLLM_PID=$!

    echo -e "  vLLM PID/PGID: ${VLLM_PID}"

    # 서버 준비 대기 (다운로드가 오래 걸려도 프로세스가 살아있는 한 계속 대기)
    local max_wait=$SERVER_START_TIMEOUT
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
        # 60초마다 진행 표시 (다운로드/컴파일 중)
        if [ $((waited % 60)) -eq 0 ] && [ $waited -gt 0 ]; then
            echo -e "  ...still starting (${waited}s, 다운로드/컴파일 중)"
        fi
        sleep 5
        waited=$((waited + 5))
    done

    echo -e "${RED}  vLLM server failed to start within ${max_wait}s${NC}"
    stop_vllm_server
    return 1
}

stop_vllm_server() {
    if [ -n "$VLLM_PID" ] && kill -0 $VLLM_PID 2>/dev/null; then
        echo -e "${YELLOW}Stopping vLLM server (PID/PGID: ${VLLM_PID})...${NC}"
        # 프로세스 그룹 전체에 SIGTERM (워커 포함). 음수 PID = 그룹.
        kill -TERM -"$VLLM_PID" 2>/dev/null || kill -TERM "$VLLM_PID" 2>/dev/null
        # 최대 30초 graceful 대기
        local w=0
        while [ $w -lt 30 ] && kill -0 $VLLM_PID 2>/dev/null; do sleep 2; w=$((w+2)); done
        # 남아있으면 그룹 강제 종료 (GPU 메모리 확실히 해제)
        kill -9 -"$VLLM_PID" 2>/dev/null || kill -9 "$VLLM_PID" 2>/dev/null
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
echo -e "GPUs: ${GPUS} (TP=${TENSOR_PARALLEL_SIZE}, mem_util=${GPU_MEM_UTIL})"
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

    # vLLM 서버 시작 (실패해도 set -e 로 전체 종료되지 않도록 if 로 감쌈)
    if ! start_vllm_server "$MODEL"; then
        echo -e "${RED}Failed to start vLLM for ${MODEL}. Skipping...${NC}"
        stop_vllm_server
        continue
    fi

    MODEL_START=$(date +%s)
    TOTAL_TASKS=${#TASKS[@]}
    CURRENT_TASK=0
    SUCCESS_COUNT=0
    FAIL_COUNT=0
    SKIP_COUNT=0

    for task in "${TASKS[@]}"; do
        CURRENT_TASK=$((CURRENT_TASK + 1))
        log_file="$LOG_DIR/${task}.log"

        # --- 재개(resume): 이미 결과(json)가 있으면 건너뜀 ---
        if [ "$SKIP_EXISTING" = "true" ] && \
           compgen -G "$PROJECT_ROOT/results/$MODEL_DIR_NAME/$task/*.json" > /dev/null 2>&1; then
            echo -e "${GREEN}[$CURRENT_TASK/$TOTAL_TASKS] Skip (already done): $task${NC}"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi

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
    echo -e "  ${GREEN}Success: ${SUCCESS_COUNT}${NC} / ${RED}Fail: ${FAIL_COUNT}${NC} / Skip: ${SKIP_COUNT} (Total: ${TOTAL_TASKS})"
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
