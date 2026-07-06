#!/bin/bash

# ============================================================================
# Single-GPU 평가 러너 — 한 모델을 GPU 1장(TP=1)에 올려서 돌린다.
# ----------------------------------------------------------------------------
# eval_finish_all.sh 의 단일 GPU 버전. 96GB 카드 1장이면 gemma(31B)/exaone(32B)/
# gpt-oss(120B MXFP4 ~61GB) 모두 TP=1 로 올라간다.
#
# 장점: GPU 마다 서로 다른 모델을 "동시에" 돌릴 수 있다.
#   GPU=0 bash run/eval/eval_1gpu.sh gemma  &
#   GPU=1 bash run/eval/eval_1gpu.sh exaone &
#   GPU=2 bash run/eval/eval_1gpu.sh gpt-oss &
#   wait
# 포트는 GPU 번호에 따라 자동으로 8010+GPU 로 달라져 충돌하지 않는다.
#
# 사용법:
#   GPU=0 bash run/eval/eval_1gpu.sh gemma            # GPU0 에서 gemma
#   GPU=1 bash run/eval/eval_1gpu.sh exaone gemma     # GPU1 에서 exaone→gemma 순차
#   bash run/eval/eval_1gpu.sh                        # 기본: GPU0 에서 gemma exaone
# ============================================================================

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============ 설정 ============
GPU="${GPU:-0}"                       # 사용할 단일 GPU 번호
GPUS="$GPU"                           # 내부적으로 CUDA_VISIBLE_DEVICES 로 쓰임
TENSOR_PARALLEL_SIZE=1                # single GPU → TP=1 고정
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
VLLM_PORT="${VLLM_PORT:-$((8010 + GPU))}"   # GPU별 포트 자동 분리 (동시 실행 대비)
VLLM_HOST="0.0.0.0"
REMOTE_URL="http://localhost:${VLLM_PORT}"
GEN_KWARGS="temperature=0.6,max_tokens=16384,top_p=0.95,top_k=20,reasoning=on"
SKIP_EXISTING=true
SERVER_START_TIMEOUT=7200
FORCE_RESET="${FORCE_RESET:-1}"       # 시작 시 해당 GPU 좀비 프로세스 강제 정리

# ============ 모델 순서 ============
DEFAULT_ORDER=(gemma exaone)
if [ "$#" -gt 0 ]; then ORDER=("$@"); else ORDER=("${DEFAULT_ORDER[@]}"); fi

model_name_for() {
    case "$1" in
        gpt-oss) echo "openai/gpt-oss-120b" ;;
        exaone)  echo "LGAI-EXAONE/EXAONE-4.0-32B" ;;
        gemma)   echo "google/gemma-4-31b-it" ;;
        solar)   echo "upstage/Solar-Open-100B" ;;
        *)       echo "" ;;
    esac
}

# gemma 는 KV 컨텍스트 제한. gpt-oss/exaone 는 기본값 사용.
extra_vllm_args_for() {
    case "$1" in
        gemma) echo "--max-model-len 40960" ;;
        *)     echo "" ;;
    esac
}

TASKS=(
    "array_formula_en_easy" "array_formula_en_medium" "array_formula_en_hard"
    "array_formula_ko_easy" "array_formula_ko_medium" "array_formula_ko_hard"
    "causal_dag_en_easy" "causal_dag_en_medium" "causal_dag_en_hard"
    "causal_dag_ko_easy" "causal_dag_ko_medium" "causal_dag_ko_hard"
    "cipher_en_easy" "cipher_en_medium" "cipher_en_hard"
    "cipher_ko_easy" "cipher_ko_medium" "cipher_ko_hard"
    "cryptarithmetic_en_easy" "cryptarithmetic_en_medium" "cryptarithmetic_en_hard"
    "cryptarithmetic_ko_easy" "cryptarithmetic_ko_medium" "cryptarithmetic_ko_hard"
    "ferryman_en_easy" "ferryman_en_medium" "ferryman_en_hard"
    "ferryman_ko_easy" "ferryman_ko_medium" "ferryman_ko_hard"
    "hanoi_en_easy" "hanoi_en_medium" "hanoi_en_hard"
    "hanoi_ko_easy" "hanoi_ko_medium" "hanoi_ko_hard"
    "inequality_en_easy" "inequality_en_medium" "inequality_en_hard"
    "inequality_ko_easy" "inequality_ko_medium" "inequality_ko_hard"
    "jamo_ko_easy" "jamo_ko_medium" "jamo_ko_hard"
)

VENV_ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo -e "${RED}Error: .venv not found at $PROJECT_ROOT/.venv${NC}"; exit 1
fi
source "$VENV_ACTIVATE"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"

VLLM_PID=""

# 지정한 단일 GPU 에 붙은 프로세스만 강제 종료 (다른 GPU 는 안 건드림)
reset_gpu() {
    local gpu="$1"
    echo -e "${YELLOW}[reset] GPU ${gpu} 초기화: 남은 프로세스 강제 종료...${NC}"
    local uuid
    uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null \
             | awk -F', ' -v i="$gpu" '$1==i{print $2}')
    [ -z "$uuid" ] && { echo "  (GPU ${gpu} uuid 조회 실패, 건너뜀)"; return; }
    local killed=0
    while IFS=',' read -r cuuid cpid; do
        cuuid=$(echo "$cuuid" | xargs); cpid=$(echo "$cpid" | xargs)
        [ -z "$cpid" ] && continue
        if [ "$cuuid" = "$uuid" ]; then
            echo -e "  kill -9 ${cpid}"
            kill -9 "$cpid" 2>/dev/null && killed=$((killed+1))
        fi
    done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null)
    echo -e "  ${killed}개 종료. VRAM 반납 대기 8s..."
    sleep 8
}

start_vllm_server() {
    local model=$1 extra=$2
    echo -e "${BLUE}Starting vLLM: ${model} on GPU ${GPU} (TP=1, port ${VLLM_PORT})...${NC}"
    setsid env CUDA_VISIBLE_DEVICES=$GPUS python -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --host "$VLLM_HOST" \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --trust-remote-code \
        $extra \
        2>&1 &
    VLLM_PID=$!
    echo -e "  vLLM PID/PGID: ${VLLM_PID}"
    local waited=0
    while [ $waited -lt $SERVER_START_TIMEOUT ]; do
        if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo -e "${GREEN}  vLLM server ready (${waited}s)${NC}"; return 0
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo -e "${RED}  vLLM server process died${NC}"; return 1
        fi
        [ $((waited % 60)) -eq 0 ] && [ $waited -gt 0 ] && echo -e "  ...still starting (${waited}s)"
        sleep 5; waited=$((waited + 5))
    done
    echo -e "${RED}  vLLM failed to start within ${SERVER_START_TIMEOUT}s${NC}"; stop_vllm_server; return 1
}

stop_vllm_server() {
    if [ -n "$VLLM_PID" ] && kill -0 $VLLM_PID 2>/dev/null; then
        echo -e "${YELLOW}Stopping vLLM (PID/PGID: ${VLLM_PID})...${NC}"
        kill -TERM -"$VLLM_PID" 2>/dev/null || kill -TERM "$VLLM_PID" 2>/dev/null
        local w=0
        while [ $w -lt 30 ] && kill -0 $VLLM_PID 2>/dev/null; do sleep 2; w=$((w+2)); done
        kill -9 -"$VLLM_PID" 2>/dev/null || kill -9 "$VLLM_PID" 2>/dev/null
        wait $VLLM_PID 2>/dev/null || true
        echo -e "${GREEN}  vLLM stopped${NC}"
    fi
    VLLM_PID=""
}

server_healthy() { curl -s --max-time 10 "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; }

trap stop_vllm_server EXIT INT TERM

OVERALL_START=$(date +%s)
echo -e "${BLUE}=== Single-GPU Eval | GPU ${GPU} | port ${VLLM_PORT} | order: ${ORDER[*]} ===${NC}"

[ "$FORCE_RESET" = "1" ] && reset_gpu "$GPU"

for KEY in "${ORDER[@]}"; do
    MODEL="$(model_name_for "$KEY")"
    if [ -z "$MODEL" ]; then
        echo -e "${RED}Unknown model key: ${KEY} — skipping${NC}"; continue
    fi
    EXTRA="$(extra_vllm_args_for "$KEY")"
    MODEL_DIR_NAME="${MODEL//\//_}"
    LOG_DIR="$PROJECT_ROOT/results/$MODEL_DIR_NAME/log"; mkdir -p "$LOG_DIR"

    echo -e "${BLUE}==== Model: ${MODEL} (GPU ${GPU}) ====${NC}"
    if ! start_vllm_server "$MODEL" "$EXTRA"; then
        echo -e "${RED}Failed to start ${MODEL}. Skipping...${NC}"; stop_vllm_server; continue
    fi

    TOTAL=${#TASKS[@]}; CUR=0; OK=0; FAIL=0; SKIP=0
    for task in "${TASKS[@]}"; do
        CUR=$((CUR+1)); log_file="$LOG_DIR/${task}.log"
        if [ "$SKIP_EXISTING" = "true" ] && \
           compgen -G "$PROJECT_ROOT/results/$MODEL_DIR_NAME/$task/*.json" > /dev/null 2>&1; then
            echo -e "${GREEN}[$CUR/$TOTAL] Skip: $task${NC}"; SKIP=$((SKIP+1)); continue
        fi
        echo -e "${YELLOW}[$CUR/$TOTAL] Evaluating: $task${NC}  (log: ${log_file})"
        if ! server_healthy; then
            echo -e "${YELLOW}  server unhealthy — restarting...${NC}"; stop_vllm_server
            if ! start_vllm_server "$MODEL" "$EXTRA"; then
                echo -e "${RED}  restart failed — aborting ${MODEL}${NC}"; FAIL=$((FAIL+1)); break
            fi
        fi
        set +e
        if python evaluation/run.py \
            --model "$MODEL" --model_router remote --remote_url "$REMOTE_URL" \
            --gen-kwargs "$GEN_KWARGS" --tasks "$task" --async --max-concurrent 5 2>&1 | tee -a "$log_file"; then
            echo -e "${GREEN}  $task done${NC}"; OK=$((OK+1))
        else
            echo -e "${RED}  $task failed${NC}"; FAIL=$((FAIL+1))
        fi
        set -e
    done

    echo -e "${BLUE}--- ${MODEL}: OK ${OK} / Fail ${FAIL} / Skip ${SKIP} (Total ${TOTAL}) ---${NC}"
    stop_vllm_server
done

echo -e "${BLUE}=== Done (GPU ${GPU}), elapsed $(( ($(date +%s)-OVERALL_START)/60 ))m ===${NC}"

# 예)
#   GPU=0 bash run/eval/eval_1gpu.sh gemma
#   GPU=0 bash run/eval/eval_1gpu.sh gemma & GPU=1 bash run/eval/eval_1gpu.sh exaone & wait
