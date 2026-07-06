#!/bin/bash

# ============================================================================
# Solar-Open-100B 전용 평가 스크립트
# ----------------------------------------------------------------------------
# 다른 모델(gemma/exaone/gpt-oss)들은 eval_gpu6_gemma.sh / eval_local_vllm_gpu47.sh
# 로 이미 돌렸고, resume(SKIP_EXISTING) 로직이 있어 재실행하면 빠진 task만 채운다.
# Solar 는 일반 vLLM 플래그로는 서버가 안 떠서 여태 결과가 없다. 이 스크립트는
# Solar 가 요구하는 전용 파서/로짓 프로세서 플래그를 붙여서 서버를 띄운다.
#
# ── 사전 준비 (중요) ────────────────────────────────────────────────────────
# Solar-Open-100B 는 스톡 vLLM 으로는 안 뜬다. 다음 중 하나가 필요:
#   (A) Upstage 커스텀 vLLM 빌드:  pip install "vllm==0.12.1.dev1+solaropen" (transformers==5.0.0)
#   (B) 도커 이미지:               upstage/vllm-solar-open:latest
#   (C) 스톡 vLLM(v0.12.x) + 수동 패치 3곳:
#        - vllm/config/model.py           : ALLOWED_MLP_LAYER_TYPES 를 ALLOWED_LAYER_TYPES 로 alias
#        - vllm/transformers_utils/config.py : 위와 동일 import 우회
#        - vllm/model_executor/models/solar_open.py : use_qk_norm -> getattr(config,'use_qk_norm',False)
#   참고: https://huggingface.co/upstage/Solar-Open-100B/discussions/25
#
# 실행:  bash run/eval/eval_solar.sh
# ============================================================================

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
# Solar-Open-100B = 102.6B MoE (활성 12B, BF16). 가중치만 ~205GB → 최소 4x80GB, 권장 8장.
# 비어있는 GPU 상황에 맞게 조정. 예: 8장이면 "0,1,2,3,4,5,6,7"
# 아래는 전부 환경변수로 덮어쓸 수 있음: GPUS="0,1,2,3" VLLM_PORT=8100 bash run/eval/eval_solar.sh
GPUS="${GPUS:-0,1,2,3}"
# 텐서 병렬 크기 (비우면 GPUS 개수로 자동 설정)
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
# GPU 메모리 사용률. 대형 모델이라 넉넉히.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
# KV 캐시/컨텍스트 길이. 문제 프롬프트 + max_tokens(16384) 를 커버하면 충분. 128k 전부 열면 OOM 위험.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
VLLM_PORT="${VLLM_PORT:-8100}"
VLLM_HOST="0.0.0.0"
REMOTE_URL="http://localhost:${VLLM_PORT}"
GEN_KWARGS="temperature=0.6,max_tokens=16384,top_p=0.95,top_k=20,reasoning=on"
# 재개(resume): 이미 결과(json)가 있는 task 는 건너뜀. 중단 후 재실행하면 이어서 진행.
SKIP_EXISTING=true
# 서버 기동 대기 최대 시간(초). 100B+ 첫 다운로드(~200GB)까지 커버.
SERVER_START_TIMEOUT=7200

# 텐서 병렬 크기 자동 계산 (GPUS 개수)
if [ -z "$TENSOR_PARALLEL_SIZE" ]; then
    TENSOR_PARALLEL_SIZE=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
fi

MODEL="upstage/Solar-Open-100B"

# 평가할 태스크 목록 — data/jsonl 에 존재하는 것과 맞춤 (다른 모델들과 동일 그리드)
TASKS=(
    "array_formula_en_easy"
    "array_formula_en_medium"
    "array_formula_en_hard"
    "array_formula_ko_easy"
    "array_formula_ko_medium"
    "array_formula_ko_hard"
    "causal_dag_en_easy"
    "causal_dag_en_medium"
    "causal_dag_en_hard"
    "causal_dag_ko_easy"
    "causal_dag_ko_medium"
    "causal_dag_ko_hard"
    "cipher_en_easy"
    "cipher_en_medium"
    "cipher_en_hard"
    "cipher_ko_easy"
    "cipher_ko_medium"
    "cipher_ko_hard"
    "cryptarithmetic_en_easy"
    "cryptarithmetic_en_medium"
    "cryptarithmetic_en_hard"
    "cryptarithmetic_ko_easy"
    "cryptarithmetic_ko_medium"
    "cryptarithmetic_ko_hard"
    "ferryman_en_easy"
    "ferryman_en_medium"
    "ferryman_en_hard"
    "ferryman_ko_easy"
    "ferryman_ko_medium"
    "ferryman_ko_hard"
    "hanoi_en_easy"
    "hanoi_en_medium"
    "hanoi_en_hard"
    "hanoi_ko_easy"
    "hanoi_ko_medium"
    "hanoi_ko_hard"
    "inequality_en_easy"
    "inequality_en_medium"
    "inequality_en_hard"
    "inequality_ko_easy"
    "inequality_ko_medium"
    "inequality_ko_hard"
    "jamo_ko_easy"
    "jamo_ko_medium"
    "jamo_ko_hard"
)
# ==============================

VENV_ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo -e "${RED}Error: .venv not found at $PROJECT_ROOT/.venv${NC}"
    exit 1
fi
source "$VENV_ACTIVATE"

# 이 GPU 노드는 huggingface.co 접속이 안 됨(DNS 차단). Solar 가 로컬 캐시에 이미
# 있으면 아래 오프라인 모드를 켜고, 아직 안 받았으면 먼저 다운로드해야 한다.
# 미리 받기(인터넷 되는 셸에서): huggingface-cli download upstage/Solar-Open-100B
if [ "${HF_OFFLINE:-1}" = "1" ]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

# Blackwell(sm_120) flashinfer JIT 컴파일용 CUDA 툴킷 경로 (nvcc)
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"

start_vllm_server() {
    local model=$1
    echo -e "${BLUE}Starting vLLM server: ${model} on GPUs ${GPUS} (TP=${TENSOR_PARALLEL_SIZE})...${NC}"

    # Solar-Open 전용 플래그:
    #   --reasoning-parser solar_open           : <think> 추론 분리 (reasoning=on 대응)
    #   --tool-call-parser solar_open + logits  : 툴콜/템플릿 제어. 우리 평가는 툴을 안 쓰지만
    #                                             SolarOpenTemplateLogitsProcessor 는 출력 포맷에 필요.
    # setsid 로 새 프로세스 그룹 → 종료 시 워커까지 그룹 단위 정리 (PGID == VLLM_PID)
    setsid env CUDA_VISIBLE_DEVICES=$GPUS python -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --host "$VLLM_HOST" \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --trust-remote-code \
        --reasoning-parser solar_open \
        --enable-auto-tool-choice \
        --tool-call-parser solar_open \
        --logits-processors vllm.model_executor.models.parallel_tool_call_logits_processor:ParallelToolCallLogitsProcessor \
        --logits-processors vllm.model_executor.models.solar_open_logits_processor:SolarOpenTemplateLogitsProcessor \
        2>&1 &
    VLLM_PID=$!

    echo -e "  vLLM PID/PGID: ${VLLM_PID}"

    local max_wait=$SERVER_START_TIMEOUT
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo -e "${GREEN}  vLLM server ready (${waited}s)${NC}"
            return 0
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo -e "${RED}  vLLM server process died (플래그/빌드 호환성 확인 → discussions/25)${NC}"
            return 1
        fi
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
        kill -TERM -"$VLLM_PID" 2>/dev/null || kill -TERM "$VLLM_PID" 2>/dev/null
        local w=0
        while [ $w -lt 30 ] && kill -0 $VLLM_PID 2>/dev/null; do sleep 2; w=$((w+2)); done
        kill -9 -"$VLLM_PID" 2>/dev/null || kill -9 "$VLLM_PID" 2>/dev/null
        wait $VLLM_PID 2>/dev/null || true
        echo -e "${GREEN}  vLLM server stopped${NC}"
    fi
    VLLM_PID=""
}

server_healthy() {
    curl -s --max-time 10 "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1
}

trap stop_vllm_server EXIT INT TERM

OVERALL_START=$(date +%s)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Solar-Open-100B Evaluation${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "GPUs: ${GPUS} (TP=${TENSOR_PARALLEL_SIZE}, mem_util=${GPU_MEM_UTIL}, max_len=${MAX_MODEL_LEN})"
echo -e "Model: ${MODEL}"
echo -e "Tasks: ${#TASKS[@]}"
echo -e "Gen kwargs: ${GEN_KWARGS}"
echo ""

MODEL_DIR_NAME="${MODEL//\//_}"
LOG_DIR="$PROJECT_ROOT/results/$MODEL_DIR_NAME/log"
mkdir -p "$LOG_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Model: ${MODEL}${NC}"
echo -e "${BLUE}========================================${NC}"

if ! start_vllm_server "$MODEL"; then
    echo -e "${RED}Failed to start vLLM for ${MODEL}. 위 사전 준비(A/B/C) 확인 필요.${NC}"
    exit 1
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

    # --- 서버 헬스체크: 죽었으면 1회 재기동, 그래도 안 되면 중단 ---
    if ! server_healthy; then
        echo -e "${YELLOW}  Server not healthy before $task — restarting vLLM...${NC}"
        stop_vllm_server
        if ! start_vllm_server "$MODEL"; then
            echo -e "${RED}  Server restart failed — aborting remaining tasks${NC}"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            break
        fi
    fi

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

stop_vllm_server

OVERALL_END=$(date +%s)
OVERALL_ELAPSED=$((OVERALL_END - OVERALL_START))

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Solar Evaluation Completed${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total elapsed: $((OVERALL_ELAPSED / 60))m $((OVERALL_ELAPSED % 60))s"

# bash run/eval/eval_solar.sh
