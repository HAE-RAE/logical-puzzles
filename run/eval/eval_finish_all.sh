#!/bin/bash

# ============================================================================
# "돌던거 다 돌게" — 미완성 평가를 gpt-oss 부터 하나씩 순서대로 완성하는 스크립트
# ----------------------------------------------------------------------------
# 모델을 아래 MODELS 순서(gpt-oss → exaone → gemma → solar)대로 하나씩 처리한다.
# 각 모델은 SKIP_EXISTING resume 로 이미 결과(json)가 있는 task 는 건너뛰고
# 빠진 task 만 이어서 채운다. 한 모델이 끝나면 서버를 내리고 다음 모델로 넘어간다.
#
# 전부 상대경로 — 스크립트 위치 기준으로 PROJECT_ROOT 를 잡으므로, 원래 서버의
# logical-puzzles/run/eval/ 아래 두고 그대로 실행하면 된다:
#     bash run/eval/eval_finish_all.sh
#
# 특정 모델만 돌리고 싶으면 인자로:  bash run/eval/eval_finish_all.sh gpt-oss exaone
#
# ── Solar 사전 준비 (중요) ──────────────────────────────────────────────────
# Solar-Open-100B 는 스톡 vLLM 으로는 서버가 안 뜬다. 다음 중 하나 필요:
#   (A) Upstage 빌드:  pip install "vllm==0.12.1.dev1+solaropen" (transformers==5.0.0)
#   (B) 도커:          upstage/vllm-solar-open:latest
#   (C) 스톡 vLLM v0.12.x + 패치 3곳 (ALLOWED_LAYER_TYPES alias 2곳, use_qk_norm getattr)
#   참고: https://huggingface.co/upstage/Solar-Open-100B/discussions/25
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

# ============ 공통 설정 ============
# 사용할 GPU 목록. 원래 gpt-oss/exaone 는 4~7번에서 돌았음. 서버 상황에 맞게 조정.
# 아래는 전부 환경변수로 덮어쓸 수 있음:
#   GPUS="4,5" VLLM_PORT=8010 bash run/eval/eval_finish_all.sh gpt-oss gemma
GPUS="${GPUS:-0,1,2,3}"
# 시작 시 대상 GPU에 남은 좀비 프로세스를 강제로 kill 하고 시작 (1=on). FORCE_RESET=0 으로 끄기.
FORCE_RESET="${FORCE_RESET:-1}"
# 텐서 병렬 크기 (비우면 GPUS 개수로 자동)
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
VLLM_PORT="${VLLM_PORT:-8010}"
VLLM_HOST="0.0.0.0"
REMOTE_URL="http://localhost:${VLLM_PORT}"
GEN_KWARGS="temperature=0.6,max_tokens=16384,top_p=0.95,top_k=20,reasoning=on"
SKIP_EXISTING=true            # 이미 결과 있는 task 는 건너뜀 (resume)
SERVER_START_TIMEOUT=7200     # 100B+ 첫 다운로드(~200GB)까지 커버

if [ -z "$TENSOR_PARALLEL_SIZE" ]; then
    TENSOR_PARALLEL_SIZE=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
fi

# ============ 모델 순서 (gpt-oss 부터 하나씩) ============
# 인자를 주면 그 키들만, 안 주면 전체를 이 순서대로.
# solar 는 이 서버 vLLM 에 solar_open 파서가 없어 제외(별도 처리). 인자 없이 실행하면 아래 3개 완주.
DEFAULT_ORDER=(gemma exaone)
if [ "$#" -gt 0 ]; then
    ORDER=("$@")
else
    ORDER=("${DEFAULT_ORDER[@]}")
fi

# 키 → 실제 HF 모델 이름
model_name_for() {
    case "$1" in
        gpt-oss) echo "openai/gpt-oss-120b" ;;
        exaone)  echo "LGAI-EXAONE/EXAONE-4.0-32B" ;;
        gemma)   echo "google/gemma-4-31b-it" ;;
        solar)   echo "upstage/Solar-Open-100B" ;;
        *)       echo "" ;;
    esac
}

# 키 → 모델별 추가 vLLM 인자 (Solar 전용 파서/로짓 프로세서, 일부는 max-model-len)
extra_vllm_args_for() {
    case "$1" in
        solar)
            echo "--max-model-len 40960 \
                  --reasoning-parser solar_open \
                  --enable-auto-tool-choice \
                  --tool-call-parser solar_open \
                  --logits-processors vllm.model_executor.models.parallel_tool_call_logits_processor:ParallelToolCallLogitsProcessor \
                  --logits-processors vllm.model_executor.models.solar_open_logits_processor:SolarOpenTemplateLogitsProcessor"
            ;;
        gemma)
            echo "--max-model-len 40960"
            ;;
        *)
            echo ""
            ;;
    esac
}

# ============ 평가할 태스크 목록 (data/jsonl 존재분과 동일 그리드) ============
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
# ==============================

VENV_ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo -e "${RED}Error: .venv not found at $PROJECT_ROOT/.venv${NC}"
    exit 1
fi
source "$VENV_ACTIVATE"

# 오프라인 모드: 이 GPU 노드는 huggingface.co 접속이 안 됨(DNS 차단).
# 이미 캐시에 받아둔 모델(gpt-oss/exaone/gemma)은 네트워크 없이 로컬 캐시로 로드된다.
# ⚠️ solar 는 아직 캐시에 없으므로, 처음 받을 때는 인터넷 되는 곳에서 미리 받거나
#    아래 두 줄을 잠깐 주석 처리하고 받아야 한다. (아래 PREDOWNLOAD 안내 참고)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"

VLLM_PID=""

# 대상 GPU(들)에 물려있는 모든 compute 프로세스를 강제 종료 → VRAM 회수.
# index → uuid 로 매핑해서 "그 GPU들에 붙은 PID"만 골라 kill 한다 (다른 GPU는 안 건드림).
reset_gpus() {
    local gpus="$1"
    echo -e "${YELLOW}[reset] GPUs ${gpus} 초기화: 남은 프로세스 강제 종료...${NC}"
    local uuids=""
    for idx in $(echo "$gpus" | tr ',' ' '); do
        local u
        u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null \
              | awk -F', ' -v i="$idx" '$1==i{print $2}')
        [ -n "$u" ] && uuids="$uuids $u"
    done
    local killed=0
    while IFS=',' read -r cuuid cpid; do
        cuuid=$(echo "$cuuid" | xargs); cpid=$(echo "$cpid" | xargs)
        [ -z "$cpid" ] && continue
        for u in $uuids; do
            if [ "$cuuid" = "$u" ]; then
                echo -e "  kill -9 ${cpid}  (on ${u})"
                kill -9 "$cpid" 2>/dev/null && killed=$((killed+1))
            fi
        done
    done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null)
    echo -e "  ${killed}개 종료. VRAM 반납 대기 8s..."
    sleep 8
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader 2>/dev/null \
      | awk -F', ' -v g="$gpus" 'BEGIN{n=split(g,a,",");for(k=1;k<=n;k++)want[a[k]]=1} want[$1]{print "  GPU "$1" free="$2}'
}

start_vllm_server() {
    local model=$1
    local extra=$2
    echo -e "${BLUE}Starting vLLM server: ${model} on GPUs ${GPUS} (TP=${TENSOR_PARALLEL_SIZE})...${NC}"

    # setsid 로 새 프로세스 그룹 → 종료 시 워커까지 그룹 단위 정리 (PGID == VLLM_PID)
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

    local max_wait=$SERVER_START_TIMEOUT
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo -e "${GREEN}  vLLM server ready (${waited}s)${NC}"
            return 0
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo -e "${RED}  vLLM server process died${NC}"
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
echo -e "${BLUE}Finish-All Evaluation (resume)${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Order: ${ORDER[*]}"
echo -e "GPUs: ${GPUS} (TP=${TENSOR_PARALLEL_SIZE}, mem_util=${GPU_MEM_UTIL}), port=${VLLM_PORT}"
echo -e "Tasks per model: ${#TASKS[@]}"
echo -e "Gen kwargs: ${GEN_KWARGS}"
echo ""

# 시작 전 대상 GPU 강제 초기화 (좀비 vLLM 잔해가 VRAM 잡고 있는 경우)
if [ "$FORCE_RESET" = "1" ]; then
    reset_gpus "$GPUS"
    echo ""
fi

for KEY in "${ORDER[@]}"; do
    MODEL="$(model_name_for "$KEY")"
    if [ -z "$MODEL" ]; then
        echo -e "${RED}Unknown model key: ${KEY} (allowed: gpt-oss exaone gemma solar) — skipping${NC}"
        continue
    fi
    EXTRA="$(extra_vllm_args_for "$KEY")"

    MODEL_DIR_NAME="${MODEL//\//_}"
    LOG_DIR="$PROJECT_ROOT/results/$MODEL_DIR_NAME/log"
    mkdir -p "$LOG_DIR"

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Model: ${MODEL}  (key=${KEY})${NC}"
    echo -e "${BLUE}========================================${NC}"

    if ! start_vllm_server "$MODEL" "$EXTRA"; then
        echo -e "${RED}Failed to start vLLM for ${MODEL}. Skipping to next model...${NC}"
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

        # resume: 이미 결과(json)가 있으면 건너뜀
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

        # 서버 헬스체크: 죽었으면 1회 재기동, 그래도 안 되면 이 모델 중단
        if ! server_healthy; then
            echo -e "${YELLOW}  Server not healthy before $task — restarting vLLM...${NC}"
            stop_vllm_server
            if ! start_vllm_server "$MODEL" "$EXTRA"; then
                echo -e "${RED}  Server restart failed — aborting remaining tasks for ${MODEL}${NC}"
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
    echo ""

    # 다음 모델을 위해 서버 종료 (GPU 메모리 해제)
    stop_vllm_server
done

OVERALL_END=$(date +%s)
OVERALL_ELAPSED=$((OVERALL_END - OVERALL_START))
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}All Done${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total elapsed: $((OVERALL_ELAPSED / 60))m $((OVERALL_ELAPSED % 60))s"

# bash run/eval/eval_finish_all.sh                 # 전체: gpt-oss → exaone → gemma → solar
# bash run/eval/eval_finish_all.sh gpt-oss         # gpt-oss 만
# bash run/eval/eval_finish_all.sh gpt-oss solar   # 원하는 것만, 원하는 순서로
