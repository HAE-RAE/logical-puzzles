#!/bin/bash
# ---------------------------------------------------------------------------
# 난이도 수정(kinship/korean_units/inequality) 후 Midm-2.0 로컬 재실행 (도그볼/H100).
#
#   eval_midm_1gpu.sh 와 동일한 vLLM 수명주기 (H100×1, TP=1, ctx 32k).
#   차이점 2가지만:
#     (1) TASKS = 난이도 수정된 3태스크 12개만
#     (2) SKIP_EXISTING=false — 기존(stale) 결과가 있어도 무조건 재실행
#   run.py 는 새 타임스탬프 JSON 을 추가하므로 dir 안에서 최신 파일이 유효 점수.
#
#   도그볼에서 실행:
#     GPU=0 bash run/eval/rerun_diff_fix_midm_dogbowl.sh
#   (다른 GPU 로 병렬 실행 시 GPU 값만 바꾸면 포트도 8010+GPU 로 자동 분리)
# ---------------------------------------------------------------------------
set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============ ① 설정 (eval_midm_1gpu.sh 본 실험 스펙과 동일) ============
MODEL="K-intelligence/Midm-2.0-Base-Instruct"
GPU="${GPU:-0}"
GPUS="$GPU"
TENSOR_PARALLEL_SIZE=1
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN=32768
VLLM_PORT="${VLLM_PORT:-$((8010 + GPU))}"
VLLM_HOST="0.0.0.0"
REMOTE_URL="http://localhost:${VLLM_PORT}"
GEN_KWARGS="temperature=0.8,top_p=0.7,top_k=20,repetition_penalty=1.05,max_tokens=14336"
SKIP_EXISTING=false          # ← 재실행: 기존 결과 있어도 다시 돈다
SERVER_START_TIMEOUT=7200
MAX_CONCURRENT="${MAX_CONCURRENT:-10}"
FORCE_RESET="${FORCE_RESET:-1}"

# ============ ② TASKS — 난이도 수정된 3태스크 12개만 ============
TASKS=(
    "kinship_ko_easy" "kinship_ko_medium" "kinship_ko_hard"
    "korean_units_ko_easy" "korean_units_ko_medium" "korean_units_ko_hard"
    "inequality_en_easy" "inequality_en_medium" "inequality_en_hard"
    "inequality_ko_easy" "inequality_ko_medium" "inequality_ko_hard"
    # hanoi 는 난이도 처리상의 문제로 벤치마크 데이터에서 제외 → 재실행 대상 아님.
)

# 파이썬 환경:
#   - Backend.AI vLLM 이미지(도그볼): vllm/torch/CUDA 가 시스템 파이썬에 이미 있음
#     → .venv 를 만들지 않고 시스템 파이썬 사용. eval 의존성만 requirements.txt 로 추가.
#   - 로컬 GPU 서버(직접 vllm 설치): .venv 가 있으면 그걸 활성화.
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi
if ! python -c "import vllm" 2>/dev/null; then
    echo -e "${RED}Error: vllm 미설치. Backend.AI vLLM 0.24.0 이미지에서 실행하거나, pip install vllm==0.24.0 하세요.${NC}"; exit 1
fi

if python - <<'PY' 2>/dev/null
from huggingface_hub import snapshot_download
snapshot_download("K-intelligence/Midm-2.0-Base-Instruct", local_files_only=True)
PY
then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi
# CUDA 는 vLLM 컨테이너가 제공(도그볼 이미지 = CUDA 12.9). 실재하는 경로만 PATH 에 추가.
# CUDA_HOME 을 직접 주고 싶으면 실행 시 `CUDA_HOME=/usr/local/cuda-12.9 bash ...` 로 넘기면 됨.
if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
fi

VLLM_PID=""

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
    echo -e "${BLUE}Starting vLLM: ${MODEL} on GPU ${GPU} (TP=1, port ${VLLM_PORT})...${NC}"
    setsid env CUDA_VISIBLE_DEVICES=$GPUS python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --host "$VLLM_HOST" \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --trust-remote-code \
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
MODEL_DIR_NAME="${MODEL//\//_}"
LOG_DIR="$PROJECT_ROOT/results/$MODEL_DIR_NAME/log"; mkdir -p "$LOG_DIR"

echo -e "${BLUE}=== Midm-2.0 RERUN(diff-fix) | GPU ${GPU} | port ${VLLM_PORT} | ${#TASKS[@]} tasks ===${NC}"

[ "$FORCE_RESET" = "1" ] && reset_gpu "$GPU"

if ! start_vllm_server; then
    echo -e "${RED}Failed to start ${MODEL}. Aborting.${NC}"; exit 1
fi

TOTAL=${#TASKS[@]}; CUR=0; OK=0; FAIL=0; SKIP=0
for task in "${TASKS[@]}"; do
    CUR=$((CUR+1)); log_file="$LOG_DIR/${task}.rerun.log"
    if [ "$SKIP_EXISTING" = "true" ] && \
       compgen -G "$PROJECT_ROOT/results/$MODEL_DIR_NAME/$task/*.json" > /dev/null 2>&1; then
        echo -e "${GREEN}[$CUR/$TOTAL] Skip: $task${NC}"; SKIP=$((SKIP+1)); continue
    fi
    echo -e "${YELLOW}[$CUR/$TOTAL] Evaluating: $task${NC}  (log: ${log_file})"
    if ! server_healthy; then
        echo -e "${YELLOW}  server unhealthy — restarting...${NC}"; stop_vllm_server
        if ! start_vllm_server; then
            echo -e "${RED}  restart failed — aborting${NC}"; FAIL=$((FAIL+1)); break
        fi
    fi
    set +e
    if python evaluation/run.py \
        --model "$MODEL" --model_router remote --remote_url "$REMOTE_URL" \
        --gen-kwargs "$GEN_KWARGS" --tasks "$task" --async --max-concurrent "$MAX_CONCURRENT" 2>&1 | tee -a "$log_file"; then
        echo -e "${GREEN}  $task done${NC}"; OK=$((OK+1))
    else
        echo -e "${RED}  $task failed${NC}"; FAIL=$((FAIL+1))
    fi
    set -e
done

echo -e "${BLUE}--- ${MODEL}: OK ${OK} / Fail ${FAIL} / Skip ${SKIP} (Total ${TOTAL}) ---${NC}"
stop_vllm_server
echo -e "${BLUE}=== Done (GPU ${GPU}), elapsed $(( ($(date +%s)-OVERALL_START)/60 ))m ===${NC}"
