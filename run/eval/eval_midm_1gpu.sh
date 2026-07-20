#!/bin/bash

# ============================================================================
# Midm-2.0-Base-Instruct (12B) 평가 러너 — GPU 1장(TP=1) 전용 (도그볼/Backend.AI)
# ----------------------------------------------------------------------------
# eval_1gpu.sh 의 Midm 전용 버전. 12B BF16 (~24GB)라 H100 1장에 여유 있게 올라간다.
# model_configs.yaml 의 본 실험 스펙을 그대로 따른다:
#   - sampling: generation_config.json 배포 설정
#     (temperature=0.8, top_p=0.7, top_k=20, repetition_penalty=1.05)
#   - ctx 32k 모델 → max_tokens=14336, vLLM --max-model-len 32768
#   - non-reasoning instruct 모델 → reasoning 플래그 없음
#
# 사전 준비 (도그볼 세션 안에서):
#   1) git clone <repo> && cd logical-puzzles
#   2) python -m venv .venv && source .venv/bin/activate
#   3) pip install -r requirements.txt && pip install -U vllm
#   ※ 첫 실행 시 HF 에서 가중치 ~24GB 다운로드 (게이트 없음, 토큰 불필요)
#
# 실행:
#   GPU=0 bash run/eval/eval_midm_1gpu.sh
#   nohup bash run/eval/eval_midm_1gpu.sh > midm_eval.out 2>&1 &   # 백그라운드 권장
#
# 재개(resume): SKIP_EXISTING=true — 중단 후 재실행하면 완료 태스크는 건너뜀.
# ============================================================================

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============ ① 설정 블록 ============
MODEL="K-intelligence/Midm-2.0-Base-Instruct"
GPU="${GPU:-0}"
GPUS="$GPU"
TENSOR_PARALLEL_SIZE=1
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN=32768                        # ctx 32k 풀 사용 (model_configs.yaml)
VLLM_PORT="${VLLM_PORT:-$((8010 + GPU))}"
VLLM_HOST="0.0.0.0"
REMOTE_URL="http://localhost:${VLLM_PORT}"
# model_configs.yaml 본 실험 스펙 (generation_config.json 배포 설정)
GEN_KWARGS="temperature=0.8,top_p=0.7,top_k=20,repetition_penalty=1.05,max_tokens=14336"
SKIP_EXISTING=true
SERVER_START_TIMEOUT=7200                  # 첫 다운로드(~24GB)까지 커버
MAX_CONCURRENT="${MAX_CONCURRENT:-10}"
FORCE_RESET="${FORCE_RESET:-1}"

# ============ ② TASKS — 본 실험 전체 93개 (message.txt 와 동일) ============
TASKS=(
    # --- 직번역 11태스크 × 6 = 66 ---
    "array_formula_en_easy" "array_formula_en_medium" "array_formula_en_hard"
    "array_formula_ko_easy" "array_formula_ko_medium" "array_formula_ko_hard"
    "causal_dag_en_easy" "causal_dag_en_medium" "causal_dag_en_hard"
    "causal_dag_ko_easy" "causal_dag_ko_medium" "causal_dag_ko_hard"
    "ferryman_en_easy" "ferryman_en_medium" "ferryman_en_hard"
    "ferryman_ko_easy" "ferryman_ko_medium" "ferryman_ko_hard"
    "hanoi_en_easy" "hanoi_en_medium" "hanoi_en_hard"
    "hanoi_ko_easy" "hanoi_ko_medium" "hanoi_ko_hard"
    "inequality_en_easy" "inequality_en_medium" "inequality_en_hard"
    "inequality_ko_easy" "inequality_ko_medium" "inequality_ko_hard"
    "logic_grid_en_easy" "logic_grid_en_medium" "logic_grid_en_hard"
    "logic_grid_ko_easy" "logic_grid_ko_medium" "logic_grid_ko_hard"
    "minesweeper_en_easy" "minesweeper_en_medium" "minesweeper_en_hard"
    "minesweeper_ko_easy" "minesweeper_ko_medium" "minesweeper_ko_hard"
    "number_baseball_en_easy" "number_baseball_en_medium" "number_baseball_en_hard"
    "number_baseball_ko_easy" "number_baseball_ko_medium" "number_baseball_ko_hard"
    "sat_puzzles_en_easy" "sat_puzzles_en_medium" "sat_puzzles_en_hard"
    "sat_puzzles_ko_easy" "sat_puzzles_ko_medium" "sat_puzzles_ko_hard"
    "sudoku_en_easy" "sudoku_en_medium" "sudoku_en_hard"
    "sudoku_ko_easy" "sudoku_ko_medium" "sudoku_ko_hard"
    "yacht_dice_en_easy" "yacht_dice_en_medium" "yacht_dice_en_hard"
    "yacht_dice_ko_easy" "yacht_dice_ko_medium" "yacht_dice_ko_hard"
    # --- 언어 특화 2태스크 × 6 = 12 ---
    "cipher_en_easy" "cipher_en_medium" "cipher_en_hard"
    "cipher_ko_easy" "cipher_ko_medium" "cipher_ko_hard"
    "cryptarithmetic_en_easy" "cryptarithmetic_en_medium" "cryptarithmetic_en_hard"
    "cryptarithmetic_ko_easy" "cryptarithmetic_ko_medium" "cryptarithmetic_ko_hard"
    # --- 한국어 전용 5태스크 × 3 = 15 ---
    "jamo_ko_easy" "jamo_ko_medium" "jamo_ko_hard"
    "saju_ko_easy" "saju_ko_medium" "saju_ko_hard"
    "kinship_ko_easy" "kinship_ko_medium" "kinship_ko_hard"
    "time_ko_easy" "time_ko_medium" "time_ko_hard"
    "korean_units_ko_easy" "korean_units_ko_medium" "korean_units_ko_hard"
)

VENV_ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo -e "${RED}Error: .venv not found at $PROJECT_ROOT/.venv${NC}"; exit 1
fi
source "$VENV_ACTIVATE"

# 첫 실행은 HF 다운로드가 필요하므로 오프라인 모드는 캐시가 있을 때만 켠다
if python - <<'PY' 2>/dev/null
from huggingface_hub import snapshot_download
snapshot_download("K-intelligence/Midm-2.0-Base-Instruct", local_files_only=True)
PY
then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"

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

echo -e "${BLUE}=== Midm-2.0 Eval | GPU ${GPU} | port ${VLLM_PORT} | ${#TASKS[@]} tasks ===${NC}"

[ "$FORCE_RESET" = "1" ] && reset_gpu "$GPU"

if ! start_vllm_server; then
    echo -e "${RED}Failed to start ${MODEL}. Aborting.${NC}"; exit 1
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
