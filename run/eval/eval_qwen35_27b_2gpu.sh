#!/bin/bash

# ============================================================================
# Qwen3.5-27B 평가 러너 — H100 2장(TP=2) 전용 (도그볼/Backend.AI 세션용)
# ----------------------------------------------------------------------------
# eval_1gpu.sh 의 2-GPU 버전. Qwen3.5-27B(dense, BF16 ~54GB)를 H100 2장에
# 텐서 병렬(TP=2)로 올리고, 시트1에서 이미 완료된 6개 태스크를 제외한
# 미완료 57개 태스크(5,700문항)만 돌린다.
#
# 사전 준비 (도그볼 세션 안에서):
#   1) git clone <repo> && cd logical-puzzles
#   2) python -m venv .venv && source .venv/bin/activate
#   3) pip install -r requirements.txt
#   4) pip install -U vllm        # Qwen3.5 는 linear-attention 하이브리드라
#                                 # 2026-02 이후 vLLM 필요. 구버전이면 서버가 안 뜬다.
#   ※ 첫 실행 시 HF 에서 가중치 ~55GB 다운로드 (Qwen 은 게이트 없음, 토큰 불필요)
#
# 실행:
#   bash run/eval/eval_qwen35_27b_2gpu.sh
#   nohup bash run/eval/eval_qwen35_27b_2gpu.sh > qwen_eval.out 2>&1 &   # 백그라운드 권장
#
# 재개(resume): SKIP_EXISTING=true 라서 중단 후 재실행하면
#   results/Qwen_Qwen3.5-27B/<task>/*.json 이 이미 있는 태스크는 건너뛴다.
# ============================================================================

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============ ① 설정 블록 ============
MODEL="Qwen/Qwen3.5-27B"
GPUS="${GPUS:-0,1}"                        # 세션에 잡힌 GPU 2장
TENSOR_PARALLEL_SIZE=2                     # H100 2장 텐서 병렬
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"    # 프롬프트 + max_tokens(16384) 커버. 전체 262k 는 불필요
VLLM_PORT="${VLLM_PORT:-8020}"
VLLM_HOST="0.0.0.0"
REMOTE_URL="http://localhost:${VLLM_PORT}"
# 팀 오픈모델 공통 생성 설정 (eval_1gpu.sh / eval_solar.sh 와 동일)
GEN_KWARGS="temperature=0.6,max_tokens=16384,top_p=0.95,top_k=20,reasoning=on"
SKIP_EXISTING=true                         # 결과 json 있는 태스크는 건너뜀 (재개용)
SERVER_START_TIMEOUT=7200                  # 첫 다운로드(~55GB)까지 커버
MAX_CONCURRENT="${MAX_CONCURRENT:-10}"     # 태스크당 동시 요청 수 (2xH100 배칭 여유 있음)

# ============ ② TASKS — 미완료 57개만 ============
# 제외(시트1 완료): array_formula, causal_dag, cipher, ferryman, logic_grid, sat_puzzles
# 제외(미완성 태스크): subway
TASKS=(
    # --- en+ko × easy/medium/hard (7태스크 × 6 = 42) ---
    "cryptarithmetic_en_easy" "cryptarithmetic_en_medium" "cryptarithmetic_en_hard"
    "cryptarithmetic_ko_easy" "cryptarithmetic_ko_medium" "cryptarithmetic_ko_hard"
    "hanoi_en_easy" "hanoi_en_medium" "hanoi_en_hard"
    "hanoi_ko_easy" "hanoi_ko_medium" "hanoi_ko_hard"
    "inequality_en_easy" "inequality_en_medium" "inequality_en_hard"
    "inequality_ko_easy" "inequality_ko_medium" "inequality_ko_hard"
    "minesweeper_en_easy" "minesweeper_en_medium" "minesweeper_en_hard"
    "minesweeper_ko_easy" "minesweeper_ko_medium" "minesweeper_ko_hard"
    "number_baseball_en_easy" "number_baseball_en_medium" "number_baseball_en_hard"
    "number_baseball_ko_easy" "number_baseball_ko_medium" "number_baseball_ko_hard"
    "sudoku_en_easy" "sudoku_en_medium" "sudoku_en_hard"
    "sudoku_ko_easy" "sudoku_ko_medium" "sudoku_ko_hard"
    "yacht_dice_en_easy" "yacht_dice_en_medium" "yacht_dice_en_hard"
    "yacht_dice_ko_easy" "yacht_dice_ko_medium" "yacht_dice_ko_hard"
    # --- ko 전용 × easy/medium/hard (5태스크 × 3 = 15) ---
    "jamo_ko_easy" "jamo_ko_medium" "jamo_ko_hard"
    "saju_ko_easy" "saju_ko_medium" "saju_ko_hard"
    "kinship_ko_easy" "kinship_ko_medium" "kinship_ko_hard"
    "time_ko_easy" "time_ko_medium" "time_ko_hard"
    "korean_units_ko_easy" "korean_units_ko_medium" "korean_units_ko_hard"
)

# ============ ③ 환경 준비 ============
VENV_ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo -e "${RED}Error: .venv not found. 상단 '사전 준비' 참고해 venv 부터 만들 것.${NC}"; exit 1
fi
source "$VENV_ACTIVATE"

# eval_1gpu.sh 와 달리 오프라인 강제(HF_HUB_OFFLINE)를 켜지 않는다 —
# 도그볼 새 세션은 가중치 첫 다운로드가 필요하기 때문. 이미 캐시가 있다면
# HF_HUB_OFFLINE=1 bash ... 로 켜서 실행해도 된다.

VLLM_PID=""

# ============ ④ vLLM 서버 기동/종료 ============
start_vllm_server() {
    echo -e "${BLUE}Starting vLLM: ${MODEL} on GPU ${GPUS} (TP=${TENSOR_PARALLEL_SIZE}, port ${VLLM_PORT})...${NC}"
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
            echo -e "${RED}  vLLM server process died — 로그 위쪽 에러 확인 (vLLM 버전/메모리)${NC}"; return 1
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

# ============ ⑤ 평가 루프 ============
OVERALL_START=$(date +%s)
MODEL_DIR_NAME="${MODEL//\//_}"
LOG_DIR="$PROJECT_ROOT/results/$MODEL_DIR_NAME/log"; mkdir -p "$LOG_DIR"

echo -e "${BLUE}=== Qwen3.5-27B Eval | GPU ${GPUS} (TP=${TENSOR_PARALLEL_SIZE}) | ${#TASKS[@]} tasks ===${NC}"

if ! start_vllm_server; then
    echo -e "${RED}vLLM 기동 실패 — 종료${NC}"; exit 1
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

echo -e "${BLUE}=== Done, elapsed $(( ($(date +%s)-OVERALL_START)/60 ))m ===${NC}"

[ $FAIL -gt 0 ] && exit 1
exit 0
