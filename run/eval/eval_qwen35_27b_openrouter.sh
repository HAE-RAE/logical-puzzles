#!/bin/bash

# ============================================================================
# Qwen3.5-27B 평가 러너 — OpenRouter 경유 (GPU 불필요)
# ----------------------------------------------------------------------------
# eval_qwen35_27b_2gpu.sh 의 API 버전. 도그볼 GPU가 없을 때 사용.
# 설정 변경으로 시트1 기존 결과(6태스크)도 무효 — 본 실험 전체 93개 태스크를
# 전부 이 러너(OpenRouter)로 돌린다. 서빙 환경 단일화로 각주 불필요.
#
# 샘플링은 로컬 스펙과 동일 (temperature=0.6, top_p=0.95, top_k=20,
# max_tokens=16384). 단 OpenRouter 프로바이더의 서빙 정밀도(FP8 등)는
# 로컬 vLLM BF16과 다를 수 있음 — 결과표 각주 대상.
#
# 실행 (로컬 Mac 또는 CPU 세션):
#   export OPENROUTER_API_KEY=sk-or-...
#   nohup bash run/eval/eval_qwen35_27b_openrouter.sh > qwen_or_eval.out 2>&1 &
#   (Mac에서 직접 돌릴 땐 잠들지 않게: caffeinate -i 를 앞에 붙이거나 전원 연결)
#
# 재개(resume): SKIP_EXISTING=true — 완료 태스크는 건너뜀.
# ============================================================================

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============ ① 설정 블록 ============
MODEL="openrouter/qwen/qwen3.5-27b"
GEN_KWARGS="temperature=0.6,top_p=0.95,top_k=20,max_tokens=32768"   # max_tokens는 model_configs.yaml 본 실험 스펙 (16384는 thinking 잘림 다수 — 스모크 테스트 확인)
SKIP_EXISTING=true
MAX_CONCURRENT="${MAX_CONCURRENT:-10}"

# .env 자동 로드 (run.py도 자체 로드하지만, 셸 레벨 가드용으로 먼저 읽음)
if [ -z "$OPENROUTER_API_KEY" ] && [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo -e "${RED}Error: OPENROUTER_API_KEY 가 없습니다 (.env 파일 또는 export 필요).${NC}"
    exit 1
fi

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
    # --- 한국어 전용 ---
    "jamo_ko_easy" "jamo_ko_medium" "jamo_ko_hard"
    "saju_ko_easy" "saju_ko_medium" "saju_ko_hard"
    "kinship_ko_easy" "kinship_ko_medium" "kinship_ko_hard"
    "time_ko_easy" "time_ko_medium" "time_ko_hard"
    "korean_units_ko_easy" "korean_units_ko_medium" "korean_units_ko_hard"
)

MODEL_DIR_NAME="${MODEL//\//_}"
LOG_DIR="$PROJECT_ROOT/results/$MODEL_DIR_NAME/log"; mkdir -p "$LOG_DIR"

echo -e "${BLUE}=== Qwen3.5-27B via OpenRouter | ${#TASKS[@]} tasks | concurrent ${MAX_CONCURRENT} ===${NC}"
echo -e "Gen kwargs: ${GEN_KWARGS}"

OVERALL_START=$(date +%s)
TOTAL=${#TASKS[@]}; CUR=0; OK=0; FAIL=0; SKIP=0
for task in "${TASKS[@]}"; do
    CUR=$((CUR+1)); log_file="$LOG_DIR/${task}.log"
    if [ "$SKIP_EXISTING" = "true" ] && \
       compgen -G "$PROJECT_ROOT/results/$MODEL_DIR_NAME/$task/*.json" > /dev/null 2>&1; then
        echo -e "${GREEN}[$CUR/$TOTAL] Skip: $task${NC}"; SKIP=$((SKIP+1)); continue
    fi
    echo -e "${YELLOW}[$CUR/$TOTAL] Evaluating: $task${NC}  (log: ${log_file})"
    set +e
    if python evaluation/run.py \
        --model "$MODEL" --model_router litellm \
        --gen-kwargs "$GEN_KWARGS" --tasks "$task" --async --max-concurrent "$MAX_CONCURRENT" 2>&1 | tee -a "$log_file"; then
        echo -e "${GREEN}  $task done${NC}"; OK=$((OK+1))
    else
        echo -e "${RED}  $task failed${NC}"; FAIL=$((FAIL+1))
    fi
    set -e
done

echo -e "${BLUE}--- OK ${OK} / Fail ${FAIL} / Skip ${SKIP} (Total ${TOTAL}) ---${NC}"
echo -e "${BLUE}=== Done, elapsed $(( ($(date +%s)-OVERALL_START)/60 ))m ===${NC}"
