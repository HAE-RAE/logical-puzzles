#!/bin/bash
# 여러 상용 모델을 한 번에 평가하는 래퍼.
# MODELS 배열에 "모델string|||gen_kwargs" 형식으로 추가하면,
# 모델마다 TASKS 전체를 task별·난이도별로 평가한다.
# (단일 모델용: eval_litellm_parallel.sh)

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============ 평가할 상용 모델들 ============
# 모델·파라미터는 run/eval/model_configs.yaml 이 single source of truth.
# 여기서 직접 수정하지 말고 YAML을 고칠 것.
# 그룹 선택: 인자로 넘기면 해당 그룹만 (기본: api_models)
#   bash run/eval/eval_litellm_multimodel.sh api_models
#   bash run/eval/eval_litellm_multimodel.sh calibration_reference api_models
GROUPS=("${@:-api_models}")

MODELS=()
while IFS= read -r line; do
    MODELS+=("$line")
done < <(python "$SCRIPT_DIR/get_model_config.py" --entries "${GROUPS[@]}")

if [ ${#MODELS[@]} -eq 0 ]; then
    echo -e "${RED}model_configs.yaml에서 모델을 못 읽음 (그룹: ${GROUPS[*]})${NC}"
    exit 1
fi
# =============================================

MAX_PARALLEL=5          # 모델 1개 안에서 동시에 돌릴 task 수
TASK_ASYNC_CONCURRENT=10 # task 1개 안에서 동시에 보낼 요청 수

TASKS=(
    ################## 직번역(11) ##################
    "array_formula_en_easy" "array_formula_en_medium" "array_formula_en_hard"
    "array_formula_ko_easy" "array_formula_ko_medium" "array_formula_ko_hard"
    "ferryman_en_easy" "ferryman_en_medium" "ferryman_en_hard"
    "ferryman_ko_easy" "ferryman_ko_medium" "ferryman_ko_hard"
    "hanoi_en_easy" "hanoi_en_medium" "hanoi_en_hard"
    "hanoi_ko_easy" "hanoi_ko_medium" "hanoi_ko_hard"
    "logic_grid_en_easy" "logic_grid_en_medium" "logic_grid_en_hard"
    "logic_grid_ko_easy" "logic_grid_ko_medium" "logic_grid_ko_hard"
    "sat_puzzles_en_easy" "sat_puzzles_en_medium" "sat_puzzles_en_hard"
    "sat_puzzles_ko_easy" "sat_puzzles_ko_medium" "sat_puzzles_ko_hard"
    "causal_dag_en_easy" "causal_dag_en_medium" "causal_dag_en_hard"
    "causal_dag_ko_easy" "causal_dag_ko_medium" "causal_dag_ko_hard"
    "inequality_en_easy" "inequality_en_medium" "inequality_en_hard"
    "inequality_ko_easy" "inequality_ko_medium" "inequality_ko_hard"
    "minesweeper_en_easy" "minesweeper_en_medium" "minesweeper_en_hard"
    "minesweeper_ko_easy" "minesweeper_ko_medium" "minesweeper_ko_hard"
    "yacht_dice_en_easy" "yacht_dice_en_medium" "yacht_dice_en_hard"
    "yacht_dice_ko_easy" "yacht_dice_ko_medium" "yacht_dice_ko_hard"
    "number_baseball_en_easy" "number_baseball_en_medium" "number_baseball_en_hard"
    "number_baseball_ko_easy" "number_baseball_ko_medium" "number_baseball_ko_hard"
    "sudoku_en_easy" "sudoku_en_medium" "sudoku_en_hard"
    "sudoku_ko_easy" "sudoku_ko_medium" "sudoku_ko_hard"

    ################## 언어 특화(2) ##################
    "cipher_en_easy" "cipher_en_medium" "cipher_en_hard"
    "cipher_ko_easy" "cipher_ko_medium" "cipher_ko_hard"
    "cryptarithmetic_en_easy" "cryptarithmetic_en_medium" "cryptarithmetic_en_hard"
    "cryptarithmetic_ko_easy" "cryptarithmetic_ko_medium" "cryptarithmetic_ko_hard"

    ################## 한국어 전용(6) ##################
    "jamo_ko_easy" "jamo_ko_medium" "jamo_ko_hard"
    "saju_ko_easy" "saju_ko_medium" "saju_ko_hard"
    "kinship_ko_easy" "kinship_ko_medium" "kinship_ko_hard"
    "time_ko_easy" "time_ko_medium" "time_ko_hard"
    "korean_units_ko_easy" "korean_units_ko_medium" "korean_units_ko_hard"
    "subway_ko_easy" "subway_ko_medium" "subway_ko_hard"
)

run_task() {
    local model=$1
    local gen_kwargs=$2
    local task=$3
    local task_num=$4
    local total=$5
    local log_dir=$6
    local log_file="$log_dir/${task}.log"

    {
        echo "========================================"
        echo "Model: $model"
        echo "Task: $task"
        echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================"
        echo ""
    } >> "$log_file"

    echo -e "${YELLOW}[$task_num/$total] $model :: $task${NC}"

    set +e
    if python evaluation/run.py \
        --model "$model" \
        --model_router litellm \
        --gen-kwargs "$gen_kwargs" \
        --tasks "$task" \
        --async \
        --max-concurrent "$TASK_ASYNC_CONCURRENT" >> "$log_file" 2>&1; then
        echo -e "${GREEN}✓ $task${NC}"
        echo "$task:SUCCESS" >> "$RESULT_FILE"
    else
        echo -e "${RED}✗ $task  (로그: $log_file)${NC}"
        echo "$task:FAIL" >> "$RESULT_FILE"
    fi
    set -e
}

GLOBAL_START=$(date +%s)

for entry in "${MODELS[@]}"; do
    MODEL="${entry%%|||*}"
    rest="${entry#*|||}"
    GEN_KWARGS="${rest%%|||*}"
    ENV_KEY="${rest##*|||}"

    # API 키 존재 확인 (셸 env 또는 .env) — 없으면 그 모델만 건너뜀
    if [ -n "$ENV_KEY" ] && [ -z "${!ENV_KEY}" ] \
        && ! grep -q "^${ENV_KEY}=." "$PROJECT_ROOT/.env" 2>/dev/null; then
        echo -e "${RED}SKIP $MODEL — $ENV_KEY 미설정 (셸 env / .env 확인)${NC}"
        continue
    fi

    MODEL_DIR_NAME="${MODEL//\//_}"
    LOG_DIR="$PROJECT_ROOT/results/$MODEL_DIR_NAME/log"
    mkdir -p "$LOG_DIR"
    RESULT_FILE="/tmp/eval_results_${MODEL_DIR_NAME//\//_}_$$"
    rm -f "$RESULT_FILE"

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}MODEL: ${MODEL}${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "Gen kwargs: ${GEN_KWARGS}"
    echo -e "Tasks: ${#TASKS[@]}개  |  병렬: ${MAX_PARALLEL}  |  로그: ${LOG_DIR}"
    echo ""

    MODEL_START=$(date +%s)
    n=0
    for task in "${TASKS[@]}"; do
        n=$((n + 1))
        run_task "$MODEL" "$GEN_KWARGS" "$task" "$n" "${#TASKS[@]}" "$LOG_DIR" &
        while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 1; done
    done
    wait

    SUCCESS=0; FAIL=0
    if [ -f "$RESULT_FILE" ]; then
        while IFS=: read -r _ result; do
            [ "$result" = "SUCCESS" ] && SUCCESS=$((SUCCESS + 1)) || FAIL=$((FAIL + 1))
        done < "$RESULT_FILE"
        rm -f "$RESULT_FILE"
    fi
    MODEL_ELAPSED=$(( $(date +%s) - MODEL_START ))
    echo -e "${BLUE}--- $MODEL 완료: ${GREEN}성공 $SUCCESS${NC} / ${RED}실패 $FAIL${NC}  (${MODEL_ELAPSED}s) ---${NC}"
    echo ""
done

TOTAL_ELAPSED=$(( $(date +%s) - GLOBAL_START ))
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}전체 완료: ${#MODELS[@]}개 모델 × ${#TASKS[@]}개 task  (총 ${TOTAL_ELAPSED}s)${NC}"
echo -e "결과: results/<model>/<task>/...json"
echo -e "${BLUE}========================================${NC}"
