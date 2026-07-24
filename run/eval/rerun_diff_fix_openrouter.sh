#!/bin/bash
# ---------------------------------------------------------------------------
# 난이도 수정(kinship/korean_units/inequality) 후 OpenRouter 모델 재실행.
#
#   대상 모델 (인자):  qwen  |  deepseek
#   대상 태스크 12개:  kinship_ko×3, korean_units_ko×3, inequality_en×3, inequality_ko×3
#     - kinship/korean_units 데이터: ff51a10 (2026-07-19 20:13) 재생성
#     - inequality 데이터:            cd830ea (2026-07-23 01:09) 재생성
#   기존 결과가 있어도 무조건 재실행(SKIP_EXISTING=false) → 새 타임스탬프 JSON 생성.
#     run.py 는 덮어쓰지 않으므로 dir 안에서 최신 파일이 유효 점수가 된다.
#
#   사용:
#     bash run/eval/rerun_diff_fix_openrouter.sh qwen
#     bash run/eval/rerun_diff_fix_openrouter.sh deepseek
# ---------------------------------------------------------------------------
set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

case "$1" in
  qwen)
    MODEL="openrouter/qwen/qwen3.5-27b"
    GEN_KWARGS="temperature=0.6,top_p=0.95,top_k=20,max_tokens=32768"
    ;;
  deepseek)
    MODEL="openrouter/deepseek/deepseek-v4-flash"
    GEN_KWARGS="temperature=1.0,top_p=1.0,max_tokens=32768"
    ;;
  *)
    echo -e "${RED}사용법: $0 [qwen|deepseek]${NC}"; exit 1 ;;
esac

MAX_CONCURRENT="${MAX_CONCURRENT:-10}"

# .env 자동 로드 (셸 레벨 가드)
if [ -z "$OPENROUTER_API_KEY" ] && [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo -e "${RED}Error: OPENROUTER_API_KEY 없음 (.env 필요).${NC}"; exit 1
fi

TASKS=(
    "kinship_ko_easy" "kinship_ko_medium" "kinship_ko_hard"
    "korean_units_ko_easy" "korean_units_ko_medium" "korean_units_ko_hard"
    "inequality_en_easy" "inequality_en_medium" "inequality_en_hard"
    "inequality_ko_easy" "inequality_ko_medium" "inequality_ko_hard"
    # hanoi 는 난이도 처리상의 문제로 벤치마크 데이터에서 제외 → 재실행 대상 아님.
)

MODEL_DIR_NAME="${MODEL//\//_}"
LOG_DIR="$PROJECT_ROOT/results/$MODEL_DIR_NAME/log"; mkdir -p "$LOG_DIR"

echo -e "${BLUE}=== RERUN(diff-fix) ${MODEL} | ${#TASKS[@]} tasks | concurrent ${MAX_CONCURRENT} ===${NC}"
echo -e "Gen kwargs: ${GEN_KWARGS}"

OVERALL_START=$(date +%s)
TODAY=$(date +%Y-%m-%d)          # 오늘 이미 끝난 태스크는 스킵 (동시성 올려 재시작 시 재실행 방지)
TOTAL=${#TASKS[@]}; CUR=0; OK=0; FAIL=0; SKIP=0
for task in "${TASKS[@]}"; do
    CUR=$((CUR+1)); log_file="$LOG_DIR/${task}.rerun.log"
    if compgen -G "$PROJECT_ROOT/results/$MODEL_DIR_NAME/$task/"*"_${TODAY}T"*.json > /dev/null 2>&1; then
        echo -e "${GREEN}[$CUR/$TOTAL] Skip (오늘 완료): $task${NC}"; SKIP=$((SKIP+1)); continue
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
