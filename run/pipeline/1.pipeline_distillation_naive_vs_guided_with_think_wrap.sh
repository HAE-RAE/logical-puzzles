#!/bin/bash
# set -e를 사용하지 않음: vLLM 서버 실패, 평가 실패 등이
# 전체 파이프라인을 중단시키지 않도록 개별 단계에서 에러 처리
set -o pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate

# .env 파일에서 환경변수 로드
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# ============ 설정 ============
GPU_ID="${GPU_ID:-3}"     # 환경변수 override: GPU_ID=0 bash run/...
VLLM_PORT="${VLLM_PORT:-8000}"
REMOTE_URL="http://localhost:${VLLM_PORT}"
GEN_KWARGS="temperature=0.6,max_tokens=16384,top_p=0.95,top_k=20,reasoning=on"

MODEL="Qwen/Qwen3-4B"
MODEL_TAG="qwen3_4b"  # 결과/모델 디렉토리 접두어

TASKS=("ferryman_en" "array_formula_en")

DISTILL_MODEL="gpt-4o-mini"
DISTILL_TEMP=0.7
DISTILL_MAX_TOKENS=8192

# Teacher 데이터 경로 suffix:
#   ""            → data/distill/{naive,guided}/            (validated만, 기본 필터링본)
#   "_unfiltered" → data/distill/{naive,guided}_unfiltered/ (wrong_answer 포함, --no-filter)
TEACHER_DIR_SUFFIX="_unfiltered"

SFT_EPOCHS=2
SFT_BATCH=2
SFT_GRAD_ACCUM=8
SFT_LR=1e-5
SFT_MAX_SEQ=8192
SFT_SEED=42
# ==============================

# TEACHER_DIR_SUFFIX == "_unfiltered"이면 generate_distillation.py에 --no-filter 전달
NO_FILTER_FLAG=""
if [ "$TEACHER_DIR_SUFFIX" = "_unfiltered" ]; then
    NO_FILTER_FLAG="--no-filter"
fi

# SFT 데이터는 항상 <think>{reasoning}</think>{final_line} 형식으로 구성 (옵션 B, parsed-split).
# 토글 제거: 비교 실험 의도가 없는 한 think wrap은 강제.
THINK_WRAP_FLAG="--think-wrap"

# 실험 메타데이터 dump 함수 (학습/평가 디렉토리에 config_experiment.json 저장)
write_experiment_config() {
    local dest_dir="$1"     # 결과 디렉토리
    local method="$2"       # naive | guided
    local task="$3"
    local teacher_file="$4"
    mkdir -p "$dest_dir"
    cat > "${dest_dir}/config_experiment.json" <<JEOF
{
  "timestamp": "$(date -Iseconds)",
  "model": "${MODEL}",
  "model_tag": "${MODEL_TAG}",
  "method": "${method}",
  "task": "${task}",
  "teacher_data": "${teacher_file}",
  "teacher_dir_suffix": "${TEACHER_DIR_SUFFIX}",
  "think_wrap": true,
  "distill_model": "${DISTILL_MODEL}",
  "distill_temp": ${DISTILL_TEMP},
  "distill_max_tokens": ${DISTILL_MAX_TOKENS},
  "sft": {
    "epochs": ${SFT_EPOCHS},
    "batch": ${SFT_BATCH},
    "grad_accum": ${SFT_GRAD_ACCUM},
    "lr": ${SFT_LR},
    "max_seq": ${SFT_MAX_SEQ},
    "seed": ${SFT_SEED}
  },
  "gen_kwargs": "${GEN_KWARGS}",
  "gpu_id": ${GPU_ID}
}
JEOF
}

# ── vLLM 서버 관리 ──
VLLM_PID=""
cleanup_vllm() {
    if [ -n "$VLLM_PID" ] && kill -0 $VLLM_PID 2>/dev/null; then
        echo -e "${YELLOW}Cleaning up vLLM (PID: ${VLLM_PID})...${NC}"
        kill $VLLM_PID 2>/dev/null
        wait $VLLM_PID 2>/dev/null || true
    fi
    VLLM_PID=""
}
trap cleanup_vllm EXIT INT TERM

start_vllm_server() {
    local model_name=$1
    local model_path=$2

    echo -e "${BLUE}    Starting vLLM: ${model_path} (GPU ${GPU_ID})...${NC}"

    local serve_args="--model $model_path --host 0.0.0.0 --port $VLLM_PORT --trust-remote-code"
    serve_args="$serve_args --served-model-name $model_name"
    serve_args="$serve_args --reasoning-parser qwen3"  # <think>...</think>를 reasoning 필드로 분리

    mkdir -p "$PROJECT_ROOT/logs"
    local vllm_log="$PROJECT_ROOT/logs/vllm_$(date +%H%M%S).log"
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m vllm.entrypoints.openai.api_server $serve_args > "$vllm_log" 2>&1 &
    VLLM_PID=$!

    local max_wait=300 waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo -e "${GREEN}    vLLM ready (${waited}s)${NC}"
            return 0
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo -e "${RED}    vLLM process died${NC}"
            VLLM_PID=""
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
    done

    echo -e "${RED}    vLLM timeout (${max_wait}s)${NC}"
    cleanup_vllm
    return 1
}

eval_task() {
    local model_name=$1
    local result_dir=$2
    local data_dir=$3
    local task=$4

    set +e
    python evaluation/run.py \
        --model "$model_name" \
        --model_router remote \
        --remote_url "$REMOTE_URL" \
        --gen-kwargs "$GEN_KWARGS" \
        --tasks "$task" \
        --data-dir "$data_dir" \
        --output-dir "$result_dir" \
        --async \
        --max-concurrent 5 2>&1
    local ret=$?
    set -e
    return $ret
}

get_accuracy() {
    local result_dir=$1
    local task=$2
    # ResultHandler saves to: output_dir/model_safe/task_name/
    # model_safe = "Qwen_Qwen3-0.6B"
    python -c "
import json, glob
# Try both patterns (with and without model subdir)
patterns = [
    '${result_dir}/*/${task}/*.json',
    '${result_dir}/${task}/*.json',
]
files = []
for p in patterns:
    files.extend(glob.glob(p))
files = sorted(files)
if not files:
    print('N/A')
else:
    with open(files[-1]) as f:
        d = json.load(f)
    s = d.get('summary',{}).get('overall',{})
    acc = s.get('accuracy', 0)
    total = s.get('total_count', s.get('total', 0))
    correct = s.get('correct_count', s.get('correct', 0))
    print(f'{acc:.4f} ({correct}/{total})')
" 2>/dev/null || echo "ERR"
}

# ════════════════════════════════════════════════════════════════
echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Naive vs Guided Distillation Comparison         ║${NC}"
echo -e "${BLUE}║  Per-Module Sequential on CUDA:${GPU_ID}                ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo -e "Student:  ${MODEL}"
echo -e "Teacher:  ${DISTILL_MODEL}"
echo -e "Modules:  ${TASKS[*]}"
echo ""

# ============ Phase 0: 퍼즐 데이터 생성 ============
echo -e "${YELLOW}[Phase 0] Generating puzzle data...${NC}"
[ ! -f "data/json/ferryman_en.jsonl" ] && python generation/ferryman_en.py --num 34 || echo "  ferryman_en: exists"
[ ! -f "data/json/hanoi_en.jsonl" ] && python generation/hanoi_en.py --num 34 --output ./data || echo "  hanoi_en: exists"
[ ! -f "data/json/array_formula_en.jsonl" ] && python generation/array_formula_en.py --num 34 --output ./data || echo "  array_formula_en: exists"
echo -e "${GREEN}[Phase 0] Done${NC}\n"

# ============ Phase 1: Train/Test 분할 ============
echo -e "${YELLOW}[Phase 1] Splitting data (80/20)...${NC}"
if [ ! -f "data/distill/split/manifest.json" ]; then
    python scripts/split_puzzle_data.py \
        --data-dir data/json \
        --output-dir data/distill/split \
        --train-ratio 0.8 --seed 42
else
    echo "  Split exists, skipping"
fi
echo -e "${GREEN}[Phase 1] Done${NC}\n"

# ============ Phase 2: GPT-4o Distillation ============
echo -e "${YELLOW}[Phase 2] Distillation via ${DISTILL_MODEL} Batch API...${NC}"

# 미완료 batch 복구 (RECOVER_BATCH_IDS 환경변수로 전달)
if [ -n "$RECOVER_BATCH_IDS" ]; then
    echo -e "  Recovering batches: $RECOVER_BATCH_IDS"
    python scripts/generate_distillation.py \
        --tasks ${TASKS[*]} \
        --data-dir data/distill/split/train \
        --output-dir data/distill \
        --methods naive guided \
        --recover-batch $RECOVER_BATCH_IDS \
        $NO_FILTER_FLAG
fi

# method별 독립적으로 체크하여 필요한 method만 실행
METHODS_TO_RUN=()
for method in naive guided; do
    NEED=false
    for task in "${TASKS[@]}"; do
        if [ ! -f "data/distill/${method}${TEACHER_DIR_SUFFIX}/teacher_data_${task}.jsonl" ]; then
            NEED=true
            break
        fi
    done
    if $NEED; then
        METHODS_TO_RUN+=("$method")
    else
        echo "  ${method}${TEACHER_DIR_SUFFIX}: all per-task files exist, skipping"
    fi
done

if [ ${#METHODS_TO_RUN[@]} -gt 0 ]; then
    python scripts/generate_distillation.py \
        --tasks ${TASKS[*]} \
        --data-dir data/distill/split/train \
        --output-dir data/distill \
        --methods ${METHODS_TO_RUN[*]} \
        --model "$DISTILL_MODEL" \
        --temperature $DISTILL_TEMP \
        --max-tokens $DISTILL_MAX_TOKENS \
        $NO_FILTER_FLAG
    if [ $? -ne 0 ]; then
        echo -e "${RED}[Phase 2] Distillation failed! Check logs.${NC}"
        exit 1
    fi
else
    echo "  All distillation data exists, skipping"
fi
echo -e "${GREEN}[Phase 2] Done${NC}\n"

# TRL 확인
if ! python -c "import trl" 2>/dev/null; then
    echo "Installing TRL..."
    uv pip install trl
fi

# ============ Phase 3~5: 모듈별 순차 학습 & 평가 ============
TEST_DATA_DIR="data/distill/split/test"

# 결과 수집용 배열
declare -A NAIVE_RESULTS
declare -A GUIDED_RESULTS

for task in "${TASKS[@]}"; do
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Module: ${task}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    NAIVE_TEACHER="data/distill/naive${TEACHER_DIR_SUFFIX}/teacher_data_${task}.jsonl"
    GUIDED_TEACHER="data/distill/guided${TEACHER_DIR_SUFFIX}/teacher_data_${task}.jsonl"
    NAIVE_SFT_DIR="data/sft_naive_${task}"
    GUIDED_SFT_DIR="data/sft_guided_${task}"
    NAIVE_MODEL_DIR="models/${MODEL_TAG}_naive_${task}"
    GUIDED_MODEL_DIR="models/${MODEL_TAG}_guided_${task}"
    NAIVE_RESULT_DIR="results/${MODEL_TAG}_naive_${task}"
    GUIDED_RESULT_DIR="results/${MODEL_TAG}_guided_${task}"

    # ── Naive: 데이터 준비 → 학습 → 평가 ──
    echo -e "\n${CYAN}  [Naive Distillation]${NC}"

    # SFT 데이터 준비
    if [ ! -f "${NAIVE_SFT_DIR}/train.jsonl" ] && [ -f "$NAIVE_TEACHER" ]; then
        echo -e "    Preparing SFT data..."
        python scripts/prepare_sft_data.py \
            --input "$NAIVE_TEACHER" \
            --output-dir "$NAIVE_SFT_DIR" \
            --train-ratio 1.0 --seed 42 \
            $THINK_WRAP_FLAG
    fi

    # 학습
    if [ ! -d "$NAIVE_MODEL_DIR" ] && [ -f "${NAIVE_SFT_DIR}/train.jsonl" ]; then
        echo -e "    Training on CUDA:${GPU_ID}..."
        if python scripts/train_sft.py \
            --model "$MODEL" \
            --train-data "${NAIVE_SFT_DIR}/train.jsonl" \
            --output-dir "$NAIVE_MODEL_DIR" \
            --epochs $SFT_EPOCHS --batch-size $SFT_BATCH \
            --gradient-accumulation-steps $SFT_GRAD_ACCUM \
            --lr $SFT_LR --max-seq-length $SFT_MAX_SEQ \
            --gpu-id $GPU_ID --seed $SFT_SEED; then
            echo -e "${GREEN}    Training complete${NC}"
        else
            echo -e "${RED}    Training FAILED for naive ${task}${NC}"
        fi
    else
        echo -e "    Model exists or no data, skipping training"
    fi

    # 평가
    NAIVE_MODEL_PATH="$PROJECT_ROOT/$NAIVE_MODEL_DIR"
    if [ -d "$NAIVE_MODEL_PATH" ]; then
        echo -e "    Evaluating..."
        if start_vllm_server "$MODEL" "$NAIVE_MODEL_PATH"; then
            eval_task "$MODEL" "$NAIVE_RESULT_DIR" "$TEST_DATA_DIR" "$task"
            cleanup_vllm
            write_experiment_config "$NAIVE_RESULT_DIR" "naive" "$task" "$NAIVE_TEACHER"
        else
            echo -e "${RED}    vLLM failed for naive ${task}, skipping eval${NC}"
        fi
    else
        echo -e "${RED}    Model not found, skipping eval${NC}"
    fi

    # ── Guided: 데이터 준비 → 학습 → 평가 ──
    echo -e "\n${CYAN}  [Guided Distillation]${NC}"

    # SFT 데이터 준비
    if [ ! -f "${GUIDED_SFT_DIR}/train.jsonl" ] && [ -f "$GUIDED_TEACHER" ]; then
        echo -e "    Preparing SFT data..."
        python scripts/prepare_sft_data.py \
            --input "$GUIDED_TEACHER" \
            --output-dir "$GUIDED_SFT_DIR" \
            --train-ratio 1.0 --seed 42 \
            $THINK_WRAP_FLAG
    fi

    # 학습
    if [ ! -d "$GUIDED_MODEL_DIR" ] && [ -f "${GUIDED_SFT_DIR}/train.jsonl" ]; then
        echo -e "    Training on CUDA:${GPU_ID}..."
        if python scripts/train_sft.py \
            --model "$MODEL" \
            --train-data "${GUIDED_SFT_DIR}/train.jsonl" \
            --output-dir "$GUIDED_MODEL_DIR" \
            --epochs $SFT_EPOCHS --batch-size $SFT_BATCH \
            --gradient-accumulation-steps $SFT_GRAD_ACCUM \
            --lr $SFT_LR --max-seq-length $SFT_MAX_SEQ \
            --gpu-id $GPU_ID --seed $SFT_SEED; then
            echo -e "${GREEN}    Training complete${NC}"
        else
            echo -e "${RED}    Training FAILED for guided ${task}${NC}"
        fi
    else
        echo -e "    Model exists or no data, skipping training"
    fi

    # 평가
    GUIDED_MODEL_PATH="$PROJECT_ROOT/$GUIDED_MODEL_DIR"
    if [ -d "$GUIDED_MODEL_PATH" ]; then
        echo -e "    Evaluating..."
        if start_vllm_server "$MODEL" "$GUIDED_MODEL_PATH"; then
            eval_task "$MODEL" "$GUIDED_RESULT_DIR" "$TEST_DATA_DIR" "$task"
            cleanup_vllm
            write_experiment_config "$GUIDED_RESULT_DIR" "guided" "$task" "$GUIDED_TEACHER"
        else
            echo -e "${RED}    vLLM failed for guided ${task}, skipping eval${NC}"
        fi
    else
        echo -e "${RED}    Model not found, skipping eval${NC}"
    fi

    # ── 모듈별 결과 출력 ──
    echo ""
    NAIVE_ACC=$(get_accuracy "$NAIVE_RESULT_DIR" "$task")
    GUIDED_ACC=$(get_accuracy "$GUIDED_RESULT_DIR" "$task")
    NAIVE_RESULTS[$task]="$NAIVE_ACC"
    GUIDED_RESULTS[$task]="$GUIDED_ACC"

    echo -e "${CYAN}  ┌────────────────────────────────────────────┐${NC}"
    echo -e "${CYAN}  │  ${task} Results                            ${NC}"
    echo -e "${CYAN}  ├────────────────────────────────────────────┤${NC}"
    echo -e "${CYAN}  │  Naive:  ${NAIVE_ACC}${NC}"
    echo -e "${CYAN}  │  Guided: ${GUIDED_ACC}${NC}"
    echo -e "${CYAN}  └────────────────────────────────────────────┘${NC}"
    echo ""

done

# ============ 전체 비교 요약 ============
echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Final Comparison: Naive vs Guided               ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo ""
printf "%-22s %20s %20s\n" "Module" "Naive" "Guided"
printf "%-22s %20s %20s\n" "──────────────────────" "────────────────────" "────────────────────"

for task in "${TASKS[@]}"; do
    printf "%-22s %20s %20s\n" "$task" "${NAIVE_RESULTS[$task]:-N/A}" "${GUIDED_RESULTS[$task]:-N/A}"
done

echo ""
echo -e "${GREEN}Experiment complete!${NC}"
echo ""
echo "Result directories:"
for task in "${TASKS[@]}"; do
    echo "  ${task}:"
    echo "    naive:  results/${MODEL_TAG}_naive_${task}/${task}/"
    echo "    guided: results/${MODEL_TAG}_guided_${task}/${task}/"
done
