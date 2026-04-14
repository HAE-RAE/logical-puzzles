#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate

# ============ 설정 ============
GPU_ID=3
VLLM_PORT=8000
REMOTE_URL="http://localhost:${VLLM_PORT}"
GEN_KWARGS="temperature=0.6,max_tokens=16384,top_p=0.95,top_k=20,reasoning=on"

MODELS=("Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B")

TASKS=(
    "cryptarithmetic" "cryptarithmetic_ko"
    "inequality" "inequality_ko"
    "minesweeper" "minesweeper_ko"
    "number_baseball" "number_baseball_ko"
    "sudoku" "sudoku_ko"
    "yacht_dice" "yacht_dice_ko"
)

# Teacher 데이터 경로 (인자로 전달)
TEACHER_DATA="${1:?Usage: $0 <teacher_data.jsonl>}"
if [ ! -f "$TEACHER_DATA" ]; then
    echo -e "${RED}Error: Teacher data not found: ${TEACHER_DATA}${NC}"
    exit 1
fi
# ==============================

# vLLM 서버 정리 함수
VLLM_PID=""
cleanup_vllm() {
    if [ -n "$VLLM_PID" ] && kill -0 $VLLM_PID 2>/dev/null; then
        echo -e "${YELLOW}Cleaning up vLLM server (PID: ${VLLM_PID})...${NC}"
        kill $VLLM_PID 2>/dev/null
        wait $VLLM_PID 2>/dev/null || true
    fi
    VLLM_PID=""
}
trap cleanup_vllm EXIT INT TERM

# vLLM 서버 시작 및 대기 (실패 시 return 1)
start_vllm_server() {
    local model=$1
    local model_path=${2:-$1}  # 로컬 경로 또는 HF 모델명

    echo -e "${BLUE}Starting vLLM server: ${model_path} on GPU ${GPU_ID}...${NC}"

    local serve_args="--model $model_path --host 0.0.0.0 --port $VLLM_PORT --trust-remote-code"
    if [ "$model_path" != "$model" ]; then
        serve_args="$serve_args --served-model-name $model"
    fi

    CUDA_VISIBLE_DEVICES=$GPU_ID python -m vllm.entrypoints.openai.api_server $serve_args 2>&1 &
    VLLM_PID=$!

    local max_wait=300 waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo -e "${GREEN}  vLLM server ready (${waited}s)${NC}"
            return 0
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo -e "${RED}  vLLM server process died${NC}"
            VLLM_PID=""
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
    done

    echo -e "${RED}  vLLM server failed to start within ${max_wait}s${NC}"
    cleanup_vllm
    return 1
}

# 태스크별 평가 실행
run_evaluation() {
    local model=$1
    local result_dir=$2

    for task in "${TASKS[@]}"; do
        echo -e "${YELLOW}  Evaluating: $task${NC}"
        set +e
        python evaluation/run.py \
            --model "$model" \
            --model_router remote \
            --remote_url "$REMOTE_URL" \
            --gen-kwargs "$GEN_KWARGS" \
            --tasks "$task" \
            --data-dir "data/sft/test_eval" \
            --output-dir "$result_dir" \
            --async \
            --max-concurrent 5 2>&1 || echo -e "${RED}  $task failed${NC}"
        set -e
    done
}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SFT Training Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Teacher data: ${TEACHER_DATA}"
echo -e "Models: ${MODELS[*]}"
echo -e "GPU: ${GPU_ID}"
echo ""

# ============ Step 1: 데이터 전처리 ============
echo -e "${YELLOW}[Step 1] Preparing SFT data (train/test split)...${NC}"
python scripts/prepare_sft_data.py \
    --input "$TEACHER_DATA" \
    --train-ratio 0.8 \
    --seed 42

# ============ Step 2: Pre-training baseline 평가 (test set) ============
echo -e "\n${YELLOW}[Step 2] Pre-training evaluation on test set...${NC}"

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT="${MODEL//\//_}"

    echo -e "\n${BLUE}Pre-training eval: ${MODEL}${NC}"

    if ! start_vllm_server "$MODEL"; then
        echo -e "${RED}Skipping pre-training eval for ${MODEL}${NC}"
        continue
    fi

    run_evaluation "$MODEL" "results/${MODEL_SHORT}_pretrain"
    cleanup_vllm
done

# ============ Step 3: TRL 설치 확인 ============
echo -e "\n${YELLOW}[Step 3] Checking TRL installation...${NC}"
if ! python -c "import trl" 2>/dev/null; then
    echo "Installing TRL..."
    uv pip install trl
fi
echo -e "${GREEN}TRL ready${NC}"

# ============ Step 4: 모델별 SFT 학습 ============
for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT="${MODEL//\//_}"

    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}Training: ${MODEL}${NC}"
    echo -e "${BLUE}========================================${NC}"

    python scripts/train_sft.py \
        --model "$MODEL" \
        --train-data "data/sft/train.jsonl" \
        --output-dir "models/${MODEL_SHORT}_sft" \
        --epochs 3 \
        --batch-size 2 \
        --gradient-accumulation-steps 8 \
        --lr 2e-5 \
        --max-seq-length 4096 \
        --gpu-id $GPU_ID \
        --seed 42

    echo -e "${GREEN}Training complete: ${MODEL}${NC}"
done

# ============ Step 5: 학습 후 평가 (SFT 모델, test set) ============
echo -e "\n${YELLOW}[Step 5] Evaluating SFT models on test set...${NC}"

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT="${MODEL//\//_}"
    SFT_MODEL_PATH="$PROJECT_ROOT/models/${MODEL_SHORT}_sft"

    if [ ! -d "$SFT_MODEL_PATH" ]; then
        echo -e "${RED}SFT model not found: ${SFT_MODEL_PATH}, skipping eval${NC}"
        continue
    fi

    echo -e "\n${BLUE}Post-training eval: ${MODEL} (SFT)${NC}"

    if ! start_vllm_server "$MODEL" "$SFT_MODEL_PATH"; then
        echo -e "${RED}Skipping post-training eval for ${MODEL}${NC}"
        continue
    fi

    run_evaluation "$MODEL" "results/${MODEL_SHORT}_sft"
    cleanup_vllm
done

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Pipeline Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Pre-training results:  results/*_pretrain/"
echo -e "Post-training results: results/*_sft/"
echo -e "All evaluations use test set only: data/sft/test_eval/"

# Usage: bash scripts/run_training_pipeline.sh data/teacher/teacher_data.jsonl
