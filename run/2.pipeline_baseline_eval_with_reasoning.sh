#!/bin/bash
# Pipeline 2: 학습 전 baseline 평가 (reasoning=on)
#
# 사용법:
#   bash run/2.pipeline_baseline_eval_with_reasoning.sh
#   GPU_ID=0 PORT=8001 MODEL=Qwen/Qwen3-4B bash run/2.pipeline_baseline_eval_with_reasoning.sh
#
# 환경변수 (모두 옵션, 기본값 있음):
#   GPU_ID    : 사용할 CUDA device (default: 0)
#   PORT      : vLLM 서버 포트 (default: 8001)
#   MODEL     : 평가할 모델 (default: Qwen/Qwen3-4B)
#   OUTPUT_DIR: 결과 저장 경로 (default: results/qwen3_4b_baseline)
#
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
source .venv/bin/activate

# ===== 환경변수 (override 가능) =====
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8001}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-results/qwen3_4b_baseline}"
GEN_KWARGS="temperature=0.6,max_tokens=16384,top_p=0.95,top_k=20,reasoning=on"

# 평가 대상 모듈 (학습 파이프라인의 TASKS와 일치시켜 학습-평가 모듈 정합성 유지)
TASKS=("ferryman_en" "array_formula_en")

# ====================================

echo "=== Baseline eval: $MODEL (reasoning=on) on GPU $GPU_ID, port $PORT ==="
echo "    output: $OUTPUT_DIR"
echo "    tasks:  ${TASKS[*]}"
echo ""

mkdir -p logs
VLLM_LOG="logs/vllm_baseline_$(date +%H%M%S).log"

CUDA_VISIBLE_DEVICES=$GPU_ID python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --host 0.0.0.0 --port $PORT \
    --trust-remote-code --served-model-name "$MODEL" \
    --reasoning-parser qwen3 > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null" EXIT INT TERM

for i in $(seq 1 180); do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "vLLM ready (${i}x2s)"; break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then echo "vLLM died (see $VLLM_LOG)"; exit 1; fi
    sleep 2
done

for task in "${TASKS[@]}"; do
    echo ""
    echo "=== Evaluating: $task ==="
    python evaluation/run.py \
        --model "$MODEL" \
        --model_router remote \
        --remote_url "http://localhost:$PORT" \
        --gen-kwargs "$GEN_KWARGS" \
        --tasks "$task" \
        --data-dir "data/distill/split/test" \
        --output-dir "$OUTPUT_DIR" \
        --async --max-concurrent 5 2>&1 || echo "  $task failed"
done

echo ""
echo "=== Baseline eval done ==="
