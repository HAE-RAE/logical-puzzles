#!/usr/bin/env bash
# 4-way 비교 + 학습 dynamics 평가
# - baseline: raw Qwen3-4B-Thinking-2507
# - SFT-naive: epoch1/2/3 LoRA adapters
# - SFT-guided: epoch1/2/3 LoRA adapters
# - GRPO:      epoch1/2/3 LoRA adapters
#
# 모든 평가는 동일 eval_60.jsonl, 동일 gen_kwargs (temp=0.6, top_p=0.95, top_k=20, N=4 sampling, pass@1 평균)

set -euo pipefail

cd "$(dirname "$0")/../../.."

# venv 활성화 (tmux/non-interactive shell에서 PATH 보장)
if [[ -f .venv/bin/activate ]]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
fi

GPU_ID="${GPU_ID:-0}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
EVAL_FILE="${EVAL_FILE:-data/array_formula_en/eval_60.jsonl}"
N_SAMPLES="${N_SAMPLES:-4}"
MAX_TOKENS="${MAX_TOKENS:-16384}"

run_eval() {
    local tag=$1; local lora_arg=$2
    local out=results/grpo_vs_sft/${tag}
    if [[ -f "${out}/summary.json" ]]; then
        echo "[skip] ${tag} (already evaluated: ${out}/summary.json)"; return
    fi
    echo "[eval] ${tag}  lora=${lora_arg}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/eval/eval_array_formula_vllm.py \
        --model "${BASE_MODEL}" \
        ${lora_arg} \
        --eval-file "${EVAL_FILE}" \
        --out-dir "${out}" \
        --n-samples ${N_SAMPLES} \
        --temperature 0.6 --top-p 0.95 --top-k 20 \
        --max-tokens ${MAX_TOKENS} --tp-size 1 --seed 42 \
        2>&1 | tee logs/eval_${tag}_$(date +%Y%m%d_%H%M%S).log
}

# baseline
run_eval baseline ""

# SFT-naive epoch ckpts
for e in 1 2 3; do
    CKPT=$(ls -d models/sft_naive/checkpoint-* 2>/dev/null | sort -V | sed -n "${e}p")
    if [[ -n "${CKPT}" ]]; then run_eval "sft_naive_ep${e}" "--lora-path ${CKPT}"; fi
done

# SFT-guided epoch ckpts
for e in 1 2 3; do
    CKPT=$(ls -d models/sft_guided/checkpoint-* 2>/dev/null | sort -V | sed -n "${e}p")
    if [[ -n "${CKPT}" ]]; then run_eval "sft_guided_ep${e}" "--lora-path ${CKPT}"; fi
done

# GRPO epoch ckpts
for e in 1 2 3; do
    CKPT=$(ls -d models/grpo/checkpoint-* 2>/dev/null | sort -V | sed -n "${e}p")
    if [[ -n "${CKPT}" ]]; then run_eval "grpo_ep${e}" "--lora-path ${CKPT}"; fi
done

echo "[ok] all evaluations complete"
echo "[hint] aggregate: python scripts/aggregate_4way_results.py"
