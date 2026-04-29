#!/bin/bash
# run/eval/eval_cipher_simple.sh
# Dedicated easy-track eval for cipher_en_simple on Qwen/Qwen3-VL-8B-Instruct.
# Regenerates puzzles then runs the standalone harness.

set -euo pipefail

cd "$(dirname "$0")/../.."

python generation/cipher_en_simple.py --num 5

python scripts/eval/run_cipher_simple_check.py \
    --model "Qwen/Qwen3-VL-8B-Instruct" \
    --data data/mini_simple/cipher_en_simple_easy.jsonl \
    --out  results/cipher_simple/run.jsonl \
    "$@"
