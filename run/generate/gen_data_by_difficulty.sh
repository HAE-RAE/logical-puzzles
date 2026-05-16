#!/usr/bin/env bash
# gen_data.sh와 동일 순서로 데이터를 생성한 뒤,
# data/jsonl/*.jsonl(이미 *_easy 등으로 끝나는 파일 제외)을 난이도별로 쪼개고,
# 합본(foo_en.jsonl)은 삭제해 task당 *_easy|medium|hard.jsonl 세 개만 남긴다.
# 난이도당 문항 수: PER_DIFF(기본 100). 총 300문항 생성기는 PER_DIFF*3 으로 호출한다.
#
# 병렬 실행 옵션:
#   WORKERS: 각 Python 생성 스크립트 내부 multiprocessing 프로세스 수 (기본 4)
#   실행 시 모든 활성 태스크를 백그라운드(&)로 동시에 띄운 뒤 wait 로 합친다.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

PER_DIFF="${PER_DIFF:-100}"
TOTAL=$((PER_DIFF * 3))
WORKERS="${WORKERS:-4}"   # per-task Python worker processes

echo "============================================"
echo "Logical Puzzles — generate + split by difficulty"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PER_DIFF=$PER_DIFF (TOTAL per generator: $TOTAL)"
echo "WORKERS=$WORKERS (Python pool per task)"
echo "============================================"
echo ""

# --- Parallel generation ---
# 활성화된 각 생성 스크립트를 백그라운드로 실행한다.
# PID 배열에 모아 두었다가 마지막에 일괄 wait한다.

PIDS=()
LOG_DIR="${PROJECT_ROOT}/logs/gen"
mkdir -p "$LOG_DIR"

_run_bg() {
  local label="$1"; shift
  local log_file="${LOG_DIR}/${label}.log"
  echo "  [START] ${label}  → ${log_file}"
  python "$@" >"$log_file" 2>&1 &
  PIDS+=($!)
}

echo "Starting parallel generation..."

# # run_py "array_formula_en..." generation/array_formula_en.py --num "$PER_DIFF"
# _run_bg "array_formula_ko" generation/array_formula_ko.py --num "$PER_DIFF" --workers "$WORKERS"

# # _run_bg "cipher_en" generation/cipher_en.py --num "$PER_DIFF" --workers "$WORKERS"
# _run_bg "cipher_ko" generation/cipher_ko.py --num "$PER_DIFF" --workers "$WORKERS"

# # _run_bg "ferryman_en" generation/ferryman_en.py --num "$PER_DIFF" --workers "$WORKERS"
# _run_bg "ferryman_ko" generation/ferryman_ko.py --num "$PER_DIFF" --workers "$WORKERS"

# _run_bg "hanoi_en" generation/hanoi_en.py --num "$PER_DIFF" --workers "$WORKERS"
# _run_bg "hanoi_ko" generation/hanoi_ko.py --num "$PER_DIFF" --workers "$WORKERS"

# # _run_bg "causal_dag_en" generation/causal_dag_en.py --num "$TOTAL" --workers "$WORKERS"
# _run_bg "causal_dag_ko" generation/causal_dag_ko.py --num "$TOTAL" --workers "$WORKERS"

# # _run_bg "logic_grid_en" generation/logic_grid_en.py --num-samples "$TOTAL" --workers "$WORKERS"
# _run_bg "logic_grid_ko" generation/logic_grid_ko.py --num-samples "$TOTAL" --workers "$WORKERS"

# _run_bg "sat_puzzle_en" generation/sat_puzzle_en.py --num-samples "$TOTAL" --workers "$WORKERS"
# _run_bg "sat_puzzle_ko" generation/sat_puzzle_ko.py --num-samples "$TOTAL" --workers "$WORKERS"



_run_bg "inequality_en"      generation/inequality_en.py      --num "$TOTAL" --workers "$WORKERS"
# # _run_bg "inequality_ko" generation/inequality_ko.py --num "$TOTAL" --workers "$WORKERS"

_run_bg "minesweeper_en"    generation/minesweeper_en.py    --num "$TOTAL" --workers "$WORKERS"
# # _run_bg "minesweeper_ko" generation/minesweeper_ko.py --num "$TOTAL" --workers "$WORKERS"

_run_bg "number_baseball_en" generation/number_baseball_en.py --num "$TOTAL" --workers "$WORKERS"
# # _run_bg "number_baseball_ko" generation/number_baseball_ko.py --num "$TOTAL" --workers "$WORKERS"

_run_bg "sudoku_en"         generation/sudoku_en.py         --num "$TOTAL" --workers "$WORKERS"
# # _run_bg "sudoku_ko" generation/sudoku_ko.py --num "$TOTAL" --workers "$WORKERS"

_run_bg "yacht_dice_en"     generation/yacht_dice_en.py     --num "$TOTAL" --workers "$WORKERS"
# _run_bg "yacht_dice_ko" generation/yacht_dice_ko.py --num "$TOTAL" --workers "$WORKERS"

_run_bg "cryptarithmetic_en" generation/cryptarithmetic_en.py --num "$TOTAL" --workers "$WORKERS"
# # _run_bg "cryptarithmetic_ko" generation/cryptarithmetic_ko.py --num "$TOTAL" --workers "$WORKERS"


_run_bg "kinship" generation/kinship.py --num "$PER_DIFF" --workers "$WORKERS"
# # _run_bg "kinship_vision" generation/kinship_vision.py --num "$PER_DIFF" --workers "$WORKERS"

echo ""
echo "Waiting for ${#PIDS[@]} background task(s) to finish..."

FAILED=()
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAILED+=("$pid")
  fi
done

echo ""
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "WARNING: ${#FAILED[@]} generation task(s) failed (pids: ${FAILED[*]})"
  echo "         Check logs in ${LOG_DIR}/"
else
  echo "All generation tasks completed successfully."
fi

echo ""
echo "============================================"
echo "Splitting combined JSONL -> *_easy|medium|hard.jsonl"
echo "============================================"
python scripts/gen/split_jsonl_by_difficulty.py --jsonl-dir "$PROJECT_ROOT/data/jsonl" --max-per-difficulty "$PER_DIFF" --delete-source

echo ""
echo "Done. JSONL outputs: data/jsonl/<stem>_{easy,medium,hard}.jsonl (combined *.jsonl removed)."
echo "Generation logs: ${LOG_DIR}/"


# bash run/generate/gen_data_by_difficulty.sh
