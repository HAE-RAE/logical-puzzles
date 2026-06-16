#!/usr/bin/env bash
# gen_data.sh와 동일 순서로 데이터를 생성한 뒤,
# data/jsonl/*.jsonl(이미 *_easy 등으로 끝나는 파일 제외)을 난이도별로 쪼개고,
# 합본(foo_en.jsonl)은 삭제해 task당 *_easy|medium|hard.jsonl 세 개만 남긴다.
# 난이도당 문항 수: PER_DIFF(기본 100). 총 300문항 생성기는 PER_DIFF*3 으로 호출한다.
#
# 병렬 실행:
#   모든 활성 태스크를 백그라운드(&)로 동시에 띄운 뒤 wait 로 합친다.
#   WORKERS 는 --workers 인자를 지원하는 생성기(sudoku/yacht_dice/number_baseball)에만 전달한다.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

PER_DIFF="${PER_DIFF:-100}"
TOTAL=$((PER_DIFF * 3))
WORKERS="${WORKERS:-4}"   # sudoku_ko / yacht_dice_ko / number_baseball_ko 전용

JSONL_DIR="${PROJECT_ROOT}/data/jsonl"

echo "============================================"
echo "Logical Puzzles — generate + split by difficulty"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PER_DIFF=$PER_DIFF (TOTAL per generator: $TOTAL)"
echo "WORKERS=$WORKERS (only for generators that accept --workers)"
echo "============================================"
echo ""

PIDS=()
LOG_DIR="${PROJECT_ROOT}/logs/gen"
mkdir -p "$LOG_DIR" "$JSONL_DIR"

# 합본 JSONL 생성 후 split.
#   $1 label (로그/표시용)
#   $2 jsonl_stem (data/jsonl/<stem>.jsonl; 기본값 = label)
#   $3+ python script args
_run_bg() {
  local label="$1"
  local jsonl_stem="${2:-$1}"
  shift 2
  local log_file="${LOG_DIR}/${label}.log"
  echo "  [START] ${label}  → ${log_file}"
  (
    if python "$@" >"$log_file" 2>&1; then
      jsonl_path="${JSONL_DIR}/${jsonl_stem}.jsonl"
      if [ -f "$jsonl_path" ]; then
        {
          echo ""
          echo "[POST] ${label} generation OK. Splitting by difficulty..."
          python scripts/gen/split_jsonl_by_difficulty.py \
            "$jsonl_path" \
            --max-per-difficulty "$PER_DIFF" --delete-source
        } >>"$log_file" 2>&1
      else
        echo "[POST] expected ${jsonl_path} not found; skipping split" >>"$log_file"
      fi
    else
      rc=$?
      echo "[POST] ${label} generation failed (rc=${rc}); skipping split" >>"$log_file"
      exit "$rc"
    fi
  ) &
  PIDS+=($!)
}

# 티어별 JSONL을 직접 쓰는 생성기 — post split 생략.
#   $1 label
#   $2+ python script args
_run_bg_direct() {
  local label="$1"
  shift
  local log_file="${LOG_DIR}/${label}.log"
  echo "  [START] ${label}  → ${log_file} (direct per-tier output, no split)"
  (
    if python "$@" >"$log_file" 2>&1; then
      echo "" >>"$log_file"
      echo "[POST] ${label} generation OK (per-tier files written by generator)." >>"$log_file"
    else
      rc=$?
      echo "[POST] ${label} generation failed (rc=${rc})" >>"$log_file"
      exit "$rc"
    fi
  ) &
  PIDS+=($!)
}

echo "Starting parallel generation..."

# --num = 난이도당 개수 (합본 JSONL → split)
# _run_bg "array_formula_ko" array_formula_ko \
#   generation/array_formula_ko.py --num "$PER_DIFF" --output ./data

# _run_bg "cipher_en" cipher_en generation/cipher_en.py --num "$PER_DIFF"
# _run_bg "cipher_ko" cipher_ko generation/cipher_ko.py --num "$PER_DIFF"

# _run_bg "ferryman_en" ferryman_en generation/ferryman_en.py --num "$PER_DIFF"
# _run_bg "ferryman_ko" ferryman_ko \
#   generation/ferryman_ko.py --num "$PER_DIFF"

# hanoi_ko: CLI 없음, 티어별 JSONL 직접 출력 (num_per_difficulty=100 고정)
# _run_bg_direct "hanoi_ko" generation/hanoi_ko.py

# _run_bg "causal_dag_en" causal_dag_en generation/causal_dag_en.py --num "$TOTAL"
# _run_bg "causal_dag_ko" causal_dag_ko \
#   generation/causal_dag_ko.py --num "$TOTAL"

# _run_bg "logic_grid_en" logic_grid_en generation/logic_grid_en.py --num-samples "$TOTAL"
# _run_bg "logic_grid_ko" logic_grid_ko \
#   generation/logic_grid_ko.py --num-samples "$TOTAL"

# 출력 stem: sat_puzzles_ko (스크립트 내부 파일명)
# _run_bg "sat_puzzle_en" sat_puzzles_en generation/sat_puzzle_en.py --num-samples "$TOTAL"
# _run_bg "sat_puzzle_ko" sat_puzzles_ko \
#   generation/sat_puzzle_ko.py --num-samples "$TOTAL"

# _run_bg "inequality_en" inequality_en generation/inequality_en.py --num "$TOTAL"
# _run_bg_direct "inequality_ko" \
#   generation/inequality_ko.py --num "$TOTAL" --outdir "$JSONL_DIR"

# _run_bg_direct "minesweeper_en" \
#   generation/minesweeper_en.py --out "$JSONL_DIR"

# _run_bg "number_baseball_en" number_baseball_en generation/number_baseball_en.py --num "$TOTAL" --workers "$WORKERS"
# _run_bg "number_baseball_ko" number_baseball_ko \
#   generation/number_baseball_ko.py --num "$TOTAL" --workers "$WORKERS"

# _run_bg "yacht_dice_en" yacht_dice_en generation/yacht_dice_en.py --num "$TOTAL" --workers "$WORKERS"
# _run_bg "yacht_dice_ko" yacht_dice_ko generation/yacht_dice_ko.py --num "$TOTAL" --workers "$WORKERS"

# _run_bg "sudoku_en" sudoku_en generation/sudoku_en.py --num "$TOTAL" --workers "$WORKERS"
# _run_bg "sudoku_ko" sudoku_ko \
#   generation/sudoku_ko.py --num "$TOTAL" --workers "$WORKERS"

# _run_bg "cryptarithmetic_en" cryptarithmetic_en generation/cryptarithmetic_en.py --num "$TOTAL"
# _run_bg "cryptarithmetic_ko" cryptarithmetic_ko generation/cryptarithmetic_ko.py --num "$TOTAL"

# kinship: 생성기가 kinship_{easy,medium,hard}.jsonl 을 직접 작성
# _run_bg_direct "kinship" generation/kinship.py --num "$PER_DIFF"
# _run_bg_direct "kinship_vision" generation/kinship_vision.py --num "$PER_DIFF"

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
echo "Sweeping for any unsplit JSONL (safety net)"
echo "============================================"
python scripts/gen/split_jsonl_by_difficulty.py \
  --jsonl-dir "$JSONL_DIR" --max-per-difficulty "$PER_DIFF" --delete-source

echo ""
echo "Done. JSONL outputs: data/jsonl/<stem>_{easy,medium,hard}.jsonl (combined *.jsonl removed)."
echo "Generation logs: ${LOG_DIR}/"


# bash run/generate/gen_data_by_difficulty.sh
