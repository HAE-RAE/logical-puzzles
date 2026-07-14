#!/bin/bash

# ============================================================================
# max_tokens 제한 없이(=컨텍스트 한도까지) 재평가.
# ----------------------------------------------------------------------------
# gen_kwargs 에서 max_tokens 를 빼면 vLLM 이 "max_model_len - 프롬프트" 만큼을
# 출력 상한으로 잡는다(사실상 무제한). truncation 때문에 hard 점수가 눌렸는지
# 확인/보정용.
#
#   - 출력은 별도 폴더(OUTPUT_DIR, 기본 results_maxtok)에 저장 → 기존 results 안 덮음.
#   - 컨텍스트는 MML(기본 131072 = 128k, 풀 컨텍스트)로 통일. 출력 상한이 사실상
#     사라진다. 대신 어려운 문제에서 매우 길게 생성 → 느리고 KV 메모리 큼.
#     느리거나 OOM 이면 MML 을 낮춰라(예: MML=98304 / 65536).
#
# 사용:
#   bash run/eval/eval_maxtokens.sh gpt-oss                 # 전 45/93 태스크
#   TASKS_ONLY=hard bash run/eval/eval_maxtokens.sh gpt-oss # hard 난이도만(빠름)
#   MML=131072 GPUS=0,1,2,3 bash run/eval/eval_maxtokens.sh gpt-oss gemma
# ============================================================================

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"; cd "$PROJECT_ROOT"

# ---- 설정 (전부 env 로 덮어쓰기 가능) ----
GPUS="${GPUS:-0,1,2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MML="${MML:-100000}"                      # max-model-len, 전 모델 통일(100k).
                                          # 네이티브 128k(gpt-oss/exaone/solar) 안이라 OK.
                                          # gemma 네이티브가 더 작으면 자동 에러 → MML 낮춰 재시도.
VLLM_PORT="${VLLM_PORT:-8020}"
OUTPUT_DIR="${OUTPUT_DIR:-results_maxtok}" # 기존 results 안 덮게 별도
# max_tokens 없음! reasoning on 유지.
GEN_KWARGS="${GEN_KWARGS:-temperature=0.6,top_p=0.95,top_k=20,reasoning=on}"
TASKS_ONLY="${TASKS_ONLY:-all}"           # all | hard  (hard 만 돌리면 빠름/저렴)
SKIP_EXISTING="${SKIP_EXISTING:-true}"
FORCE_RESET="${FORCE_RESET:-1}"
SERVER_START_TIMEOUT=7200
[ -z "$TENSOR_PARALLEL_SIZE" ] && TENSOR_PARALLEL_SIZE=$(echo "$GPUS" | tr ',' '\n' | grep -c .)

DEFAULT_ORDER=(gpt-oss gemma exaone)
if [ "$#" -gt 0 ]; then ORDER=("$@"); else ORDER=("${DEFAULT_ORDER[@]}"); fi

model_name_for(){ case "$1" in
  gpt-oss) echo "openai/gpt-oss-120b";; exaone) echo "LGAI-EXAONE/EXAONE-4.0-32B";;
  gemma) echo "google/gemma-4-31b-it";; solar) echo "upstage/Solar-Open-100B";; *) echo "";; esac; }
extra_vllm_args_for(){ case "$1" in
  solar) echo "--reasoning-parser solar_open --enable-auto-tool-choice --tool-call-parser solar_open --logits-processors vllm.model_executor.models.parallel_tool_call_logits_processor:ParallelToolCallLogitsProcessor --logits-processors vllm.model_executor.models.solar_open_logits_processor:SolarOpenTemplateLogitsProcessor";;
  *) echo "";; esac; }

# 태스크 목록 (공통 45 그리드). TASKS_ONLY=hard 면 hard 만.
ALL=(
 array_formula causal_dag cipher cryptarithmetic ferryman hanoi inequality
)
TASKS=()
for c in "${ALL[@]}"; do for l in en ko; do for d in easy medium hard; do
  if [ "$TASKS_ONLY" = "all" ] || [ "$d" = "$TASKS_ONLY" ]; then TASKS+=("${c}_${l}_${d}"); fi
done; done; done
for d in easy medium hard; do
  if [ "$TASKS_ONLY" = "all" ] || [ "$d" = "$TASKS_ONLY" ]; then TASKS+=("jamo_ko_${d}"); fi
done
# 특정 태스크만 지정하고 싶으면 ONLY_TASKS 로 덮어쓰기(공백 구분).
#   ONLY_TASKS="hanoi_en_hard hanoi_ko_hard" bash run/eval/eval_maxtokens.sh gpt-oss
[ -n "$ONLY_TASKS" ] && TASKS=($ONLY_TASKS)

VENV="$PROJECT_ROOT/.venv/bin/activate"; [ -f "$VENV" ] || { echo -e "${RED}.venv 없음${NC}"; exit 1; }
source "$VENV"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"; export PATH="$CUDA_HOME/bin:$PATH"
VLLM_PID=""

reset_gpus(){ local u=""; for i in $(echo "$1"|tr ',' ' '); do
    x=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null|awk -F', ' -v i="$i" '$1==i{print $2}'); [ -n "$x" ]&&u="$u $x"; done
  echo -e "${YELLOW}[reset] GPU $1 프로세스 정리...${NC}"
  while IFS=',' read -r cu cp; do cu=$(echo "$cu"|xargs);cp=$(echo "$cp"|xargs);[ -z "$cp" ]&&continue
    for z in $u; do [ "$cu" = "$z" ]&&{ echo "  kill $cp"; kill -9 "$cp" 2>/dev/null; }; done
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null); sleep 8; }

start(){ local model=$1 extra=$2
  echo -e "${BLUE}vLLM: ${model} GPU ${GPUS} TP=${TENSOR_PARALLEL_SIZE} max_len=${MML} port=${VLLM_PORT}${NC}"
  setsid env CUDA_VISIBLE_DEVICES=$GPUS python -m vllm.entrypoints.openai.api_server \
    --model "$model" --host 0.0.0.0 --port "$VLLM_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MML" --trust-remote-code $extra 2>&1 &
  VLLM_PID=$!; local w=0
  while [ $w -lt $SERVER_START_TIMEOUT ]; do
    curl -s "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1 && { echo -e "${GREEN}  ready (${w}s)${NC}"; return 0; }
    kill -0 $VLLM_PID 2>/dev/null || { echo -e "${RED}  server died${NC}"; return 1; }
    sleep 5; w=$((w+5)); done; return 1; }
stop(){ [ -n "$VLLM_PID" ]&&kill -0 $VLLM_PID 2>/dev/null&&{ kill -TERM -"$VLLM_PID" 2>/dev/null||kill -TERM "$VLLM_PID" 2>/dev/null
    local w=0; while [ $w -lt 30 ]&&kill -0 $VLLM_PID 2>/dev/null;do sleep 2;w=$((w+2));done
    kill -9 -"$VLLM_PID" 2>/dev/null||kill -9 "$VLLM_PID" 2>/dev/null; wait $VLLM_PID 2>/dev/null||true; }; VLLM_PID=""; }
healthy(){ curl -s --max-time 10 "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; }
trap stop EXIT INT TERM

echo -e "${BLUE}=== MAX-TOKENS(무제한) 재평가 | order:${ORDER[*]} | tasks:${TASKS_ONLY}(${#TASKS[@]}) | out:${OUTPUT_DIR} ===${NC}"
echo -e "gen_kwargs: ${GEN_KWARGS}  (max_tokens 없음 → 컨텍스트 ${MML} 까지)"
[ "$FORCE_RESET" = "1" ] && reset_gpus "$GPUS"

for KEY in "${ORDER[@]}"; do
  MODEL="$(model_name_for "$KEY")"; [ -z "$MODEL" ]&&{ echo -e "${RED}unknown: $KEY${NC}"; continue; }
  EXTRA="$(extra_vllm_args_for "$KEY")"; MDIR="${MODEL//\//_}"
  LOG_DIR="$PROJECT_ROOT/$OUTPUT_DIR/$MDIR/log"; mkdir -p "$LOG_DIR"
  echo -e "${BLUE}==== ${MODEL} ====${NC}"
  start "$MODEL" "$EXTRA" || { echo -e "${RED}start 실패 → skip${NC}"; stop; continue; }
  cur=0; ok=0; fail=0; skip=0
  for task in "${TASKS[@]}"; do
    cur=$((cur+1)); lf="$LOG_DIR/${task}.log"
    if [ "$SKIP_EXISTING" = "true" ] && compgen -G "$PROJECT_ROOT/$OUTPUT_DIR/$MDIR/$task/*.json" >/dev/null 2>&1; then
      echo -e "${GREEN}[$cur/${#TASKS[@]}] skip: $task${NC}"; skip=$((skip+1)); continue; fi
    echo -e "${YELLOW}[$cur/${#TASKS[@]}] $task${NC}"
    healthy || { stop; start "$MODEL" "$EXTRA" || { fail=$((fail+1)); break; }; }
    set +e
    python evaluation/run.py --model "$MODEL" --model_router remote \
      --remote_url "http://localhost:${VLLM_PORT}" --gen-kwargs "$GEN_KWARGS" \
      --tasks "$task" --output-dir "$OUTPUT_DIR" --async --max-concurrent 5 2>&1 | tee -a "$lf"
    [ ${PIPESTATUS[0]} -eq 0 ] && { ok=$((ok+1)); } || { fail=$((fail+1)); }
    set -e
  done
  echo -e "${BLUE}--- ${MODEL}: ok $ok fail $fail skip $skip / ${#TASKS[@]} ---${NC}"
  stop
done
echo -e "${BLUE}=== done → ${OUTPUT_DIR}/ ===${NC}"

# 결과 위치가 다르므로(results_maxtok) 그래프는:
#   python -c "..."  또는 plot 스크립트의 결과 경로를 바꿔서 비교
