"""
GPT-4o Distillation 데이터 생성 (OpenAI Batch API)

두 가지 방식으로 teacher CoT를 생성한다:
  - naive:  문제만 제공 → GPT-4o가 자유롭게 풀이
  - guided: 문제 + solution log 제공 → GPT-4o가 로그 기반 자연어 CoT 생성

OpenAI Batch API를 사용하여 50% 할인된 비용으로 비동기 처리한다.
"""

import argparse
import json
import re
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("openai 패키지가 필요합니다: pip install openai")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Evaluator SYSTEM_PROMPT (평가 시 student가 받는 것과 동일) ──────────────
EVAL_SYSTEM_PROMPTS = {
    "ferryman_en": """### Instructions
You are an expert at solving boat navigation problems.

### Rules
1. Analyze all given navigation regulations step by step.
2. Apply all speed limits, mandatory rest stops, and cargo regulations in your calculations.
3. Explain your reasoning clearly, then present your final conclusion in the format below.

### Output format
End with $\\boxed{X hours Y minutes}$.""",

    "hanoi_en": """### Instructions
You must answer ONLY in the format (disk, from, to).

### Rules
Follow the Hanoi puzzle given in the user message.

### Output format
(disk, from, to) — e.g. (1, 0, 2)""",

    "array_formula_en": """### Instructions
You are a spreadsheet/Excel expert.
Analyze the given table data and answer the question accurately.

### Rules
1. For numeric results, answer with only the number (no units, commas, or currency symbols)
2. For decimals, truncate unless otherwise specified
3. For text answers, provide the exact value only
4. Briefly explain your reasoning, then end with "Final answer: [answer]"

### Output format
End your response with a line: Final answer: [answer]""",

    "yacht_dice_en": """### Instructions
You are an expert at solving Yacht Dice optimization problems.

### Rules
Yacht Dice is a dice game where you roll 5 dice for 12 rounds and assign each round to a scoring category.

Scoring Categories:
- Aces through Sixes: Sum of dice showing that number
- Three-of-a-Kind: Sum of all dice if at least 3 match
- Four-of-a-Kind: Sum of all dice if at least 4 match
- Full House: 25 points for exactly 3 of one number and 2 of another
- Small Straight: 30 points for 4 consecutive numbers
- Large Straight: 40 points for 5 consecutive numbers
- Yacht: 50 points for all 5 dice showing the same number

Upper Section Bonus: If the sum of Aces through Sixes is 63 or more, add 35 bonus points.

Each category can only be used once.

### Output format
CRITICAL: Your very last line MUST be in this exact format:
Answer: [number]""",
}

# ── Guided Distillation 용 시스템 프롬프트 ────────────────────────────────
GUIDED_SYSTEM_PROMPT = """You are an expert puzzle solver creating educational step-by-step solutions.
You are given a problem and its verified solution log.
Your job is to produce a natural, human-readable chain-of-thought explanation
that follows the solution log's reasoning closely.

IMPORTANT RULES:
- Do NOT deviate from the solution log's calculations or final answer.
- Your explanation must arrive at exactly the same answer as the ground truth.
- Write as if you are solving the problem yourself, but use the solution log as your guide.
- End with the exact answer in the format specified by the problem."""


# ── Answer 파싱 (evaluator 로직 재사용) ──────────────────────────────────
def parse_answer_ferryman(response: str):
    """X hours Y minutes → 'X hours Y minutes'

    LaTeX 표기(`\\text{...}`, `\\,`)를 먼저 제거해 `\\boxed{6 \\text{ hours } 21 \\text{ minutes}}`
    같이 boxed 내부에 중첩 괄호가 있는 형태도 파싱되도록 한다.
    """
    cleaned = re.sub(r'\\text\s*\{\s*([^{}]*?)\s*\}', r'\1', response)
    cleaned = re.sub(r'\\[,;!]', ' ', cleaned)

    m = re.search(
        r'\\boxed\{\s*(\d+)\s*hours?\s*(?:and\s+)?(\d+)\s*minutes?\s*\}',
        cleaned, re.IGNORECASE)
    if m:
        return f"{int(m.group(1))} hours {int(m.group(2))} minutes"

    # fallback: "X hours [and] Y minutes" 패턴
    m = re.search(
        r'(\d+)\s*hours?\s*(?:and\s+)?(\d+)\s*minutes?',
        cleaned, re.IGNORECASE)
    if m:
        return f"{int(m.group(1))} hours {int(m.group(2))} minutes"
    return None


def parse_answer_hanoi(response: str):
    """(disk, from, to) 튜플 — 마지막 매칭 우선 (CoT 중간 숫자 오파싱 방지)"""
    matches = re.findall(r'\((\d+),\s*(\d+),\s*(\d+)\)', response)
    if matches:
        last = matches[-1]
        return f"({last[0]}, {last[1]}, {last[2]})"
    return None


def parse_answer_array_formula(response: str):
    """Final answer: [value]"""
    patterns = [
        r"[Ff]inal\s*[Aa]nswer\s*[:：]\s*(.+?)(?:\n|$)",
        r"[Aa]nswer\s*[:：]\s*(.+?)(?:\n|$)",
    ]
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def parse_answer_yacht_dice(response: str):
    """Answer: [number]"""
    matches = re.findall(
        r'(?:Answer|Output|Final\s*Answer)\s*[:\s]*(\d+)',
        response, re.IGNORECASE)
    if matches:
        return matches[-1]
    return None


ANSWER_PARSERS = {
    "ferryman_en": parse_answer_ferryman,
    "hanoi_en": parse_answer_hanoi,
    "array_formula_en": parse_answer_array_formula,
    "yacht_dice_en": parse_answer_yacht_dice,
}


def normalize_answer(answer):
    """답변을 비교 가능한 문자열로 정규화"""
    s = str(answer).strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s


def validate_answer(task_name: str, parsed: str, ground_truth: str) -> bool:
    """파싱된 답변과 ground truth 비교"""
    if parsed is None:
        return False
    norm_parsed = normalize_answer(parsed)
    norm_gt = normalize_answer(ground_truth)

    # 직접 비교
    if norm_parsed == norm_gt:
        return True

    # 숫자 비교 (array_formula, yacht_dice)
    try:
        return abs(float(norm_parsed) - float(norm_gt)) < 0.01
    except (ValueError, TypeError):
        pass

    # ferryman: "X hours Y minutes" 비교
    if task_name == "ferryman_en":
        def extract_hm(s):
            m = re.search(r'(\d+)\s*hours?\s+(\d+)\s*minutes?', s, re.IGNORECASE)
            return (int(m.group(1)), int(m.group(2))) if m else None
        hm_parsed = extract_hm(norm_parsed)
        hm_gt = extract_hm(norm_gt)
        if hm_parsed and hm_gt:
            return hm_parsed == hm_gt

    return False


# ── Batch API 요청 생성 ──────────────────────────────────────────────────
def build_batch_request(custom_id: str, model: str, messages: list,
                        temperature: float, max_tokens: int) -> dict:
    """OpenAI Batch API 형식의 요청 한 건"""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    }


def create_naive_request(puzzle: dict, task_name: str, model: str,
                         temperature: float, max_tokens: int) -> dict:
    system_prompt = EVAL_SYSTEM_PROMPTS[task_name]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": puzzle["question"]},
    ]
    custom_id = f"naive__{task_name}__{puzzle['id']}"
    return build_batch_request(custom_id, model, messages, temperature, max_tokens)


TASK_OUTPUT_FORMATS = {
    "ferryman_en": "End with $\\boxed{X hours Y minutes}$.",
    "hanoi_en": "End with the answer in format: (disk, from, to)",
    "array_formula_en": "End with: Final answer: [answer]",
    "yacht_dice_en": "End with: Answer: [number]",
}


def create_guided_request(puzzle: dict, task_name: str, model: str,
                          temperature: float, max_tokens: int) -> dict:
    output_fmt = TASK_OUTPUT_FORMATS.get(task_name, "")
    user_content = f"""## Problem
{puzzle['question']}

## Verified Solution Log
{puzzle['solution']}

## Ground Truth Answer
{puzzle['answer']}

Rewrite the solution log as a clear, natural chain-of-thought explanation.
Your explanation must arrive at exactly this answer: {puzzle['answer']}
{output_fmt}"""

    messages = [
        {"role": "system", "content": GUIDED_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    custom_id = f"guided__{task_name}__{puzzle['id']}"
    return build_batch_request(custom_id, model, messages, temperature, max_tokens)


# ── Batch 실행 / 폴링 ───────────────────────────────────────────────────
def upload_and_submit_batch(client: OpenAI, batch_input_path: Path,
                            description: str) -> str:
    """batch input 파일 업로드 → batch 생성 → batch_id 반환"""
    print(f"Uploading batch input: {batch_input_path}")
    with open(batch_input_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    print(f"  File ID: {file_obj.id}")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )
    print(f"  Batch ID: {batch.id} (status: {batch.status})")
    return batch.id


def poll_batch(client: OpenAI, batch_id: str, interval: int = 30) -> dict:
    """batch 완료까지 폴링"""
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        completed = batch.request_counts.completed if batch.request_counts else 0
        total = batch.request_counts.total if batch.request_counts else 0
        print(f"  [{time.strftime('%H:%M:%S')}] {status} — {completed}/{total}")

        if status in ("completed", "failed", "expired", "cancelled"):
            return batch
        time.sleep(interval)


def download_results(client: OpenAI, batch, output_path: Path) -> list:
    """batch 결과 다운로드 → JSON 리스트 반환"""
    if batch.status != "completed":
        print(f"Batch not completed: {batch.status}")
        if batch.error_file_id:
            errors = client.files.content(batch.error_file_id).text
            error_path = output_path.parent / "batch_errors.jsonl"
            error_path.write_text(errors)
            print(f"  Errors saved: {error_path}")
        return []

    content = client.files.content(batch.output_file_id).text
    output_path.write_text(content)
    print(f"  Results saved: {output_path}")

    results = []
    for line in content.strip().split("\n"):
        if line.strip():
            results.append(json.loads(line))
    return results


# ── 결과 처리 ────────────────────────────────────────────────────────────
def process_results(results: list, puzzles_by_id: dict, no_filter: bool = False):
    """batch 결과를 파싱하여 teacher_data + raw_responses 생성

    no_filter=False (기본): validated(GT 일치) 샘플만 teacher_data에 포함
    no_filter=True: 파싱 성공한 모든 샘플(validated + wrong_answer) 포함.
                    parse_failed(파서가 답을 못 뽑은 경우)는 항상 제외.

    teacher_record에는 항상 `validated`, `parsed_answer`를 함께 기록해
    사후에 필터링을 다시 걸 수 있도록 한다.
    """
    teacher_data = []
    raw_responses = []
    stats = {"total": 0, "validated": 0, "rejected": 0, "parse_failed": 0,
             "included": 0}

    for result in results:
        custom_id = result["custom_id"]
        parts = custom_id.split("__", 2)
        if len(parts) != 3:
            continue
        method, task_name, puzzle_id = parts

        response_body = result.get("response", {}).get("body", {})
        choices = response_body.get("choices", [])
        if not choices:
            stats["parse_failed"] += 1
            continue

        assistant_text = choices[0].get("message", {}).get("content", "")
        puzzle = puzzles_by_id.get(f"{task_name}__{puzzle_id}")
        if puzzle is None:
            continue

        stats["total"] += 1
        ground_truth = str(puzzle["answer"])

        # 답변 파싱 및 검증
        parser = ANSWER_PARSERS.get(task_name)
        parsed = parser(assistant_text) if parser else None
        is_valid = validate_answer(task_name, parsed, ground_truth)

        raw_record = {
            "id": puzzle_id,
            "task": task_name,
            "method": method,
            "ground_truth": ground_truth,
            "parsed_answer": str(parsed) if parsed else None,
            "validated": is_valid,
            "response": assistant_text,
        }
        raw_responses.append(raw_record)

        if is_valid:
            stats["validated"] += 1
        else:
            stats["rejected"] += 1
            if parsed is None:
                stats["parse_failed"] += 1

        # 포함 조건: parse_failed 제외 + (validated 또는 no_filter)
        include = (parsed is not None) and (is_valid or no_filter)
        if include:
            teacher_record = {
                "task": task_name,
                "query": puzzle["question"],
                "answer": assistant_text,
                "id": puzzle_id,
                "difficulty": puzzle.get("difficulty", "unknown"),
                "validated": is_valid,
                "parsed_answer": str(parsed) if parsed else None,
            }
            teacher_data.append(teacher_record)
            stats["included"] += 1

    return teacher_data, raw_responses, stats


def save_teacher_data(teacher_data: list, raw_responses: list, stats: dict,
                      method_dir: Path):
    """teacher_data를 combined + per-task 파일로 저장"""
    method_dir.mkdir(parents=True, exist_ok=True)

    # combined
    teacher_path = method_dir / "teacher_data.jsonl"
    with open(teacher_path, "w", encoding="utf-8") as f:
        for item in teacher_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # per-task
    by_task = {}
    for item in teacher_data:
        by_task.setdefault(item["task"], []).append(item)

    for task_name, items in by_task.items():
        task_path = method_dir / f"teacher_data_{task_name}.jsonl"
        with open(task_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  {task_name}: {len(items)} validated samples -> {task_path.name}")

    # raw responses
    raw_path = method_dir / "raw_responses.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for item in raw_responses:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  Combined: {teacher_path} ({len(teacher_data)} total)")
    return by_task


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate distillation data via OpenAI Batch API")
    parser.add_argument("--tasks", nargs="+",
                        default=["ferryman_en", "hanoi_en", "array_formula_en", "yacht_dice_en"])
    parser.add_argument("--data-dir", default="data/distill/split/train")
    parser.add_argument("--output-dir", default="data/distill")
    parser.add_argument("--methods", nargs="+", default=["naive", "guided"],
                        choices=["naive", "guided"])
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Batch polling interval in seconds")
    # 결과 처리 전용 모드 (이미 batch가 완료된 경우)
    parser.add_argument("--process-only", type=str, default=None,
                        help="Skip batch creation; process existing batch output file")
    # batch 복구 모드 (서버에 남아있는 batch 결과를 다운로드)
    parser.add_argument("--recover-batch", type=str, nargs="+", default=None,
                        metavar="BATCH_ID",
                        help="Recover batch results from OpenAI server. "
                             "Provide batch IDs (e.g., --recover-batch batch_xxx batch_yyy)")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable GT validation filter. Includes parsed "
                             "but incorrect responses (wrong_answer) alongside "
                             "validated ones. parse_failed is still excluded. "
                             "Outputs to data/distill/{method}_unfiltered/ "
                             "to keep the filtered artifacts separate.")
    args = parser.parse_args()

    # no_filter일 때 출력 디렉토리 접미사
    dir_suffix = "_unfiltered" if args.no_filter else ""

    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 퍼즐 로드 ──
    puzzles_by_id = {}
    all_puzzles = {}  # task_name → list of puzzles

    for task_name in args.tasks:
        jsonl_path = data_dir / f"{task_name}.jsonl"
        if not jsonl_path.exists():
            print(f"[SKIP] {jsonl_path} not found")
            continue
        items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    items.append(item)
                    puzzles_by_id[f"{task_name}__{item['id']}"] = item
        all_puzzles[task_name] = items
        print(f"Loaded {len(items)} puzzles for {task_name}")

    if not all_puzzles:
        print("No puzzles loaded. Check --data-dir and --tasks.")
        return

    # ── process-only 모드 ──
    if args.process_only:
        print(f"\n=== Processing existing batch output: {args.process_only} ===")
        raw_results = []
        with open(args.process_only, "r") as f:
            for line in f:
                if line.strip():
                    raw_results.append(json.loads(line))

        for method in args.methods:
            method_results = [r for r in raw_results if r["custom_id"].startswith(f"{method}__")]
            teacher_data, raw_responses, stats = process_results(
                method_results, puzzles_by_id, no_filter=args.no_filter)

            method_dir = output_dir / f"{method}{dir_suffix}"
            print(f"\n[{method}{dir_suffix}] Stats: {json.dumps(stats, indent=2)}")
            save_teacher_data(teacher_data, raw_responses, stats, method_dir)

        return

    # ── recover-batch 모드: 서버에서 기존 batch 결과 다운로드 ──
    if args.recover_batch:
        client = OpenAI()
        all_results = []

        for batch_id in args.recover_batch:
            print(f"\n=== Recovering batch: {batch_id} ===")
            batch = client.batches.retrieve(batch_id)
            print(f"  Status: {batch.status}, "
                  f"completed: {batch.request_counts.completed}/{batch.request_counts.total}")

            if batch.status != "completed":
                print(f"  Batch not completed yet. Polling...")
                batch = poll_batch(client, batch_id, interval=args.poll_interval)

            if batch.status == "completed":
                # method를 batch의 첫 결과에서 추론
                batch_output_path = output_dir / f"batch_output_recovered_{batch_id[-8:]}.jsonl"
                results = download_results(client, batch, batch_output_path)
                all_results.extend(results)
                print(f"  Downloaded {len(results)} results")
            else:
                print(f"  Batch failed: {batch.status}")

        if all_results:
            for method in args.methods:
                method_results = [r for r in all_results
                                  if r["custom_id"].startswith(f"{method}__")]
                if not method_results:
                    continue
                teacher_data, raw_responses, stats = process_results(
                    method_results, puzzles_by_id, no_filter=args.no_filter)
                method_dir = output_dir / f"{method}{dir_suffix}"
                print(f"\n[{method}{dir_suffix}] Stats: {json.dumps(stats, indent=2)}")
                save_teacher_data(teacher_data, raw_responses, stats, method_dir)

        return

    # ── Batch 요청 생성 ──
    client = OpenAI()  # OPENAI_API_KEY env var

    for method in args.methods:
        print(f"\n{'='*60}")
        print(f"  Method: {method}")
        print(f"{'='*60}")

        batch_requests = []
        for task_name, puzzles in all_puzzles.items():
            for puzzle in puzzles:
                if method == "naive":
                    req = create_naive_request(puzzle, task_name, args.model,
                                               args.temperature, args.max_tokens)
                else:
                    req = create_guided_request(puzzle, task_name, args.model,
                                                args.temperature, args.max_tokens)
                batch_requests.append(req)

        print(f"Total requests: {len(batch_requests)}")

        # batch input 파일 작성
        batch_input_path = output_dir / f"batch_input_{method}.jsonl"
        with open(batch_input_path, "w", encoding="utf-8") as f:
            for req in batch_requests:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
        print(f"Batch input saved: {batch_input_path}")

        # batch 업로드 & 제출
        batch_id = upload_and_submit_batch(
            client, batch_input_path,
            description=f"distillation_{method}_{args.model}")

        # 폴링
        print(f"\nPolling batch {batch_id}...")
        batch = poll_batch(client, batch_id, interval=args.poll_interval)

        # 결과 다운로드
        batch_output_path = output_dir / f"batch_output_{method}.jsonl"
        raw_results = download_results(client, batch, batch_output_path)

        if not raw_results:
            print(f"[{method}] No results. Check batch status.")
            continue

        # 결과 처리
        teacher_data, raw_responses, stats = process_results(
            raw_results, puzzles_by_id, no_filter=args.no_filter)

        method_dir = output_dir / f"{method}{dir_suffix}"
        print(f"\n[{method}{dir_suffix}] Stats: {json.dumps(stats, indent=2)}")
        save_teacher_data(teacher_data, raw_responses, stats, method_dir)

    # ── 전체 통계 저장 ──
    print("\nDone!")


if __name__ == "__main__":
    main()
