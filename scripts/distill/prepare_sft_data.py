"""
SFT 데이터 전처리 스크립트

Teacher 모델이 생성한 데이터(task/query/answer)를 로드하여:
1. task별 evaluator SYSTEM_PROMPT 매핑
2. chat message format 변환
3. 80/20 train/test split
4. train.jsonl, per-task test eval 파일 저장
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _lib import PROJECT_ROOT

# evaluator에서 사용하는 SYSTEM_PROMPT 매핑 (task_name → prompt)
SYSTEM_PROMPTS = {
    "cryptarithmetic": """You are an expert puzzle solver specializing in cryptarithmetic problems.

Rules:
- Each letter represents a unique digit (0-9)
- Different letters must map to different digits
- Leading letters cannot be zero
- '*' represents an unknown letter that could be any letter

Solve the puzzle and provide your answer in this exact format:
Answer: [number]""",

    "inequality": """You are an expert puzzle solver specializing in logical constraint puzzles.

Solve the inequality puzzle by filling blanks with numbers.
Each number must be used exactly once.
Inequality symbols (< or >) between positions must be satisfied.

Provide your answer in this exact format:
Answer: [numbers separated by spaces]""",

    "minesweeper": """You are solving a Minesweeper puzzle. Analyze the grid using logical reasoning and deduce the exact location of all mines.

Output the sum of linear indices (row * columns + col) for all mine positions as a single integer.

Answer: [number]""",

    "number_baseball": """You are an expert puzzle solver specializing in logical deduction games like Bulls and Cows (Number Baseball).

Rules:
- "Strike" means a digit is correct AND in the correct position
- "Ball" means a digit is correct BUT in the wrong position
- Find the secret number that satisfies ALL hints

Provide your answer in this exact format:
Answer: [the secret number]""",

    "sudoku": """You are a logic puzzle expert specializing in Sudoku.

Solve the Sudoku puzzle following standard rules:
- Each row must contain digits 1-9 exactly once
- Each column must contain digits 1-9 exactly once
- Each 3x3 box must contain digits 1-9 exactly once

Provide your answer in the exact format requested.""",

    "yacht_dice": """You are an expert at solving Yacht Dice optimization problems.

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

CRITICAL: Your very last line MUST be in this exact format:
Answer: [number]""",

    "ferryman": """### Instructions
You are an expert at solving boat navigation problems.

### Rules
1. Analyze all given navigation regulations step by step.
2. Apply all speed limits, mandatory rest stops, and cargo regulations in your calculations.
3. Explain your reasoning clearly, then present your final conclusion in the format below.

### Output format
End with $\\boxed{X hours Y minutes}$.""",

    "hanoi": """### Instructions
You must answer ONLY in the format (disk, from, to).

### Rules
Follow the Hanoi puzzle given in the user message.

### Output format
(disk, from, to) — e.g. (1, 0, 2)""",

    "array_formula": """### Instructions
You are a spreadsheet/Excel expert.
Analyze the given table data and answer the question accurately.

### Rules
1. For numeric results, answer with only the number (no units, commas, or currency symbols)
2. For decimals, truncate unless otherwise specified
3. For text answers, provide the exact value only
4. Briefly explain your reasoning, then end with "Final answer: [answer]"

### Output format
End your response with a line: Final answer: [answer]""",
}

# _ko 및 _en 태스크는 동일한 system prompt 사용
for base in list(SYSTEM_PROMPTS.keys()):
    SYSTEM_PROMPTS[f"{base}_ko"] = SYSTEM_PROMPTS[base]
    SYSTEM_PROMPTS[f"{base}_en"] = SYSTEM_PROMPTS[base]


def load_jsonl(path: Path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # 필수 필드 검증
            for field in ("task", "query", "answer"):
                if field not in item:
                    raise ValueError(f"Line {line_num}: missing required field '{field}'")
            items.append(item)
    return items


# ── <think> wrap 용 task별 최종답변 split 패턴 ─────────────────────────────
# answer 텍스트에서 "최종 답변 라인"의 시작 위치를 찾기 위한 정규식.
# 매칭되는 가장 뒤쪽(last)을 기준으로 split한다.
FINAL_ANSWER_SPLIT_PATTERNS = {
    # ferryman: LaTeX \boxed{... hours ... minutes} — 마지막 \boxed 기준
    "ferryman_en": re.compile(
        r"\\?\$?\\?\[?\s*\\boxed\{[^{}]*(?:\{[^}]*\}[^{}]*)*\}", re.IGNORECASE),
    # array_formula: "Final answer: X" — 마지막 라인 기준
    "array_formula_en": re.compile(
        r"(?:Final\s*[Aa]nswer|[Aa]nswer)\s*[:：]\s*.*",
    ),
    # hanoi: 별도 처리 (split_hanoi 함수 참조). 위 패턴 사용 안 함.
}

# hanoi 전용: (N, N, N) 튜플 라인 매칭 (라인 기준)
_HANOI_TUPLE_LINE = re.compile(r"^[\s]*\(?\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)?\s*[.;]?\s*$")


def build_final_line(task: str, parsed_answer: str) -> str:
    """parsed_answer로 task별 정규 최종답변 라인 재조립"""
    if task == "ferryman_en":
        return f"\\boxed{{{parsed_answer}}}"
    if task == "hanoi_en":
        # parsed_answer already "(d, f, t)"
        return parsed_answer
    if task == "array_formula_en":
        return f"Final answer: {parsed_answer}"
    return parsed_answer  # fallback


def _split_hanoi(answer: str, parsed_answer: str):
    """hanoi 전용 split — reasoning 본문이 손상되지 않도록 보수적으로 분리.

    teacher 응답 안에 (N, N, N) 형식 move 튜플이 reasoning 중간에 다수 등장.
    원본 정규식은 "마지막 매치"로 split하여 reasoning이 거의/전부 잘려나가는 문제.

    개선 전략:
    1) "Final answer:" / "Therefore" / "Thus" / "answer is" 등의 마커 뒤
       나오는 마지막 튜플을 우선 split point로 사용.
    2) 마커가 없으면 마지막 줄(line) 단위로 검사하여 "튜플만 있는 라인"의 경우만 split.
    3) 그것도 없으면 raw answer를 reasoning으로 두고 final_line은 parsed_answer로 부착.
    """
    final_line = build_final_line("hanoi_en", parsed_answer)

    # 1) 마커 기반 split
    marker_pat = re.compile(
        r"(?:Final\s*[Aa]nswer|Therefore|Thus|[Aa]nswer\s+is|[Aa]nswer\s*[:：])",
    )
    marker_matches = list(marker_pat.finditer(answer))
    if marker_matches:
        marker_pos = marker_matches[-1].start()
        # 마커 앞쪽을 reasoning으로 사용
        reasoning = answer[:marker_pos].rstrip()
        if reasoning:
            return reasoning, final_line

    # 2) 라인 단위: 마지막에 "튜플만 있는 라인"이 있으면 그 위까지를 reasoning으로
    lines = answer.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if _HANOI_TUPLE_LINE.match(lines[i]):
            reasoning = "\n".join(lines[:i]).rstrip()
            if reasoning:
                return reasoning, final_line
            break

    # 3) Fallback: reasoning = 전체 answer, final_line은 별도 부착
    return answer.rstrip(), final_line


def split_reasoning_and_final(answer: str, task: str, parsed_answer: str):
    """teacher answer를 (reasoning, final_line)으로 분리.

    - answer 내 task별 최종답변 패턴의 "마지막" 매치를 기준으로 split
    - reasoning = 마지막 매치 이전 + 이후 텍스트(해설 꼬리)는 버림
    - final_line은 parsed_answer로 재조립해 형식 통일
    - 패턴을 찾지 못하면 reasoning=answer 전체로 대체 (fallback)
    - hanoi는 전용 처리 (_split_hanoi)로 reasoning이 빈 문자열이 되는 문제 회피
    """
    if task == "hanoi_en":
        return _split_hanoi(answer, parsed_answer)

    pattern = FINAL_ANSWER_SPLIT_PATTERNS.get(task)
    if pattern is None:
        return answer.rstrip(), build_final_line(task, parsed_answer)

    matches = list(pattern.finditer(answer))
    if not matches:
        return answer.rstrip(), build_final_line(task, parsed_answer)

    last = matches[-1]
    reasoning = answer[: last.start()].rstrip()
    final_line = build_final_line(task, parsed_answer)
    return reasoning, final_line


def to_chat_format(item: dict, think_wrap: bool = False) -> dict:
    """Teacher 데이터를 chat message format으로 변환.

    think_wrap=True 일 때 assistant content를
        <think>\n{reasoning}\n</think>\n\n{final_line}
    형식으로 재구성한다 (옵션 B, parsed-split).
    """
    task = item["task"]
    system_prompt = SYSTEM_PROMPTS.get(task)
    if system_prompt is None:
        raise ValueError(f"Unknown task '{task}'. Available: {list(SYSTEM_PROMPTS.keys())}")

    if think_wrap:
        parsed_answer = item.get("parsed_answer")
        if not parsed_answer:
            # parsed_answer 없으면 wrap 불가 → raw 유지
            assistant_content = item["answer"]
        else:
            reasoning, final_line = split_reasoning_and_final(
                item["answer"], task, str(parsed_answer))
            assistant_content = f"<think>\n{reasoning}\n</think>\n\n{final_line}"
    else:
        assistant_content = item["answer"]

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["query"]},
            {"role": "assistant", "content": assistant_content},
        ],
        "task": task,
        "id": item.get("id", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data from teacher model JSONL")
    parser.add_argument("--input", required=True, help="Input teacher data JSONL path (task/query/answer format)")
    parser.add_argument("--output-dir", default="data/sft", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--think-wrap", action="store_true",
                        help="Wrap teacher reasoning inside <think>...</think> blocks "
                             "and append a normalized final-answer line built from "
                             "parsed_answer (option B, parsed-split).")
    args = parser.parse_args()

    random.seed(args.seed)

    input_path = PROJECT_ROOT / args.input
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 로드
    print(f"Loading teacher data from {input_path}...")
    items = load_jsonl(input_path)
    print(f"Loaded {len(items)} samples")

    # task별 그룹핑
    by_task: dict[str, list] = {}
    for item in items:
        by_task.setdefault(item["task"], []).append(item)

    all_train = []
    # evaluation/run.py 호환용 per-task test 파일 (원본 query/answer 포맷)
    test_eval_dir = output_dir / "test_eval"
    test_eval_dir.mkdir(parents=True, exist_ok=True)
    stats = {}

    for task_name, task_items in sorted(by_task.items()):
        # shuffle & split
        random.shuffle(task_items)
        split_idx = int(len(task_items) * args.train_ratio)
        train_raw = task_items[:split_idx]
        test_raw = task_items[split_idx:]

        # chat format 변환 (train만)
        train_chat = [to_chat_format(it, think_wrap=args.think_wrap) for it in train_raw]
        all_train.extend(train_chat)

        # per-task test eval 파일 저장 (evaluation/run.py가 읽는 형식)
        # evaluation/run.py는 {task_name}.jsonl에서 id, question, answer, difficulty 등을 읽음
        eval_path = test_eval_dir / f"{task_name}.jsonl"
        with open(eval_path, "w", encoding="utf-8") as f:
            for item in test_raw:
                eval_record = {
                    "id": item.get("id", ""),
                    "question": item["query"],
                    "answer": item.get("answer", ""),
                    "difficulty": item.get("difficulty", "unknown"),
                }
                f.write(json.dumps(eval_record, ensure_ascii=False) + "\n")

        stats[task_name] = {
            "total": len(task_items),
            "train": len(train_raw),
            "test": len(test_raw),
        }
        print(f"[OK] {task_name}: {len(task_items)} total → {len(train_raw)} train / {len(test_raw)} test")

    # 전체 train 셔플
    random.shuffle(all_train)

    # 저장
    train_path = output_dir / "train.jsonl"
    stats_path = output_dir / "split_stats.json"

    with open(train_path, "w", encoding="utf-8") as f:
        for item in all_train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_train": len(all_train),
            "total_test": sum(s["test"] for s in stats.values()),
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "per_task": stats,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {train_path} ({len(all_train)} samples)")
    print(f"Saved: {test_eval_dir}/ (per-task test files for evaluation/run.py)")
    print(f"Saved: {stats_path}")


if __name__ == "__main__":
    main()
