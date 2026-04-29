"""
Teacher distillation을 위한 OpenAI Batch API 입력 jsonl 생성

두 mode:
  - naive : system + user(question 만)            → teacher가 자유 풀이
  - guided: system + user(question + GT + solution) → teacher가 GT 기반 reasoning 재서술

응답 format을 통제하기 위해 system prompt 끝에 명시적 지시를 추가한다:
  "Wrap your reasoning in <think>...</think> and end with 'Final answer: X'."
이렇게 해야 student(Qwen3-Thinking)의 inference 출력 분포와 일치하는 학습 신호가 만들어진다.

학습 instance(=question + tables) 단위로 1 응답 호출. n=3000 호출/방식.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _lib import PROJECT_ROOT, THINK_FORMAT_INSTRUCTION

sys.path.insert(0, str(PROJECT_ROOT))
from evaluation.evaluators.array_formula import ArrayFormulaEvaluator
from generation.array_formula_en import puzzle_to_prompt

GUIDED_SUFFIX = (
    "\n\n### Reference solution outline (intermediate steps only — final number deliberately omitted)\n"
    "Use the following step-by-step outline as a structural guide for your reasoning. "
    "Re-express it as your own thorough chain-of-thought inside <think>...</think>, "
    "performing each calculation yourself in detail, then derive the final answer.\n"
    "{solution}"
)


def strip_final_from_solution(sol: str) -> str:
    """solution log에서 마지막 'Final answer:' 라인을 제거."""
    import re as _re
    return _re.sub(
        r"\n?\s*Final\s*answer\s*[:：].*$", "", sol, flags=_re.IGNORECASE | _re.DOTALL
    ).rstrip()


def build_request(custom_id: str, model: str, system: str, user: str,
                  max_completion_tokens: int) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": max_completion_tokens,
        },
    }


def load_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["naive", "guided"], required=True)
    parser.add_argument("--train-file", default="data/array_formula_en/train_3k.jsonl")
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    args = parser.parse_args()

    train_path = PROJECT_ROOT / args.train_file
    out_path = PROJECT_ROOT / args.out_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(train_path)
    ev = ArrayFormulaEvaluator()
    ev._task_name = "array_formula_en"
    base_system = ev.SYSTEM_PROMPT  # student와 동일한 prompt
    system = base_system + THINK_FORMAT_INSTRUCTION

    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            custom_id = f"{args.mode}__{row['id']}"
            base_user = puzzle_to_prompt(row)  # tables + question 통합 (student inference와 동일)
            if args.mode == "naive":
                user = base_user
            else:  # guided
                user = base_user + GUIDED_SUFFIX.format(
                    solution=strip_final_from_solution(str(row.get("solution", "")).strip()),
                )
            req = build_request(
                custom_id=custom_id,
                model=args.model,
                system=system,
                user=user,
                max_completion_tokens=args.max_completion_tokens,
            )
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[ok] wrote {n_written} requests to {out_path}")


if __name__ == "__main__":
    main()
