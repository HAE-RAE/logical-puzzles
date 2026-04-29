"""
OpenAI Batch API 결과를 SFT 학습 jsonl 로 변환

처리 흐름:
  1. batch_id 로 status 확인 (in_progress 면 중단)
  2. output_file_id 다운로드
  3. 응답에서 <think>...</think> + final answer 추출
  4. assistant content를 `<think>{r}</think>\\n\\nFinal answer: {a}` 로 정규화
  5. drop 통계 + SFT messages jsonl 출력

학습 신호 일관성: 응답이 <think> tag를 포함하지 않으면 본문 전체를 reasoning으로 wrap
                  (단, final answer 추출은 무조건 정규식 매칭 성공해야 함; 실패 시 drop)
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _lib import (
    PROJECT_ROOT,
    THINK_RE,
    FINAL_RE,
    load_jsonl,
    get_openai_client,
    ensure_dotenv,
)

sys.path.insert(0, str(PROJECT_ROOT))
from evaluation.evaluators.array_formula import ArrayFormulaEvaluator
from generation.array_formula_en import puzzle_to_prompt


def index_train(train_rows):
    return {f"naive__{r['id']}": r for r in train_rows} | {
        f"guided__{r['id']}": r for r in train_rows
    }


def parse_response(text: str):
    """returns (reasoning, final_answer) or (None, None) if final answer not found."""
    m_think = THINK_RE.search(text)
    if m_think:
        reasoning = m_think.group(1).strip()
        # final answer는 think 종료 이후에서 검색
        post = text[m_think.end():]
        m_final = FINAL_RE.search(post)
    else:
        # think tag 없음: 마지막 final answer 라인까지를 reasoning, 그 라인을 final로
        m_final = FINAL_RE.search(text)
        if m_final:
            reasoning = text[: m_final.start()].strip()
        else:
            reasoning = None
    if m_final is None:
        return None, None
    final = m_final.group(1).strip()
    return reasoning, final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-file", required=True, help="submit_distill_batch.py 메타")
    parser.add_argument("--train-file", default="data/array_formula_en/train_3k.jsonl")
    parser.add_argument("--out-sft", required=True, help="<mode> SFT messages jsonl")
    parser.add_argument("--out-stats", required=True, help="drop/통계 json")
    parser.add_argument("--raw-output", help="raw batch output 저장 경로(선택)")
    args = parser.parse_args()

    ensure_dotenv()
    client = get_openai_client()

    meta = json.loads((PROJECT_ROOT / args.meta_file).read_text())
    batch_id = meta["batch_id"]
    print(f"[batch] retrieving {batch_id} ...")
    batch = client.batches.retrieve(batch_id)
    print(f"[batch] status={batch.status}  request_counts={batch.request_counts}")

    if batch.status != "completed":
        print(f"[stop] batch not completed yet (status={batch.status}). Re-run later.", file=sys.stderr)
        sys.exit(2)

    out_id = batch.output_file_id
    err_id = batch.error_file_id

    raw = client.files.content(out_id).text
    if args.raw_output:
        rp = PROJECT_ROOT / args.raw_output
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(raw, encoding="utf-8")
        print(f"[raw] saved to {rp}")

    train_rows = load_jsonl(PROJECT_ROOT / args.train_file)
    train_idx = index_train(train_rows)

    ev = ArrayFormulaEvaluator()
    ev._task_name = "array_formula_en"
    base_system = ev.SYSTEM_PROMPT  # student 학습 시 받을 system prompt

    out_sft_path = PROJECT_ROOT / args.out_sft
    out_sft_path.parent.mkdir(parents=True, exist_ok=True)

    drop_reasons = Counter()
    matched_gt = 0
    total = 0
    written = 0

    with out_sft_path.open("w", encoding="utf-8") as fo:
        for line in raw.splitlines():
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            cid = row.get("custom_id")
            puzzle = train_idx.get(cid)
            if puzzle is None:
                drop_reasons["unknown_custom_id"] += 1
                continue

            resp = row.get("response") or {}
            body = resp.get("body") or {}
            choices = body.get("choices") or []
            if not choices:
                drop_reasons["no_choice"] += 1
                continue
            content = (choices[0].get("message") or {}).get("content") or ""
            if not content.strip():
                drop_reasons["empty_content"] += 1
                continue

            reasoning, final = parse_response(content)
            if final is None:
                drop_reasons["no_final_answer"] += 1
                continue
            if reasoning is None or not reasoning.strip():
                drop_reasons["no_reasoning"] += 1
                continue

            # GT 일치 통계 (drop 하지는 않음 — 기존 파이프라인 unfiltered 정책)
            try:
                eq = abs(float(final) - float(puzzle["answer"])) < 1e-6
            except Exception:
                eq = str(final).strip().lower() == str(puzzle["answer"]).strip().lower()
            if eq:
                matched_gt += 1

            assistant_content = f"<think>\n{reasoning}\n</think>\n\nFinal answer: {final}"
            sft_row = {
                "puzzle_id": puzzle["id"],
                "difficulty": puzzle["difficulty"],
                "problem_type": puzzle["type"],
                "gt_answer": puzzle["answer"],
                "teacher_final": final,
                "matched_gt": eq,
                "messages": [
                    {"role": "system", "content": base_system},
                    {"role": "user", "content": puzzle_to_prompt(puzzle)},
                    {"role": "assistant", "content": assistant_content},
                ],
            }
            fo.write(json.dumps(sft_row, ensure_ascii=False) + "\n")
            written += 1

    stats = {
        "batch_id": batch_id,
        "input_total": total,
        "written_sft_rows": written,
        "drop_reasons": dict(drop_reasons),
        "gt_match_count": matched_gt,
        "gt_match_rate": (matched_gt / written) if written else 0.0,
    }
    (PROJECT_ROOT / args.out_stats).write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[stats]", json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"[ok] sft jsonl: {out_sft_path}")


if __name__ == "__main__":
    main()
