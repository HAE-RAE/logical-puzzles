"""
Batch 본 호출 전 비용 견적 (옵션 B)

각 mode 10건 sample 호출 → 응답/사용 토큰 측정 → 6 000건 환산 견적.
sync API 호출 (Batch SLA 24h 우회). reasoning model이면 reasoning_tokens 별도 보고.
"""

import json
import statistics
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluators.array_formula import ArrayFormulaEvaluator
from generation.array_formula_en import puzzle_to_prompt

# prepare_distill_batch.py 와 동일 instruction
THINK_FORMAT_INSTRUCTION = (
    "\n\n### Response format (STRICTLY MANDATORY — your response is invalid otherwise)\n"
    "You MUST produce an extensive, thorough chain-of-thought:\n"
    "  - Identify EVERY row from EVERY table that is relevant to the question.\n"
    "  - Show EVERY arithmetic step explicitly (do not skip intermediate calculations).\n"
    "  - State each filtering or grouping criterion before applying it.\n"
    "  - Aim for a detailed, fully verifiable trace (typically 500-1500 reasoning tokens). "
    "A short answer is INSUFFICIENT — show your work.\n"
    "Even if a reference answer or solution log is provided in the user message, "
    "you MUST STILL produce the full step-by-step <think> trace. "
    "Responses without proper <think>...</think> reasoning are REJECTED.\n"
    "\n"
    "Output structure (exactly):\n"
    "<think>\n"
    "  ...detailed multi-step reasoning here, listing each row, each calculation...\n"
    "</think>\n"
    "Final answer: <the answer>\n"
    "Do not write anything after this line."
)

GUIDED_SUFFIX = (
    "\n\n### Reference solution outline (intermediate steps only — final number deliberately omitted)\n"
    "Use the following step-by-step outline as a structural guide for your reasoning. "
    "Re-express it as your own thorough chain-of-thought inside <think>...</think>, "
    "performing each calculation yourself in detail, then derive the final answer.\n"
    "{solution}"
)


def strip_final_from_solution(sol: str) -> str:
    """solution log에서 마지막 'Final answer:' 라인을 제거해 GT 직접 노출 방지."""
    import re as _re
    return _re.sub(
        r"\n?\s*Final\s*answer\s*[:：].*$", "", sol, flags=_re.IGNORECASE | _re.DOTALL
    ).rstrip()

MODEL = "gpt-5.4-mini"
N_PER_MODE = 10
MAX_COMPLETION = 8192


def main():
    load_dotenv()
    client = OpenAI()

    train = [json.loads(l) for l in (PROJECT_ROOT / "data/array_formula_en/train_3k.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    sample = train[:N_PER_MODE]

    ev = ArrayFormulaEvaluator()
    ev._task_name = "array_formula_en"
    system = ev.SYSTEM_PROMPT + THINK_FORMAT_INSTRUCTION

    results = {}
    samples_dump = {}
    for mode in ["naive", "guided"]:
        in_tok, out_tok, reasoning_tok, total_tok = [], [], [], []
        gt_match = 0
        bodies = []
        for i, p in enumerate(sample):
            base_user = puzzle_to_prompt(p)
            user = base_user if mode == "naive" else (
                base_user + GUIDED_SUFFIX.format(
                    solution=strip_final_from_solution(str(p.get("solution","")).strip()),
                )
            )
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_completion_tokens=MAX_COMPLETION,
                )
            except Exception as e:
                print(f"[{mode}] {i+1}/{N_PER_MODE} ERROR: {e}", file=sys.stderr)
                continue
            usage = resp.usage
            in_tok.append(usage.prompt_tokens)
            out_tok.append(usage.completion_tokens)
            total_tok.append(usage.total_tokens)
            rtok = 0
            if usage.completion_tokens_details is not None:
                rtok = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
            reasoning_tok.append(rtok)
            content = resp.choices[0].message.content
            bodies.append({"puzzle_id": p["id"], "gt": str(p["answer"]), "content": content,
                           "in_tok": usage.prompt_tokens, "out_tok": usage.completion_tokens})
            # GT match (Final answer 라인 추출 → eval)
            import re
            m = re.search(r"[Ff]inal\s*[Aa]nswer\s*[:：]\s*(.+?)(?:\n|$)", content)
            if m:
                pred = m.group(1).strip()
                try: ok = abs(float(pred) - float(p["answer"])) < 1e-6
                except: ok = pred.strip().lower() == str(p["answer"]).strip().lower()
                if ok: gt_match += 1
            print(f"[{mode}] {i+1:2d}/{N_PER_MODE}  in={usage.prompt_tokens:5d}  out={usage.completion_tokens:5d}  match={'Y' if (m and ok) else '-'}")
        samples_dump[mode] = bodies

        results[mode] = {
            "n": len(in_tok),
            "in_avg": statistics.mean(in_tok) if in_tok else 0,
            "out_avg": statistics.mean(out_tok) if out_tok else 0,
            "out_max": max(out_tok) if out_tok else 0,
            "reasoning_avg": statistics.mean(reasoning_tok) if reasoning_tok else 0,
            "total_avg": statistics.mean(total_tok) if total_tok else 0,
            "gt_match": gt_match,
            "gt_match_rate": gt_match / len(in_tok) if in_tok else 0,
        }

    # Batch API 단가 (gpt-5-mini standard에 50% 할인 가정 — 추측)
    # gpt-5-mini standard: input $0.25/1M, output $2.00/1M (추측)
    P_IN = 0.125 / 1_000_000
    P_OUT = 1.000 / 1_000_000

    print("\n=== 6 000건(naive 3k + guided 3k) 환산 견적 (Batch 50% 할인 가정 추측) ===")
    grand = 0.0
    for mode, r in results.items():
        in_total = r["in_avg"] * 3000
        out_total = r["out_avg"] * 3000
        cost = in_total * P_IN + out_total * P_OUT
        grand += cost
        print(f"  {mode:6s}: in_avg={r['in_avg']:.0f}  out_avg={r['out_avg']:.0f} (reasoning {r['reasoning_avg']:.0f})  → ${cost:.2f}")
    print(f"  TOTAL ≈ ${grand:.2f}")

    out_path = PROJECT_ROOT / "results/distill_cost_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": MODEL, "n_per_mode": N_PER_MODE,
        "results": results,
        "estimated_total_cost_6k_batch_USD": grand,
        "_pricing_assumption_USD_per_M": {"input": 0.125, "output": 1.000, "note": "추측"},
        "samples": samples_dump,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    main()
