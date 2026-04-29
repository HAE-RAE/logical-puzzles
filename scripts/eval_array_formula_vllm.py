"""
array_formula_en 평가 스크립트 (vLLM, N-sampling pass@1 평균)

GRPO-vs-SFT 비교 실험의 4개 조건(baseline / SFT-naive / SFT-guided / GRPO)에서
공통으로 재사용. ArrayFormulaEvaluator의 parse/check 로직을 그대로 사용.

설계 원칙:
- 동일 generation 설정 (Qwen3-Thinking 권장값 + N=4 sampling)
- pass@1 = (정답 sample 수 / N)의 puzzle 평균
- 난이도 / problem_type 별 break-down 자동 보고
- chat template은 모델의 tokenizer를 그대로 사용 (Thinking 모델은 <think>\\n까지 자동 prepend)
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import PROJECT_ROOT, load_jsonl, save_jsonl as write_jsonl

sys.path.insert(0, str(PROJECT_ROOT))
from evaluation.evaluators.array_formula import ArrayFormulaEvaluator
from generation.array_formula_en import puzzle_to_prompt


def build_prompts(puzzles, tokenizer, evaluator):
    """각 puzzle을 chat template 적용한 prompt 문자열 리스트로 변환.

    user content는 generator의 puzzle_to_prompt(p)로 만든다 (tables + question 통합).
    평가/학습/Batch 호출 모두에서 동일 prompt를 써야 distribution이 일치한다.
    """
    prompts = []
    for p in puzzles:
        sys_p = evaluator._get_system_prompt(p)
        user_text = puzzle_to_prompt(p)
        msgs = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": user_text},
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(text)
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--eval-file", default="data/array_formula_en/eval_60.jsonl")
    parser.add_argument("--out-dir", required=True, help="results output dir")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--tp-size", type=int, default=1, help="tensor parallel size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-mem-util", type=float, default=0.85)
    parser.add_argument("--lora-path", default=None,
                        help="LoRA adapter dir (None = base model 평가). vLLM LoRARequest로 로드.")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_path = PROJECT_ROOT / args.eval_file
    puzzles = load_jsonl(eval_path)
    print(f"[eval] model={args.model}  n_puzzles={len(puzzles)}  n_samples={args.n_samples}")

    # vLLM (lazy import)
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_tokens + 4096,  # input + output 여유
        seed=args.seed,
        dtype="bfloat16",
        enable_lora=args.lora_path is not None,
        max_lora_rank=64 if args.lora_path is not None else None,
    )
    lora_request = None
    if args.lora_path:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest(lora_name=Path(args.lora_path).name,
                                   lora_int_id=1, lora_path=args.lora_path)
    sp = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    evaluator = ArrayFormulaEvaluator()
    evaluator._task_name = "array_formula_en"

    prompts = build_prompts(puzzles, tokenizer, evaluator)
    print(f"[eval] generating {len(prompts)} prompts × n={args.n_samples} (lora={args.lora_path}) ...")
    t0 = time.time()
    outputs = llm.generate(prompts, sp, lora_request=lora_request) if lora_request else llm.generate(prompts, sp)
    gen_sec = time.time() - t0
    print(f"[eval] generation done in {gen_sec:.1f}s")

    # parse + score
    per_puzzle = []
    for puzzle, out in zip(puzzles, outputs):
        sample_results = []
        for s in out.outputs:
            text = s.text
            predicted = evaluator._parse_answer(text, puzzle)
            correct, _ = evaluator._check_answer(puzzle["answer"], predicted)
            sample_results.append({
                "predicted": str(predicted) if predicted is not None else None,
                "correct": correct,
                "completion_len_tokens": len(s.token_ids),
                "raw_response": text,
            })
        n_correct = sum(1 for s in sample_results if s["correct"])
        pass_at_1 = n_correct / len(sample_results)
        per_puzzle.append({
            "puzzle_id": puzzle["id"],
            "difficulty": puzzle["difficulty"],
            "problem_type": puzzle["type"],
            "expected": str(puzzle["answer"]),
            "n_samples": len(sample_results),
            "n_correct": n_correct,
            "pass_at_1": pass_at_1,
            "samples": sample_results,
        })

    # summary
    def avg(values):
        return sum(values) / len(values) if values else 0.0

    by_diff = defaultdict(list)
    by_ptype = defaultdict(list)
    overall = []
    for p in per_puzzle:
        overall.append(p["pass_at_1"])
        by_diff[p["difficulty"]].append(p["pass_at_1"])
        by_ptype[p["problem_type"]].append(p["pass_at_1"])

    summary = {
        "model": args.model,
        "lora_path": args.lora_path,
        "eval_file": str(eval_path),
        "n_puzzles": len(per_puzzle),
        "n_samples_per_puzzle": args.n_samples,
        "gen_kwargs": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "overall_pass_at_1": avg(overall),
        "by_difficulty": {k: {"n": len(v), "pass_at_1": avg(v)} for k, v in by_diff.items()},
        "by_problem_type": {k: {"n": len(v), "pass_at_1": avg(v)} for k, v in by_ptype.items()},
        "wall_clock_sec": gen_sec,
    }

    write_jsonl(out_dir / "per_puzzle.jsonl", per_puzzle)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[eval] summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
