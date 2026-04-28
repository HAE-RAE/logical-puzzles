"""Standalone iteration harness for the cipher_en_simple track.

Loads Qwen/Qwen3-VL-8B-Instruct locally, runs the 5 simple-cipher puzzles,
parses each answer with the production CipherEvaluator parser and reports
N/5 accuracy plus a per-puzzle pass/fail table.

Why a separate harness (not evaluation/run.py)?
- The user explicitly asked for a dedicated track that doesn't touch the
  existing cipher_en pipeline. This file only reads the new
  cipher_en_simple jsonl and only loads one model.
- Faster turnaround for the iterate-difficulty loop: load the model once,
  reuse via a JSON-controlled regenerate-and-rerun cycle if needed.

Usage
-----
    python scripts/run_cipher_simple_check.py
    python scripts/run_cipher_simple_check.py --quant 4
    python scripts/run_cipher_simple_check.py \
        --data data/mini_simple/cipher_en_simple_easy.jsonl \
        --out  results/cipher_simple/run_001.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force stdout/stderr to UTF-8 so model output containing em-dashes / Korean
# characters does not crash the harness on Windows (default cp949 console).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from evaluation.evaluators.cipher import CipherEvaluator


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DATA_DIR = PROJECT_ROOT / "data" / "mini_simple"
RESULTS_DIR = PROJECT_ROOT / "results" / "cipher_simple"

# Per-difficulty pass band (inclusive). Used to set the harness exit code
# and to decide whether the difficulty is calibrated. The exit code is 0 when
# the run is in band, 1 otherwise — convenient for shell-driven iteration.
TARGET_BAND = {
    "easy":   (7, 9),    # 7-9/10
    "medium": (4, 6),    # 4-6/10
    "hard":   (2, 4),    # 2-4/10
}


SYSTEM_PROMPT = (
    "You are a careful cipher-decoding assistant. The user will give you a "
    "ciphertext and the algorithm used. Decode it step by step, showing your "
    "work briefly, then on the FINAL line write exactly:\n"
    "원문: WORD\n"
    "where WORD is an uppercase English word with no spaces. The CipherEvaluator "
    "only reads the last `원문:` line, so you may reason freely above it."
)


def load_puzzles(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_model(model_name: str, quant: Optional[int],
                gpu_max_gib: Optional[float] = None,
                cpu_max_gib: float = 32.0) -> Tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
    print(f"[load] {model_name}")
    # Reduce CUDA fragmentation; helps bnb 4-bit loaders that spike memory
    # transiently while quantising layer-by-layer.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    load_kwargs: Dict = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if gpu_max_gib is not None:
        load_kwargs["max_memory"] = {0: f"{gpu_max_gib}GiB", "cpu": f"{cpu_max_gib}GiB"}
    if quant == 4:
        from transformers import BitsAndBytesConfig
        # Skip the vision tower / patch-merger from quantisation. Qwen3-VL's
        # vision components have non-standard linear layouts that have caused
        # bnb 4-bit loaders to segfault on this RTX 5080 setup at ~30% load.
        # We're doing text-only inference anyway so the vision side can stay
        # bf16 (small footprint).
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["visual", "merger", "vision_tower",
                                    "patch_embed", "lm_head"],
        )
        # dtype is required for non-quantised modules (the skipped vision
        # parts) — keep bf16.
    elif quant == 8:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["visual", "merger", "vision_tower",
                                    "patch_embed", "lm_head"],
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
    model.eval()
    print(f"[load] device={next(model.parameters()).device} dtype={next(model.parameters()).dtype}")
    return model, processor


def trim_to_last_answer_line(raw: str) -> str:
    """Keep only the content of the LAST `원문:` (or `Answer:`) line.

    The production CipherEvaluator's KO regex `[가-힣\\s]+` crosses line
    boundaries, so a response like:

        → 원문: 고양이

        원문: 고양이

    yields a captured group of `고양이\\n\\n원문` instead of `고양이`. We
    sidestep the issue without modifying the production parser by truncating
    `raw` to just the final `원문:`-prefixed line + everything after it
    (which is normally empty for greedy decoding).
    """
    import re
    matches = list(re.finditer(r'원문\s*[:：]', raw))
    if not matches:
        return raw
    return raw[matches[-1].start():]


@torch.inference_mode()
def generate_one(model, processor, system_prompt: str, user_prompt: str,
                 max_new_tokens: int = 256) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user",   "content": [{"type": "text", "text": user_prompt}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    )
    new_tokens = out[0, inputs.input_ids.shape[1]:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy",
                    help="Selects data file under data/mini_simple/ and the target pass band")
    ap.add_argument("--lang", choices=["en", "ko"], default="en",
                    help="Selects cipher_<lang>_simple_<difficulty>.jsonl")
    ap.add_argument("--data", default=None,
                    help="Override data file (otherwise derived from --difficulty/--lang)")
    ap.add_argument("--out", default=None,
                    help="Override output file (otherwise results/cipher_simple/run_<difficulty>.jsonl)")
    ap.add_argument("--quant", type=int, default=None, choices=[None, 4, 8])
    ap.add_argument("--gpu-max-gib", type=float, default=None,
                    help="Cap GPU memory (GiB) and offload remainder to CPU")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--tag", default=None,
                    help="Optional tag appended to the output filename, e.g. iter1")
    args = ap.parse_args()

    diff = args.difficulty
    lang = args.lang
    band = TARGET_BAND[diff]

    data_path = Path(args.data) if args.data else (DATA_DIR / f"cipher_{lang}_simple_{diff}.jsonl")
    if args.out:
        out_path = Path(args.out)
    else:
        suffix = f"_{args.tag}" if args.tag else ""
        out_path = RESULTS_DIR / f"run_{lang}_{diff}{suffix}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    puzzles = load_puzzles(data_path)
    print(f"[data] {len(puzzles)} puzzles from {data_path}")

    model, processor = build_model(args.model, args.quant, gpu_max_gib=args.gpu_max_gib)

    evaluator = CipherEvaluator()  # only used as a parser/scorer here

    correct = 0
    rows_out: List[Dict] = []
    for idx, p in enumerate(puzzles, start=1):
        t0 = time.time()
        raw = generate_one(model, processor, SYSTEM_PROMPT, p["question"],
                           max_new_tokens=args.max_new_tokens)
        dt = time.time() - t0
        # Pre-process: keep only the last `원문:` line so the production
        # KO parser doesn't accidentally swallow the next paragraph.
        for_parse = trim_to_last_answer_line(raw)
        predicted = evaluator._parse_answer(for_parse, p)
        is_correct, _ = evaluator._check_answer(p["answer"], predicted)
        if is_correct:
            correct += 1

        status = "PASS" if is_correct else "FAIL"
        print(f"  [{idx}/{len(puzzles)}] {status} | gold={p['answer']:<8} "
              f"pred={str(predicted):<8} | {dt:5.1f}s")
        if not is_correct:
            print(f"      raw: {raw[:200]}")

        rows_out.append({
            "id": p.get("id"),
            "gold": p["answer"],
            "predicted": predicted,
            "correct": bool(is_correct),
            "latency_s": round(dt, 2),
            "raw": raw,
        })

    acc = correct / len(puzzles) if puzzles else 0.0
    lo, hi = band
    in_band = lo <= correct <= hi
    summary = {
        "model": args.model,
        "difficulty": diff,
        "data": str(data_path),
        "total": len(puzzles),
        "correct": correct,
        "accuracy": acc,
        "target_band": [lo, hi],
        "in_band": bool(in_band),
    }

    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_summary": summary}, ensure_ascii=False) + "\n")
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[result] {diff}: {correct}/{len(puzzles)} ({acc:.1%}) | "
          f"target band [{lo},{hi}] -> {'IN BAND' if in_band else ('TOO HARD' if correct < lo else 'TOO EASY')}")
    print(f"[out] {out_path}")
    return 0 if in_band else 1


if __name__ == "__main__":
    raise SystemExit(main())
