"""
Qwen3-4B-Thinking GRPO 학습 (TRL 1.x, vLLM rollout)

설계 (DeepSeek-R1 + Open-R1 하이퍼파라미터를 4B 스케일로 다운):
  - prompt set: SFT와 동일한 train_3k.jsonl (data-source 통제)
  - reward (학습 원리상 sparse + dense 혼합):
      * correctness: ArrayFormulaEvaluator로 정답 비교 → 1.0 / 0.0
      * format    : <think>...</think> 1회 정확 매칭 + Final answer 라인 → 0.1
  - generation : temp=1.0 (R1과 동일, exploration 위해), max_completion=8192
  - group size : G=8 (4B + 4×A100 메모리)
  - β (KL)     : 0.001
  - clip ε     : 0.2 (TRL 기본; R1의 10은 GRPO clip 정의가 다름)
  - lr         : 1e-6
  - 3 epoch    : 사용자 지시 (SFT와 동일 epoch 수)
  - save per epoch
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluators.array_formula import ArrayFormulaEvaluator
from generation.array_formula_en import puzzle_to_prompt

THINK_RE = re.compile(r"^[^<]*<think>(.+?)</think>\s*\n*(.*)$", re.DOTALL)
FINAL_RE = re.compile(r"[Ff]inal\s*[Aa]nswer\s*[:：]\s*.+", re.IGNORECASE)


def build_dataset(train_jsonl: Path):
    """GRPO는 prompt만 필요. answer는 reward 계산용 메타데이터로 보존."""
    ev = ArrayFormulaEvaluator()
    ev._task_name = "array_formula_en"
    sys_p = ev.SYSTEM_PROMPT
    rows = []
    for line in train_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        rows.append({
            "prompt": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": puzzle_to_prompt(p)},
            ],
            # 모든 메타필드는 string 으로 통일 (pyarrow 타입 추론 충돌 방지: number/text answer 혼재)
            "gt_answer": str(p["answer"]),
            "answer_type": str(p.get("answer_type", "number")),
            "puzzle_id": str(p["id"]),
            "difficulty": str(p["difficulty"]),
        })
    return Dataset.from_list(rows)


_evaluator = ArrayFormulaEvaluator()
_evaluator._task_name = "array_formula_en"


def correctness_reward(completions, gt_answer, answer_type, **kwargs):
    """1.0 if parsed answer matches GT else 0.0"""
    rewards = []
    for comp, gt, atype in zip(completions, gt_answer, answer_type):
        # completions are list[ChatMessage] when conversational; we want assistant text
        if isinstance(comp, list):
            text = comp[-1]["content"]
        else:
            text = comp
        puzzle_stub = {"answer_type": atype, "answer": gt}
        predicted = _evaluator._parse_answer(text, puzzle_stub)
        correct, _ = _evaluator._check_answer(gt, predicted)
        rewards.append(1.0 if correct else 0.0)
    return rewards


def format_reward(completions, **kwargs):
    """0.1 if response contains <think>...</think> exactly once and a Final answer line."""
    rewards = []
    for comp in completions:
        if isinstance(comp, list):
            text = comp[-1]["content"]
        else:
            text = comp
        # Qwen3-Thinking은 chat template이 <think>\n을 prepend하므로
        # completion 본문은 reasoning ... </think> ... Final answer: X 형태로 시작
        # 즉 본문에 </think>가 정확히 1번 나오고 final answer 패턴이 그 뒤에 있어야 함.
        n_close = text.count("</think>")
        has_final = bool(FINAL_RE.search(text))
        if n_close == 1 and has_final:
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--train-data", default="data/array_formula_en/train_3k.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=8, help="GRPO group size")
    parser.add_argument("--max-completion-length", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.001, help="KL coefficient")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--use-vllm", action="store_true", help="use vLLM rollout (recommended)")
    parser.add_argument("--vllm-max-model-len", type=int, default=12288,
                        help="vLLM KV cache max seq len (prompt + completion). Qwen3-Thinking default 262144 is too large.")
    parser.add_argument("--vllm-gpu-mem-util", type=float, default=0.3)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)  # α/r = 1.0 (SFT와 동일 ratio 유지하려면 SFT script도 동일하게 변경)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    train_path = PROJECT_ROOT / args.train_data

    dataset = build_dataset(train_path)
    logger.info(f"loaded {len(dataset)} prompts")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    cfg = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_completion_length=args.max_completion_length,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.epochs,
        seed=args.seed,
        report_to=[],
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        use_vllm=args.use_vllm,
        vllm_max_model_length=args.vllm_max_model_len,
        vllm_gpu_memory_utilization=args.vllm_gpu_mem_util,
        vllm_enable_sleep_mode=True,  # 학습 step 동안 vLLM weight offload → 단일 GPU OOM 회피
    )

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=cfg,
        train_dataset=dataset,
        reward_funcs=[correctness_reward, format_reward],
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    logger.info(f"done. final model at {output_dir / 'final'}")


if __name__ == "__main__":
    main()
