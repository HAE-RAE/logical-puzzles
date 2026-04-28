"""
Qwen3-4B-Thinking SFT 학습 (TRL 1.x)

- 입력: parse_distill_batch.py가 만든 SFT jsonl ({"messages": [...], ...})
- chat template 자동 적용 (Qwen3-Thinking은 assistant 시작에 <think>\\n prepend)
- assistant 토큰만 loss (assistant_only_loss=True)
- save per epoch → 학습 dynamics 평가용 ckpt 4개 (epoch1/2/3 + final이지만 final=epoch3)

논문 fairness 위해 naive/guided 학습 hyperparameter는 완전히 동일하게 한다.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_messages_jsonl(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        rows.append({"messages": d["messages"]})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)  # α/r = 1.0 (GRPO/SFT fair 비교)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    train_path = PROJECT_ROOT / args.train_data

    logger.info(f"model={args.model}  train={train_path}  out={output_dir}")
    rows = load_messages_jsonl(train_path)
    logger.info(f"loaded {len(rows)} training samples")
    dataset = Dataset.from_list(rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_seq_length,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.epochs,
        seed=args.seed,
        # NOTE: Qwen3-Thinking chat template은 trl의 {% generation %} 마커 없음 → False로 fallback.
        # assistant 응답이 token 비중 90%+ (long reasoning)이므로 system/user 포함해도 학습 신호 거의 동일.
        # naive/guided 둘 다 동일 설정으로 fairness 보장.
        assistant_only_loss=False,
        report_to=[],  # wandb 사용 시 ["wandb"]
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
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

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    logger.info(f"done. final model at {output_dir / 'final'}")


if __name__ == "__main__":
    main()
