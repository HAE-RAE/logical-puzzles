"""Unified TRL SFT training script.

Two modes via --thinking flag:

  Standard (default):
    Full finetuning, assistant_only_loss=True, save once at the end.
    Suited for small Qwen3 base models on canonical SFT data.

  Thinking (--thinking):
    LoRA finetuning, save per epoch, assistant_only_loss=False.
    Suited for Qwen3-*-Thinking with long-reasoning teacher traces.
    NOTE: Qwen3-Thinking chat template lacks {% generation %} markers, so we
    fall back to assistant_only_loss=False — assistant tokens dominate (>90%
    of long reasoning) so learning signal is nearly identical.

Usage:
  # standard
  python scripts/train_sft.py --model Qwen/Qwen3-0.6B --train-data data/sft/train.jsonl

  # thinking + LoRA
  python scripts/train_sft.py --thinking --model Qwen/Qwen3-4B-Thinking-2507 \\
      --train-data data/distill/sft_train.jsonl --output-dir models/q4b_thinking_sft
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import PROJECT_ROOT, setup_logger

logger = setup_logger(__name__)


def load_messages_jsonl(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            d = json.loads(line)
            rows.append({"messages": d["messages"]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--thinking", action="store_true",
                        help="Thinking mode: LoRA + save_strategy=epoch + assistant_only_loss=False")
    parser.add_argument("--model", required=True,
                        help="Base model id (e.g. Qwen/Qwen3-0.6B or Qwen/Qwen3-4B-Thinking-2507)")
    parser.add_argument("--train-data", default="data/sft/train.jsonl",
                        help="JSONL with {messages: [...]} entries")
    parser.add_argument("--output-dir", default=None,
                        help="Default: models/{model_short}_sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", "--per-device-batch-size", dest="batch_size",
                        type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad-accum", "--gradient-accumulation-steps", dest="grad_accum",
                        type=int, default=8)
    parser.add_argument("--lr", type=float, default=None,
                        help="Default: 2e-5 standard / 1e-5 thinking")
    parser.add_argument("--max-seq-length", type=int, default=None,
                        help="Default: 4096 standard / 8192 thinking")
    parser.add_argument("--gpu-id", type=int, default=None,
                        help="If set, exports CUDA_VISIBLE_DEVICES (standard mode default 3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100,
                        help="(standard mode) checkpoint save interval; ignored in thinking mode (epoch-based)")
    # LoRA (thinking only)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64,
                        help="α/r = 1.0 (GRPO/SFT fair 비교)")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    # Mode-aware defaults
    if args.lr is None:
        args.lr = 1e-5 if args.thinking else 2e-5
    if args.max_seq_length is None:
        args.max_seq_length = 8192 if args.thinking else 4096
    if args.gpu_id is None and not args.thinking:
        args.gpu_id = 3
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / args.output_dir
    else:
        model_short = args.model.replace("/", "_")
        output_dir = PROJECT_ROOT / "models" / f"{model_short}_sft"

    train_path = Path(args.train_data)
    if not train_path.is_absolute():
        train_path = PROJECT_ROOT / args.train_data

    logger.info(f"mode={'thinking' if args.thinking else 'standard'}  model={args.model}")
    logger.info(f"train={train_path}  out={output_dir}")

    logger.info("Loading training data...")
    rows = load_messages_jsonl(train_path)
    logger.info(f"Loaded {len(rows)} training samples")
    dataset = Dataset.from_list(rows)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model...")
    model_kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
    if args.thinking:
        model_kwargs["attn_implementation"] = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    if args.thinking:
        sft_config = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
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
            assistant_only_loss=False,
            report_to=[],
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
        )
        peft_cfg = None
        try:
            from peft import LoraConfig
            peft_cfg = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
        except ImportError:
            raise SystemExit("--thinking mode requires `peft` (pip install peft)")
        trainer = SFTTrainer(
            model=model, args=sft_config, train_dataset=dataset,
            processing_class=tokenizer, peft_config=peft_cfg,
        )
        logger.info("Starting training (thinking + LoRA)...")
        trainer.train()
        trainer.save_model(str(output_dir / "final"))
        tokenizer.save_pretrained(str(output_dir / "final"))
        logger.info(f"done. final model at {output_dir / 'final'}")
    else:
        sft_config = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            max_length=args.max_seq_length,
            logging_steps=args.logging_steps,
            save_strategy="no",
            bf16=True,
            seed=args.seed,
            report_to="none",
            gradient_checkpointing=True,
            assistant_only_loss=True,
        )
        trainer = SFTTrainer(
            model=model, args=sft_config, train_dataset=dataset,
            processing_class=tokenizer,
        )
        logger.info("Starting training (standard SFT)...")
        trainer.train()
        logger.info(f"Saving final model to {output_dir}...")
        trainer.save_model(str(output_dir))
        logger.info("Training complete!")


if __name__ == "__main__":
    main()
