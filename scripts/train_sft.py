"""
TRL SFT 학습 스크립트

- Qwen3 모델에 대해 completion-only SFT 수행
- system prompt + question은 마스킹, teacher의 answer(추론 포함)만 학습
- chat template 적용 후 response token만 loss 계산
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def load_sft_data(path: Path):
    """JSONL 파일에서 messages 필드 로드"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                items.append({"messages": item["messages"]})
    return items


def main():
    parser = argparse.ArgumentParser(description="TRL SFT Training for Puzzle Solving")
    parser.add_argument("--model", required=True, help="Base model name (e.g., Qwen/Qwen3-0.6B)")
    parser.add_argument("--train-data", default="data/sft/train.jsonl", help="Training data path")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: models/{model_name}_sft)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--gpu-id", type=int, default=3, help="GPU device ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging interval")
    parser.add_argument("--save-steps", type=int, default=100, help="Checkpoint save interval")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # 출력 디렉토리
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        model_short = args.model.replace("/", "_")
        output_dir = PROJECT_ROOT / "models" / f"{model_short}_sft"

    train_data_path = PROJECT_ROOT / args.train_data

    logger.info(f"Model: {args.model}")
    logger.info(f"Train data: {train_data_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"GPU: {args.gpu_id}")

    # 데이터 로드
    logger.info("Loading training data...")
    train_items = load_sft_data(train_data_path)
    train_dataset = Dataset.from_list(train_items)
    logger.info(f"Loaded {len(train_dataset)} training samples")

    # 토크나이저 로드
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 모델 로드
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # SFT 설정 (assistant_only_loss: system/user 토큰은 -100 마스킹, assistant만 loss 계산)
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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

    # Trainer 생성
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # 학습 시작
    logger.info("Starting training...")
    trainer.train()

    # 최종 모델 저장
    logger.info(f"Saving final model to {output_dir}...")
    trainer.save_model(str(output_dir))

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
