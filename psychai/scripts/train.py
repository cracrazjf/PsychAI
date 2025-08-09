"""
PsychAI CLI - Train

Thin command-line wrapper around PsychAI.Trainer.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from ..training import Trainer
from ..config import TextTrainingConfig
from ..config.settings import SettingsConfig
from ..utils import ensure_training_dirs
from ..data import load_json, validate_format


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a language model with PsychAI")

    # Data disk path
    p.add_argument("--data-disk-path", default="/root/autodl-tmp", help="Path of your data disk")

    # Data
    p.add_argument("--data-name", default=None, help="Data name")
    p.add_argument("--train-data", required=True, help="Path to training data (JSON in chat or instruction format)")
    p.add_argument("--eval-data", help="Optional path to eval data (JSON)")
    p.add_argument("--data-format", choices=["chat", "instruction"], default="chat")

    # Model
    p.add_argument("--model-name", default=None, help="model name")
    p.add_argument("--model-ref", default=None, help="HF model name or local path")
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--model-type", default=None, help="e.g., llama")

    # LoRA
    p.add_argument("--lora-rank", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--lora-dropout", type=float, default=None)
    p.add_argument("--no-lora", action="store_true", help="Disable LoRA application")

    # Training
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--scheduler", default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--eval-steps", type=int, default=None)
    p.add_argument("--save-steps", type=int, default=None)
    p.add_argument("--save-total-limit", type=int, default=None)
    p.add_argument("--logging-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)

    # Output
    p.add_argument("--output-dir", default=None, help="Where to write checkpoints and logs")

    # Performance
    p.add_argument("--unsloth", action="store_true", help="Use Unsloth if available")

    # Config overlay
    p.add_argument("--config", help="Optional JSON config file to pre-populate fields")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> TextTrainingConfig:
    cfg_overrides: Dict[str, Any] = {}
    if args.data_disk_path is not None:
        cfg_overrides["DATA_DISK_PATH"] = args.data_disk_path
    if args.output_dir is not None:
        cfg_overrides["OUTPUT_DIR"] = args.output_dir
    if args.model_ref is not None:
        cfg_overrides["MODEL_NAME"] = args.model_ref
    if args.max_length is not None:
        cfg_overrides["MAX_LENGTH"] = args.max_length
    if args.model_type is not None:
        cfg_overrides["MODEL_TYPE"] = args.model_type
    if args.data_name is not None:
        cfg_overrides["DATA_NAME"] = args.data_name
    if args.model_ref is not None:
        cfg_overrides["MODEL_PATH"] = args.model_ref

    # LoRA
    if args.lora_rank is not None:
        cfg_overrides["LORA_RANK"] = args.lora_rank
    if args.lora_alpha is not None:
        cfg_overrides["LORA_ALPHA"] = args.lora_alpha
    if args.lora_dropout is not None:
        cfg_overrides["LORA_DROPOUT"] = args.lora_dropout

    # Train knobs
    if args.epochs is not None:
        cfg_overrides["NUM_EPOCHS"] = args.epochs
    if args.batch_size is not None:
        cfg_overrides["BATCH_SIZE"] = args.batch_size
    if args.grad_accum_steps is not None:
        cfg_overrides["GRAD_ACCUM_STEPS"] = args.grad_accum_steps
    if args.lr is not None:
        cfg_overrides["LEARNING_RATE"] = args.lr
    if args.weight_decay is not None:
        cfg_overrides["WEIGHT_DECAY"] = args.weight_decay
    if args.scheduler is not None:
        cfg_overrides["LR_SCHEDULER"] = args.scheduler
    if args.warmup_steps is not None:
        cfg_overrides["WARMUP_STEPS"] = args.warmup_steps
    if args.eval_steps is not None:
        cfg_overrides["EVAL_STEPS"] = args.eval_steps
    if args.save_steps is not None:
        cfg_overrides["SAVE_STEPS"] = args.save_steps
    if args.save_total_limit is not None:
        cfg_overrides["SAVE_TOTAL_LIMIT"] = args.save_total_limit
    if args.logging_steps is not None:
        cfg_overrides["LOGGING_STEPS"] = args.logging_steps
    if args.seed is not None:
        cfg_overrides["RANDOM_STATE"] = args.seed

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            file_cfg = json.load(f)
        # Only take known keys to avoid typos silently being ignored
        for key, value in file_cfg.items():
            cfg_overrides[key] = value

    return TextTrainingConfig(**cfg_overrides)


def load_dataset(path: str, data_format: str) -> List[Any]:  # type: ignore[name-defined]
    data = load_json(path)
    if data_format not in ("chat", "instruction"):
        raise ValueError("data-format must be 'chat' or 'instruction'")
    if not validate_format(data, data_format):
        raise ValueError(f"Data at {path} is not valid {data_format} format")
    return data


def main() -> None:
    args = parse_args()

    config = build_config(args)
    ensure_training_dirs(config, SettingsConfig())

    train_data = load_dataset(args.train_data, args.data_format)
    eval_data = load_dataset(args.eval_data, args.data_format) if args.eval_data else None

    trainer = Trainer(config)
    trainer.load_model_and_tokenizer(
        model_name=config.MODEL_NAME,
        use_unsloth=args.unsloth,
        apply_lora=not args.no_lora,
        for_training=True,
    )

    trainer.train(train_data=train_data, eval_data=eval_data, output_dir=config.OUTPUT_DIR)


if __name__ == "__main__":
    main()


