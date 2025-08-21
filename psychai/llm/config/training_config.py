"""
Minimal text training configuration.

This class intentionally keeps only the fields used by the current Trainer
while remaining easy to extend. Users can subclass and override defaults.
"""

import os
from typing import List


class TrainingConfig:
    # Paths
    DATA_DISK_PATH = os.getenv("DATA_DISK_PATH", None)
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", None)
    LOGGING_DIR = os.getenv("LOGGING_DIR", None)
    MODELS_PATH = os.getenv("MODELS_PATH", None)
    MODEL_PATH = os.getenv("MODEL_PATH", None)

    # Data
    DATA_NAME = os.getenv("DATA_NAME", None)
    TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", None)
    EVAL_DATA_PATH = os.getenv("EVAL_DATA_PATH", None)

    # Random State
    RANDOM_STATE = int(os.getenv("RANDOM_STATE", "66"))

    # Model
    USE_UNSLOTH = os.getenv("USE_UNSLOTH", True)
    MODEL_NAME = os.getenv("MODEL_NAME", None)
    CHAT_TEMPLATE = os.getenv("CHAT_TEMPLATE", None)
    FULL_FINETUNING = os.getenv("FULL_FINETUNING", False)
    LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", True)
    DTYPE = os.getenv("DTYPE", None)
    TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", None)
    REASONING_EFFORT = os.getenv("REASONING_EFFORT", None) # only for gpt-oss

    # LoRA
    APPLY_LORA = os.getenv("APPLY_LORA", True)
    LORA_RANK = int(os.getenv("LORA_RANK", 8))
    LORA_ALPHA = int(os.getenv("LORA_ALPHA", 16))
    LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", 0.05))
    LORA_TARGET_MODULES: List[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj",      # MLP layers
    ]
    BIAS = os.getenv("BIAS", "none")
    USE_GRADIENT_CHECKPOINTING = os.getenv("USE_GRADIENT_CHECKPOINTING", "unsloth")
    USE_RSLORA = os.getenv("USE_RSLORA", False)
    LOFTQ_CONFIG = os.getenv("LOFTQ_CONFIG", None)
    
    # Training
    TASK_TYPE = os.getenv("TASK_TYPE", None)
    MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", 64))
    PER_DEVICE_TRAIN_BATCH_SIZE = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", 16))
    GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", 1))
    WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", 10))
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 10))
    MAX_STEPS = int(os.getenv("MAX_STEPS", 100))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-4))
    OPTIMIZER = os.getenv("OPTIMIZER", None)
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 0.01))
    LR_SCHEDULER = os.getenv("LR_SCHEDULER", None)

    # Evaluation / Saving / Logging
    EVAL_STRATEGY = "steps"
    EVAL_STEPS = int(os.getenv("EVAL_STEPS", 100))
    SAVE_MODEL = os.getenv("SAVE_MODEL", None)
    SAVE_STEPS = int(os.getenv("SAVE_STEPS", 100))
    SAVE_TOTAL_LIMIT = int(os.getenv("SAVE_TOTAL_LIMIT", 100))
    LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", 100))
    LOAD_BEST_MODEL_AT_END = os.getenv("LOAD_BEST_MODEL_AT_END", True)
    METRIC_FOR_BEST_MODEL = os.getenv("METRIC_FOR_BEST_MODEL", "eval_loss")
    GREATER_IS_BETTER = os.getenv("GREATER_IS_BETTER", False)
    REPORT_TO = os.getenv("REPORT_TO", "none")

    def __init__(self, **overrides):
        for k, v in overrides.items():
            setattr(self, k, v)


