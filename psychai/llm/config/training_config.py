"""
Minimal text training configuration.

This class intentionally keeps only the fields used by the current Trainer
while remaining easy to extend. Users can subclass and override defaults.
"""

import os
from typing import List


class TrainingConfig:
    # Paths
    DATA_DISK_PATH = None
    OUTPUT_DIR = None
    LOGGING_DIR = None
    MODEL_PATH = None
    MODEL_SAVE_PATH = None

    # Data
    DATA_NAME = None
    DATA_TYPE = None
    TRAIN_DATA_PATH = None
    EVAL_DATA_PATH = None

    # Random State
    RANDOM_STATE = 66

    # Model
    USE_UNSLOTH = True
    MODEL_NAME = None
    REASONING = False
    CHAT_TEMPLATE = None
    FULL_FINETUNING = False
    LOAD_IN_4BIT = True
    DTYPE = None
    TRUST_REMOTE_CODE = None
    REASONING_EFFORT = None # only for gpt-oss

    # LoRA
    APPLY_LORA = True
    LORA_RANK = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES: List[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj",      # MLP layers
    ]
    BIAS = "none"
    USE_GRADIENT_CHECKPOINTING = "unsloth"
    USE_RSLORA = False
    LOFTQ_CONFIG = None
    
    # Training
    TASK_TYPE = None
    MAX_SEQ_LENGTH = 64
    PER_DEVICE_TRAIN_BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 1
    WARMUP_STEPS = 10
    NUM_EPOCHS = 10
    MAX_STEPS = 100
    LEARNING_RATE = 1e-4
    OPTIMIZER = None
    WEIGHT_DECAY = 0.01
    LR_SCHEDULER = None

    # Evaluation / Saving / Logging
    EVAL_STRATEGY = "steps"
    PER_DEVICE_EVAL_BATCH_SIZE = 1
    EVAL_ACCUMULATION_STEPS = 4
    EVAL_STEPS = 100
    SAVE_MODEL = None
    SAVE_STRATEGY = "steps"
    SAVE_STEPS = 100
    SAVE_TOTAL_LIMIT = 100
    LOGGING_STEPS = 100
    LOAD_BEST_MODEL_AT_END = True
    METRIC_FOR_BEST_MODEL = "eval_loss"
    GREATER_IS_BETTER = False
    REPORT_TO = "none"

    def __init__(self, **overrides):
        for k, v in overrides.items():
            setattr(self, k, v)


