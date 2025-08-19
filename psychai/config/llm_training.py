"""
Minimal text training configuration.

This class intentionally keeps only the fields used by the current Trainer
while remaining easy to extend. Users can subclass and override defaults.
"""

import os
from typing import List


class LLMTrainingConfig:
    # Paths
    DATA_DISK_PATH = os.getenv("DATA_DISK_PATH", None)
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", None)
    LOGS_DIR = os.getenv("LOGS_DIR", None)
    MODELS_PATH = os.getenv("MODELS_PATH", None)
    MODEL_PATH = os.getenv("MODEL_PATH", None)

    # Data
    DATA_NAME = os.getenv("DATA_NAME", None)
    TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", None)
    EVAL_DATA_PATH = os.getenv("EVAL_DATA_PATH", None)

    # Random State
    RANDOM_STATE = int(os.getenv("RANDOM_STATE", "66"))

    # Model
    MODEL_NAME = os.getenv("MODEL_NAME", None)
    MODEL_TYPE = os.getenv("MODEL_TYPE", None)
    FULL_FINETUNING = os.getenv("FULL_FINETUNING", False)
    LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", True)
    DTYPE = os.getenv("DTYPE", None)
    GRADIENT_CHECKPOINTING = os.getenv("GRADIENT_CHECKPOINTING", "unsloth")
    TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", None)
    REASONING_EFFORT = os.getenv("REASONING_EFFORT", None) # only for gpt-oss

    # Text Model specific
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", 64))
    TASK_TYPE = os.getenv("TASK_TYPE", None)

    # LoRA
    USE_RSLORA = os.getenv("USE_RSLORA", False)
    LOFTQ_CONFIG = os.getenv("LOFTQ_CONFIG", None)
    LORA_RANK = int(os.getenv("LORA_RANK", 8))
    LORA_ALPHA = int(os.getenv("LORA_ALPHA", 16))
    LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", 0.05))
    LORA_TARGET_MODULES: List[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj",      # MLP layers
    ]
    BIAS = os.getenv("BIAS", "none")
    
    # Training
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 10))
    MAX_STEPS = int(os.getenv("MAX_STEPS", 100))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
    GRAD_ACCUM_STEPS = int(os.getenv("GRAD_ACCUM_STEPS", 1))
    GRAD_CLIP = os.getenv("GRAD_CLIP", None)
    OPTIMIZER = os.getenv("OPTIMIZER", None)
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-4))
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 0.01))
    LR_SCHEDULER = os.getenv("LR_SCHEDULER", None)
    WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", 10))
    STEP_SIZE = int(os.getenv("STEP_SIZE", 10))
    GAMMA = float(os.getenv("GAMMA", 0.1))

    # Evaluation / Saving / Logging
    EVAL_STRATEGY = "steps"
    EVAL_STEPS = int(os.getenv("EVAL_STEPS", 100))
    SAVE_MODEL = os.getenv("SAVE_MODEL", None)
    SAVE_STEPS = int(os.getenv("SAVE_STEPS", 100))
    SAVE_TOTAL_LIMIT = int(os.getenv("SAVE_TOTAL_LIMIT", 100))
    LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", 100))

    def __init__(self, **overrides):
        for k, v in overrides.items():
            setattr(self, k, v)


