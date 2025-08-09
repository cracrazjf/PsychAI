"""
Minimal text training configuration.

This class intentionally keeps only the fields used by the current Trainer
while remaining easy to extend. Users can subclass and override defaults.
"""

import os
from typing import List


class TextTrainingConfig:
    # Paths
    DATA_DISK_PATH = os.getenv("DATA_DISK_PATH", "./autodl-tmp")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(DATA_DISK_PATH, "outputs"))
    LOGS_DIR = os.getenv("LOGS_DIR", os.path.join(DATA_DISK_PATH, "logs/training"))
    MODELS_PATH = os.getenv("MODELS_PATH", os.path.join(DATA_DISK_PATH, "models"))
    MODEL_PATH = os.getenv("MODEL_PATH", None)

    # Model
    MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    MODEL_TYPE = os.getenv("MODEL_TYPE", "llama")
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
    RANDOM_STATE = int(os.getenv("RANDOM_STATE", "66"))

    # Data
    DATA_NAME = os.getenv("DATA_NAME", "data")

    # LoRA
    LORA_RANK = int(os.getenv("LORA_RANK", "16"))
    LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))
    LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.1"))
    LORA_TARGET_MODULES: List[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj",      # MLP layers
    ]
    SAVE_MODEL = os.getenv("SAVE_MODEL", "true").lower() == "true"

    # Training
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
    GRAD_ACCUM_STEPS = int(os.getenv("GRAD_ACCUM_STEPS", "8"))
    OPTIMIZER = os.getenv("OPTIMIZER", "adamw_8bit")
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
    LR_SCHEDULER = os.getenv("LR_SCHEDULER", "linear")
    WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "30"))

    # Evaluation / Saving / Logging
    EVAL_STRATEGY = "steps"
    EVAL_STEPS = int(os.getenv("EVAL_STEPS", "50"))
    SAVE_STEPS = int(os.getenv("SAVE_STEPS", "100"))
    SAVE_TOTAL_LIMIT = int(os.getenv("SAVE_TOTAL_LIMIT", "3"))
    LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", "10"))

    def __init__(self, **overrides):
        for k, v in overrides.items():
            setattr(self, k, v)


