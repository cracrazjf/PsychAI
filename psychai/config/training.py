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
    PRETRAINED = os.getenv("PRETRAINED", None)
    GRADIENT_CHECKPOINTING = os.getenv("GRADIENT_CHECKPOINTING", None)
    TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", None)

    # Text Model specific
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", 64))

    # Vision Model specific
    CLASS_TO_IDX = os.getenv("CLASS_TO_IDX", None)
    IMAGE_PIXEL_NAME = os.getenv("IMAGE_PIXEL_NAME", None)
    IMAGE_LAYOUT = os.getenv("IMAGE_LAYOUT", None)
    IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", 48))
    IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", 48))
    IMAGE_CHANNELS = int(os.getenv("IMAGE_CHANNELS", 3))

    IMAGE_PATH_NAME = os.getenv("IMAGE_PATH_NAME", None)
    IMAGE_ROOT_PATH = os.getenv("IMAGE_ROOT_PATH", None)

    IMAGE_LABEL_NAME = os.getenv("IMAGE_LABEL_NAME", None)
    IMAGE_TEXT_NAME = os.getenv("IMAGE_TEXT_NAME", None)
    IMAGE_META_NAME = os.getenv("IMAGE_META_NAME", None)
    TO_RGB = os.getenv("TO_RGB", True)

    USE_AMP = os.getenv("USE_AMP", None)
    GRADIENT_CLIP_NORM = float(os.getenv("GRADIENT_CLIP_NORM", 1.0))

    TASK_TYPE = os.getenv("TASK_TYPE", None)
    NUM_CLASSES = int(os.getenv("NUM_CLASSES", 100))
    CLASS_WEIGHTS = os.getenv("CLASS_WEIGHTS", None)
    LABEL_SMOOTHING = float(os.getenv("LABEL_SMOOTHING", 0.0))

    DATALOADER_WORKERS = int(os.getenv("DATALOADER_WORKERS", 6))
    PIN_MEMORY = os.getenv("PIN_MEMORY", None)
    DROP_LAST = os.getenv("DROP_LAST", None)

    # LoRA
    LORA_RANK = int(os.getenv("LORA_RANK", 8))
    LORA_ALPHA = int(os.getenv("LORA_ALPHA", 16))
    LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", 0.05))
    LORA_TARGET_MODULES: List[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj",      # MLP layers
    ]
    SAVE_MODEL = os.getenv("SAVE_MODEL", None)

    # Training
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 10))
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
    SAVE_STEPS = int(os.getenv("SAVE_STEPS", 100))
    SAVE_TOTAL_LIMIT = int(os.getenv("SAVE_TOTAL_LIMIT", 100))
    LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", 100))

    def __init__(self, **overrides):
        for k, v in overrides.items():
            setattr(self, k, v)


