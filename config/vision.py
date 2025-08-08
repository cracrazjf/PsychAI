"""
Minimal vision training configuration.
Add fields only when we implement the vision trainer.
"""

import os
from typing import Tuple, List


class VisionTrainingConfig:
    # Paths 
    DATA_DISK_PATH = os.getenv("DATA_DISK_PATH", "./autodl-tmp")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(DATA_DISK_PATH, "weights"))
    MODELS_PATH = os.getenv("MODELS_PATH", os.path.join(DATA_DISK_PATH, "models"))
    MODEL_PATH = os.getenv("MODEL_PATH", None)

    # Model
    MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))
    RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

    # Data
    DATA_NAME = os.getenv("DATA_NAME", "data")

    # Image preprocessing
    IMAGE_SIZE: Tuple[int, int] = (224, 224)
    MEAN: List[float] = [0.485, 0.456, 0.406]
    STD: List[float] = [0.229, 0.224, 0.225]
    AUG_FLIP = True
    AUG_RANDOM_CROP = False

    # Training
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "5"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    GRAD_ACCUM_STEPS = int(os.getenv("GRAD_ACCUM_STEPS", "1"))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
    WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "100"))

    # Evaluation / Saving / Logging
    EVAL_STEPS = int(os.getenv("EVAL_STEPS", "100"))
    SAVE_STEPS = int(os.getenv("SAVE_STEPS", "200"))
    SAVE_TOTAL_LIMIT = int(os.getenv("SAVE_TOTAL_LIMIT", "3"))
    LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", "20"))

    def __init__(self, **overrides):
        for k, v in overrides.items():
            setattr(self, k, v)


