"""
Minimal audio training configuration.
Add fields only when we implement the audio trainer.
"""

import os


class AudioTrainingConfig:
    # Paths
    DATA_DISK_PATH = os.getenv("DATA_DISK_PATH", "./autodl-tmp")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(DATA_DISK_PATH, "weights"))
    MODELS_PATH = os.getenv("MODELS_PATH", os.path.join(DATA_DISK_PATH, "models"))
    MODEL_PATH = os.getenv("MODEL_PATH", None)

    # Model
    MODEL_NAME = os.getenv("MODEL_NAME", "openai/whisper-tiny")
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", "16000"))
    RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

    # Data
    DATA_NAME = os.getenv("DATA_NAME", "data")

    # Audio processing
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
    TARGET_LENGTH = int(os.getenv("TARGET_LENGTH", "16000"))
    FEATURE_TYPE = os.getenv("FEATURE_TYPE", "mfcc")  # mfcc | mel | chroma

    # Training
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "5"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
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


