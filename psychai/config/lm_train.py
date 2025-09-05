import os
from typing import List


class LM_TrainConfig:
    # Paths
    DATA_DISK_PATH = None
    OUTPUT_DIR = None
    LOGGING_DIR = None
    MODEL_PATH = None
    TOKENIZER_PATH = None
    MODEL_SAVE_PATH = None

    # Data
    DATA_NAME = None
    DATA_TYPE = None
    TRAIN_DATA_PATH = None
    EVAL_DATA_PATH = None
    SEQUENCE_LENGTH = None
    DATA_PROCESS_BATCH_SIZE = None
    DATA_PROCESS_NUM_PROC = None

    # Random State
    RANDOM_STATE = 66

    # Model
    MODEL_NAME = None
    CUSTOMIZED_MODEL = None
    TASK = None
    FULL_FINETUNING = False
    DTYPE = None
    TRUST_REMOTE_CODE = None # only for gpt-oss
    
    # Training
    PER_DEVICE_TRAIN_BATCH_SIZE = None
    GRADIENT_ACCUMULATION_STEPS = None
    WARMUP_RATIO = None
    NUM_EPOCHS = None
    MAX_STEPS = None
    CALLBACKS = None
    LEARNING_RATE = None
    OPTIMIZER = None
    WEIGHT_DECAY = None
    LR_SCHEDULER = None
    FP16 = None


    # Evaluation / Saving / Logging
    EVAL_STRATEGY = None
    COMPUTE_METRICS = None
    EVAL_STEPS = None
    EVAL_ACCUMULATION_STEPS = None
    PER_DEVICE_EVAL_BATCH_SIZE = None
    SAVE_STRATEGY = None
    SAVE_MODEL = None
    SAVE_STEPS = None
    SAVE_TOTAL_LIMIT = None
    LOGGING_STEPS = None
    REPORT_TO = None
    LOAD_BEST_MODEL_AT_END = None
    METRIC_FOR_BEST_MODEL = None
    GREATER_IS_BETTER = None

    def __init__(self, **overrides):
        for k, v in overrides.items():
            setattr(self, k, v)

    def create_directories(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        print(f"Output directory created: {self.OUTPUT_DIR}")
        os.makedirs(self.LOGGING_DIR, exist_ok=True)
        print(f"Logging directory created: {self.LOGGING_DIR}")
        os.makedirs(self.MODEL_SAVE_PATH, exist_ok=True)
        print(f"Model save directory created: {self.MODEL_SAVE_PATH}")


