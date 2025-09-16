import os
from typing import List

class LM_EvalConfig:
    MODEL_ROOT = None
    TASK = "causal_lm"
    CUSTOMIZED_MODEL = False
    TRUST_REMOTE_CODE = True
    SEQUENCE_LENGTH = 2
    OVERLAPPING_SEQUENCES = True

    def __init__(self, **overrides):
        for k, v in overrides.items():
            setattr(self, k, v)