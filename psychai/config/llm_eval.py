import os
from typing import List

class LLM_EvalConfig:
    DATA_ROOT = None
    MODELS_ROOT = None
    MODEL_CACHE_ROOT = None
    USE_UNSLOTH = True
    MAX_SEQ_LENGTH = 1024
    LOAD_IN_4BIT = True
    DTYPE = None

    GENERATE_ARGS = {
        "max_new_tokens": 128,
        "temperature": 0.7,
        "do_sample": True,
        "top_p": 0.95,
        "top_k": 50,
        "reasoning_effort": None
    }
    PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
            ### Instruction:
            {}
            
            ### Input:
            {}
            
            ### Response:
            {}"""
    BATCH_SIZE = 16
    OUTPUT_DIR = "results"
    SAVE_SUMMARY = True
    
    def __init__(self, **overrides):
        for key, value in overrides.items():
            setattr(self, key, value)
        