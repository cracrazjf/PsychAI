# Import main classes for easy access
from .models import ModelLoader, load_model, load_model_unsloth, apply_lora, apply_lora_unsloth
from .training import Trainer
from .data import (
    load_json, save_json, train_test_split, validate_format,
    convert_to_chat_format, load_csv_as_chat,
    load_csv_as_instruction, convert_to_instruction_format,
)
from .evaluation import (
    Evaluator, ModelManager, benchmark_text, compare_text, interactive_text
)
from .utils import (
    print_memory_usage,
    ensure_training_dirs,
    ensure_cache_dirs,
    ensure_all_dirs,
)

# Make key classes available at package level
__all__ = [
    # Models
    "ModelLoader",
    "load_model",
    "load_model_unsloth",
    "apply_lora",
    "apply_lora_unsloth",
    
    # Training
    "Trainer",
    
    # Data
    "load_json",
    "save_json",
    "train_test_split",
    "validate_format",
    "convert_to_chat_format",
    "load_csv_as_chat",
    "load_csv_as_instruction",
    "convert_to_instruction_format",
    
    # Evaluation
    "Evaluator",
    "ModelManager",
    "benchmark_text",
    "compare_text",
    "interactive_text",
    
    # Utilities
    "print_memory_usage",
    "ensure_training_dirs",
    "ensure_cache_dirs",
    "ensure_all_dirs",
    
    # Version
    "__version__",
]