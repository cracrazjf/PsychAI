__version__ = "0.1.0"
__author__ = "Jingfeng(Craig) Zhang"
__email__ = "jz44@illinois.edu"

# Import main classes for easy access
from .config import SettingsConfig, TextTrainingConfig, VisionTrainingConfig, AudioTrainingConfig
from .models import ModelLoader, load_model, load_model_unsloth, apply_lora, apply_lora_unsloth
from .training import Trainer
from .data import (
    load_json, save_json, train_test_split, validate_format,
    convert_to_chat_format, load_csv_as_chat, load_image, load_audio,
    load_csv_as_instruction, convert_to_instruction_format,
)
from .evaluation import (
    TextEvaluator, ModelManager, benchmark_text, compare_text, interactive_text
)
from .utils import (
    print_memory_usage,
    ensure_training_dirs,
    ensure_cache_dirs,
    ensure_all_dirs,
)

# Make key classes available at package level
__all__ = [
    # Configuration
    "SettingsConfig",
    "TextTrainingConfig",
    "VisionTrainingConfig",
    "AudioTrainingConfig",
    
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
    "load_image",
    "load_audio",
    
    # Evaluation
    "TextEvaluator",
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