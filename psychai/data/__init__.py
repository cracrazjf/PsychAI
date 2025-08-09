"""
Data module - Simple data utilities

Essential functions for loading and processing text, image, and audio data.
Focus on the 80% use case with clean, simple functions.
"""

# Core text data functions
from .text import (
    load_json, save_json, train_test_split, validate_format,
    convert_to_chat_format, load_csv_as_chat, combine_datasets,
    sample_data, print_data_stats, convert_to_instruction_format,
    load_csv_as_instruction
)

# Image data functions
from .image import (
    load_image, load_image_dataset, create_image_chat_sample,
    process_image_for_model, get_image_info
)

# Audio data functions  
from .audio import (
    load_audio, load_audio_dataset, create_audio_chat_sample,
    process_audio_for_model, get_audio_info
)

# Multimodal functions
from .multimodal import (
    create_multimodal_chat_sample, load_multimodal_dataset,
    convert_multimodal_to_chat, get_multimodal_stats
)

__all__ = [
    # Core functions
    "load_json",
    "save_json", 
    "train_test_split",
    "validate_format",
    "convert_to_chat_format",
    "load_csv_as_chat",
    "load_csv_as_instruction",
    "convert_to_instruction_format",
    "combine_datasets",
    "sample_data",
    "print_data_stats",
    
    # Image functions
    "load_image",
    "load_image_dataset",
    "create_image_chat_sample",
    "process_image_for_model",
    "get_image_info",
    
    # Audio functions
    "load_audio",
    "load_audio_dataset", 
    "create_audio_chat_sample",
    "process_audio_for_model",
    "get_audio_info",
    
    # Multimodal functions
    "create_multimodal_chat_sample",
    "load_multimodal_dataset",
    "convert_multimodal_to_chat",
    "get_multimodal_stats",
]