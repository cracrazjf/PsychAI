"""
Training Configuration

Generic training configuration class with environment variable support.
Provides sensible defaults while allowing full customization.
"""

import os
from typing import List, Optional, Union


class SettingsConfig:
    # =============================================================================
    # CACHE AND ENVIRONMENT SETTINGS
    # =============================================================================
    
    # Base data path
    DATA_DISK_PATH = None

    # Transformers cache directory
    TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", None)

    # Hugging Face datasets cache directory
    HF_DATASETS_CACHE = os.getenv("HF_DATASETS_CACHE", None)

    # Hugging Face home directory
    HF_HOME = os.getenv("HF_HOME", None)

    # Torch cache directory
    TORCH_HOME = os.getenv("TORCH_HOME", None)
    
    # CUDA visible devices
    CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", None)

    # Hugging Face Mirror (if using autodl)
    HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
    
    # Hugging Face token (must be provided via environment)
    HF_TOKEN = os.getenv("HF_TOKEN", None)

    def __init__(self, **overrides):
        for k, v in overrides.items():
            setattr(self, k, v)
    
    def setup_environment(cls):
        """Setup environment variables and cache directories"""
        
        # Create cache directories
        for cache_dir in [cls.TRANSFORMERS_CACHE, cls.HF_DATASETS_CACHE, cls.HF_HOME, cls.TORCH_HOME]:
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
        
        # Set environment variables
        env_vars = {
            "TRANSFORMERS_CACHE": cls.TRANSFORMERS_CACHE,
            "HF_DATASETS_CACHE": cls.HF_DATASETS_CACHE,
            "HF_HOME": cls.HF_HOME,
            "TORCH_HOME": cls.TORCH_HOME,
        }
        
        # Optional environment variables
        if cls.HF_ENDPOINT:
            env_vars["HF_ENDPOINT"] = cls.HF_ENDPOINT
        if cls.HF_TOKEN:
            env_vars["HUGGING_FACE_HUB_TOKEN"] = cls.HF_TOKEN
        if cls.CUDA_VISIBLE_DEVICES:
            env_vars["CUDA_VISIBLE_DEVICES"] = cls.CUDA_VISIBLE_DEVICES
        
        # Apply environment variables
        for key, value in env_vars.items():
            if value:
                os.environ[key] = str(value)
        
        print("Environment setup completed!")
        
    def login_huggingface(cls):
        """Login to Hugging Face if token is provided"""
        if cls.HF_TOKEN:
            try:
                from huggingface_hub import login
                if cls.HF_ENDPOINT is not None:
                    os.environ["HF_ENDPOINT"] = cls.HF_ENDPOINT
                login(token=cls.HF_TOKEN)
                print("Hugging Face login completed!")
            except ImportError:
                print("huggingface_hub not installed, skipping login")
            except Exception as e:
                print(f"Hugging Face login failed: {e}")
        else:
            print("No HF token provided, skipping login")