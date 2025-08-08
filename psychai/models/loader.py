"""
Model loading utilities

This module provides functions for loading various types of language models:
- Standard HuggingFace models
- Unsloth-optimized models  
- Local and remote models
- Quantized models for memory efficiency
"""

import torch
import logging
from typing import Tuple, Optional, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.INFO)

# Check for optional dependencies
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("⚠️ Unsloth not available. Install with: pip install unsloth")

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training


class ModelLoader:
    """Main model loading class with various loading strategies"""
    
    def __init__(self):
        self.loaded_models = {}
        
    def load_model(
        self, 
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", 
        for_training: bool = False,
        use_unsloth: bool = False,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Load a model using the appropriate strategy
        
        Args:
            model_name: Name or path of the model
            for_training: Whether model will be used for training
            use_unsloth: Whether to use Unsloth optimization
            **kwargs: Additional arguments passed to specific loaders
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if use_unsloth and UNSLOTH_AVAILABLE:
            return load_model_unsloth(model_name, for_training=for_training, **kwargs)
        else:
            return load_model(model_name, for_training=for_training, **kwargs)


def load_model(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", 
    model_path: str = None,
    for_training: bool = False
) -> Tuple[Any, Any]:
    """
    Load model using standard HuggingFace transformers
    
    Args:
        model_name: Name or path of the model to load
        model_path: Local path to model (overrides model_name if provided)
        for_training: Whether model will be used for training (enables quantization)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name} from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path if model_path else model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if for_training:
        print("Loading model for training...")
        
        if torch.cuda.is_available():
            # Use CUDA with 4-bit quantization for training
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path if model_path else model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            model = prepare_model_for_kbit_training(model)
        else:
            # Use CPU fallback
            model = AutoModelForCausalLM.from_pretrained(
                model_path if model_path else model_name,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
    else:
        print("Loading model for inference...")
        
        if torch.cuda.is_available():
            print("Using CUDA for inference.")
            model = AutoModelForCausalLM.from_pretrained(
                model_path if model_path else model_name,
                torch_dtype=torch.float16,
                device_map="cuda"
            )
        else:
            print("Using CPU for inference.")
            model = AutoModelForCausalLM.from_pretrained(
                model_path if model_path else model_name,
                torch_dtype=torch.float32,
                device_map="cpu"
            )

    return model, tokenizer


def load_model_unsloth(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", 
    model_path: str = None,
    max_seq_length: int = 512, 
    load_in_4bit: bool = True, 
    for_training: bool = True
) -> Tuple[Any, Any]:
    """
    Load model using Unsloth for memory efficiency and speed
    
    Args:
        model_name: Name or path of the model to load
        max_seq_length: Maximum sequence length
        model_path: Local path to model (overrides model_name if provided)
        load_in_4bit: Whether to use 4-bit quantization
        for_training: Whether model will be used for training
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        ImportError: If Unsloth is not available
    """
    if not UNSLOTH_AVAILABLE:
        raise ImportError("Unsloth is not installed. Please install it first.")
    
    # Use local path if provided, otherwise use model_name
    model_path = model_path if model_path else model_name
    
    quantization_str = "4-bit" if load_in_4bit else "16-bit"
    print(f"🚀 Loading {model_name} with Unsloth from: {model_path} ({quantization_str})")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect optimal dtype
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
    )
    
    if not for_training:
        FastLanguageModel.for_inference(model)
    
    return model, tokenizer