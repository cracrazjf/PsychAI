"""
LoRA (Low-Rank Adaptation) utilities

This module provides functions for applying LoRA to models:
- Standard PEFT LoRA application
- Unsloth-optimized LoRA application
"""

from typing import List, Any, Optional

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

from peft import LoraConfig, get_peft_model


def apply_lora(
    model: Any,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM"
) -> Any:
    """
    Apply LoRA to a model using standard PEFT
    
    Args:
        model: The model to apply LoRA to
        rank: LoRA rank (dimensionality of adaptation)
        alpha: LoRA alpha (scaling factor)
        dropout: LoRA dropout rate
        target_modules: Which modules to apply LoRA to
        bias: LoRA bias setting
        task_type: Type of task for PEFT
        
    Returns:
        Model with LoRA applied
    """
    if target_modules is None:
        # Default target modules for most transformer models
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias=bias,
        task_type=task_type
    )
    
    return get_peft_model(model, lora_config)


def apply_lora_unsloth(
    model: Any,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    use_gradient_checkpointing: str = "unsloth",
    random_state: int = 42,
    use_rslora: bool = False,
    loftq_config: Optional[Any] = None
) -> Any:
    """
    Apply LoRA using Unsloth for better optimization
    
    Args:
        model: The model to apply LoRA to
        rank: LoRA rank (dimensionality of adaptation)
        alpha: LoRA alpha (scaling factor)
        dropout: LoRA dropout rate
        target_modules: Which modules to apply LoRA to
        bias: LoRA bias setting
        use_gradient_checkpointing: Gradient checkpointing method
        random_state: Random seed for reproducibility
        use_rslora: Whether to use RSLoRA
        loftq_config: LoFT-Q configuration
        
    Returns:
        Model with LoRA applied using Unsloth optimization
        
    Raises:
        ImportError: If Unsloth is not available
    """
    if not UNSLOTH_AVAILABLE:
        raise ImportError("Unsloth is not installed. Please install it first.")
    
    if target_modules is None:
        # Default target modules for Unsloth (typically more comprehensive)
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    print("🔧 Applying LoRA with Unsloth...")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        target_modules=target_modules,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
        use_rslora=use_rslora,
        loftq_config=loftq_config,
    )
    
    return model