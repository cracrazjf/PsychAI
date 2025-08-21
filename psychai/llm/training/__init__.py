"""
Training module - Training pipeline and configuration

This module provides:
- Training pipeline using SFTTrainer
- Training configuration management
- Memory management utilities
- Training metrics and monitoring
"""

from .trainer import Trainer

__all__ = [
    "Trainer",
]