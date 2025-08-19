"""
PsychAI: A modular framework for psychology research with AI

This package provides tools for:
- Large Language Model (LLM) training and evaluation
- Vision model training and evaluation  
- Simple neural network implementations
- Utility functions for data processing and visualization

Main modules:
- psychai.llm: LLM training, evaluation, and utilities
- psychai.vision: Vision model training and evaluation
- psychai.simple_nn: Simple neural network implementations
- psychai.config: Configuration management
- psychai.utils: General utilities
"""

__version__ = "0.1.0"
__author__ = "Jingfeng(Craig) Zhang"  
__email__ = "jz44@illinois.edu"

# Import main submodules for easy access
from . import llm
from . import vision  
from . import simple_nn
from . import config
from . import utils

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "llm",
    "vision",
    "simple_nn", 
    "config",
    "utils",
]
