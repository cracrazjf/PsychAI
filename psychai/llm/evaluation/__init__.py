"""
Evaluation module

Exports a simple, text-focused evaluator and runner utilities.
Audio/vision evaluators can be added later with the same interface.
"""

from .evaluator import (
    Evaluator
)
__all__ = [
    "Evaluator"
]