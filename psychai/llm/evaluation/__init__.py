"""
Evaluation module

Exports a simple, text-focused evaluator and runner utilities.
Audio/vision evaluators can be added later with the same interface.
"""

from .evaluator import (
    Evaluator,
    ModelManager,   
    benchmark_text,
    compare_text,
    interactive_text
)

# Backwards compatibility: keep old names if imported elsewhere
try:  # pragma: no cover
    from .evaluator import Evaluator as _LegacyEvaluator, ModelEvaluator as _LegacyModelEvaluator
except Exception:  # if file removed later
    _LegacyEvaluator = None
    _LegacyModelEvaluator = None

__all__ = [
    "Evaluator",    
    "ModelManager",
    "benchmark_text",
    "compare_text",
    "interactive_text",
]