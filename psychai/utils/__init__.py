"""
Utils module - General utilities and helper functions

This module provides:
- Memory management utilities
- Logging and monitoring
- File I/O helpers
- General purpose utilities
"""

from .memory import (
    print_memory_usage, clear_cache, get_memory_stats
)
from .paths import (
    ensure_training_dirs, ensure_cache_dirs, ensure_all_dirs
)

__all__ = [
    # Memory utilities
    "print_memory_usage",
    "clear_cache", 
    "get_memory_stats",
    # Paths utilities
    "ensure_training_dirs",
    "ensure_cache_dirs",
    "ensure_all_dirs",
]