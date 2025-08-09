"""
Directory utilities

Helpers to create and validate required directories for training and caching.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from ..config.text import TextTrainingConfig
from ..config.settings import SettingsConfig


def _ensure_dir(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    os.makedirs(path, exist_ok=True)
    return path


def ensure_training_dirs(config: TextTrainingConfig) -> Dict[str, str]:
    """
    Ensure key training directories exist based on TextTrainingConfig.
    Returns a mapping of logical names to absolute paths.
    """
    paths: Dict[str, str] = {}

    for key in ("DATA_DISK_PATH", "OUTPUT_DIR", "LOGS_DIR", "MODELS_PATH"):
        value = getattr(config, key, None)
        ensured = _ensure_dir(value)
        if ensured:
            paths[key] = os.path.abspath(ensured)

    return paths


def ensure_cache_dirs(settings: SettingsConfig, set_env: bool = False) -> Dict[str, str]:
    """
    Ensure cache directories from SettingsConfig exist. Optionally set env vars.
    """
    cache_keys = [
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "HF_HOME",
        "TORCH_HOME",
    ]

    paths: Dict[str, str] = {}
    for key in cache_keys:
        value = getattr(settings, key, None)
        ensured = _ensure_dir(value)
        if ensured:
            abspath = os.path.abspath(ensured)
            paths[key] = abspath

    if set_env:
        for key, value in paths.items():
            os.environ[key] = value

    return paths


def ensure_all_dirs(
    config: TextTrainingConfig, settings: Optional[SettingsConfig] = None, set_env: bool = True
) -> Dict[str, Dict[str, str]]:
    """
    Ensure both training and cache directories.
    If settings is not provided, uses SettingsConfig with defaults.
    """
    training_paths = ensure_training_dirs(config)
    settings_obj = settings or SettingsConfig()
    cache_paths = ensure_cache_dirs(settings_obj, set_env=set_env)
    return {"training": training_paths, "cache": cache_paths}


