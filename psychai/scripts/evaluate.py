"""
PsychAI CLI - Evaluate (Interactive)

Launches the interactive text evaluation REPL.
"""

from __future__ import annotations

import argparse
from typing import Optional

from ..evaluation import interactive_text
from ..config import TextTrainingConfig
from ..config.settings import SettingsConfig
from ..utils import ensure_cache_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive evaluation (REPL) for PsychAI")
    p.add_argument("--models-root", default=None, help="Directory with local models (defaults to config MODELS_PATH)")
    p.add_argument("--data-root", default=None, help="Directory with datasets (defaults to config DATA_DISK_PATH)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ensure_cache_dirs(SettingsConfig())

    # Defaults from training config
    default_models_root = TextTrainingConfig.MODELS_PATH
    default_data_root = TextTrainingConfig.DATA_DISK_PATH

    models_root = args.models_root or default_models_root
    data_root = args.data_root or default_data_root

    interactive_text(models_root=models_root, data_root=data_root)


if __name__ == "__main__":
    main()


