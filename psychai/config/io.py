from dataclasses import replace
from typing import Any, Dict
import yaml
from .config import ModelConfig, DataConfig, OptimConfig, LoggingConfig, TrainingConfig, EvaluationConfig

def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def update_config(dc, updates: Dict[str, Any]):
    fields = {f.name for f in dc.__dataclass_fields__.values()}
    kwargs = {}
    for k, v in updates.items():
        if not hasattr(dc, k):
            raise ValueError(f"Unknown config field: {k} for {type(dc).__name__}")
        cur_val = getattr(dc, k)
        # nested dict -> recurse if current value is dataclass
        if isinstance(cur_val, (ModelConfig, DataConfig, OptimConfig, LoggingConfig, TrainingConfig, EvaluationConfig)) and isinstance(v, dict):
            kwargs[k] = update_config(cur_val, v)
        else:
            kwargs[k] = v
    return replace(dc, **kwargs)

def load_config(
    yaml_path: str | None = None,
    overrides: Dict[str, Any] | None = None,
) -> TrainingConfig:
    cfg = TrainingConfig()  # defaults in code

    # 1) YAML overrides
    if yaml_path is not None:
        yaml_cfg = load_yaml_config(yaml_path)
        cfg = update_config(cfg, yaml_cfg)

    # 2) manual / CLI overrides (flattened: "model.hidden_size": 512)
    if overrides:
        nested: Dict[str, Any] = {}
        for key, value in overrides.items():
            parts = key.split(".")
            d = nested
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = value
        cfg = update_config(cfg, nested)

    return cfg
