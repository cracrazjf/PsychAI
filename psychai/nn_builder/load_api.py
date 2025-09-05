# ===== load_api.py =====
import json, os
from typing import Any, Dict, Optional, Union

import torch

try:
    from safetensors.torch import load_file as load_safetensors
    _HAS_SAFETENSORS = True
except Exception:
    _HAS_SAFETENSORS = False


# --------- load config

def load_config(load_dir: str) -> Dict[str, Any]:
    cfg_path = os.path.join(load_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"No config.json found under {load_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------- rebuild spec + model

def build_spec_from_config(config: Dict[str, Any]):
    from .nn_builder import ModelSpec  # adjust import path to your code

    spec_dict = config["spec"]
    spec = ModelSpec(
        vocab_size=spec_dict.get("vocab_size"),
        image_shape=tuple(spec_dict["image_shape"]) if spec_dict.get("image_shape") else None,
    )
    for layer_spec in spec_dict["layers"]:
        # already has "_name" and params
        spec.layers.append(layer_spec)
        spec._names.add(layer_spec["_name"])
    return spec


def from_pretrained(
    load_dir: str,
    *,
    strict: bool = True,
    map_location: Optional[Union[str, torch.device]] = None,
    torch_dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.nn.Module:
    """
    Load config + weights from a directory.
    Returns a ready-to-use nn.Module (CausalLMWrapper if task=="causal-lm").
    """
    from .nn_builder import Model, CausalLMWrapper  # adjust imports

    # 1) Load config
    config = load_config(load_dir)

    # 2) Rebuild spec
    spec = build_spec_from_config(config)

    # 3) Build model
    model = Model(spec)

    # 5) Locate weights file
    weights_path = None
    if _HAS_SAFETENSORS:
        candidate = os.path.join(load_dir, "model.safetensors")
        if os.path.exists(candidate):
            weights_path = candidate
    if weights_path is None:
        candidate = os.path.join(load_dir, "pytorch_model.bin")
        if os.path.exists(candidate):
            weights_path = candidate
    if weights_path is None:
        raise FileNotFoundError(f"No weights file found under {load_dir}")

    # 6) Load state_dict
    if weights_path.endswith(".safetensors"):
        sd = load_safetensors(weights_path, device="cpu")
    else:
        sd = torch.load(weights_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(sd, strict=strict)

    if missing or unexpected:
        print(f"[from_pretrained] Missing keys: {missing}")
        print(f"[from_pretrained] Unexpected keys: {unexpected}")

    # 7) Move / cast
    if device is not None:
        model.to(device)
    if torch_dtype is not None:
        model.to(dtype=torch_dtype)

    return model
