# ===== save_api.py =====
import json, os, sys, time, hashlib
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Tuple, Union, Optional

import torch

try:
    from safetensors.torch import save_file as save_safetensors  # optional
    _HAS_SAFETENSORS = True
except Exception:
    _HAS_SAFETENSORS = False


# --------- helpers

def _jsonify(obj: Any) -> Any:
    """
    Make an object JSON-serializable:
    - Convert tuples to lists
    - Dataclasses -> dict
    - Tensors -> shape + dtype summary (not data)
    - torch.dtype/device -> str
    - Sets -> sorted lists
    - Fallback: use str(obj)
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, set):
        return sorted([_jsonify(x) for x in obj])
    if is_dataclass(obj):
        return _jsonify(asdict(obj))
    if isinstance(obj, torch.dtype):
        return str(obj).replace('torch.', '')
    if isinstance(obj, torch.device):
        return str(obj)
    if torch.is_tensor(obj):
        return {
            "__tensor__": True,
            "dtype": str(obj.dtype).replace('torch.', ''),
            "shape": list(obj.shape),
        }
    # tuples like (C,H,W), etc.
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode('utf-8', errors='ignore')
    # last resort
    return str(obj)


def _canonical_json_bytes(d: Dict[str, Any]) -> bytes:
    return json.dumps(d, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')


def _sha256_of_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --------- config extraction

def extract_spec_dict(model_or_spec) -> Dict[str, Any]:
    """
    Return a JSON-ready dict describing your ModelSpec:
    - { 'vocab_size', 'image_shape', 'layers': [...] }
    Accepts:
      - ModelSpec instance
      - Model (with .spec attribute)
      - raw list[dict] of layers
    """
    # Case 1: ModelSpec-like
    if hasattr(model_or_spec, 'layers') and hasattr(model_or_spec, 'vocab_size'):
        spec_layers = model_or_spec.layers
        vocab_size = getattr(model_or_spec, 'vocab_size', None)
        image_shape = getattr(model_or_spec, 'image_shape', None)
    # Case 2: Model with .spec
    elif hasattr(model_or_spec, 'spec'):
        spec = model_or_spec.spec
        return extract_spec_dict(spec)
    # Case 3: already a list[dict]
    elif isinstance(model_or_spec, list):
        spec_layers = model_or_spec
        vocab_size = None
        image_shape = None
    else:
        raise TypeError("extract_spec_dict: expected ModelSpec, Model, or list[dict].")

    # Important: your layer specs already contain "_name", "type", and params.
    # We only need to deep-convert to JSON-friendly.
    spec_dict = {
        "vocab_size": _jsonify(vocab_size),
        "image_shape": _jsonify(image_shape),
        "layers": _jsonify(spec_layers),
    }
    return spec_dict


def build_config_dict(
    model_or_spec,
    *,
    model_type: str = "custom",
    config_version: int = 1,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compose the top-level config.json dict (architecture + metadata).
    """
    spec_dict = extract_spec_dict(model_or_spec)

    meta = {
        "time_saved_unix": int(time.time()),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "uses_safetensors": _HAS_SAFETENSORS,
    }
    if extra_metadata:
        meta.update(_jsonify(extra_metadata))

    # Hash a canonical representation of spec only (not weights)
    arch_hash = _sha256_of_bytes(_canonical_json_bytes(spec_dict))

    cfg = {
        "model_type": model_type,       # e.g., "custom-elman"
        "config_version": config_version,
        "spec": spec_dict,              # <- your ModelSpec as a dict
        "metadata": meta,
        "arch_hash": arch_hash,
    }
    return cfg


# --------- save APIs

def save_config(save_dir: str, config_dict: Dict[str, Any]) -> str:
    """
    Write config.json to save_dir. Returns the config path.
    """
    os.makedirs(save_dir, exist_ok=True)
    cfg_path = os.path.join(save_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    return cfg_path


def _split_state_dict_for_safetensors(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Safetensors requires tensors only; filter out non-tensors if any.
    """
    return {k: v for k, v in state_dict.items() if torch.is_tensor(v)}


def save_pretrained(
    model: torch.nn.Module,
    save_dir: str,
    *,
    model_or_spec_for_config: Optional[Union[object, List[dict]]] = None,
    model_type: str = "custom",
    config_version: int = 1,
    extra_metadata: Optional[Dict[str, Any]] = None,
    weights_filename: Optional[str] = None,
    prefer_safetensors: bool = True,
) -> Dict[str, str]:
    """
    Save both config.json and model weights (HF-like).
    - `model` can be your CausalLMWrapper or bare Model; state_dict() is saved as-is.
    - `model_or_spec_for_config` is used to build the architecture config; if None, we try
       to infer from `model` (looking for .spec).
    - Returns paths of written files.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1) Build config
    if model_or_spec_for_config is None:
        # Try to infer spec from model
        if hasattr(model, "model") and hasattr(model.model, "spec"):
            target_for_cfg = model.model.spec
        elif hasattr(model, "spec"):
            target_for_cfg = model.spec
        else:
            raise ValueError(
                "save_pretrained: Please pass `model_or_spec_for_config=` "
                "or ensure `model` (or model.model) has a `.spec` attribute."
            )
    else:
        target_for_cfg = model_or_spec_for_config

    config_dict = build_config_dict(
        target_for_cfg,
        model_type=model_type,
        config_version=config_version,
        extra_metadata=extra_metadata,
    )
    cfg_path = save_config(save_dir, config_dict)

    # 2) Save weights
    sd = model.state_dict()
    paths = {"config": cfg_path}

    if prefer_safetensors and _HAS_SAFETENSORS:
        tensors_only = _split_state_dict_for_safetensors(sd)
        if weights_filename is None:
            weights_filename = "model.safetensors"
        weights_path = os.path.join(save_dir, weights_filename)
        save_safetensors(tensors_only, weights_path)
        paths["weights"] = weights_path
        # Optional: save a hash of the weights file
        with open(weights_path, "rb") as f:
            weights_hash = _sha256_of_bytes(f.read())
        meta_path = os.path.join(save_dir, "meta.json")
        meta_obj = {
            "weights_format": "safetensors",
            "weights_file": os.path.basename(weights_path),
            "weights_sha256": weights_hash,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_obj, f, indent=2, ensure_ascii=False)
        paths["meta"] = meta_path
    else:
        if weights_filename is None:
            weights_filename = "pytorch_model.bin"
        weights_path = os.path.join(save_dir, weights_filename)
        torch.save(sd, weights_path)
        paths["weights"] = weights_path

    return paths
