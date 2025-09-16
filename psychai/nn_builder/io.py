# ===== save_api.py =====
import json, os, sys, time, hashlib
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Tuple, Union, Optional

import torch
import torch.nn as nn

try:
    from safetensors.torch import save_file as save_safetensors
    from safetensors.torch import load_file as load_safetensors
    _HAS_SAFETENSORS = True
except Exception:
    _HAS_SAFETENSORS = False

__all__ = ["save_config", "save_pretrained", "load_config", "build_spec_from_config", "from_pretrained"]

# --------- helpers

def _jsonify(obj: Any) -> Any:
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
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode('utf-8', errors='ignore')
    return str(obj)

def _canonical_json_bytes(d: Dict[str, Any]) -> bytes:
    return json.dumps(d, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')

def _sha256_of_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --------- config extraction

def extract_spec_dict(model_or_spec) -> Dict[str, Any]:
    if hasattr(model_or_spec, 'layers') and hasattr(model_or_spec, 'vocab_size') and hasattr(model_or_spec, 'image_shape'):
        spec_layers = model_or_spec.layers
        vocab_size = getattr(model_or_spec, 'vocab_size', None)
        image_shape = getattr(model_or_spec, 'image_shape', None)
    elif hasattr(model_or_spec, 'spec'):
        spec = model_or_spec.spec
        return extract_spec_dict(spec)
    elif isinstance(model_or_spec, list):
        spec_layers = model_or_spec
        vocab_size = None
        image_shape = None
    else:
        raise TypeError("extract_spec_dict: expected ModelSpec, Model, or list[dict].")

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

def unwrap_model(m: nn.Module) -> nn.Module:
    while True:
        # 1) torch.nn wrappers
        if hasattr(m, "module") and isinstance(getattr(m, "module"), nn.Module):
            m = m.module
            continue
        # 2) your custom wrapper with .model
        if hasattr(m, "model") and isinstance(getattr(m, "model"), nn.Module):
            m = m.model
            continue
        # 3) PEFT (optional)
        try:
            from peft import PeftModel
            if isinstance(m, PeftModel):
                base = m.get_base_model()
                if isinstance(base, nn.Module):
                    m = base
                    continue
        except Exception:
            pass
        break
    return m

# --------- save APIs

def save_config(save_dir: str, config_dict: Dict[str, Any]) -> str:
    os.makedirs(save_dir, exist_ok=True)
    cfg_path = os.path.join(save_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    return cfg_path

def _split_state_dict_for_safetensors(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
    sd = unwrap_model(model).state_dict()
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


# --------- load config

def load_config(load_dir: str) -> Dict[str, Any]:
    cfg_path = os.path.join(load_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"No config.json found under {load_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

# --------- rebuild spec + model

def build_spec_from_config(config: Dict[str, Any]):
    from .nn_builder import ModelSpec

    spec_dict = config["spec"]
    spec = ModelSpec(
        vocab_size=spec_dict.get("vocab_size"),
        image_shape=tuple(spec_dict["image_shape"]) if spec_dict.get("image_shape") else None,
    )
    for layer_spec in spec_dict["layers"]:
        # already has "name" and params
        spec.layers.append(layer_spec)
        spec.names.add(layer_spec["name"])
    return spec


def from_pretrained(
    load_dir: str,
    *,
    strict: bool = True,
    map_location: Optional[Union[str, torch.device]] = None,
    torch_dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.nn.Module:
    from .nn_builder import Model

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
