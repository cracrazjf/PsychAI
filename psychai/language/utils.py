import os, io, json, time, shutil, tempfile, random, sys, hashlib, csv
from pathlib import Path
from typing import Union, Callable, Iterator, Dict, List, Any, Optional, TextIO
import numpy as np
import torch

try:
    from safetensors.torch import save_file as _save_safetensors
    from safetensors.torch import load_file as _load_safetensors
    _HAS_SAFETENSORS = True
except Exception:
    _HAS_SAFETENSORS = False

# Data utils for loading/saving datasets

def _default_delimiter_for(path: str | Path, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    p = Path(path)
    if p.suffix == ".tsv" or "".join(p.suffixes[-2:]).endswith(".tsv.gz") or "".join(p.suffixes[-2:]).endswith(".tsv.zst"):
        return "\t"
    return ","

def _open_text(path: str | Path, *, encoding: str = "utf-8", errors: str = "ignore") -> TextIO:
    p = Path(path)
    suffixes = "".join(p.suffixes[-2:])  # handle .jsonl.gz, .csv.zst, etc.

    if suffixes.endswith(".gz"):
        import gzip
        return io.TextIOWrapper(gzip.open(p, "rb"), encoding=encoding, errors=errors)

    if suffixes.endswith(".zst"):
        if zstd is None:
            raise RuntimeError("zstandard not installed; cannot read .zst files. Try `pip install zstandard`.")
        fh = p.open("rb")
        dctx = zstd.ZstdDecompressor()  # type: ignore
        return io.TextIOWrapper(dctx.stream_reader(fh), encoding=encoding, errors=errors)

    # Plain text
    return p.open("r", encoding=encoding, errors=errors)

def read_csv(
    path: str | Path,
    *,
    delimiter: Optional[str] = None,
    encoding: str = "utf-8",
    errors: str = "ignore",
    skip_blank: bool = True,
) -> Iterator[Dict]:
    delim = _default_delimiter_for(path, delimiter)
    with _open_text(path, encoding=encoding, errors=errors) as f:
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            if skip_blank and not any(v and str(v).strip() for v in row.values()):
                continue
            yield dict(row)

def read_jsonl(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    errors: str = "ignore",
    skip_blank: bool = True,
) -> Iterator[Dict]:
    with _open_text(path, encoding=encoding, errors=errors) as f:
        for i, line in enumerate(f):
            if skip_blank and not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:  # pragma: no cover
                raise ValueError(f"Invalid JSON on line {i+1} of {path}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Line {i+1} of {path} is not a JSON object (got {type(obj).__name__}).")
            yield obj

def read_json(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    errors: str = "ignore",
) -> Iterator[Dict]:
    with _open_text(path, encoding=encoding, errors=errors) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:  # pragma: no cover
            raise ValueError(f"Invalid JSON in {path}: {e}") from e

    if isinstance(data, list):
        for i, obj in enumerate(data):
            if not isinstance(obj, dict):
                raise ValueError(f"Element {i} in {path} is not a JSON object (got {type(obj).__name__}).")
            yield obj
    elif isinstance(data, dict):
        yield data
    else:
        raise ValueError(f"Top-level value in {path} must be an object or array of objects (got {type(data).__name__}).")

def stable_id(*parts: Any, digest_size: int = 16) -> str:
    h = hashlib.blake2b(digest_size=digest_size)
    for p in parts:
        if isinstance(p, (bytes, bytearray)):
            h.update(p)
        else:
            h.update(str(p).encode("utf-8", errors="ignore"))
        h.update(b"|")
    return h.hexdigest()

def infer_file_type(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return "csv"
    elif ext == ".jsonl":
        return "jsonl"
    elif ext == ".json":
        return "json"
    else:
        raise ValueError(f"Invalid file type: {ext}")

# Train utils for saving/loading checkpoints
def _atomic_write_bytes(dst_path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(dst_path)) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, dst_path)

def _atomic_write_json(dst_path: str, obj: Dict[str, Any]) -> None:
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_write_bytes(dst_path, data)

def to_serializable(v):
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v

def clean_dir(path: str) -> None:
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)

def _tensor_state_dict_cpu_only(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu()
    return out

def save_checkpoint(
    run_dir: str,
    model: torch.nn.Module,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,             # any .state_dict()/.load_state_dict()
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    tokenizer: Optional[object] = None,             # HF-like tokenizer with .save_pretrained(dir)
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,       # e.g., {"val_loss": 1.23, "accuracy": 0.91}
    is_best: bool = False,
    max_to_keep: int = 3,                           # how many epoch_* dirs to retain
) -> str:
    """
    Saves a full training checkpoint under run_dir/checkpoints/epoch_XXXX/.
    Also updates 'last/' (and 'best/' if is_best) mirrors and prunes old checkpoints.
    Returns the path to the created checkpoint directory.
    """
    ckpt_root = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_root, exist_ok=True)
    tag = f"epoch_{(epoch if epoch is not None else 0):04d}"
    ckpt_dir = os.path.join(ckpt_root, tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1) Save model weights (.safetensors preferred)
    weights_path = os.path.join(ckpt_dir, "model.safetensors" if _HAS_SAFETENSORS else "pytorch_model.bin")
    state = _tensor_state_dict_cpu_only(model.state_dict())
    if _HAS_SAFETENSORS:
        _save_safetensors(state, weights_path)
    else:
        torch.save(state, weights_path)

    # 2) Optimizer / Scheduler / Scaler (optional)
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
    if scaler is not None:
        torch.save(scaler.state_dict(), os.path.join(ckpt_dir, "scaler.pt"))

    # 3) RNG states (optional but great for determinism)
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state().tolist(),
        "torch_cuda_all": [t.tolist() for t in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
    }
    torch.save(rng_state, os.path.join(ckpt_dir, "rng.pt"))

    # 4) Trainer state (lightweight metadata)
    trainer_state = {
        "epoch": int(epoch) if epoch is not None else None,
        "global_step": int(global_step) if global_step is not None else None,
        "is_best": bool(is_best),
        "time_saved_unix": int(time.time()),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "metrics": {k: to_serializable(v) for (k, v) in (metrics or {}).items()},
    }
    _atomic_write_json(os.path.join(ckpt_dir, "trainer_state.json"), trainer_state)

    # 5) Tokenizer snapshot (optional; small but handy to keep with ckpt)
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tok_dir = os.path.join(ckpt_dir, "tokenizer")
        clean_dir(tok_dir)
        tokenizer.save_pretrained(tok_dir)

    # 6) Update 'last/' mirror
    last_dir = os.path.join(ckpt_root, "last")
    clean_dir(last_dir)
    shutil.copytree(ckpt_dir, last_dir)

    # 7) Update 'best/' mirror if requested
    if is_best:
        best_dir = os.path.join(ckpt_root, "best")
        clean_dir(best_dir)
        shutil.copytree(ckpt_dir, best_dir)

    # 8) Retention policy: keep only newest N epoch_* dirs (preserve mirrors separately)
    epoch_dirs = [d for d in os.listdir(ckpt_root) if d.startswith("epoch_") and os.path.isdir(os.path.join(ckpt_root, d))]
    # sort by epoch index
    epoch_dirs_sorted = sorted(epoch_dirs, key=lambda s: int(s.split("_")[1]))
    excess = max(0, len(epoch_dirs_sorted) - max_to_keep)
    for d in epoch_dirs_sorted[:excess]:
        clean_dir(os.path.join(ckpt_root, d))

    return ckpt_dir

def load_checkpoint(
    ckpt_dir: str,
    model: torch.nn.Module,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,                 # any .load_state_dict()
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: str = "auto",                         # "auto" | "cpu" | "cuda"
    strict: bool = True,                                # pass to model.load_state_dict
    load_rng: bool = False,                             # restore RNG states if available
    weights_filename: Optional[str] = None,             # override file name if needed
) -> Dict[str, Any]:
    """
    Load a training checkpoint created by `save_checkpoint(...)`.

    Args:
        ckpt_dir:           Path to a checkpoint directory (e.g., ".../checkpoints/epoch_0003" or ".../checkpoints/best").
        model:              Model instance with the same architecture; will have its state_dict loaded.
        optimizer:          If provided, loads optimizer state from optimizer.pt (if present).
        scheduler:          If provided, loads LR scheduler state from scheduler.pt (if present).
        scaler:             If provided, loads AMP GradScaler state from scaler.pt (if present).
        map_location:       "auto" (cuda if available else cpu), or "cpu" or "cuda".
        strict:             Whether to enforce that the keys in state_dict match the model.
        load_rng:           If True, restore RNG states (python, numpy, torch CPU/CUDA) from rng.pt (if present).
        weights_filename:   Optional override for weights file name.

    Returns:
        trainer_state (dict): Contents of trainer_state.json if present; empty dict otherwise.
    """
    if map_location == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif map_location in ("cpu", "cuda"):
        device = map_location
    else:
        device = map_location  # allow explicit torch device strings

    # ---- 1) Load model weights ----
    if weights_filename is None:
        safepath = os.path.join(ckpt_dir, "model.safetensors")
        binpath  = os.path.join(ckpt_dir, "pytorch_model.bin")
        if os.path.exists(safepath):
            weights_path = safepath
            use_safetensors = True
        elif os.path.exists(binpath):
            weights_path = binpath
            use_safetensors = False
        else:
            raise FileNotFoundError(f"No weights file found in {ckpt_dir} (looked for model.safetensors / pytorch_model.bin)")
    else:
        weights_path = os.path.join(ckpt_dir, weights_filename)
        use_safetensors = weights_path.endswith(".safetensors")

    if use_safetensors:
        if not _HAS_SAFETENSORS:
            raise RuntimeError("Tried to load a .safetensors file but safetensors is not installed.")
        state_dict = _load_safetensors(weights_path)        # tensors come CPU; fine for load_state_dict
    else:
        state_dict = torch.load(weights_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        if missing:
            print(f"[load_checkpoint] Warning: missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
        if unexpected:
            print(f"[load_checkpoint] Warning: unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    # Move model to target device (optional but typical)
    try:
        model.to(device)
    except Exception:
        # If your code manages device moves elsewhere, you can ignore this.
        pass

    # ---- 2) Optimizer / Scheduler / Scaler (optional) ----
    def _maybe_load(obj, filename):
        path = os.path.join(ckpt_dir, filename)
        if obj is not None and os.path.exists(path):
            state = torch.load(path, map_location=device)
            obj.load_state_dict(state)
            return True
        return False

    _maybe_load(optimizer, "optimizer.pt")
    if scheduler is not None and hasattr(scheduler, "load_state_dict"):
        _maybe_load(scheduler, "scheduler.pt")
    _maybe_load(scaler, "scaler.pt")

    # ---- 3) RNG states (optional) ----
    if load_rng:
        rng_path = os.path.join(ckpt_dir, "rng.pt")
        if os.path.exists(rng_path):
            rng = torch.load(rng_path, map_location="cpu")
            try:
                if "python" in rng and rng["python"] is not None:
                    random.setstate(rng["python"])
                if "numpy" in rng and rng["numpy"] is not None:
                    np.random.set_state(rng["numpy"])
                if "torch_cpu" in rng and rng["torch_cpu"] is not None:
                    torch.random.set_rng_state(torch.tensor(rng["torch_cpu"], dtype=torch.uint8))
                if "torch_cuda_all" in rng and rng["torch_cuda_all"] and torch.cuda.is_available():
                    for i, st in enumerate(rng["torch_cuda_all"]):
                        torch.cuda.set_rng_state(torch.tensor(st, dtype=torch.uint8), device=i)
            except Exception as e:
                print(f"[load_checkpoint] Warning: failed to restore RNG states: {e}")

    # ---- 4) Trainer state JSON (optional) ----
    trainer_state_path = os.path.join(ckpt_dir, "trainer_state.json")
    trainer_state = {}
    if os.path.exists(trainer_state_path):
        try:
            with open(trainer_state_path, "r", encoding="utf-8") as f:
                trainer_state = json.load(f)
        except Exception as e:
            print(f"[load_checkpoint] Warning: failed to read trainer_state.json: {e}")

    return trainer_state