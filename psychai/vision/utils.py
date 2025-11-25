import json
import os
import hashlib
import csv
from pathlib import Path
from typing import Any, Iterator, List, Dict, Tuple, Optional
import numpy as np
import torch
from PIL import Image
from datasets import Dataset, DatasetDict


def _to_numpy(x: Any) -> np.ndarray:

    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, list):
        return np.array(x)
    else:
        raise TypeError(f"Expected numpy array, torch tensor, or list; got {type(x).__name__}")

def _to_hwc(arr: np.ndarray, layout: Optional[str]) -> np.ndarray:
    if arr.ndim == 2:
        # This is a grayscale image, we add a channel dimension
        return arr[..., None]
    if arr.ndim != 3:
        # This is not a valid image array
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")
    h, w, c = arr.shape
    if layout is not None:
        if layout == "HWC":
            return arr
        elif layout == "CHW":
            return np.transpose(arr, (1, 2, 0))
        else:
            raise ValueError(f"Invalid layout: {layout}")
    if c in (1, 3, 4):
        return arr
    if h in (1, 3, 4):
        return np.transpose(arr, (1, 2, 0))
    raise ValueError(
        f"Cannot infer channel layout for shape {arr.shape}. "
        "Provide layout='HWC' or layout='CHW'."
    )

def _sanitize_range_dtype(arr: np.ndarray) -> Tuple[np.ndarray, str]:
    if np.issubdtype(arr.dtype, np.integer):
        if arr.dtype != np.uint8:
            info = np.iinfo(arr.dtype)
            if info.max > 255:
                arr = np.clip(arr, 0, info.max).astype(np.float32)
                arr = np.round(arr * (255.0 / info.max)).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr, "uint8"

    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmin >= 0.0 and vmax <= 1.0:
        rng = "float01"
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0 + 0.5).astype(np.uint8)
    elif vmin >= -1.0 and vmax <= 1.0:
        rng = "-1..1"
        arr = np.clip((arr + 1.0) * 0.5, 0.0, 1.0)
        arr = (arr * 255.0 + 0.5).astype(np.uint8)
    else:
        rng = "float255"
        arr = np.clip(arr, 0.0, 255.0)
        arr = (arr + 0.5).astype(np.uint8)
    return arr, rng

def pixels_to_pil(
    pixels: Any,
    *,
    layout: Optional[str] = None,
    assume_bgr: bool = False,
    keep_alpha: bool = False,
) -> Image.Image:

    arr = _to_numpy(pixels)

    # 1) Infer + convert to HWC
    arr_hwc = _to_hwc(arr, layout)

    # 2) Sanitize dtype/range
    arr_hwc = _sanitize_range_dtype(arr_hwc)

    # 3) Channel handling
    C = arr_hwc.shape[2]
    if C == 1:
        return Image.fromarray(arr_hwc[..., 0], mode="L")

    if C == 3:
        if assume_bgr:
            arr_hwc = arr_hwc[..., ::-1]
        return Image.fromarray(arr_hwc, mode="RGB")

    if C == 4:
        if not keep_alpha:
            rgb = arr_hwc[..., :3]
            if assume_bgr:
                rgb = rgb[..., ::-1]
            return Image.fromarray(rgb, mode="RGB")
        else:
            rgb = arr_hwc[..., :3]
            a = arr_hwc[..., 3]
            if assume_bgr:
                rgb = rgb[..., ::-1]
            rgba = np.dstack([rgb, a])
            return Image.fromarray(rgba, mode="RGBA")

    raise ValueError(f"Unsupported channel count after HWC conversion: {C}")