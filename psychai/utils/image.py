from pathlib import Path
from typing import List, Sequence, Optional, Union, Dict, Any
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import json
import re

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in EXTS

def xywh_to_xyxy(b: List[float]) -> List[float]:
    x, y, w, h = b
    return [x, y, x + w, y + h]

def to_hwc(pixels, width=None, height=None, channels=3, layout="HWC"):
    """
    Convert various pixel layouts to H×W×C (uint8).
    layout:
      - "HWC": already interleaved per pixel (RGBRGB...)
      - "CHW": channel-first
      - "CIFAR": R-block, G-block, B-block
    """
    arr = np.array(pixels)

    # Ensure uint8
    if arr.dtype.kind not in "ui":
        if arr.max() <= 1.0:
            arr = arr * 255
        arr = arr.astype(np.uint8)

    if arr.ndim == 1:
        if width is None or height is None:
            raise ValueError("Need width/height for flat arrays")
        if channels == 1:
            return arr.reshape(height, width, 1)

        if layout.upper() == "HWC":
            return arr.reshape(height, width, channels)
        elif layout.upper() == "CHW":
            return arr.reshape(channels, height, width).transpose(1, 2, 0)
        elif layout.upper() == "CIFAR":
            return arr.reshape(channels, height, width).transpose(1, 2, 0)
        else:
            raise ValueError(f"Unknown layout: {layout}")

    elif arr.ndim == 2:
        return arr[:, :, None]  # H×W×1 grayscale

    elif arr.ndim == 3:
        if layout.upper() == "CHW" and arr.shape[0] in (1, 3):
            return arr.transpose(1, 2, 0)
        return arr  # assume already HWC

    else:
        raise ValueError(f"Unsupported ndim={arr.ndim}")

def pixels_to_pil(
    pixels: Any,
    width: Optional[int],
    height: Optional[int],
    channels: int = 3,
):
    """
    Accepts: 
      - a flat string "70 80 82 ..." or "[70, 80, ...]"
      - list/tuple of numbers or strings
      - nested lists (H×W or H×W×C)
      - numpy arrays
    Returns a PIL.Image in L or RGB mode.
    """

    # ---- 1) Coerce to a numeric 1D/2D/3D ndarray ----
    if isinstance(pixels, np.ndarray):
        arr = pixels
        if arr.dtype.kind in "USO":  # unicode/bytes/object -> parse to float
            arr = arr.astype(object).ravel()
            arr = np.array([float(x) for x in arr], dtype=np.float32)
        # else already numeric
    elif isinstance(pixels, str):
        s = pixels.strip()
        # Prefer JSON parsing if it looks like a JSON array
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                arr = np.array(parsed, dtype=np.float32)
            except Exception:
                # Fallback to whitespace parsing
                s = re.sub(r"[\[\],]", " ", s)
                arr = np.fromstring(s, sep=" ", dtype=np.float32)
        else:
            # Remove brackets/commas and split on whitespace
            s = re.sub(r"[\[\],]", " ", s)
            arr = np.fromstring(s, sep=" ", dtype=np.float32)
    else:
        # list/tuple/etc.
        try:
            if len(pixels) > 0 and isinstance(pixels[0], str):
                arr = np.array([float(x) for x in pixels], dtype=np.float32)
            else:
                arr = np.array(pixels, dtype=np.float32)
        except TypeError:
            # e.g., nested lists
            arr = np.array(pixels)

    # ---- 2) Handle range/dtype to uint8 ----
    if arr.size == 0:
        raise ValueError("Empty pixel array.")

    if arr.dtype.kind not in "ui":  # floats or others
        # If it looks normalized (0..1), scale to 0..255
        amax = float(np.nanmax(arr))
        if amax <= 1.0:
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.astype(np.uint8)
    else:
        # already integer—ensure 0..255
        arr = arr.clip(0, 255).astype(np.uint8)

    # ---- 3) Reshape ----
    if arr.ndim == 1:
        if width is None or height is None:
            raise ValueError("Flat pixel array requires explicit width and height in DatasetConfig.image_size.")
        if channels == 1:
            arr = arr.reshape(height, width)
            img = Image.fromarray(arr, mode="L")
            return img
        else:
            arr = arr.reshape(height, width, channels)
            return Image.fromarray(arr, mode="RGB")

    elif arr.ndim == 2:
        img = Image.fromarray(arr, mode="L")
        return img

    elif arr.ndim == 3:
        # Assume H×W×C
        if arr.shape[2] == 1:
            img = Image.fromarray(arr.squeeze(-1), mode="L")
            return img
        elif arr.shape[2] == 3:
            return Image.fromarray(arr, mode="RGB")
        else:
            raise ValueError(f"Unsupported channel count: {arr.shape[2]}")

    else:
        raise ValueError(f"Unsupported pixel array ndim={arr.ndim}")


def show_image(
    image: Union[str, Path, Image.Image],
    labels: Optional[Sequence[Union[int, str]]] = None,     # same length as bboxes
    bboxes: Optional[Sequence[Sequence[float]]] = None,     # each box length 4
    captions: Optional[Union[str, Sequence[str]]] = None,   # optional
    class_to_name: Optional[Dict[int, str]] = None,         # map id->name if labels are ints
    figsize=(10, 8),
    linewidth: float = 2.0,
    font_size: int = 10,
) -> None:
    """
    Render image with optional bounding boxes + labels + captions.

    Args:
      image: path or PIL.Image
      labels: list of labels (str or int). If int and class_to_name provided, will map to names.
      bboxes: list of boxes; must match len(labels) if labels provided.
      captions: str or list[str] to show in title.
      boxes_format: "xywh" (COCO) or "xyxy".
      class_to_name: optional mapping for int labels -> human-readable names.
    """
    img = _to_pil(image)
    W, H = img.size

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis("off")

    # draw boxes (if any)
    if bboxes is not None and len(bboxes) > 0:
        if labels is None:
            labels = ["" for _ in bboxes]
        if len(labels) != len(bboxes):
            raise ValueError(f"labels and bboxes must have same length; got {len(labels)} vs {len(bboxes)}")

        for lab, box in zip(labels, bboxes):
            x1, y1, x2, y2 = box

            # clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)

            rect = patches.Rectangle(
                (x1, y1),
                max(1.0, x2 - x1),
                max(1.0, y2 - y1),
                fill=False,
                linewidth=linewidth
            )
            ax.add_patch(rect)

            # label text
            if lab is not None and lab != "":
                if isinstance(lab, int) and class_to_name is not None:
                    txt = class_to_name.get(lab, str(lab))
                else:
                    txt = str(lab)
                ax.text(
                    x1, max(0, y1 - 3),
                    txt,
                    fontsize=font_size,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
                )
    elif labels is not None:
        for lab in labels:
            if isinstance(lab, int) and class_to_name is not None:
                txt = class_to_name.get(lab, str(lab))
            else:
                txt = str(lab)
            ax.set_title(txt)

    # title with captions (optional)
    title_bits: List[str] = [Path(image).name if isinstance(image, (str, Path)) else "image"]
    if captions:
        if isinstance(captions, str):
            title_bits.append(f"caption: {captions}")
        else:
            title_bits.append(f"captions: {len(captions)}")
    plt.show()


def show_images(
    images: Sequence[Union[str, Path, Image.Image]],
    labels_list: Optional[Sequence[Optional[Sequence[Union[int, str]]]]] = None,  # list per image
    bboxes_list: Optional[Sequence[Optional[Sequence[Sequence[float]]]]] = None,  # list per image
    captions_list: Optional[Sequence[Optional[Union[str, Sequence[str]]]]] = None, # list per image
    class_to_name: Optional[Dict[int, str]] = None,
    linewidth: float = 2.0,
    font_size: int = 10,
    max_per_row: int = 6,
    figsize_per_img=(4, 4),
) -> None:
    """
    Display multiple images in a grid (max_per_row per row) with optional bboxes, labels, captions.
    """
    n = len(images)
    rows = math.ceil(n / max_per_row)
    fig, axs = plt.subplots(
        rows,
        min(max_per_row, n),
        figsize=(figsize_per_img[0] * min(max_per_row, n), figsize_per_img[1] * rows)
    )
    if rows == 1:
        axs = [axs] if n == 1 else axs
    axs = np.array(axs).reshape(-1)  # flatten for indexing

    for idx, img in enumerate(images):
        ax = axs[idx]
        pil_img = _to_pil(img)
        W, H = pil_img.size
        ax.imshow(pil_img)
        ax.axis("off")

        labels = labels_list[idx] if labels_list is not None else None
        bboxes = bboxes_list[idx] if bboxes_list is not None else None
        captions = captions_list[idx] if captions_list is not None else None

        # draw bounding boxes if provided
        if bboxes is not None and len(bboxes) > 0:
            if labels is None:
                labels = ["" for _ in bboxes]
            for lab, box in zip(labels, bboxes):
                x1, y1, x2, y2 = box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W - 1, x2), min(H - 1, y2)

                rect = patches.Rectangle(
                    (x1, y1),
                    max(1.0, x2 - x1),
                    max(1.0, y2 - y1),
                    fill=False,
                    color="red",
                    linewidth=linewidth
                )
                ax.add_patch(rect)

                if lab is not None and lab != "":
                    if isinstance(lab, int) and class_to_name is not None:
                        txt = class_to_name.get(lab, str(lab))
                    else:
                        txt = str(lab)
                    ax.text(
                        x1, max(0, y1 - 3),
                        txt,
                        fontsize=font_size,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
                    )
        elif labels is not None:
            if isinstance(labels, int):
                labels = [labels]
            # No boxes → just title with label
            txt = ", ".join(
                class_to_name.get(l, str(l)) if isinstance(l, int) and class_to_name else str(l)
                for l in labels
            )
            ax.set_title(txt, fontsize=font_size)

        # Add captions in title if given
        if captions:
            if isinstance(captions, str):
                ax.set_title(f"{ax.get_title()} | {captions}", fontsize=font_size)
            else:
                ax.set_title(f"{ax.get_title()} | captions: {len(captions)}", fontsize=font_size)

    # Hide any unused subplots
    for j in range(n, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()
