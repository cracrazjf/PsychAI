import json
from PIL import Image
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterator, Optional, List
from .utils import pixels_to_pil, read_json, read_jsonl, read_csv, to_hwc

def load_any(file_path: str, 
             file_type: str = "json", 
             task: str = "image_classification",
             *,
             image_root: Optional[str] = None, 
             pixel_layout: Optional[str] = None,
             label2id: Optional[Dict[str, int]] = None,
             ) -> Iterator[Dict]:

    if file_type == "csv":
        row_dicts = read_csv(file_path)
    elif file_type == "jsonl":
        row_dicts = read_jsonl(file_path)
    elif file_type == "json":
        row_dicts = read_json(file_path)
    else:
        raise ValueError(f"Invalid file type: {file_type}")

    for row in row_dicts:
        if "image_path" in row:
            if image_root is None:
                raise ValueError("image_root is required when image_path is provided")
            row["image"] = Image.open(os.path.join(image_root, row["image_path"]))
        elif "image_pixels" in row:
            row["image"] = pixels_to_pil(row["image_pixels"], pixel_layout=pixel_layout)
        if task == "image_classification":
            if "labels" not in row:
                raise ValueError("labels is required when task is image_classification")
            try:
                row["labels"] = int(row["labels"])
            except ValueError:
                if label2id is not None:
                    row["labels"] = label2id[row["labels"]]
                else:
                    raise ValueError(f"labels is not integers and label2id is not provided")
        elif task == "detection":
            if "annotations" not in row:
                raise ValueError("annotations is required when task is detection")
            if "bboxes" not in row["annotations"]:
                raise ValueError("bboxes is required when task is detection")
            if "labels" not in row["annotations"]:
                raise ValueError("labels is required when task is detection")
            if isinstance(row["annotations"]["labels"], list):
                try:
                    row["annotations"]["labels"] = [int(label) for label in row["annotations"]["labels"]]
                except ValueError:
                    if label2id is not None:
                        row["annotations"]["labels"] = [label2id[label] for label in row["annotations"]["labels"]]
                    else:
                        raise ValueError(f"labels is not integers and label2id is not provided")
            else:
                raise ValueError(f"labels must be a list")
        yield row

# COCO to JSONL
def load_coco(
    file_path: str,
    image_root: str,
    task: str = "detection",
    bbox_format: str = "xywh"
):
    p = Path(file_path)
    with p.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    id2label = {int(c["id"]): str(c["name"]) for c in coco.get("categories", [])}
    label2id = {v: k for k, v in id2label.items()}
    img_map = {img["id"]: img["file_name"] for img in coco["images"]}

    ann_map = defaultdict(list)
    for ann in coco.get("annotations", []):
        ann_map[ann["image_id"]].append(ann)

    def row_iter() -> Iterator[Dict]:
        for img_id, file_name in img_map.items():
            anns = ann_map.get(img_id, [])
            boxes: List[List[float]] = []
            categories:   List[int] = []
            iscrowd: List[int] = []
            areas:   List[float] = []
            segments: List = []

            for ann in anns:
                if "bbox" in ann:
                    if bbox_format == "xywh":
                        boxes.append(ann["bbox"])
                    elif bbox_format == "xyxy":
                        x, y, w, h = ann["bbox"]
                        boxes.append([x, y, x + w, y + h])
                categories.append(ann["category_id"])
                if "iscrowd" in ann:
                    iscrowd.append(float(ann["iscrowd"]))
                if "area" in ann:
                    areas.append(float(ann["area"]))
                if "segmentation" in ann:
                    segments.append(ann["segmentation"])

                row: {
                    "image_id": img_id,
                    "image_path": os.path.join(image_root, file_name),
                    "annotations": {
                        "categories": categories,
                    }
                }
                if task == "detection":
                    row["annotations"]["boxes"] = boxes
                    row["annotations"]["iscrowd"] = iscrowd
                    row["annotations"]["areas"] = areas
                elif task == "segmentation":
                    row["annotations"]["segments"] = segments
                yield row

    return row_iter(), label2id, id2label


