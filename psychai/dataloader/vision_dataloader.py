from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, List, Callable
from pathlib import Path
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from ..utils.image import to_hwc, pixels_to_pil, is_image
from ..utils.utils import is_all_int
import hashlib
import csv
from functools import partial
import pickle

@dataclass
class Record:
    id: str
    image: Any
    label: Optional[Any] = None
    text: Optional[str] = None

def stable_id(*parts: Any) -> str:
    """Deterministic ID via BLAKE2b over input parts."""
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        if isinstance(p, (bytes, bytearray)):
            h.update(p)
        else:
            h.update(str(p).encode("utf-8", errors="ignore"))
            h.update(b"|")
    return h.hexdigest()

def normalize_label(label, class_to_idx: Optional[Dict[str, int]] = None):
    """Map labels into numeric space if needed."""
    if label is None:
        return None
    if isinstance(label, list):
        return [class_to_idx[str(l)] if class_to_idx else l for l in label]
    return class_to_idx[str(label)] if class_to_idx else label

def get_image_info(obj: Dict[str, Any], *, config) -> Image.Image:
    image = None
    if config.IMAGE_PATH_NAME is not None:
        image_path = config.IMAGE_PATH_NAME
        root = config.IMAGE_ROOT_PATH
        if obj.get(image_path) is not None:
            img_path = Path(image_path)
            if root is not None:
                img_path = Path(root) / img_path
            img_path = str(img_path)
            image = Image.open(img_path)
    elif config.IMAGE_PIXEL_NAME is not None:
        image_pixel = config.IMAGE_PIXEL_NAME
        image_layout = config.IMAGE_LAYOUT
        image_width = config.IMAGE_WIDTH
        image_height = config.IMAGE_HEIGHT
        image_channels = config.IMAGE_CHANNELS
        if obj.get(image_pixel) is not None:
            image_pixels = obj.get(image_pixel)
            if image_layout != "HWC":
                image_pixels = to_hwc(image_pixels, width=image_width, height=image_height, channels=image_channels)
            image = pixels_to_pil(image_pixels, image_width, image_height, image_channels)
    return image

def to_record(obj: Dict[str, Any], *, config, source: str) -> Record:
    image = get_image_info(obj, config=config)

    label = None
    if config.IMAGE_LABEL_NAME is not None:
        label_name = config.IMAGE_LABEL_NAME
        if obj.get(label_name) is not None:
            label = obj.get(label_name)
            if not is_all_int(label):
                if config.CLASS_TO_IDX is not None:
                    label = normalize_label(label, config.CLASS_TO_IDX)
                else:
                    raise ValueError(f"Label {label_name} is not all ints and no class_to_idx is provided")
            else:
                label = int(label)
    text = None
    if config.IMAGE_TEXT_NAME is not None:
        text_name = config.IMAGE_TEXT_NAME
        if obj.get(text_name) is not None:
            text = obj.get(text_name)

    meta = None
    if config.IMAGE_META_NAME is not None:
        meta_name = config.IMAGE_META_NAME
        if obj.get(meta_name) is not None:
            if obj.get("image_id") is not None:
                meta = {"image_id": obj.get("image_id"), meta_name: obj.get(meta_name)}
            else:
                meta = {meta_name: obj.get(meta_name)}

    id = stable_id(source, text)
    return Record(id, image, label, text, meta)

def read_jsonl(path: str, config) -> Iterator[Record]:
    if path.endswith(".jsonl"):
        with Path(path).open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                if obj.get("image_id") is not None:
                    source = f"{path}#{obj.get('image_id')}"
                else:
                    source = f"{path}#{i}"
                yield to_record(obj, config=config, source=source)
    else:
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
            for i, obj in enumerate(data):
                if obj.get("image_id") is not None:
                    source = f"{path}#{obj.get('image_id')}"
                else:
                    source = f"{path}#{i}"
                yield to_record(obj, config=config, source=source)

def read_csv(path:str, config) -> Iterator[Record]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if row.get("image_id") is not None:
                source = f"{path}#{row.get('image_id')}"
            else:
                source = f"{path}#{i}"
            yield to_record(row, config=config, source=source)

def read_folder(path: str, config) -> Iterator[Record]:
    classes = [d.name for d in Path(path).iterdir() if d.is_dir()]
    classes = sorted(classes)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    for cls_name in classes:
        cls_dir = Path(path) / cls_name
        for p in cls_dir.rglob("*"):
            if is_image(p):
                id = stable_id(path, cls_name, p.name)
                yield Record(
                    id=id,
                    image=Image.open(p),
                    label=class_to_idx[cls_name],
                    text=None,
                    meta=None
                )

def prepare_and_split_data(records: List[Record], train_p=0.7, val_p: Optional[float] = None):
    train, val, test = [], [], []
    for r in records:
        # prefer r.id if you added it; else pull from meta
        rid = getattr(r, "id", None) or (r.meta or {}).get("id")
        if not rid:
            raise ValueError("Record missing stable id; add one during to_fieldmap().")
        bucket = (int(rid[:8], 16) % 1000) / 1000.0
        if val_p is None:
            if bucket < train_p:
                train.append(r)
            else:
                test.append(r)
        else:
            if bucket < train_p:
                train.append(r)
            elif bucket < train_p + val_p:
                val.append(r)
            else:
                test.append(r)
    if val_p is None:
        print(f"split {len(train)} data into train, {len(test)} data into test")
        return train, test
    else:   
        print(f"split {len(train)} data into train, {len(val)} data into val, {len(test)} data into test")
        return train, val, test

class ImageDataset(Dataset):
    def __init__(self, records: List[Record], transform, config):
        self.records = records
        self.transform = transform
        self.config = config
        self.to_rgb = config.TO_RGB

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        image = r.image
        if self.to_rgb:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = r.label
        text = r.text
        return {"image": image, "label": label, "text": text}

def _collate_classification(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = [b["image"] for b in batch]
    if isinstance(images[0], torch.Tensor):
        try:
            images = torch.stack(images, dim=0)
        except Exception:
            pass
    result: Dict[str, Any] = {"image": images}
    if "label" in batch[0]:
        labels = [b.get("label") for b in batch]
        if labels[0] is not None:
            result["label"] = torch.tensor(labels, dtype=torch.long)
    return result

def _collate_detection(batch: List[Dict[str, Any]], processor: Any, train: bool, device: str) -> Dict[str, Any]:
    images = [b["image"] for b in batch]
    annotations = [b["meta"] for b in batch]
    if train:
        processed = processor(images=images, annotations=annotations, return_tensors="pt")
        processed["pixel_values"] = processed["pixel_values"].to(device)
        return processed
    else:
        processed = processor(images=images, return_tensors="pt")
        sizes = [(im.height, im.width) for im in images]
        img_ids = [m.get("image_id", i) for i, m in enumerate(annotations)]
        return {"pixel_values": processed["pixel_values"], "sizes": sizes, "image_ids": img_ids}

def _collate_segmentation(batch: List[Dict[str, Any]], processor: Any, device: str) -> Dict[str, Any]:
    images = [b["image"] for b in batch]
    segmentation_maps = [b["meta"] for b in batch]
    processed = processor(images=images, segmentation_maps=segmentation_maps, return_tensors="pt")
    processed["pixel_values"] = processed["pixel_values"].to(device)
    processed["labels"] = processed["labels"].to(device)
    processed["_orig_sizes"] = [(im.height, im.width) for im in images]
    return processed

def create_dataloader(config, 
                      *,
                      train_data: List[Record], 
                      eval_data: Optional[List[Record]] = None, 
                      train_transform: Optional[Callable] = None,
                      eval_transform: Optional[Callable] = None,
                      processor: Optional[Any] = None,
                      device: str =  "cuda" if torch.cuda.is_available() else "cpu"):

    task_type = config.TASK_TYPE
    batch_size = config.BATCH_SIZE
    num_workers = config.DATALOADER_WORKERS
    pin_memory = config.PIN_MEMORY
    drop_last = config.DROP_LAST   

    
    train_dataset = ImageDataset(train_data, train_transform, config)
    eval_dataset = ImageDataset(eval_data, eval_transform, config) if eval_data is not None else None
    print(f"Found {len(train_dataset)} numbers of train data, Found {len(eval_dataset)} numbers of eval data")

    if pin_memory is None:
        pin_memory = torch.cuda.is_available() 

    if task_type == "classification":
        _collate_fn = partial(_collate_classification)
    elif task_type == "detection":
        _collate_fn = partial(_collate_detection, processor=processor, device=device)
    elif task_type == "segmentation":
        _collate_fn = partial(_collate_segmentation, processor=processor, device=device)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=_collate_fn,
    )

    if task_type == "detection":
        _collate_fn = partial(_collate_detection, processor=processor, device=device, train=False)

    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=_collate_fn,
        )
    else:
        eval_loader = None
    return train_loader, eval_loader

# COCO to JSONL
def coco_to_jsonl(
    coco_json_path: str,
    out_path: Optional[str] = None,
    include_segmentations: bool = False,
    include_captions: bool = False
):
    coco_json_path = Path(coco_json_path)
    
    with coco_json_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    # Map IDs
    cat_map = {c["id"]  : c["name"] for c in coco.get("categories", [])}
    img_map = {img["id"]: img["file_name"] for img in coco["images"]}
    num_classes = coco.get("categories", [])[-1]["id"]

    if out_path is None:
        return {v: k for k, v in cat_map.items()}, cat_map, num_classes
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Group annotations by image
    ann_map = defaultdict(list)
    for ann in coco.get("annotations", []):
        rec = {"category_id": ann["category_id"]}
        if "bbox" in ann:
            rec["bbox"] = ann["bbox"]  # COCO format [x, y, w, h]
        if "area" in ann:
            rec["area"] = ann["area"]
        if "iscrowd" in ann:
            rec["iscrowd"] = ann["iscrowd"]
        if include_segmentations and "segmentation" in ann:
            rec["segmentation"] = ann["segmentation"]
        ann_map[ann["image_id"]].append(rec)

    # Group captions by image if requested
    cap_map = defaultdict(list)
    if include_captions and "captions" in coco:
        for cap in coco["captions"]:
            cap_map[cap["image_id"]].append(cap["caption"])

    # Write JSONL
    with out_path.open("w", encoding="utf-8") as out_f:
        for img_id, fname in img_map.items():
            record = {
                "image_id": img_id,
                "image_path": str(fname),
                "labels": list({ann["category_id"] for ann in ann_map[img_id]}),
                "annotations": ann_map[img_id],
            }
            if include_captions and cap_map[img_id]:
                record["captions"] = cap_map[img_id]
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {out_path} with {len(img_map)} records.")
    return {v: k for k, v in cat_map.items()}, cat_map, num_classes

# CIFAR-100 to JSONL
def cifar_to_jsonl(
    meta_path: str,
    cifar_path: Optional[str] = None,    
    meta_label_key: str = b'fine_label_names',
    data_key: str = b'data',
    data_label_key: str = b'fine_labels',
    out_path: str = None,
):
    with open(meta_path, 'rb') as fo:
        meta = pickle.load(fo, encoding='bytes')
    idx_to_class = {k: v.decode('utf-8') for k, v in enumerate(meta[meta_label_key])}
    class_to_idx = {v: k for k, v in idx_to_class.items()}
    if out_path is None:
        return class_to_idx, idx_to_class

    with open(cifar_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    if out_path is not None:
        save_to_jsonl(
            dict[data_key],
            [idx_to_class[i] for i in dict[data_label_key]],
            out_path
        )
    return class_to_idx, idx_to_class

