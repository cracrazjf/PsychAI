import json
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch

def load_folders(
        folder_path: str,
        jsonl_path: str,
        exts={".jpg", ".jpeg", ".png"}):
    
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder not found: {folder}")
    subdirs = [d for d in folder.iterdir() if d.is_dir()]
    labeled = len(subdirs) > 0

    items = []
    class_to_idx = {}

    if labeled:
        class_names = sorted([d.name for d in subdirs])
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    else:
        class_to_idx = None

    count = 0
    with open(jsonl_path, "w") as f:
        if labeled:
            for class_name in class_names:
                for img_path in (folder / class_name).iterdir():
                    if img_path.suffix.lower() in exts:
                        img = Image.open(img_path).convert("RGB")
                        px = transform(img).tolist()

                        entry = {
                            "path": str(img_path),
                            "pixel_values": px,
                            "label": class_to_idx[class_name],
                            "class_name": class_name,
                        }

                        f.write(json.dumps(entry) + "\n")
                        count += 1
        else:
            for img_path in folder.iterdir():
                if img_path.suffix.lower() in exts:
                    img = Image.open(img_path).convert("RGB")
                    px = transform(img).tolist()

                    entry = {
                        "path": str(img_path),
                        "pixel_values": px
                    }

                    f.write(json.dumps(entry) + "\n")
                    count += 1

    print(f"Wrote {count} images to {jsonl_path}")
    return class_to_idx

        