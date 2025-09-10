from __future__ import annotations

from optparse import Option
from typing import Any, Dict, Optional
from PIL import Image
import torch

try:
    import timm  # type: ignore
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
except Exception as e:
    raise ImportError("timm is required. Install with extras: psychai[vision-timm]") from e

try:
    from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoModelForSemanticSegmentation, AutoConfig
except Exception as e:
    raise ImportError("transformers is required. Install with extras: psychai[vision-hf]") from e

class Vision_ModelManager:
    def __init__(self):
        self.model_name = None
        self.task = None
        self.model = None
        self.train_transform = None
        self.eval_transform = None

    def load_model(self, 
                   model_name: str,
                   task: str,
                   platform: str,
                   *,
                   model_path: Optional[str] = None,
                   pretrained: Optional[bool] = True,
                   checkpoint_path: Optional[str] = None,
                   label2id: Optional[Dict[str, int]] = None,
                   trust_remote_code: Optional[bool] = None):
        self.free_memory()
        self.model_name = model_name
        self.task = task
        if platform == "timm":
            self.model, self.train_transform, self.eval_transform = load_model_from_timm(model_name, pretrained, checkpoint_path, label2id)
    
    def free_memory(self) -> None:
        if hasattr(self, 'model') and self.model is not None:
            try:
                del self.model
                print("✅ Current model deleted")
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()
        self.model = None
        self.train_transform = None
        self.eval_transform = None
        self.model_name = None
        self.task = None
        print("✅ Cache cleared")

def load_model_from_timm(model_name: str, pretrained: bool, checkpoint_path: Optional[str] = None, label2id: Optional[Dict[str, int]] = None):
    print(f"🚀 Loading model from {model_name} with pretrained={pretrained} and checkpoint_path={checkpoint_path}")
    if label2id is None:
        model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)
    else:
        num_classes = len(label2id)
        model = timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained, checkpoint_path=checkpoint_path)
    print(f"😎 Model loaded")
    cfg = resolve_data_config({}, model=model)
    train_transform = create_transform(**cfg, is_training=True)
    eval_transform = create_transform(**cfg, is_training=False)
    print(f"😎 Transforms created")
    print_timm_model_info(model)
    return model, train_transform, eval_transform

def print_timm_model_info(model):
    print("=" * 60)
    print(f"📦 Model Class : {model.__class__.__name__}")
    print("=" * 60)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔢 Total Params    : {total_params:,}")
    print(f"🟢 Trainable Params: {trainable_params:,}")
    print("=" * 60)

    if hasattr(model, "get_classifier"):
        head = model.get_classifier()
        print("🎯 Classifier head:")
        print(head)
        if hasattr(head, "in_features"):
            print(f"   • in_features : {head.in_features}")
        if hasattr(head, "out_features"):
            print(f"   • out_features: {head.out_features}")
    else:
        print("🎯 No classifier head found")

    print("=" * 60)
