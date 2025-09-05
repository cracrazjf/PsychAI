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
    def __init__(self, model_name: str, loading_from :str = "timm", pretrained = True ,task: str = "classification"):
        self.model_name = model_name
        self.loading_from = loading_from
        self.pretrained = pretrained
        self.model = None
        self.train_transform = None
        self.eval_transform = None
        self.processor = None
        self.task = task

    def load_model(self, num_classes: Optional[int] = None,
                    class_names: Optional[List] = None,
                    trust_remote_code: Optional[bool] = None):
        self.free_memory()
        if self.loading_from == "timm":
            self.load_model_from_timm(num_classes)
        elif self.loading_from == "hf":
            self.load_model_from_hf(trust_remote_code, class_names)
        print("Model loaded successfully")
    
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
        self.processor = None
        self.model_name = None
        print("✅ Cache cleared")
    
    def print_timm_model_info(self, input_size=(1,3,224,224)):
        """
        Pretty print summary of a TIMM model:
        - Model name / type
        - Input size
        - Number of params (total/trainable)
        - Classifier head details
        - Children modules
        """
        print("=" * 60)
        print(f"📦 Model Class : {self.model.__class__.__name__}")
        print(f"🖼️ Input Size  : {input_size}")
        print("=" * 60)

        # Param counts
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"🔢 Total Params    : {total_params:,}")
        print(f"🟢 Trainable Params: {trainable_params:,}")
        print("=" * 60)

        # Classifier / head (if present)
        if hasattr(self.model, "get_classifier"):
            head = self.model.get_classifier()
            print("🎯 Classifier head:")
            print(head)
            if hasattr(head, "in_features"):
                print(f"   • in_features : {head.in_features}")
            if hasattr(head, "out_features"):
                print(f"   • out_features: {head.out_features}")
        else:
            print("🎯 No classifier head found")

        print("=" * 60)

    def load_model_from_timm(self, num_classes: Optional[int] = None):
        if num_classes is None:
            self.model = timm.create_model(self.model_name, pretrained=self.pretrained)
        else:
            self.model = timm.create_model(self.model_name, pretrained=self.pretrained, num_classes=num_classes)
        cfg = resolve_data_config({}, model=self.model)
        self.train_transform = create_transform(**cfg, is_training=True)
        self.eval_transform = create_transform(**cfg, is_training=False)
        self.print_timm_model_info(input_size=(1,3,224,224))

    def print_hf_model_config(self, config):
        """
        Nicely print key fields of Hugging Face configs for detection/segmentation models.
        """
        cfg = config.to_dict()
        print("=" * 60)
        print(f"📦 Model Type: {cfg.get('model_type', 'unknown')}")
        print(f"🧩 Task : {task}")
        print("=" * 60)

        # Common fields
        print(f"🔢 num_labels        : {cfg.get('num_labels')}")
        print(f"📖 id2label          : {cfg.get('id2label')}")
        print(f"📖 label2id          : {cfg.get('label2id')}")
        print()

        if self.task == "detection":
            # DETR / detection-specific fields
            print("⚡ Detection-specific:")
            print(f"   • num_queries     : {cfg.get('num_queries')}")
            print(f"   • hidden_size     : {cfg.get('hidden_size')}")
            print(f"   • encoder_layers  : {cfg.get('encoder_layers')}")
            print(f"   • decoder_layers  : {cfg.get('decoder_layers')}")
            print(f"   • attention_heads : {cfg.get('encoder_attention_heads')}")
            print(f"   • backbone        : {cfg.get('backbone', 'N/A')}")
            print()

        elif self.task == "segmentation":
            # SegFormer / semantic segmentation fields
            print("🎨 Segmentation-specific:")
            print(f"   • backbone        : {cfg.get('backbone', 'N/A')}")
            print(f"   • hidden_sizes    : {cfg.get('hidden_sizes')}")
            print(f"   • depths          : {cfg.get('depths')}")
            print(f"   • strides         : {cfg.get('strides')}")
            print(f"   • decoder_hidden_size : {cfg.get('decoder_hidden_size')}")
            print()

        print("=" * 60)
        print("🔧 Other important fields:")
        keep_keys = ["image_size", "pad_token_id", "dropout", 
                    "hidden_dropout_prob", "attention_probs_dropout_prob"]
        for k in keep_keys:
            if k in cfg:
                print(f"   • {k}: {cfg[k]}")

        print("=" * 60)

    def load_model_from_hf(self, trust_remote_code: bool, class_names: Optional[List] = None): 
        self.processor = AutoImageProcessor.from_pretrained(self.model_name, use_fast=True)
        if class_names is not None:
            id2label   = {i: name for i, name in enumerate(class_names)}
            label2id   = {v: k for k, v in id2label.items()}
            num_classes = len(class_names)

        if self.task == "detection":
            if num_classes is not None:
                self.model = AutoModelForObjectDetection.from_pretrained(model_name, 
                                                                    num_labels=num_classes,
                                                                    id2label=id2label,
                                                                    label2id=label2id,
                                                                    ignore_mismatched_sizes=True,
                                                                    trust_remote_code=trust_remote_code,
                                                                    pretrained=self.pretrained)
            else:
                self.model = AutoModelForObjectDetection.from_pretrained(model_name, pretrained=self.pretrained)
        elif self.task == "segmentation":
            if num_classes is not None:
                self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name, 
                                                                        num_labels=num_classes,
                                                                        id2label=id2label,
                                                                        label2id=label2id,
                                                                        ignore_mismatched_sizes=True,
                                                                        trust_remote_code=trust_remote_code,
                                                                        pretrained=self.pretrained)
            else:
                self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name, pretrained=self.pretrained)
        else:
            raise ValueError(f"Invalid task: {task}")
