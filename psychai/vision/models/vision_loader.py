from __future__ import annotations

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

def print_timm_model_info(model: nn.Module, input_size=(1,3,224,224)):
    """
    Pretty print summary of a TIMM model:
    - Model name / type
    - Input size
    - Number of params (total/trainable)
    - Classifier head details
    - Children modules
    """
    print("=" * 60)
    print(f"📦 Model Class : {model.__class__.__name__}")
    print(f"🖼️ Input Size  : {input_size}")
    print("=" * 60)

    # Param counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔢 Total Params    : {total_params:,}")
    print(f"🟢 Trainable Params: {trainable_params:,}")
    print("=" * 60)

    # Classifier / head (if present)
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

def load_pretrained_timm(config: Dict[str, Any]):
    num_classes = config.NUM_CLASSES
    model_name = config.MODEL_NAME
    pretrained = config.PRETRAINED
    if num_classes is None:
        model = timm.create_model(model_name, pretrained=pretrained)
    else:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    cfg = resolve_data_config({}, model=model)
    transform = create_transform(**cfg, is_training=True)
    transform_val = create_transform(**cfg, is_training=False)
    
    print_timm_model_info(model, input_size=(1,3,224,224))

    return model, transform, transform_val

def print_hf_model_config(config, task_type: str):
    """
    Nicely print key fields of Hugging Face configs for detection/segmentation models.
    """
    cfg = config.to_dict()
    print("=" * 60)
    print(f"📦 Model Type: {cfg.get('model_type', 'unknown')}")
    print(f"🧩 Task Type : {task_type}")
    print("=" * 60)

    # Common fields
    print(f"🔢 num_labels        : {cfg.get('num_labels')}")
    print(f"📖 id2label          : {cfg.get('id2label')}")
    print(f"📖 label2id          : {cfg.get('label2id')}")
    print()

    if task_type == "detection":
        # DETR / detection-specific fields
        print("⚡ Detection-specific:")
        print(f"   • num_queries     : {cfg.get('num_queries')}")
        print(f"   • hidden_size     : {cfg.get('hidden_size')}")
        print(f"   • encoder_layers  : {cfg.get('encoder_layers')}")
        print(f"   • decoder_layers  : {cfg.get('decoder_layers')}")
        print(f"   • attention_heads : {cfg.get('encoder_attention_heads')}")
        print(f"   • backbone        : {cfg.get('backbone', 'N/A')}")
        print()

    elif task_type == "segmentation":
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

def load_pretrained_hf_vision(config: Dict[str, Any]):
    model_name = config.MODEL_NAME
    pretrained = config.PRETRAINED
    num_classes = config.NUM_CLASSES
    
    trust_remote_code = config.TRUST_REMOTE_CODE
    task_type = config.TASK_TYPE

    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    if num_classes is not None:
        id2label   = {i: name for i, name in enumerate(config.CLASS_NAMES)}
        label2id   = {v: k for k, v in id2label.items()}

    if task_type == "detection":
        if num_classes is not None:
            model = AutoModelForObjectDetection.from_pretrained(model_name, 
                                                                num_labels=num_classes,
                                                                id2label=id2label,
                                                                label2id=label2id,
                                                                ignore_mismatched_sizes=True,
                                                                trust_remote_code=trust_remote_code)
        else:
            model = AutoModelForObjectDetection.from_pretrained(model_name, pretrained=pretrained)
    elif task_type == "segmentation":
        if num_classes is not None:
            model = AutoModelForSemanticSegmentation.from_pretrained(model_name, 
                                                                    num_labels=num_classes,
                                                                    id2label=id2label,
                                                                    label2id=label2id,
                                                                    ignore_mismatched_sizes=True,
                                                                    trust_remote_code=trust_remote_code)
        else:
            model = AutoModelForSemanticSegmentation.from_pretrained(model_name, pretrained=pretrained)
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    print_hf_model_config(model.config, task_type)
    return model, processor
