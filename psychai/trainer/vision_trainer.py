from __future__ import annotations

from ...config import TrainingConfig
from typing import Any, Optional, List
from ..model_manager import Vision_ModelManager
from ..dataloader import create_dataloader_hf, Record
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.optim import AdamW, SGD

class Vision_Trainer:
    def __init__(self, config: TrainingConfig = None):
        """
        Initialize trainer with configuration
        
        Args:
            config: Training configuration instance
        """
        self.config = config
        self.model_manager = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_model(self):
        loading_from = self.config.loading_from
        model_name = self.config.MODEL_NAME
        pretrained = self.config.PRETRAINED
        task = self.config.TASK
        trust_remote_code = self.config.TRUST_REMOTE_CODE
        num_classes = self.config.NUM_CLASSES
        class_names = self.config.CLASS_NAMES
        self.model_manager = Vision_ModelManager(model_name, loading_from, pretrained, task)
        self.model_manager.load_model(num_classes, class_names, trust_remote_code)

    def _create_optimizer(self, model):
        if self.config.OPTIMIZER == "adamw":
            optimizer = AdamW(model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY)
        elif self.config.OPTIMIZER == "sgd":
            optimizer = SGD(model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.OPTIMIZER}")
        return optimizer

    def _train_detection_epoch(self, epoch, model, optimizer, train_loader, device, amp_dtype):
        total_loss = 0
        model.train()
        for batch_idx, enc in enumerate(train_loader):
            pixel_values = enc["pixel_values"].to(device, non_blocking=True)
        labels = enc["labels"]  # list[dict] stays on CPU; HF handles it
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type=="cuda")):
            out = model(pixel_values=pixel_values, labels=labels)
            loss = out.loss
            loss.backward(); optimizer.step()
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
        return avg_loss

    def _train_segmentation_epoch(self, epoch, model, optimizer, train_loader, device, amp_dtype):
        total_loss = 0
        model.train()
        for batch_idx, enc in enumerate(train_loader):
            pixel_values = enc["pixel_values"].to(device, non_blocking=True)
            labels = enc["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type=="cuda")):
                out = model(pixel_values=pixel_values, labels=labels)
                loss = out.loss
                loss.backward(); optimizer.step()
                total_loss += loss.item()
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self, train_data: List[Record], 
                        eval_data: Optional[List[Record]] = None, 
                        output_dir: Optional[str] = None) -> Any:
        self.model, self.processor = load_pretrained_hf_vision(self.config)
        self.model.to(self.device)
        optimizer = self._create_optimizer(self.model)
        bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        amp_dtype = torch.bfloat16 if bf16_ok else torch.float16


        train_loader, eval_loader = create_dataloader(
            config=self.config,
            train_data=train_data,
            eval_data=eval_data,
            processor=self.processor,
            device=self.device
        )
        # if self.config.NUM_CLASSES is not None:
        #     self._adapt_new_classifier(self.model)

        for epoch in range(self.config.NUM_EPOCHS):
            if self.config.TASK_TYPE == "detection":
                train_loss = self._train_detection_epoch(epoch, self.model, optimizer, train_loader, self.device, amp_dtype)
            elif self.config.TASK_TYPE == "segmentation":
                train_loss = self._train_segmentation_epoch(epoch, self.model, optimizer, train_loader, self.device, amp_dtype)
            else:
                raise ValueError(f"Unsupported task type: {self.config.TASK_TYPE}")
