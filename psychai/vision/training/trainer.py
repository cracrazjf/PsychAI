from __future__ import annotations

from ...config import TrainingConfig
from typing import Any, Optional, List
from ..models import load_pretrained_hf_vision
from ..data.dataloader import create_dataloader, Record
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.optim import AdamW, SGD

class HFVisionTrainer:
    def __init__(self, config: TrainingConfig = None):
        """
        Initialize trainer with configuration
        
        Args:
            config: Training configuration instance
        """
        self.config = config
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _adapt_new_classifier(self, model):
        if self.config.TASK_TYPE == "detection":
            head = getattr(model, "class_embed", None) or getattr(model, "class_labels_classifier", None)
            if head is None:
                raise ValueError("No class embed or class labels classifier found in model")
            in_features = getattr(model, head).in_features
            new_head = nn.Linear(in_features, self.config.NUM_CLASSES+1)
            _init_classifier_weights(new_head)
            if head == "class_embed":
                model.class_embed = new_head
            elif head == "class_labels_classifier":
                model.class_labels_classifier = new_head
            model.config.num_labels = self.config.NUM_CLASSES
            model.config.id2label = {i: name for i, name in enumerate(self.config.CLASS_NAMES)}
            model.config.label2id = {v: k for k, v in model.config.id2label.items()}

        elif self.config.TASK_TYPE == "segmentation":
            head = model.decode_head.classifier if hasattr(model, "decode_head") else model.sem_seg_head.classifier
            if head is None:
                raise ValueError("No decode head or sem seg head found in model")
            in_channels = head.in_channels
            model.decode_head.classifier = torch.nn.Conv2d(in_channels, self.config.NUM_CLASSES+1, kernel_size=1)
            _init_decoder_weights(model.decode_head.classifier)

            model.config.num_labels = self.config.NUM_CLASSES
            model.config.id2label = {i: name for i, name in enumerate(self.config.CLASS_NAMES)}
            model.config.label2id = {v: k for k, v in model.config.id2label.items()}
        else:
            raise ValueError(f"Unsupported task type: {self.config.TASK_TYPE}")

        def _init_classifier_weights(new_head):
            if self.config.INIT_WEIGHTS == "xavier_uniform":
                nn.init.xavier_uniform_(new_head.weight)
                nn.init.zeros_(new_head.bias)
            elif self.config.INIT_WEIGHTS == "kaiming_uniform":
                nn.init.kaiming_uniform_(new_head.weight)
                nn.init.zeros_(new_head.bias)
            else:
                raise ValueError(f"Unsupported initialization method: {self.config.INIT_WEIGHTS}")

        def _init_decoder_weights(decoder):
            if self.config.INIT_WEIGHTS == "xavier_uniform":
                nn.init.xavier_uniform_(decoder.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(decoder.bias)
            elif self.config.INIT_WEIGHTS == "kaiming_uniform":
                nn.init.kaiming_uniform_(decoder.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(decoder.bias)
            else:
                raise ValueError(f"Unsupported initialization method: {self.config.INIT_WEIGHTS}")

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
