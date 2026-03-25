import gc
import timm
import torch
import random
import numpy as np
import json
import os
from timm.data import resolve_model_data_config
from torch.utils.data import DataLoader
from typing import Any, Optional
from ..nn_builder import ClassificationModelWrapper, Model, build_spec_from_config, from_pretrained, load_config, save_pretrained
from ..language.utils import to_serializable, save_checkpoint, clean_dir

class ModelManager:
    def __init__(self):
        self.model = None,
        self.model_name = None
        self.model_path = None
        self.model_type = None
        self.model_wrapper = None
        self.model_config = None

    def load_model(self, 
                   model_name: str, 
                   model_path: str,
                   model_type: str,
                   wrapper: Optional[str] = None,
                   device = "cpu",
                   task: str = "classification") -> None:
        
        self.free_memory()

        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type
        self.wrapper = wrapper

        if "custom" in model_type:
            wrapper_map = {
                "classification": ClassificationModelWrapper,
            }
            ctor = wrapper_map.get(wrapper, None)
            try:
                self.model = from_pretrained(self.model_path)
            except Exception:
                print(f"Model not found, rebuilding model from config")

                config = load_config(self.model_path)
                model = build_spec_from_config(config)  
                self.model = Model(model)
            print(self.model.summary())
            if ctor is not None:
                self.model = ctor(self.model)
                print(f"Model wrapped with {ctor}")
        elif "timm" in model_type:
            self.model = timm.create_model(model_name, pretrained=True, features_only= (task== "feature_extraction"))
            self.model_config = resolve_model_data_config(model=self.model)
        self.model.to(device)

        print(f"Model {model_name} loaded on {device}.")

    def free_memory(self) -> None:
        if hasattr(self, 'model') and self.model is not None:
            try:
                del self.model
                print("Current model deleted")
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()

        self.model = None
        self.model_name = None
        self.model_path = None
        self.model_type = None
        self.model_config = None
        print("Cache cleared")

class TrainingManager:
    def __init__(self, 
                 cfg):
         self.cfg = cfg
         self.mm = ModelManager()

    def configure_optimizer(self):
        if self.cfg.optim.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(self.mm.model.parameters(), 
                              lr=self.cfg.optim.lr, 
                              weight_decay=self.cfg.optim.weight_decay)
        elif self.cfg.optim.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.mm.model.parameters(), 
                                         lr=self.cfg.optim.lr, 
                                         weight_decay=self.cfg.optim.weight_decay)
        elif self.cfg.optim.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.mm.model.parameters(), 
                            lr=self.cfg.optim.lr, 
                            momentum=self.cfg.optim.momentum,
                            weight_decay=self.cfg.optim.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.cfg.optim.optimizer}")
        
        if self.cfg.optim.lr_scheduler == "multistep":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                    milestones=self.cfg.optim.lr_steps,
                    gamma=self.cfg.optim.gamma,
                )
        else:
            self.scheduler = None

    def train_epoch(self, epoch: int, train_loader: DataLoader, val_loader: DataLoader):
        device = next(self.mm.model.parameters()).device
        self.mm.model.train()

        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            inputs = batch['pixel_values'].to(device)
            labels = batch['labels'].long().to(device)

            if self.cfg.model.wrapper == "classification":
                outputs = self.mm.model(inputs, labels=labels)
                epoch_loss += outputs["loss"].item()
            else:
                raise NotImplementedError(f"Task {self.cfg.task} not implemented yet")

            self.optimizer.zero_grad()
            outputs["loss"].backward()
            if self.cfg.optim.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.mm.model.parameters(), self.cfg.optim.grad_clip)
            self.optimizer.step()

            if self.cfg.logging.log_strategy == "step":
                if (step + 1) % self.cfg.logging.log_interval == 0:
                    train_info = {"epoch": epoch_loss / (step + 1), "step": step + 1}
                    with open(self.log_path, "a") as f:
                        f.write(json.dumps(train_info) + "\n")

                if (step + 1) % self.cfg.logging.eval_interval == 0:
                    self.evaluate(val_loader, self.eval_fn, epoch, step=step + 1, eval_path=self.eval_path)

        if self.scheduler is not None:
            self.scheduler.step()
        return {"epoch_loss": epoch_loss / len(train_loader)}

    def evaluate(self, 
                 dataloader: DataLoader, 
                 eval_fn: Optional[Any]=None, 
                 epoch: Optional[int]=0, 
                 step: Optional[int]=0, 
                 eval_path: Optional[str]=None):
        
        device = next(self.mm.model.parameters()).device
        self.mm.model.eval()

        weights = None
        if self.cfg.logging.return_weights:
            weights = self.mm.model.get_weights()

        eval_loss = 0.0
        accuracy = 0.0
        total = 0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                idxs = batch.get("idx", None)
                inputs = batch['pixel_values'].to(device)
                labels = batch.get("labels", None)
                if labels is not None:
                    labels = batch['labels'].long().to(device)

                if self.cfg.model.wrapper == "classification":
                    outputs = self.mm.model(inputs, labels=labels, return_embeds=self.cfg.logging.return_embeddings)
                    eval_loss += outputs["loss"].item()
                    preds = outputs["logits"].argmax(dim=-1)
                    accuracy += (preds == labels).sum().item()
                    total += labels.size(0)
                    if self.cfg.logging.return_embeddings:
                        embeds = outputs["embeds"][self.cfg.logging.layer_of_interest][self.cfg.logging.embed_type]
                    else:
                        embeds = outputs["embeds"]
                else:
                    embeds = self.mm.model(inputs)
                    
                if eval_fn is not None:
                    eval_fn(self.mm, 
                            self.cfg, 
                            idxs,
                            outputs["inputs"], 
                            outputs["labels"] if "labels" in outputs else None, 
                            outputs["logits"],
                            preds,
                            embeds,
                            weights)

        eval_info = {"epoch": epoch + 1, "step": step, "batch": i, "eval_loss": eval_loss / len(dataloader), "accuracy": accuracy / total}   
        if eval_path is not None:
            os.makedirs(os.path.dirname(eval_path), exist_ok=True)        
            with open(eval_path, "w") as f:
                f.write(json.dumps(eval_info) + "\n")

    def train(self, train_loader: DataLoader, val_loader: DataLoader, eval_fn: Optional[Any] = None):
        self.eval_fn = eval_fn
        for run in range(self.cfg.num_runs):
            seed = self.cfg.seed + run
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            self.run_dir = None
            self.log_path = None
            if self.cfg.exp_dir is not None:
                self.run_dir = os.path.join(self.cfg.exp_dir, f"run_{run+1}")
                os.makedirs(self.run_dir, exist_ok=True)
                self.log_path = os.path.join(self.run_dir, "log.jsonl")
                self.eval_path = os.path.join(self.run_dir, "eval_results.json")
                with open(self.log_path, "w") as f:
                    f.write("")
                with open(self.eval_path, "w") as f:
                    f.write("")
        
            self.mm.load_model(
                model_name=self.cfg.model.name,
                model_path=self.cfg.model.path,
                model_type=self.cfg.model.model_type,
                wrapper=self.cfg.model.wrapper,
                device=self.cfg.device,
                task=self.cfg.task
            )

            self.configure_optimizer()

            for epoch in range(self.cfg.num_epochs):
                train_info = self.train_epoch(epoch, train_loader, val_loader)

                if self.cfg.logging.eval_strategy == "epoch":
                    if (epoch + 1) % self.cfg.logging.log_interval == 0:
                        with open(self.log_path, "a") as f:
                            f.write(json.dumps(train_info) + "\n")

                    if (epoch + 1) % self.cfg.logging.eval_interval == 0:
                        self.evaluate(val_loader, self.eval_fn, epoch, step=0, eval_path=self.eval_path)

                if (epoch + 1) % self.cfg.logging.save_interval == 0:
                        save_checkpoint(self.run_dir, 
                                        self.mm.model, 
                                        optimizer=self.optimizer,
                                        scaler=None,
                                        epoch=epoch,
                                        max_to_keep=self.cfg.logging.save_total_limit,
                                        prefer_safetensors=self.cfg.logging.prefer_safetensors
                                        )

            if self.cfg.logging.save_model:
                    save_dir = os.path.join(self.run_dir, "export")
                    clean_dir(save_dir)
                    save_pretrained(self.mm.model, save_dir, prefer_safetensors=self.cfg.logging.prefer_safetensors)
                    print(f"model saved!")



                

