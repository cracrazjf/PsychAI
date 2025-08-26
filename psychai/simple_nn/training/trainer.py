from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from time import perf_counter
from tqdm.auto import tqdm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from ...config import TrainingConfig
from ...simple_nn.data.dataloader import Record
from torch.utils.data import DataLoader
from ...utils.utils import cuda_memory_stats

class BaseAdapter:
    def forward(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    def loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
    def metrics(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return {}

# 1) Image classification (logits [B,C], labels [B])
class ClassificationAdapter(BaseAdapter):
    def __init__(self, class_weights: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def forward(self, model, batch):
        return {"logits": model(batch["image"])}

    def loss(self, outputs, batch):
        return F.cross_entropy(
            outputs["logits"], batch["label"],
            weight=self.class_weights, label_smoothing=self.label_smoothing
        )

    def metrics(self, outputs, batch):
        pred = outputs["logits"].argmax(1)
        acc = (pred == batch["label"]).float().mean().item()
        return {"acc": acc}

# 2) Regression (pred [B,D], y [B,D] or [B])
class RegressionAdapter(BaseAdapter):
    def __init__(self): self.crit = nn.MSELoss()
    def forward(self, model, batch):
        return {"logits": model(batch["inputs"])}
    def loss(self, outputs, batch):
        return self.crit(outputs["pred"], batch["labels"])

# 3) Language modeling (word prediction & classification)
class LanguageAdapter(BaseAdapter):
    def __init__(self, task: str, ignore_index: int = -100):
        self.task = task
        self.ignore_index = ignore_index

    def forward(self, model, batch):
        # Expect model to accept (input_ids, attention_mask)-> logits [B,T,V]
        logits = model(batch["input_ids"], batch.get("attention_mask"))  # your LSTM/MLP signature
        return {"logits": logits}

    def loss(self, outputs, batch):
        B, T, V = outputs["logits"].shape
        logits = outputs["logits"].reshape(B*T, V)
        labels = batch["labels"].reshape(B*T)
        return F.cross_entropy(logits, labels, ignore_index=self.ignore_index)

    def metrics(self, outputs, batch):
        with torch.no_grad():
            if self.task == "causal_lm" or self.task == "masked_lm":
                B, T, V = outputs["logits"].shape
                logits = outputs["logits"].reshape(B*T, V)
                labels = batch["labels"].reshape(B*T)
                valid = labels != self.ignore_index
                n = valid.sum().item()
                if n == 0:
                    return {"ppl": float("nan")}
                ce = F.cross_entropy(logits[valid], labels[valid], reduction="mean").item()
                return {"ppl": math.exp(ce)}
            elif self.task == "seq_cls":
                B, T, C = outputs["logits"].shape
                pred = outputs["logits"].argmax(-1)
                mask = batch["labels"] != self.ignore_index
                correct = (pred[mask] == batch["labels"][mask]).sum().item()
                total = mask.sum().item()
                return {"tok_acc": correct / max(1, total)}
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")

class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model_manager = None
        self.task = config.TASK

    def setup_amp(self):
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            self.autocast_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            self.autocast_dtype = torch.float16
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.autocast_dtype = None

    def get_adapter(self) -> BaseAdapter:
        class_weights = self.config.CLASS_WEIGHTS
        label_smoothing = self.config.LABEL_SMOOTHING
        ignore_index = self.config.IGNORE_INDEX
        adapter_map = {
            "image_cls": ClassificationAdapter(class_weights=class_weights, label_smoothing=label_smoothing),
            "reg_lm": RegressionAdapter(),
            "causal_lm": LanguageAdapter(task="causal_lm", ignore_index=ignore_index),
            "masked_lm": LanguageAdapter(task="masked_lm", ignore_index=ignore_index),
            "seq_cls": LanguageAdapter(task_type="seq_cls", ignore_index=ignore_index),
        }
        return adapter_map[self.task]
    
    def get_model(self) -> nn.Module:
        if "image" in self.task:
            pass
        else:
            from ...simple_nn.models.language_loader import ModelManager
            model_name = self.config.MODEL_NAME
            trust_remote_code = self.config.TRUST_REMOTE_CODE
            self.model_manager = ModelManager(model_name, trust_remote_code, self.task_type)
            self.model_manager.load_model()

    def get_dataloader(self, train_data: List[Record], eval_data: Optional[List[Record]] = None) -> Tuple[DataLoader, DataLoader]:
        if self.config.MODEL_TYPE == "timm":
            print(f"Creating dataloader for {self.config.MODEL_TYPE}")
            from ...vision.data.dataloader import create_dataloader
            if eval_data is not None:
                return create_dataloader(self.config, 
                                         train_data=train_data, 
                                         eval_data=eval_data, 
                                         train_transform=self.transform, 
                                         eval_transform=self.transform_val, 
                                         device=self.device)
            else:
                return create_dataloader(self.config, 
                                         train_data=train_data, 
                                         train_transform=self.transform, 
                                         device=self.device)
        elif self.config.MODEL_TYPE == "hf-language":
            print(f"Creating dataloader for {self.config.MODEL_TYPE}")
            from ...simple_nn.data.dataloader import create_dataloader_hf
            if eval_data is not None:
                return create_dataloader_hf(self.config, train_data=train_data, eval_data=eval_data)
            else:
                return create_dataloader_hf(self.config, train_data=train_data)
        else:
            raise ValueError(f"Unsupported model type: {self.config.MODEL_TYPE}")

    def _autocast(self):
        return torch.amp.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None)

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def _set_optimizer(self, model):
        if self.config.OPTIMIZER == "adamw":
            self.opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.config.OPTIMIZER == "sgd":
            self.opt = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.OPTIMIZER}")
        if self.lr_scheduler == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.config.NUM_EPOCHS, eta_min=1e-6)
        elif self.lr_scheduler == "step":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=self.config.STEP_SIZE, gamma=self.config.GAMMA)
        else:
            raise ValueError(f"Unsupported lr scheduler: {self.lr_scheduler}")

    def _get_lr(self, opt):
        return opt.param_groups[0].get("lr", float("nan"))
    
    def train_step(self, batch):
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        batch = self._to_device(batch)
        if self.use_amp and (self.scaler is not None):
            # AMP with GradScaler (FP16 on older CUDA)
            with self._autocast():
                outputs = self.adapter.forward(self.model, batch)
                loss = self.adapter.loss(outputs, batch)
            self.scaler.scale(loss).backward()
            if self.config.GRAD_CLIP:
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            self.scaler.step(self.opt)
            self.scaler.update()
            return float(loss.item())

        elif self.use_amp:
            # AMP without GradScaler (e.g., BF16 on Ampere/Hopper)
            with self._autocast():
                outputs = self.adapter.forward(self.model, batch)
                loss = self.adapter.loss(outputs, batch)
            loss.backward()
            if self.config.GRAD_CLIP:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            self.opt.step()
            return float(loss.item())

        else:
            # Pure FP32
            outputs = self.adapter.forward(self.model, batch)
            loss = self.adapter.loss(outputs, batch)
            loss.backward()
            if self.config.GRAD_CLIP:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            self.opt.step()
            return float(loss.item())

    def train_epoch(self, train_loader, epoch, eval_loader=None) -> Dict[str, float]:
        self.model.train()
        eval_steps = self.config.EVAL_STEPS
        start = perf_counter()
        total_loss = 0.0; n_steps = 0
        ema = None; beta = 0.98

        pbar = tqdm(train_loader, desc=f"🔧 train (epoch {epoch:02d})", leave=False, dynamic_ncols=True)
        for batch in pbar:            
            step_loss = self.train_step(batch)

            n_steps += 1
            total_loss += step_loss
            avg = total_loss / n_steps
            ema = step_loss if ema is None else (beta * ema + (1 - beta) * step_loss)
            ema_bc = ema / (1 - beta**n_steps)
            lr = self._get_lr(self.opt)
            mem = cuda_memory_stats()
            postfix = {
                "loss": f"{step_loss:.4f}",
                "avg": f"{avg:.4f}",
                "ema": f"{ema_bc:.4f}",
                "lr": f"{lr:.2e}",
            }
            if mem is not None:
                postfix["mem(GB)"] = f"{mem:.2f}"

            pbar.set_postfix(postfix)

            if eval_loader is not None and eval_steps > 0 and n_steps % eval_steps == 0:
                va = self.validate(eval_loader, epoch)
                print(f"epoch {epoch:02d}: " + "  ".join([f"{k}={v:.4f}" for k, v in va.items()]))
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        wall = perf_counter() - start
        final = {
            "loss": total_loss / max(1, n_steps),
            "steps": n_steps,
            "time_s": wall,
            "steps_per_s": n_steps / max(wall, 1e-8),
            "lr": self._get_lr(self.opt),
        }
        print(f"🏁 Training completed in {wall:.2f}s")
        return final

    @torch.no_grad()
    def validate(self, loader, epoch) -> Dict[str, float]:
        self.model.eval()
        start = perf_counter()
        total_loss = 0.0; n_steps = 0
        agg_metrics: Dict[str, float] = {}

        pbar = tqdm(loader, desc=f"🧪 validating (epoch {epoch:02d})", leave=False, dynamic_ncols=True)
        for batch in pbar:
            batch = self._to_device(batch)
            if self.use_amp:
                with self._autocast():
                    outputs = self.adapter.forward(self.model, batch)
                    loss = self.adapter.loss(outputs, batch)
            else:
                outputs = self.adapter.forward(self.model, batch)
                loss = self.adapter.loss(outputs, batch)

            step_loss = float(loss.item())
            total_loss += step_loss
            n_steps += 1
            m = self.adapter.metrics(outputs, batch)
            for k, v in m.items():
                agg_metrics[k] = agg_metrics.get(k, 0.0) + float(v) * step_loss

            avg_loss = total_loss / max(1, n_steps)
            pbar.set_postfix({
                "loss": f"{step_loss:.4f}",
                "avg":  f"{avg_loss:.4f}"
            })
        wall = perf_counter() - start
        # average metrics over steps
        agg_metrics = {k: v / max(1, n_steps) for k, v in agg_metrics.items()}
        agg_metrics["val_loss"] = total_loss / max(1, n_steps)
        agg_metrics["val_steps"] = n_steps
        agg_metrics["time_s"] = wall
        agg_metrics["steps_per_s"]= n_steps / max(wall, 1e-8)
        print(f"🏁 Validation completed in {wall:.2f}s")
        return agg_metrics

    def train(self, train_data, eval_data=None):
        self.adapter = self.get_adapter()
        self.get_model()
        self._set_optimizer(self.model)
        train_loader, val_loader = self.get_dataloader(train_data, eval_data)
        num_epochs = self.config.NUM_EPOCHS
        for epoch in range(num_epochs):
            tr = self.train_epoch(train_loader, epoch, val_loader)
            if val_loader is not None:
                va = self.validate(val_loader, epoch)
                print(f"epoch {epoch:02d}: loss={tr['loss']:.4f}  " +
                      "  ".join([f"{k}={v:.4f}" for k, v in va.items()]))
            else:
                print(f"epoch {epoch:02d}: loss={tr['loss']:.4f}")
