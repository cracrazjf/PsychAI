import gc
import timm
import torch
from timm.data import resolve_model_data_config
from torch.utils.data import DataLoader
from typing import Any, Optional

class ModelManager:
    def __init__(self):
        self.model = None,
        self.model_name = None
        self.model_path = None
        self.model_type = None
        self.model_config = None

    def load_model(self, 
                   model_name: str, 
                   model_path: str,
                   model_type: str,
                   device,
                   task: str = "classification") -> None:
        
        self.free_memory()

        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type
        
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
                 cfg,
                 eval_fn=None):
         self.cfg = cfg
         self.mm = ModelManager()

    def evaluate(self, 
                 dataloader: DataLoader, 
                 eval_fn: Optional[Any], 
                 epoch: Optional[int], 
                 step: Optional[int] = 0, 
                 eval_path: Optional[str] = None):
        
        device = next(self.mm.model.parameters()).device
        self.mm.model.eval()

        preds_per_batch = []
        labels_per_batch = []
        inputs_per_batch = []
        logits_per_batch = []
        embeddings = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['pixel_values'].to(device)
                labels = batch.get("labels", None)
                if labels is not None:
                    labels = torch.tensor(batch['labels']).long().to(device)

                if self.cfg.task == "classification":
                    logits = self.mm.model(inputs)
                    preds = logits.argmax(dim=-1)
                    preds_per_batch.append(preds.cpu())
                    if labels is not None:
                        labels_per_batch.append(labels.cpu())
                    inputs_per_batch.append(inputs.cpu())
                    if cfg.logging.return_logits:
                        logits_per_batch.append(logits.cpu())
                else:
                    feats = self.mm.model(inputs)
                    embeddings.append(feats[self.cfg.logging.layer_of_interest].cpu())

        if self.cfg.task == "classification":
            if eval_fn is not None:
                eval_results = eval_fn(self.mm, 
                                        self.cfg, 
                                        inputs_per_batch, 
                                        labels_per_batch, 
                                        logits_per_batch, 
                                        preds_per_batch, 
                                        embeddings)
                
                return eval_results



                

