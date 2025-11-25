from pprint import pformat
from tqdm import tqdm
from .lm_mm import LM_ModelManager
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from itertools import chain
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from pathlib import Path

class LM_Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_manager = LM_ModelManager()

    def load_model_and_tokenizer(self, model_path):
        self.model_manager.load_model(self.cfg.model.name,
                                      model_path,
                                      self.cfg.model.task,
                                      custom=self.cfg.model.customized_model,
                                      tokenizer_path=model_path,
                                      trust_remote_code=self.cfg.model.trust_remote_code,
                                      random_seed=None,
                                      weight_init=None)
    
    def tokenize_data(self, path: str):
        dataset = load_dataset(path=path, split="train")

        def tokenize_function(batch):
            return self.model_manager.tokenizer(batch["text"], add_special_tokens=False, truncation=False)
        tokenized_dataset = dataset.map(tokenize_function, 
                                        batched=True, 
                                        batch_size=self.cfg.data.data_process_batch_size, 
                                        num_proc=self.cfg.data.data_process_num_proc)
        return tokenized_dataset

    def prepare_data(self, 
                     dataset):

        collator = DataCollatorForLanguageModeling(tokenizer=self.model_manager.tokenizer, 
                                                   mlm=self.cfg.model.task == "masked_lm")

        def _concatenate(dataset):
            def flatten(x):
                if len(x) > 0 and not isinstance(x[0], (list, tuple)):
                    return x
                return list(chain.from_iterable(x))
            input_ids = flatten(dataset["input_ids"])
            attention_mask = flatten(dataset["attention_mask"])
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        def _create_windows(dataset: dict,
                            sequence_length, 
                            stride: int | None = None, 
                            pad_left: bool = False,
                            drop_last=False):
            
            assert sequence_length > 1, "sequence_length must be >= 2 for causal LM"

            input_ids = dataset["input_ids"]    
            attention_mask = dataset["attention_mask"]

            if stride == None:
                stride = sequence_length

            if pad_left:
                if self.model_manager.tokenizer.bos_token_id is None:
                    raise ValueError("Tokenizer does not have a bos_token_id")
                # pad_len = sequence_length - 1
                # pad_ids = [self.model_manager.tokenizer.bos_token_id] * pad_len
                # pad_mask = [0] * pad_len
                input_ids = [self.model_manager.tokenizer.pad_token_id] + input_ids
                attention_mask = [0] + attention_mask

            n = len(input_ids)

            windows = {"input_ids": [], "attention_mask": []}


            for start in range(0, n, stride):
                end = start + sequence_length
                ids = input_ids[start:end]
                msk = attention_mask[start:end]

                # pad if last window is too short
                if len(ids) < sequence_length:
                    if drop_last:
                        break
                    if self.model_manager.tokenizer.pad_token_id is None:
                        raise ValueError("Tokenizer does not have a pad_token_id")
                    pad_len = sequence_length - len(ids)
                    ids = ids + [self.model_manager.tokenizer.pad_token_id] * pad_len
                    msk = msk + [0] * pad_len

                windows["input_ids"].append(ids)
                windows["attention_mask"].append(msk)

                if end >= n:
                    break

            return windows
        
       
        dataset = _concatenate(dataset)
        windows = _create_windows(dataset,
                                  self.cfg.data.sequence_length,
                                  stride=self.cfg.data.stride,
                                  pad_left=self.cfg.data.pad_left,
                                  drop_last=self.cfg.data.drop_last)

        dataset = Dataset.from_dict(windows)

        dataloader = DataLoader(dataset, 
                                batch_size=self.cfg.data.batch_size, 
                                shuffle=False, 
                                collate_fn=collator, 
                                drop_last=self.cfg.data.drop_last,
                                num_workers=self.cfg.data.num_workers
                                )
        return dataloader

    def build_embeddings_map(self,input_ids, attention_mask, representations, batch_id):
        mask = attention_mask[:,:-1].bool()
        input_ids = input_ids[:,:-1].detach().cpu()

        representations_map = {}
        b_idx, t_idx = mask.nonzero(as_tuple=True)
        for bi, ti in zip(b_idx.tolist(), t_idx.tolist()):
            input_id = int(input_ids[bi, ti])
            map = {"token_id": input_id, 
                   "token_string": self.model_manager.tokenizer.decode([input_id]), 
                   "layers": {}}
            for layer_name, layer_repr in representations.items():
                layer_bucket = {}
                for key, tensor in layer_repr.items():
                    if isinstance(tensor, tuple):
                        vec = tuple(t[bi, ti].detach().cpu().numpy() for t in tensor)
                    elif hasattr(tensor, "detach"):
                        vec = tensor[bi, ti].detach().cpu().numpy()
                    layer_bucket[key] = vec
                map["layers"][layer_name] = layer_bucket
            representations_map[(batch_id, (bi, ti))] = map
        return representations_map

    def evaluate_language_model(self, compute_metrics=None):
        exp_root = Path(self.cfg.experiment_directory)
        exported_models = list(exp_root.rglob("export"))
        eval_results = {}

        with tqdm(total=len(exported_models), desc="Evaluating models", position=0, dynamic_ncols=True) as pbar:
            for subdir in exported_models:
                self.load_model_and_tokenizer(subdir)
                
                weights = self.model_manager.model.base_model.get_weights()

                if self.cfg.data.test_path is None:
                    eval_results[str(subdir)] = {"weights": weights}
                    pbar.update(1)
                    continue
                else:
                    test_dataset = self.tokenize_data(self.cfg.data.test_path)
                    test_loader = self.prepare_data(test_dataset)

                    if self.cfg.training_method == "continuous":
                        state = {}
                    else:
                        state = None

                    embeddings_map = {}
                    labels, preds, logits = [], [], []

                    self.model_manager.model.eval()
                    with tqdm(total=len(test_loader), desc="Evaluating test data", position=1, dynamic_ncols=True) as test_pbar:
                        for i, batch in enumerate(test_loader):
                            with torch.no_grad():
                                outputs = self.model_manager.model.forward(input_ids=batch["input_ids"], 
                                                                        attention_mask=batch["attention_mask"], 
                                                                        labels=batch["labels"],
                                                                        return_embeds=True, 
                                                                        state=state)
                                if self.cfg.training_method == "continuous":
                                    state = outputs["state"]

                            labels.extend(outputs["labels"].detach().cpu().numpy())
                            logits.extend(outputs["logits"].detach().cpu().numpy())
                            preds.extend(torch.argmax(outputs["logits"], dim=-1).detach().cpu().numpy())

                            embeddings_map = self.build_embeddings_map(batch["input_ids"], 
                                                                      batch["attention_mask"], 
                                                                      outputs["embeds"],
                                                                      i)
                            
                            embeddings_map.update(embeddings_map)

                        if compute_metrics is not None:
                            metric_info = compute_metrics(test_dataset, 
                                                        np.stack(labels), 
                                                        np.stack(preds), 
                                                        np.stack(logits), 
                                                        self.model_manager.tokenizer, 
                                                        self.cfg)
                            
                            if "accuracy" in metric_info:
                                    tqdm.write("accuracy metrics:\n" + pformat(metric_info["accuracy"]))
                            else:
                                raise ValueError("Metric info must contain 'accuracy' key")
                            
                        test_pbar.update(1)

                    eval_results[str(subdir)] = {"embeddings": embeddings_map, "weights": weights}
                pbar.update(1)
        return eval_results