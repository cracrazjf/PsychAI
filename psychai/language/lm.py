from curses import wrapper
import gc
import torch
import os
import json
import random
import numpy as np
from tqdm import tqdm
from pprint import pformat
from itertools import chain
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from typing import Dict, Optional, Any
from transformers import DataCollatorForLanguageModeling
from .utils import to_serializable, save_checkpoint, load_checkpoint, clean_dir
from ..nn_builder import from_pretrained, save_pretrained, load_config, build_spec_from_config, Model, CausalLMWrapper

try:
    from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
except Exception as e:
    raise ImportError("transformers is required. Install with extras: psychai[simple-nn]") from e


class ModelManager:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.model_path = None
        self.model_type = None
        self.wrapper = None

        self.tokenizer = None
        self.tokenizer_path = None

    def load_model(self, 
                   model_name: str, 
                   model_path: str = None, 
                   model_type: str = None,
                   wrapper: str = "causal_lm",
                   device:str = "cpu",
                   *,
                   tokenizer_path: Optional[str] = None, 
                   trust_remote_code: Optional[bool] = True):
        self.free_memory()

        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type
        self.wrapper = wrapper
        self.tokenizer_path = tokenizer_path

        if "custom" in model_name.lower() or "custom" in model_type.lower():
            self.load_custom_model()
        else:
            self.load_hf_model()
    
        self.model.to(device)

    def free_memory(self) -> None:
        if hasattr(self, 'model') and self.model is not None:
            try:
                del self.model
                print("Current model deleted")
            except Exception:
                pass
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            try:
                del self.tokenizer
                print("Current tokenizer deleted")
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()

        self.model = None
        self.model_name = None
        self.model_path = None
        self.model_type = None
        self.wrapper = None

        self.tokenizer = None
        self.tokenizer_path = None
        print("Cache cleared")

    def load_custom_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path) if self.tokenizer_path is not None else AutoTokenizer.from_pretrained(self.model_path)

        wrapper_map = {
            "causal_lm": CausalLMWrapper
        }
        ctor = wrapper_map.get(self.wrapper)

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

    def load_hf_model(self):
        if self.model_path is None:
            self.model_path = self.model_name

        print(f"Loading model and tokenizer from {self.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)

        wrapper_map = {
            "causal_lm": AutoModelForCausalLM,
            "masked_lm": AutoModelForMaskedLM,
            "seq_cls": AutoModelForSequenceClassification,
            "tok_cls": AutoModelForTokenClassification,
        }
        ctor = wrapper_map.get(self.wrapper)
        if ctor is None:
            raise ValueError(f"Unknown wrapper: {self.wrapper}")

        self.model = ctor.from_pretrained(self.model_path)
        self.print_hf_model()

    def print_hf_model(self):
            cfg = self.model.config.to_dict()
            print("== Model ==")
            print(f"model_type        : {cfg.get('model_type')}")
            print(f"n_layer           : {cfg.get('n_layer') or cfg.get('num_hidden_layers')}")
            print(f"n_head            : {cfg.get('n_head')  or cfg.get('num_attention_heads')}")
            print(f"n_embd/hidden     : {cfg.get('n_embd')  or cfg.get('hidden_size')}")
            print(f"vocab_size        : {cfg.get('vocab_size')} (tokenizer: {len(self.tokenizer)})")
            print(f"max_position      : {cfg.get('n_positions') or cfg.get('max_position_embeddings')}")
            print(f"pad_token_id      : {cfg.get('pad_token_id')}  eos_token_id: {cfg.get('eos_token_id')}")
            total = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print("== Params ==")
            print(f"total/trainable   : {total:,} / {trainable:,}")


class TrainingManager:
    def __init__(self, cfg):
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
                            weight_decay=self.cfg.optim.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.cfg.optim.optimizer}")

    def tokenize_data(self, path: str):
        if os.path.isfile(path):
            dataset = load_dataset("jsonl", data_files=path, split="train")
        else:
            dataset = load_dataset(path=path, split="train")

        def _tokenize_function(batch):
            return self.mm.tokenizer(batch["text"], add_special_tokens=False, truncation=False)

        tokenized_dataset = dataset.map(_tokenize_function, 
                                        batched=True, 
                                        batch_size=self.cfg.data.data_process_batch_size, 
                                        num_proc=self.cfg.data.data_process_num_proc)
        return tokenized_dataset

    def prepare_data(self, dataset: Dataset, shuffle_dataset: bool, shuffle_dataloader: bool, seed: int):
        if shuffle_dataset:
            dataset = dataset.shuffle(seed=seed)
        
        def _concatenate(dataset):
            def flatten(x):
                if len(x) > 0 and not isinstance(x[0], (list, tuple)):
                    return x
                return list(chain.from_iterable(x))
            
            input_ids = flatten(dataset["input_ids"])
            attention_mask = flatten(dataset["attention_mask"])
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        
        def _create_windows(dataset: dict,
                            window_size: int | None = None, 
                            stride: int | None = None, 
                            pad_left: bool = False,
                            pad_id = None,
                            drop_last=False):
            
            if self.cfg.task == "masked_lm":
                assert window_size > 1, "window_size must be >= 2 for causal LM"

            input_ids = dataset["input_ids"]    
            attention_mask = dataset["attention_mask"]

            if stride == None:
                stride = window_size

            if pad_left:
                if pad_id is None:
                    raise ValueError("pad_id must be provided when pad_left is True")
                input_ids = [pad_id] + input_ids
                attention_mask = [0] + attention_mask

            n = len(input_ids)

            windows = {"input_ids": [], "attention_mask": []}

            for start in range(0, n, stride):
                end = start + window_size
                ids = input_ids[start:end]
                msk = attention_mask[start:end]

                if len(ids) < window_size:
                    if drop_last:
                        break
                    if pad_id is None:
                        raise ValueError("pad_id must be provided when drop_last is False")
                    pad_len = window_size - len(ids)
                    ids = ids + [pad_id] * pad_len
                    msk = msk + [0] * pad_len

                windows["input_ids"].append(ids)
                windows["attention_mask"].append(msk)

                if end >= n:
                    break

            return windows
        
        concatenated_dataset = _concatenate(dataset)
        windows = _create_windows(concatenated_dataset,
                                  window_size=self.cfg.data.window_size,
                                  stride=self.cfg.data.stride,
                                  pad_left=self.cfg.data.pad_left,
                                  pad_id=self.mm.tokenizer.pad_token_id,
                                  drop_last=self.cfg.data.drop_last)
        
        final_dataset = Dataset.from_dict(windows)
        
        collator = DataCollatorForLanguageModeling(tokenizer=self.mm.tokenizer, 
                                                   mlm=self.cfg.wrapper == "masked_lm")
        
        g = torch.Generator().manual_seed(seed)
        dataloader = DataLoader(final_dataset,
                                collate_fn=collator,
                                batch_size=self.cfg.data.batch_size,
                                num_workers=self.cfg.data.num_workers,
                                drop_last=self.cfg.data.drop_last,
                                shuffle=shuffle_dataloader,
                                generator=g)
        return dataloader

    def train_epoch(self, 
                    dataloader: DataLoader, 
                    epoch: int, 
                    val_loader: Optional[DataLoader] = None, 
                    eval_fn: Optional[Any] = None, 
                    eval_path: Optional[str] = None,
                    log_path: Optional[str] = None) -> Dict[str, Any]:
        
        device = next(self.mm.model.parameters()).device
        self.mm.model.train()

        epoch_loss = 0.0
        recurrent_state = {} if self.cfg.bp_method == "continuous" else None
        with tqdm(total=len(dataloader),
                  desc=f"Train Epoch {epoch + 1}/{self.cfg.num_epochs}",
                  position=1,
                  leave=False,
                  ncols=100) as batch_bar:
            for i, batch in enumerate(dataloader):
                self.optimizer.zero_grad()

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = self.mm.model.forward(input_ids=input_ids, 
                                                attention_mask=attention_mask, 
                                                labels=labels, 
                                                recurrent_state=recurrent_state,
                                                detach_state=True)
                
                if self.cfg.bp_method == "continuous":
                    recurrent_state = outputs["recurrent_state"]

                loss = outputs["loss"]
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                batch_bar.update(1)
                batch_bar.set_postfix(loss=float(epoch_loss / (i + 1)), lr=self.optimizer.param_groups[0]["lr"])

                if self.cfg.interval_strategy == "step":
                    if (i + 1) % self.cfg.eval_interval == 0:
                        if val_loader is not None:
                            batch_bar.clear()
                            self.evaluate(dataloader=val_loader, 
                                          eval_fn=eval_fn, 
                                          epoch=epoch, 
                                          step=i + 1, 
                                          eval_path=eval_path)
                            batch_bar.refresh()
                    if (i + 1) % self.cfg.logging.log_interval == 0:
                        with open(log_path, "a") as f:
                            f.write(json.dumps({"epoch": epoch + 1, "step": i + 1, "train_loss": epoch_loss / (i + 1)}) + "\n")

        return {"epoch": epoch + 1, "train_loss": epoch_loss / len(dataloader)}
                    
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
        weights = {}
        embedding_maps = []
        
        if self.cfg.logging.return_weights:
            weights = self.mm.model.base_model.get_weights()

        recurrent_state = {} if self.cfg.bp_method == "continuous" else None

        def _collect_embeddings(input_ids, attention_mask, embeddings):
            batch_size = input_ids.shape[0]
            embeddings_list = [[] for _ in range(batch_size)]

            b_idx, t_idx = attention_mask.nonzero(as_tuple=True)

            for bi, ti in zip(b_idx.tolist(), t_idx.tolist()):
                token_id = int(input_ids[bi, ti])

                token_entry = {
                    "token_id": token_id,
                    "token_str": self.mm.tokenizer.decode([token_id]),
                    "embedding": embeddings[bi, ti].detach().cpu(),
                }
                embeddings_list[bi].append(token_entry)
            return embeddings_list
        
        eval_loss = 0
        with tqdm(total=len(dataloader),
                  desc=f"Evaluating",
                  position=1,
                  leave=True,
                  ncols=100,
                ) as eval_bar:
            for i, batch in enumerate(dataloader):
                with torch.no_grad():
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch.get("labels", None)
                    if labels is not None:
                        labels = labels.to(device)

                    outputs = self.mm.model.forward(input_ids=input_ids, 
                                                    attention_mask=attention_mask, 
                                                    labels=labels, 
                                                    recurrent_state=recurrent_state,
                                                    detach_state=False,
                                                    return_embeds=self.cfg.logging.return_embeddings)
                    
                    if self.cfg.bp_method == "continuous":
                        recurrent_state = outputs["recurrent_state"]
                    
                    loss = outputs["loss"]
                    eval_loss += loss.item()

                    logits = outputs["logits"]
                    preds = torch.argmax(logits, dim=-1)

                    inputs_per_batch.append(outputs["input_ids"].cpu())
                    labels_per_batch.append(outputs["labels"].cpu())
                    logits_per_batch.append(logits.cpu())
                    preds_per_batch.append(preds.cpu())

                    if self.cfg.logging.return_embeddings:
                        embedding_maps.append(_collect_embeddings(batch["input_ids"], 
                                                                  batch["attention_mask"], 
                                                                  outputs["embeds"][self.cfg.logging.layer_of_interest][self.cfg.logging.embed_type]))

                    eval_bar.update(1)
                    eval_bar.set_postfix({"loss": f"{eval_loss / (i + 1):.4f}"})

            eval_info = {"epoch": epoch + 1, "step": step, "eval_loss": eval_loss / len(dataloader)}

            if eval_fn is not None:
                eval_bar.clear()
                eval_result = eval_fn(self.mm, 
                                      self.cfg, 
                                      inputs_per_batch, 
                                      labels_per_batch, 
                                      logits_per_batch, 
                                      preds_per_batch, 
                                      embedding_maps, 
                                      weights)
                eval_bar.refresh()
                if isinstance(eval_result, dict):
                    if "accuracy" in eval_result:
                        tqdm.write("accuracy:\n" + pformat(eval_result["accuracy"]))
                    for key, value in eval_result.items():
                        if isinstance(value, dict):
                            for k, v in value.items():
                                value[k] = to_serializable(v)
                        eval_info[key] = to_serializable(value)
                    if eval_path is not None:
                        with open(eval_path, "a") as f:
                            f.write(json.dumps(eval_info) + "\n")

                return eval_result

    def generate(self, prompt: str | torch.Tensor, max_new_tokens: int = 50, temperature: float = 1.0, top_k: int = 50, return_logits: bool = False):
        device = next(self.mm.model.parameters()).device
        self.mm.model.eval()
        if isinstance(prompt, torch.Tensor):
            input_ids = prompt.to(device)
        else:
            input_ids = self.mm.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)

        generated = input_ids
        logits_list = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.mm.model(input_ids=generated)
                logits = outputs["logits"]
                next_token_logits = logits[:, -1, :]

                if temperature is not None and temperature > 0.0 and temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                if return_logits:
                    logits_list.append(next_token_logits.clone().cpu().numpy())

                if top_k is None or top_k <= 0:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    k = min(top_k, probs.size(-1))
                    top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
                    sampled_idx_in_topk = torch.multinomial(top_k_probs, num_samples=1)
                    next_token = top_k_indices.gather(-1, sampled_idx_in_topk)
                generated = torch.cat((generated, next_token), dim=1)

        output_tokens = generated.cpu()
        decoded_text = [
                        self.mm.tokenizer.decode(seq.tolist(), skip_special_tokens=True)
                        for seq in output_tokens
                    ]
        if return_logits:
            return decoded_text, logits_list
        else:
            return decoded_text

    def train(self, weight_init_fn=None , eval_fn=None):
        for run in range(self.cfg.num_runs):
            seed = self.cfg.seed + run
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            run_dir = None
            log_path = None
            if self.cfg.exp_dir is not None:
                run_dir = os.path.join(self.cfg.exp_dir, f"run_{run+1}")
                os.makedirs(run_dir, exist_ok=True)
                log_path = os.path.join(run_dir, "log.jsonl")
                eval_path = os.path.join(run_dir, "eval_results.json")
                with open(log_path, "w") as f:
                    f.write("")
                with open(eval_path, "w") as f:
                    f.write("")

            tqdm.write(f"===== Run {run + 1}/{self.cfg.num_runs} =====")
            self.mm.load_model(self.cfg.model.name,
                               self.cfg.model.path,
                               self.cfg.model.model_type,
                               task=self.cfg.task,
                               device=self.cfg.device,
                               tokenizer_path=self.cfg.model.tokenizer_path,
                               trust_remote_code=self.cfg.model.trust_remote_code)
            
            if weight_init_fn is not None:
                weight_init_fn(self.mm.model)

            self.configure_optimizer()

            train_dataset = self.tokenize_data(self.cfg.data.train_path)
            train_dataloader = None
            if not self.cfg.data.shuffle_dataset:
                train_dataloader = self.prepare_data(train_dataset, 
                                                     shuffle_dataset=False, 
                                                     shuffle_dataloader=self.cfg.data.shuffle_dataloader, 
                                                     seed=seed) 

            val_dataloader = None
            if self.cfg.data.val_path is not None:
                val_dataset = self.tokenize_data(self.cfg.data.val_path)
                val_dataloader = self.prepare_data(val_dataset, 
                                                   shuffle_dataset=False, 
                                                   shuffle_dataloader=False, 
                                                   seed=seed)

            with tqdm(total=self.cfg.num_epochs,
                      desc=f"Run {run+1}/{self.cfg.num_runs}",
                      position=0,
                      dynamic_ncols=True,
                    ) as epoch_bar:
                for epoch in range(self.cfg.num_epochs):
                    if self.cfg.data.shuffle_dataset:
                        train_dataloader = self.prepare_data(train_dataset, 
                                                             shuffle_dataset=True, 
                                                             shuffle_dataloader=self.cfg.data.shuffle_dataloader, 
                                                             seed=seed + epoch)

                    train_info =self.train_epoch(train_dataloader, 
                                                 epoch, 
                                                 val_loader=val_dataloader, 
                                                 eval_fn=eval_fn, 
                                                 eval_path=eval_path, 
                                                 log_path=log_path)

                    if self.cfg.eval_strategy == "epoch":
                        if (epoch+1) % self.cfg.eval_interval == 0:
                            if val_dataloader is not None:
                                epoch_bar.clear()
                                self.evaluate(val_dataloader, eval_fn, epoch, step=len(val_dataloader), eval_path=eval_path)
                                epoch_bar.refresh()
                    
                        if (epoch + 1) % self.cfg.logging.log_interval == 0:
                            with open(log_path, "a") as f:
                                f.write(json.dumps(train_info) + "\n")

                    if (epoch + 1) % self.cfg.save_interval == 0:
                        save_checkpoint(run_dir, 
                                        self.mm.model, 
                                        optimizer=self.optimizer,
                                        scaler=None,
                                        tokenizer=self.mm.tokenizer,
                                        epoch=epoch,
                                        max_to_keep=self.cfg.logging.save_total_limit
                                        )
                    epoch_bar.update(1)
            
            if self.cfg.save_model:
                save_dir = os.path.join(run_dir, "export")
                clean_dir(save_dir)
                save_pretrained(self.mm.model, save_dir)
                self.mm.tokenizer.save_pretrained(save_dir)
                print(f"model saved!")