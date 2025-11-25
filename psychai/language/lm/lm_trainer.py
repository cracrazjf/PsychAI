import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pprint import pformat
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from typing import Tuple, Optional
from itertools import chain
from datasets import load_dataset, Dataset
from .lm_mm import LM_ModelManager
from ..utils import to_serializable, save_checkpoint, load_checkpoint, clean_dir
from ...nn_builder import save_pretrained


class LM_Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_manager = LM_ModelManager()

    def create_optimizer(self):
        if self.cfg.optim.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.model_manager.model.parameters(), 
                              lr=self.cfg.optim.lr, 
                              weight_decay=self.cfg.optim.weight_decay)
        elif self.cfg.optim.optimizer == "adam":
            optimizer = torch.optim.Adam(self.model_manager.model.parameters(), 
                                         lr=self.cfg.optim.lr, 
                                         weight_decay=self.cfg.optim.weight_decay)
        elif self.cfg.optim.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model_manager.model.parameters(), 
                            lr=self.cfg.optim.lr, 
                            weight_decay=self.cfg.optim.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.cfg.optim.optimizer}")
        return optimizer

    def load_model_and_tokenizer(self, random_seed: int | None = None):
        self.model_manager.load_model(self.cfg.model.name,
                                      self.cfg.model.path,
                                      self.cfg.model.task,
                                      custom=self.cfg.model.customized_model,
                                      tokenizer_path=self.cfg.model.tokenizer_path,
                                      trust_remote_code=self.cfg.model.trust_remote_code,
                                      random_seed=random_seed,
                                      weight_init=self.cfg.model.weight_init)

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
                     dataset,
                     shuffle_dataset: bool, 
                     shuffle_dataloader: bool,
                     random_seed: int):
        if shuffle_dataset:
            dataset = dataset.shuffle(seed=random_seed)

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
                                shuffle=shuffle_dataloader, 
                                collate_fn=collator, 
                                drop_last=self.cfg.data.drop_last,
                                num_workers=self.cfg.data.num_workers
                                )
        return dataloader

    def train(self, test_metric=None):
        for run in range(self.cfg.num_runs):
            tqdm.write(f"===== Run {run + 1}/{self.cfg.num_runs} =====")
            self.load_model_and_tokenizer(random_seed=self.cfg.seed + run)
            optimizer = self.create_optimizer()

            train_dataset = self.tokenize_data(self.cfg.data.train_path)
            val_dataset = self.tokenize_data(self.cfg.data.val_path) if self.cfg.data.val_path is not None else None

            run_dir = None
            log_file = None
            if self.cfg.experiment_directory is not None:
                run_dir = os.path.join(self.cfg.experiment_directory, f"run_{run}")
                os.makedirs(run_dir, exist_ok=True)
                log_file = os.path.join(run_dir, "train_log.jsonl")
                with open(log_file, "w") as f:
                    pass

            train_loader = None
            if not self.cfg.data.shuffle_dataset:
                train_loader = self.prepare_data(train_dataset, False, self.cfg.data.shuffle_dataloader, self.cfg.seed + run)
            val_loader = self.prepare_data(val_dataset, False, False, 0) if val_dataset is not None else None

            with tqdm(total=self.cfg.num_epochs,
                      desc=f"Run {run+1}/{self.cfg.num_runs}",
                      position=0,
                      dynamic_ncols=True,
                    ) as epoch_bar:
                for epoch in range(self.cfg.num_epochs):
                    if train_loader is None:
                        train_loader = self.prepare_data(train_dataset, 
                                                        self.cfg.data.shuffle_dataset, 
                                                        self.cfg.data.shuffle_dataloader, 
                                                        self.cfg.seed + epoch)

                    def _train_epoch(model, train_loader, optimizer):
                        model.train()
                        running_loss = 0

                        if self.cfg.training_method == "continuous":
                            state = {}
                        else:
                            state = None

                        with tqdm(
                            total=len(train_loader),
                            desc=f"Train Epoch {epoch + 1}/{self.cfg.num_epochs}",
                            position=1,
                            leave=False,
                            ncols=100
                            ) as batch_bar:

                            for i, batch in enumerate(train_loader):
                                optimizer.zero_grad()
                                outputs = model.forward(input_ids=batch["input_ids"], 
                                                        attention_mask=batch["attention_mask"], 
                                                        labels=batch["labels"], 
                                                        state=state,
                                                        detach_state=True)
                                if self.cfg.training_method == "continuous":
                                    state = outputs["state"]

                                loss = outputs["loss"]
                                loss.backward()
                                optimizer.step()
                                running_loss += loss.item()

                                batch_bar.update(1)
                                if (i + 1) % getattr(self.cfg, "log_interval", 50) == 0:
                                    batch_bar.set_postfix(
                                        loss=float(running_loss / (i + 1)),
                                        lr=optimizer.param_groups[0]["lr"],
                                    )

                        return {"train_loss": running_loss / len(train_loader)}

                    train_loss = _train_epoch(self.model_manager.model, train_loader, optimizer)

                    def _eval_epoch(model, val_loader, test_metric, eval_dataset, epoch):
                        tqdm.write(f"Evaluating at epoch {epoch + 1}...")
                        labels, preds, logits = [], [], []

                        if self.cfg.training_method == "continuous":
                            state = {}
                        else:
                            state = None

                        model.eval()
                        running_loss = 0

                        with tqdm(
                            total=len(val_loader),
                            desc=f"Eval Epoch {epoch + 1}/{self.cfg.num_epochs}",
                            position=1,
                            leave=False,
                            ncols=100,
                        ) as eval_bar:
                            for i, batch in enumerate(val_loader):
                                with torch.no_grad():
                                    outputs = model.forward(input_ids=batch["input_ids"], 
                                                            attention_mask=batch["attention_mask"], 
                                                            labels=batch["labels"], 
                                                            state=state,
                                                            detach_state=True)
                                    if self.cfg.training_method == "continuous":
                                        state = outputs["state"]
                                        
                                loss = outputs["loss"]
                                running_loss += loss.item()

                                labels.extend(outputs["labels"].detach().cpu().numpy())
                                logits.extend(outputs["logits"].detach().cpu().numpy())
                                preds.extend(torch.argmax(outputs["logits"], dim=-1).detach().cpu().numpy())

                                eval_bar.update(1)
                                if (i + 1) % getattr(self.cfg, "log_interval", 50) == 0:
                                    eval_bar.set_postfix(
                                        {"loss": f"{running_loss / (i + 1):.4f}"}
                                    )

                            record = {
                                "epoch": epoch + 1,
                                "eval_loss": running_loss / len(val_loader)
                                }
                            if test_metric is not None:
                                metric_info = test_metric(eval_dataset, 
                                                        np.stack(labels), 
                                                        np.stack(preds), 
                                                        np.stack(logits), 
                                                        self.model_manager.tokenizer,
                                                        self.cfg)
                                if "accuracy" in metric_info:
                                    tqdm.write("accuracy metrics:\n" + pformat(metric_info["accuracy"]))
                                else:
                                    raise ValueError("Metric info must contain 'accuracy' key")
                                
                                for key, value in metric_info.items():
                                    for k, v in value.items():
                                        value[k] = to_serializable(v)
                                    record[key] = to_serializable(value)
                            if log_file is not None:
                                with open(log_file, "a") as f:
                                    f.write(json.dumps(record) + "\n")

                    if (epoch+1) % self.cfg.logging.eval_interval == 0:
                        if val_loader is not None:
                            _eval_epoch(self.model_manager.model, val_loader, test_metric, val_dataset, epoch)
                        save_checkpoint(run_dir, 
                                        self.model_manager.model, 
                                        optimizer=optimizer,
                                        scaler=None,
                                        tokenizer=self.model_manager.tokenizer,
                                        epoch=epoch,
                                        max_to_keep=self.cfg.logging.save_total_limit
                                        )
                        
                    epoch_bar.update(1)
            
            if self.cfg.logging.save_model:
                save_dir = os.path.join(run_dir, "export")
                clean_dir(save_dir)
                save_pretrained(self.model_manager.model, save_dir)
                self.model_manager.tokenizer.save_pretrained(save_dir)
                print(f"model saved!")