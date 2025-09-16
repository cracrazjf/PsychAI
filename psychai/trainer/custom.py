from psychai.model_manager.language import LM_ModelManager
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from typing import Tuple, Optional
from itertools import chain
from datasets import Dataset
import torch
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from psychai.trainer.utils import to_serializable, save_checkpoint, load_checkpoint, clean_dir
from psychai.nn_builder.io import save_pretrained
from tqdm import tqdm
import numpy as np
from pprint import pprint
import os
import json

class Custom_Trainer:
    def __init__(self, config):
        self.config = config
        self.model_manager = None

    #general functions
    def create_optimizer(self):
        if self.config.OPTIMIZER == "adamw":
            optimizer = AdamW(self.model_manager.model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY)
        elif self.config.OPTIMIZER == "sgd":
            optimizer = SGD(self.model_manager.model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.OPTIMIZER}")

        scheduler = None
        if self.config.LEARNING_RATE_SCHEDULER is not None:
            if self.config.LEARNING_RATE_SCHEDULER == "step":
                scheduler = StepLR(optimizer, step_size=self.config.LR_STEPS, gamma=self.config.GAMMA)
            elif self.config.LEARNING_RATE_SCHEDULER == "exponential":
                scheduler = ExponentialLR(optimizer, gamma=self.config.GAMMA)
            else:
                raise ValueError(f"Unsupported learning rate scheduler: {self.config.LEARNING_RATE_SCHEDULER}")
        return optimizer, scheduler
    
    # language model functions
    def load_lanaguge_model_and_tokenizer(self):
        self.model_manager = LM_ModelManager()
        self.model_manager.load_model(self.config.MODEL_NAME,
                                      self.config.MODEL_PATH, 
                                      self.config.TASK, 
                                      custom=self.config.CUSTOMIZED_MODEL,
                                      tokenizer_path=self.config.TOKENIZER_PATH,
                                      trust_remote_code=self.config.TRUST_REMOTE_CODE)
        
    def tokenize_language_data(self, dataset_path) -> Tuple[Dataset, Dataset, DataCollatorForLanguageModeling]:
        if self.model_manager.tokenizer.pad_token is None: self.model_manager.tokenizer.pad_token = self.model_manager.tokenizer.eos_token
        dataset = load_dataset("json", data_files=dataset_path, split="train")

        def tokenize_function(batch):
            return self.model_manager.tokenizer(batch["text"], add_special_tokens=False, truncation=False)
        
        task = self.config.TASK
        if task == "causal_lm":
            collator = DataCollatorForLanguageModeling(tokenizer=self.model_manager.tokenizer, mlm=False)
        elif task == "masked_lm":
            collator = DataCollatorForLanguageModeling(tokenizer=self.model_manager.tokenizer, mlm=True)
        else:
            raise ValueError(f"Unsupported task type: {task}")

        # Tokenize dataset
        print(f"tokenizing dataset")
        tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=self.config.DATA_PROCESS_BATCH_SIZE, num_proc=self.config.DATA_PROCESS_NUM_PROC)

        return tokenized_dataset, collator

    def create_dataloader(self, dataset, shuffle_dataset, shuffle_dataloader, collator, epoch: Optional[int] = None):

        def _concatenate_input_ids_and_attention_mask(dataset):
            input_ids_lists = dataset["input_ids"]
            attn_mask_lists = dataset["attention_mask"]
            input_ids = list(chain.from_iterable(input_ids_lists))
            attention = list(chain.from_iterable(attn_mask_lists))
            return {"input_ids": input_ids, "attention_mask": attention}

        def _create_sliding_windows(input_ids, attention_mask, sequence_length, pad_token_id):
            assert sequence_length > 1, "sequence_length must be >= 2 for causal LM"
            n = len(input_ids)
            windows = {"input_ids": [], "attention_mask": []}
            for start in range(0, n, 1):
                end = start + sequence_length
                ids = input_ids[start:end]
                msk = attention_mask[start:end]

                # pad if last window is too short
                if len(ids) < sequence_length:
                    pad_len = sequence_length - len(ids)
                    ids = ids + [pad_token_id] * pad_len
                    msk = msk + [0] * pad_len

                windows["input_ids"].append(ids)
                windows["attention_mask"].append(msk)

                if end >= n:
                    break
            return windows
        
        def _create_nonoverlapping_windows(input_ids, attention_mask, sequence_length, pad_token_id):
            assert sequence_length > 1, "sequence_length must be >= 2 for causal LM"
            windows = {"input_ids": [], "attention_mask": []}

            for i in range(0, len(input_ids), sequence_length):
                ids = input_ids[i : i + sequence_length]
                mask = attention_mask[i : i + sequence_length]

                if len(ids) < sequence_length:
                    pad_len = sequence_length - len(ids)
                    ids = ids + [pad_token_id] * pad_len
                    mask = mask + [0] * pad_len

                windows["input_ids"].append(ids)
                windows["attention_mask"].append(mask)
            return windows
        
        if shuffle_dataset:
            dataset = dataset.shuffle(seed=epoch)
        dataset = _concatenate_input_ids_and_attention_mask(dataset)
        if self.config.OVERLAPPING_SEQUENCES:
            windows = _create_sliding_windows(dataset["input_ids"], dataset["attention_mask"], self.config.SEQUENCE_LENGTH, self.model_manager.tokenizer.pad_token_id)
        else:
            windows = _create_nonoverlapping_windows(dataset["input_ids"], dataset["attention_mask"], self.config.SEQUENCE_LENGTH, self.model_manager.tokenizer.pad_token_id)
        dataset = Dataset.from_dict(windows)
        dataloader = DataLoader(dataset, batch_size=self.config.BATCH_SIZE, 
                                shuffle=shuffle_dataloader, collate_fn=collator,
                                pin_memory=self.config.PIN_MEMORY, drop_last=self.config.DROP_LAST)
        return dataloader

    def train_language_model(self, compute_metrics=None):
        # create legacy train flag
        legacy_train = self.config.SEQUENCE_LENGTH == 2
        print(f"Detected sequence length equals to 2, using legacy train mode")

        for run in range(self.config.NUM_RUNS):
            best_acc = float("-inf") 
            # load model and tokenizer
            self.load_lanaguge_model_and_tokenizer()

            # tokenize dataset
            train_dataset, collator = self.tokenize_language_data(self.config.TRAIN_DATA_PATH)
            if self.config.EVAL_DATA_PATH is not None:
                eval_dataset, collator = self.tokenize_language_data(self.config.EVAL_DATA_PATH)
            else:
                eval_dataset = None

            # select optimizer
            optimizer, scheduler = self.create_optimizer()

            # create runs directory
            if self.config.PROJECT_PATH is not None:
                run_dir = os.path.join(self.config.PROJECT_PATH, f"run_{run}")
                os.makedirs(run_dir, exist_ok=True)
                log_file = os.path.join(run_dir, "train_metrics_info.jsonl")
                with open(log_file, "w") as f:
                    pass
            else:
                log_file = None
            
            # if not shuffling dataset, create train dataloader here
            if not self.config.SHUFFLE_DATASET:
                train_loader = self.create_dataloader(train_dataset, False, self.config.SHUFFLE_DATALOADER, collator)
            # create eval dataloader no matter what 
            eval_loader = self.create_dataloader(eval_dataset, False, False, collator) if eval_dataset is not None else None

            # training loop
            loop = tqdm(range(self.config.NUM_EPOCHS), leave=True, desc=f"Epoch", ncols=100)
            for epoch in loop:
                if self.config.SHUFFLE_DATASET:
                    train_loader = self.create_dataloader(train_dataset, True, self.config.SHUFFLE_DATALOADER, collator, epoch)

                # show examples of train dataset
                if epoch == 0:
                    print(f"showing examples of train dataset")
                    for i, batch in enumerate(train_loader):
                        if i < 3:
                            print
                            print(f"Example input sequences{i}: {self.model_manager.tokenizer.batch_decode(batch['input_ids'][:,:-1])}")
                            print(f"Example label{i}: {self.model_manager.tokenizer.batch_decode(batch['labels'][:,1:])}")
                            print("--------------------------------")
                        elif i > len(train_loader) - 4:
                            print(f"Example input sequences{i}: {self.model_manager.tokenizer.batch_decode(batch['input_ids'][:,:-1])}")
                            labels_for_decode = batch['labels'][:,1:].clone()
                            pad_token_id = self.model_manager.tokenizer.pad_token_id
                            labels_for_decode[labels_for_decode == -100] = pad_token_id
                            print(f"Example label{i}: {self.model_manager.tokenizer.batch_decode(labels_for_decode)}")
                            print("--------------------------------")
                
                def _train_epoch(model, train_loader, optimizer):
                    if legacy_train:
                        state = {}
                    model.train()
                    running_loss = 0
                    for i, batch in enumerate(train_loader):
                        optimizer.zero_grad()
                        if legacy_train:
                            outputs = model.forward(input_ids=batch["input_ids"], 
                                                    attention_mask=batch["attention_mask"], 
                                                    labels=batch["labels"], 
                                                    state=state,
                                                    detach_state=True)
                        else:
                            outputs = model.forward(input_ids=batch["input_ids"], 
                                                    attention_mask=batch["attention_mask"], 
                                                    labels=batch["labels"])
                        loss = outputs["loss"]
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        if legacy_train:
                            state = outputs["state"]
                    return {"loss": f"{running_loss / (i + 1):.4f}",
                            "lr": f"{optimizer.param_groups[0]['lr']:.6f}"}

                train_info = _train_epoch(self.model_manager.model, train_loader, optimizer)
                if scheduler is not None and self.config.LEARNING_RATE_SCHEDULER in ["step", "exponential"]:
                    scheduler.step()

                loop.set_postfix(train_info)

                def _eval_epoch(model, eval_loader, compute_metrics, eval_dataset, epoch):
                    print(f"\n evaluating model...")
                    all_labels = []
                    all_preds = []
                    all_logits = []
                    if legacy_train:
                        state = {}
                    model.eval()
                    running_loss = 0
                    loop = tqdm(eval_loader, leave=True, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}", ncols=100)
                    for i, batch in enumerate(loop):
                        if legacy_train:
                            with torch.no_grad():
                                outputs = model.forward(input_ids=batch["input_ids"], 
                                                        attention_mask=batch["attention_mask"], 
                                                        labels=batch["labels"], 
                                                        state=state,
                                                        detach_state=True)
                        else:
                            with torch.no_grad():
                                outputs = model.forward(input_ids=batch["input_ids"], 
                                                        attention_mask=batch["attention_mask"], 
                                                        labels=batch["labels"])
                        loss = outputs["loss"]
                        running_loss += loss.item()
                        all_labels.extend(outputs["labels"].detach().cpu().numpy())
                        logits = outputs["logits"]
                        all_logits.extend(logits.detach().cpu().numpy())
                        preds = torch.argmax(logits, dim=-1)
                        all_preds.extend(preds.detach().cpu().numpy())
                        if legacy_train:
                            state = outputs["state"]
                        loop.set_postfix({"loss": f"{running_loss / (i + 1):.4f}"})
                    all_labels = np.stack(all_labels)
                    all_logits = np.stack(all_logits)
                    all_preds = np.stack(all_preds)
                    record = {
                            "epoch": epoch+1,
                            "loss": running_loss / (i + 1)
                        }
                    if compute_metrics is not None:
                        metric_info = compute_metrics(eval_dataset, all_labels, all_preds, all_logits, self.model_manager.tokenizer)
                        if self.config.METRIC_FOR_BEST_MODEL is not None:
                            metric_for_best = metric_info['accuracy'][self.config.METRIC_FOR_BEST_MODEL]
                        else:
                            metric_for_best = None
                        print(f"evaluation metrics: ")
                        pprint(metric_info['accuracy'], compact=True)
                        for key, value in metric_info.items():
                            for k, v in value.items():
                                value[k] = to_serializable(v)
                            record[key] = to_serializable(value)
                    if log_file is not None:
                        with open(log_file, "a") as f:
                            f.write(json.dumps(record) + "\n")
                    return metric_for_best

                if eval_dataset is not None:
                    if (epoch+1) % self.config.EVAL_STEPS == 0 or epoch == 0:
                        metric_for_best = _eval_epoch(self.model_manager.model, eval_loader, compute_metrics, eval_dataset, epoch)
                        is_best = False
                        if metric_for_best is not None and (metric_for_best > best_acc):
                            print(f"New best metric: {metric_for_best}")
                            best_acc = metric_for_best
                            is_best = True
                        ckpt_dir = save_checkpoint(run_dir, self.model_manager.model, 
                                            optimizer=optimizer,
                                            scheduler=scheduler,
                                            scaler=None,
                                            tokenizer=self.model_manager.tokenizer,
                                            epoch=epoch,
                                            is_best=is_best,
                                            max_to_keep=self.config.SAVE_TOTAL_LIMIT
                                            )
            if self.config.LOAD_BEST_MODEL_AT_END:
                best_ckpt = os.path.join(run_dir, "checkpoints", "best")
                _ = load_checkpoint(best_ckpt, model=self.model_manager.model, map_location="auto", strict=True, load_rng=False)
                print(f"Loaded best model from: {best_ckpt}")

            if self.config.SAVE_MODEL:
                if self.config.LOAD_BEST_MODEL_AT_END:
                    print(f"Saving best model to: {os.path.join(run_dir, 'export')}")
                else:
                    print(f"Saving last model to: {os.path.join(run_dir, 'export')}")
                save_dir = os.path.join(run_dir, "export")
                clean_dir(save_dir)
                save_pretrained(self.model_manager.model, save_dir)
                self.model_manager.tokenizer.save_pretrained(save_dir)
                print(f"model saved!")