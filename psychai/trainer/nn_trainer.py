from torch._dynamo.polyfills import predicate
from transformers import Trainer, TrainingArguments
from psychai.model_manager.lm_manager import LM_ModelManager
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from functools import partial
from typing import Tuple
from itertools import chain
from datasets import Dataset
from psychai.artificial_dataset.xAyBz import XAYBZ
import torch
from torch.optim import AdamW, SGD
from dataclasses import dataclass
import numpy

class NN_Trainer:
    def __init__(self, config):
        self.config = config
        self.model_manager = None

    def load_lanaguge_model_and_tokenizer(self):
        self.model_manager = LM_ModelManager()
        self.model_manager.load_model(self.config.MODEL_NAME, self.config.MODEL_PATH, self.config.TASK, self.config.CUSTOMIZED_MODEL, self.config.TOKENIZER_PATH, self.config.TRUST_REMOTE_CODE)
        
    def create_nonoverlapping_sequences(self, input_ids, attention_mask, sequence_length, pad_token_id):
        print(f"📝 creating nonoverlapping sequences with sequence length {sequence_length}, pad token id {pad_token_id}")
        blocks = {"input_ids": [], "attention_mask": []}

        for i in range(0, len(input_ids), sequence_length):
            ids = input_ids[i : i + sequence_length]
            mask = attention_mask[i : i + sequence_length]

            if len(ids) < sequence_length:
                pad_len = sequence_length - len(ids)
                ids = ids + [pad_token_id] * pad_len
                mask = mask + [0] * pad_len

            blocks["input_ids"].append(ids)
            blocks["attention_mask"].append(mask)
        return blocks
    
    def create_overlapping_sequences(self, input_ids, attention_mask, sequence_length, pad_token_id, stride = 1):
        print(f"📝 creating overlapping sequences with sequence length {sequence_length}, pad token id {pad_token_id}, and stride {stride}")
        assert sequence_length > 1, "sequence_length must be >= 2 for causal LM"
        n = len(input_ids)
        windows_input_ids, windows_attention_mask = [], []
        for start in range(0, n, stride):
            end = start + sequence_length
            ids = input_ids[start:end]
            msk = attention_mask[start:end]

            # pad if last window is too short
            if len(ids) < sequence_length:
                pad_len = sequence_length - len(ids)
                ids = ids + [pad_token_id] * pad_len
                msk = msk + [0] * pad_len

            windows_input_ids.append(ids)
            windows_attention_mask.append(msk)

            if end >= n:
                break
        return {"input_ids": windows_input_ids, "attention_mask": windows_attention_mask}

    def concatenate_input_ids_and_attention_mask(self, dataset):
        input_ids_lists = dataset["input_ids"]
        attn_mask_lists = dataset["attention_mask"]
        input_ids = list(chain.from_iterable(input_ids_lists))
        attention = list(chain.from_iterable(attn_mask_lists))
        return input_ids, attention

    def prepare_language_data(self) -> Tuple[DataLoader, DataLoader]:
        if self.model_manager.tokenizer.pad_token is None: self.model_manager.tokenizer.pad_token = self.model_manager.tokenizer.eos_token
        train_dataset = load_dataset("json", data_files=self.config.TRAIN_DATA_PATH, split="train")
        if self.config.EVAL_DATA_PATH is not None:
            eval_dataset = load_dataset("json", data_files=self.config.EVAL_DATA_PATH, split="train")
        else:
            eval_dataset = None
        
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
        print(f"📝 tokenizing train dataset")
        tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=self.config.DATA_PROCESS_BATCH_SIZE, num_proc=self.config.DATA_PROCESS_NUM_PROC)
        if eval_dataset is not None:
            print(f"📝 tokenizing eval dataset")
            tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, batch_size=self.config.DATA_PROCESS_BATCH_SIZE, num_proc=self.config.DATA_PROCESS_NUM_PROC, remove_columns=eval_dataset.column_names)
        else:
            tokenized_eval_dataset = None

        if self.config.OVERLAPPING_SEQUENCE:
            return tokenized_train_dataset, tokenized_eval_dataset, collator
        else:
            train_dataset_input_ids, train_dataset_attention = self.concatenate_input_ids_and_attention_mask(tokenized_train_dataset)
            train_sequences = self.create_nonoverlapping_sequences(train_dataset_input_ids, train_dataset_attention, self.config.SEQUENCE_LENGTH, self.model_manager.tokenizer.pad_token_id)
            final_train_dataset = Dataset.from_dict(train_sequences)
            if tokenized_eval_dataset is not None:
                eval_dataset_input_ids, eval_dataset_attention = self.concatenate_input_ids_and_attention_mask(tokenized_eval_dataset)
                eval_sequences = self.create_nonoverlapping_sequences(eval_dataset_input_ids, eval_dataset_attention, self.config.SEQUENCE_LENGTH, self.model_manager.tokenizer.pad_token_id)
                final_eval_dataset = Dataset.from_dict(eval_sequences)
            else:
                final_eval_dataset = None
            return final_train_dataset, final_eval_dataset, collator

    def create_training_arguments(self):
        return TrainingArguments(
                num_train_epochs=self.config.NUM_EPOCHS,
                per_device_train_batch_size=self.config.PER_DEVICE_TRAIN_BATCH_SIZE,
                learning_rate=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY,
                seed=self.config.RANDOM_STATE,
                eval_strategy=self.config.EVAL_STRATEGY,
                eval_steps=self.config.EVAL_STEPS,
                include_for_metrics = ["inputs"],
                logging_strategy=self.config.LOGGING_STRATEGY,
                logging_steps=self.config.LOGGING_STEPS,
                dataloader_drop_last=False
            )

    def train(self, compute_metrics=None):
        self.load_lanaguge_model_and_tokenizer()
        training_args = self.create_training_arguments()
        if self.config.OVERLAPPING_SEQUENCE:
            tokenized_train_dataset, tokenized_eval_dataset, collator = self.prepare_language_data()
            self.legacy_train(tokenized_train_dataset, tokenized_eval_dataset, collator, compute_metrics)
        else:
            self.load_lanaguge_model_and_tokenizer()
            train_dataset, eval_dataset, collator = self.prepare_language_data()
            trainer = Trainer(
                model=self.model_manager.model,
                tokenizer=self.model_manager.tokenizer,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=collator,
                compute_metrics=compute_metrics
            )
            trainer.train()

    def train_language_model(self, train_dataset, eval_dataset, collator, compute_metrics=None):
        # select optimizer
        if self.config.OPTIMIZER == "adamw":
            optimizer = AdamW(self.model_manager.model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY)
        elif self.config.OPTIMIZER == "sgd":
            optimizer = SGD(self.model_manager.model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.OPTIMIZER}")
        
        for epoch in range(self.config.NUM_EPOCHS):
            if epoch == 0:
                print(f"‼️ training will start with shuffling train dataset with seed of epoch each time")
            shuffled_train_dataset = train_dataset.shuffle(seed=epoch)
            train_input_ids, train_attention = self.concatenate_input_ids_and_attention_mask(shuffled_train_dataset)
            train_sequences = self.create_overlapping_sequences(train_input_ids, train_attention, 
                                                                   self.config.SEQUENCE_LENGTH, 
                                                                   self.model_manager.tokenizer.pad_token_id,
                                                                   self.config.OVERLAPPING_STRIDE)
            final_train_dataset = Dataset.from_dict(train_sequences)
            train_loader = DataLoader(final_train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, collate_fn=collator)
            
            if epoch == 0:
                print(f"✅ created train loader with {len(train_loader)} batches, batch size {self.config.BATCH_SIZE}, and shuffle is turned off")
                print(f"📄 showing train example sequences:")
                for i, batch in enumerate(train_loader):
                    if i < 3:
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
            if eval_dataset is not None:
                eval_input_ids, eval_attention = self.concatenate_input_ids_and_attention_mask(eval_dataset)
                eval_sequences = self.create_overlapping_sequences(eval_input_ids, eval_attention, 
                                                                   self.config.SEQUENCE_LENGTH, 
                                                                   self.model_manager.tokenizer.pad_token_id,
                                                                   self.config.OVERLAPPING_STRIDE)
                final_eval_dataset = Dataset.from_dict(eval_sequences)
                eval_loader = DataLoader(final_eval_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, collate_fn=collator)
                
                if epoch == 0:
                    print(f"✅ created eval loader with {len(eval_loader)} batches, batch size {self.config.BATCH_SIZE}, and shuffle is turned off")
                    print(f"📄 showing eval example sequences:")
                    for i, batch in enumerate(eval_loader):
                        if i < 3:
                            print(f"Example input sequences{i}: {self.model_manager.tokenizer.batch_decode(batch['input_ids'][:,:-1])}")
                            print(f"Example label{i}: {self.model_manager.tokenizer.batch_decode(batch['labels'][:,1:])}")
                            print("--------------------------------")
                        elif i > len(eval_loader) - 4:
                            print(f"Example input sequences{i}: {self.model_manager.tokenizer.batch_decode(batch['input_ids'][:,:-1])}")
                            labels_for_decode = batch['labels'][:,1:].clone()
                            pad_token_id = self.model_manager.tokenizer.pad_token_id
                            labels_for_decode[labels_for_decode == -100] = pad_token_id
                            print(f"Example label{i}: {self.model_manager.tokenizer.batch_decode(labels_for_decode)}")
                            print("--------------------------------")
            else:
                eval_loader = None
            if self.config.LEGACY_TRAIN:
                print(f"‼️ Legacy training activated")
                print(f"‼️ This is specifically for RNN model. If you are not using RNN model, please set legacy_train to False")
                print(f"‼️ Legacy training will keep the state of the model between batches")
                state = {}
                print(f"✅ created state first time for epoch {epoch}: state is empty")
            self.model_manager.model.train()
            total_loss = 0
            all_inputs = []
            all_preds = []
            all_labels = []
            for i, batch in enumerate(train_loader):
                all_inputs.extend(self.model_manager.tokenizer.batch_decode(batch["input_ids"][:,:-1]))
                optimizer.zero_grad()
                if self.config.LEGACY_TRAIN:
                    outputs = self.model_manager.model.forward(input_ids=batch["input_ids"], 
                                                               attention_mask=batch["attention_mask"], 
                                                               labels=batch["labels"], 
                                                               state=state,
                                                               detach_state=True)
                else:
                    outputs = self.model_manager.model.forward(input_ids=batch["input_ids"], 
                                                               attention_mask=batch["attention_mask"], 
                                                               labels=batch["labels"])
                loss = outputs["loss"]
                logits = outputs["logits"]
                labels = outputs["labels"]
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(self.model_manager.tokenizer.batch_decode(preds))
                labels_for_decode = labels.clone()
                labels_for_decode[labels_for_decode == -100] = self.model_manager.tokenizer.pad_token_id
                all_labels.extend(self.model_manager.tokenizer.batch_decode(labels_for_decode))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if self.config.LEGACY_TRAIN:
                    state = outputs["state"]
            if compute_metrics is not None:
                metrics = compute_metrics(all_inputs, all_preds, all_labels)
            