from __future__ import annotations

from read_any import read_json, read_jsonl, read_csv
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Iterable, Optional, Sequence, Any, Iterator, Callable
from collections import Counter
import os
import hashlib
import json
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from functools import partial
from pathlib import Path

def load_any(file_path: str, 
             file_type: str = "json",
             task: str = "causal_lm") -> Iterator[Dict]:

    if file_type == "csv":
        row_dicts = read_csv(file_path)
    elif file_type == "jsonl":
        row_dicts = read_jsonl(file_path)
    elif file_type == "json":
        row_dicts = read_json(file_path)
    else:
        raise ValueError(f"Invalid file type: {file_type}")
    
    if task == "causal_lm":
        required_keys = ["text"]
    elif task == "masked_lm":
        required_keys = ["text"]
    elif task == "text_classification":
        required_keys = ["text", "label"]
    elif task == "token_classification":
        required_keys = ["tokens", "label"]
    else:
        raise ValueError(f"Invalid task: {task}, must be 'causal_lm', 'masked_lm', 'text_classification', or 'token_classification'")

    for row in row_dicts:
        for key in required_keys:
            if key not in row:
                raise ValueError(f"Missing required key: {key}")
        yield row

def load_childes(source: str, *, participant: str = "CHI", by_utterance: bool = True, match: Optional[str] = None) -> Iterator[Dict]:
    try:
        import pylangacq
    except Exception as e:
        raise RuntimeError("pylangacq not installed; pip install pylangacq") from e

    rdr = pylangacq.read_chat(source)
    participant = participant if isinstance(participant, list) else [participant]
    if match:
        rdr = rdr.filter(match=match)
    if by_utterance:
        utts_in_files = rdr.words(participants=participant, by_utterances=True, by_files=True)
        file_paths   = [Path(p) for p in rdr.file_paths()] 
        for i, (utts, file_path) in enumerate(zip(utts_in_files, file_paths)):
            child_name = file_path.parent.name
            for words in utts:
                if not words: continue
                text = " ".join(words)
                meta = {"participant": participant, "corpus": source, "child_name": child_name, "utterance_index": i}
                yield {text: text, "meta": meta}
    else:
        meta = {"participant": participant, "corpus": source}
        all_words = rdr.words(participants=participant)
        text = " ".join(all_words)
        yield {text: text, "meta": meta}

# def create_language_dataloader(config,
                        model_manager) -> Tuple[DataLoader, DataLoader]:

    if model_manager.tokenizer.pad_token is None: model_manager.tokenizer.pad_token = model_manager.tokenizer.eos_token
    train_dataset = load_dataset("json", data_files=config.TRAIN_DATA_PATH, split="train")
    if config.EVAL_DATA_PATH is not None:
        eval_dataset = load_dataset("json", data_files=config.EVAL_DATA_PATH, split="train")
    else:
        eval_dataset = None
    
    def tokenize_function(batch):
        return model_manager.tokenizer(batch["text"], add_special_tokens=False, truncation=False)
    
    def create_inputs_and_labels(examples, sequence_length, pad_token_id):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        blocks = {"input_ids": [], "attention_mask": []}

        for i in range(0, len(concatenated["input_ids"]), sequence_length):
            ids = concatenated["input_ids"][i : i + sequence_length]
            mask = concatenated["attention_mask"][i : i + sequence_length]

            if len(ids) < sequence_length:
                pad_len = sequence_length - len(ids)
                ids = ids + [pad_token_id] * pad_len
                mask = mask + [0] * pad_len


            blocks["input_ids"].append(ids)
            blocks["attention_mask"].append(mask)
        blocks["labels"] = blocks["input_ids"].copy()

        return blocks

    train_tok_ds = train_dataset.map(tokenize_function, batched=True, batch_size=config.DATA_PROCESS_BATCH_SIZE, num_proc=config.DATA_PROCESS_NUM_PROC, remove_columns=train_dataset.column_names)

    prepare_data_fn =partial(create_inputs_and_labels, sequence_length=config.SEQUENCE_LENGTH, pad_token_id=model_manager.tokenizer.pad_token_id)
    train_lm_ds  = train_tok_ds.map(prepare_data_fn, batched=True, batch_size=config.BATCH_SIZE, num_proc=config.DATA_PROCESS_NUM_PROC)
    if eval_dataset is not None:
        eval_tok_ds = eval_dataset.map(tokenize_function, batched=True, batch_size=config.DATA_PROCESS_BATCH_SIZE, num_proc=config.DATA_PROCESS_NUM_PROC, remove_columns=eval_dataset.column_names)
        eval_lm_ds  = eval_tok_ds.map(prepare_data_fn, batched=True, batch_size=config.DATA_PROCESS_BATCH_SIZE, num_proc=config.DATA_PROCESS_NUM_PROC)
    
    task = config.TASK
    if task == "causal_lm":
        collator = DataCollatorForLanguageModeling(tokenizer=model_manager.tokenizer, mlm=False)
    elif task == "masked_lm":
        collator = DataCollatorForLanguageModeling(tokenizer=model_manager.tokenizer, mlm=True)
    else:
        raise ValueError(f"Unsupported task type: {task}")

    
    train_loader = DataLoader(train_lm_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collator)
    if eval_dataset is not None:
        val_loader   = DataLoader(eval_lm_ds,  batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collator)
    else:
        val_loader = None

    return train_loader, val_loader