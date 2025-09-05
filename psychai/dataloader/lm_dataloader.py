from __future__ import annotations

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
import gzip
from pathlib import Path
import io
import csv
import zstandard as zstd

def _open_text(path: str) -> Iterator[str]:
    """Yield lines from (possibly compressed) single-file text."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f: yield line
    elif path.endswith(".zst"):
        if zstd is None:
            raise RuntimeError("zstandard not installed; cannot read .zst")
        with open(path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as r:
                for line in io.TextIOWrapper(r, encoding="utf-8", errors="ignore"):
                    yield line
    else:
        with open(path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f: yield json.loads(line)

def normalize_text(s: str) -> str:
    # keep simple & predictable; customize as needed
    return " ".join(s.strip().split())

def stable_id(*parts: Any) -> str:
    """Deterministic ID via BLAKE2b over input parts."""
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        if isinstance(p, (bytes, bytearray)):
            h.update(p)
        else:
            h.update(str(p).encode("utf-8", errors="ignore"))
            h.update(b"|")
    return h.hexdigest()

@dataclass
class Record:
    id: str
    text: str
    meta: Dict[str, Any]

def to_record(text: str, source: str, meta: Optional[Dict[str, Any]] = None) -> Record:
    text = normalize_text(text)
    rid = stable_id(source, text)
    return Record(id=rid, text=text, meta=meta or {"source": source})

def read_txt_lines(path: str) -> Iterator[Record]:
    for i, line in enumerate(_open_text(path)):
        line = line.strip()
        if line:
            yield to_record(line, source=f"{line}#{i}", meta={"source": path})

def read_jsonl(path: str, text_key: str = "text") -> Iterator[Record]:
    for i, line in enumerate(_open_text(path)):
        if not isinstance(line, dict):
            raise ValueError(f"each line must be a dictionary")
        text = line.get(text_key) or line.get("content") or line.get("body") or ""
        if text.strip():
            yield to_record(text, source=f"{text}#{i}", meta={"source": path})

def read_csv(path: str, text_col: str = "text", delimiter: str = None) -> Iterator[Record]:
    sep = delimiter or ("\t" if path.endswith(".tsv") else ",")
    with open(path, "rt", encoding="utf-8", errors="ignore") as f:
        rows = csv.DictReader(f, delimiter=sep)
        cols = rows.fieldnames or []
        for i, row in enumerate(rows):
            text = row.get(text_col, "")
            if text.strip():
                yield to_record(text, source=f"{text}#{i}", meta={"source": path})            

def read_childes(source: str, *, participant: str = "CHI", by_utterance: bool = True, match: Optional[str] = None) -> Iterator[Record]:
    try:
        import pylangacq
    except Exception as e:
        raise RuntimeError("pylangacq not installed; pip install pylangacq") from e

    rdr = pylangacq.read_chat(source)
    participant = participant if isinstance(participant, list) else [participant]
    if match:
        rdr = rdr.filter(match=match)  # e.g., child/folder name
    # words(..., by_utterances=True) returns List[List[str]]
    if by_utterance:
        utts_in_files = rdr.words(participants=participant, by_utterances=True, by_files=True)
        file_paths   = [Path(p) for p in rdr.file_paths()] 
        for i, (utts, file_path) in enumerate(zip(utts_in_files, file_paths)):
            child_name = file_path.parent.name
            for words in utts:
                if not words: continue
                text = " ".join(words)
                meta = {"participant": participant, "corpus": source, "child_name": child_name, "utterance_index": i}
                yield to_record(text, source=f"{participant}_{text}#utt{i}", meta=meta)
    else:
        # one record per file: join all words
        all_words = rdr.words(participants=participant)
        text = " ".join(all_words)
        yield to_record(text, source=f"{source}", meta={"participant": participant, "corpus": source})

def split_data(
    records: Iterator[Record],
    save_path: Optional[str] = None,
    test_ratio: float = 0.2,
    validation_ratio: float = 0.0
):
    """
    Split dataset into train/test JSONL files deterministically using stable hash.

    Args:
        records: Iterator of dicts, each record must have a unique id (or id_key field).
        train_path: Output path for training set (.jsonl).
        test_path: Output path for test set (.jsonl).
        test_ratio: Proportion of records to put in test set.
        id_key: Key in record used as stable identifier (defaults to "id").
    """
    train_records = []
    test_records = []
    validation_records = []
    if save_path is None:
        for rec in records:
            sid = str(rec.id)

            # hash → float between 0 and 1
            h = hashlib.md5(sid.encode("utf-8")).hexdigest()
            p = int(h, 16) / 16**32  

            if p < test_ratio:
                test_records.append(rec)
            elif p < test_ratio + validation_ratio:
                validation_records.append(rec)
            else:
                train_records.append(rec)
        
        return train_records, validation_records, test_records
    else:
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, "train.jsonl"), "w", encoding="utf-8") as f_train, \
            open(os.path.join(save_path, "test.jsonl"), "w", encoding="utf-8") as f_test, \
            open(os.path.join(save_path, "validation.jsonl"), "w", encoding="utf-8") as f_validation:

            train_records = []
            test_records = []
            validation_records = []
            for rec in records:
                # pick something stable to hash
                sid = str(rec.id)

                # hash → float between 0 and 1
                h = hashlib.md5(sid.encode("utf-8")).hexdigest()
                p = int(h, 16) / 16**32  

                if p < test_ratio:
                    test_records.append(rec)
                    f_test.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                elif p < test_ratio + validation_ratio:
                    validation_records.append(rec)
                    f_validation.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                else:
                    train_records.append(rec)
                    f_train.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

        print(f"✅ Wrote {len(train_records)} train records to {os.path.join(save_path, 'train.jsonl')}, {len(test_records)} test records to {os.path.join(save_path, 'test.jsonl')}")
        print(f"✅ Wrote {len(validation_records)} validation records to {os.path.join(save_path, 'validation.jsonl')}")
        return train_records, validation_records, test_records

def create_language_dataloader(config,
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