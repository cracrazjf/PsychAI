from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Optional, Sequence, Any, Iterator, Callable
from ...config.training import TrainingConfig
from collections import Counter
import math
import random
import hashlib
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import itertools
import gzip
import io
import csv
import zstandard as zstd
from ...simple_nn.models.language_loader import ModelManager

PAD, BOS, EOS, UNK, CLS, SEP, MASK = "<pad>", "<bos>", "<eos>", "<unk>", "<cls>", "<sep>", "<mask>"

@dataclass
class VocabPack:
    stoi: Dict[str, int]
    itos: List[str]
    pad_id: int
    bos_id: int
    eos_id: int
    unk_id: int
    cls_id: int
    sep_id: int
    mask_id: int

def tokenize(s: str, level: str = "word") -> List[str]:
    return list(s) if level == "char" else normalize_text(s).split()

def build_vocab(texts: Iterable[str], level="word", min_freq: int = 1,
                extra_tokens: Sequence[str] = ()) -> VocabPack:
    cnt = Counter()
    for t in texts:
        cnt.update(tokenize(t, level))
    # reserve specials at fixed indices
    itos = [PAD, BOS, EOS, UNK, CLS, SEP, MASK] + [tok for tok, f in cnt.items() if f >= min_freq and tok not in {PAD,BOS,EOS,UNK,CLS,SEP,MASK}]
    itos = list(dict.fromkeys(list(itos) + list(extra_tokens)))  # preserve order, unique
    stoi = {t: i for i, t in enumerate(itos)}
    return VocabPack(
        stoi=stoi, itos=itos,
        pad_id=stoi[PAD], bos_id=stoi[BOS], eos_id=stoi[EOS], unk_id=stoi[UNK],
        cls_id=stoi[CLS], sep_id=stoi[SEP], mask_id=stoi[MASK],
    )

def encode(tokens: List[str], vocab: VocabPack, unk_ok: bool = True) -> List[int]:
    if unk_ok:
        return [vocab.stoi.get(t, vocab.unk_id) for t in tokens]
    missing = [t for t in tokens if t not in vocab.stoi]
    if missing:
        raise KeyError(f"Unknown tokens encountered: {missing[:5]}...")
    return [vocab.stoi[t] for t in tokens]

def decode(ids: Iterable[int], vocab: VocabPack, strip_specials: bool = True) -> List[str]:
    toks = [vocab.itos[i] if 0 <= i < len(vocab.itos) else UNK for i in ids]
    return [t for t in toks if not (strip_specials and t in {PAD,BOS,EOS,CLS,SEP,MASK})]

def add_specials(
    ids: List[int],
    *,
    add_bos: bool = False,
    add_eos: bool = False,
    bos_id: int,
    eos_id: int
) -> List[int]:
    out = ids
    if add_bos: out = [bos_id] + out
    if add_eos: out = out + [eos_id]
    return out

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
            for line in f: yield line

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

def pad_sequences(seqs: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad 1D LongTensors to [B, T] with pad_id; also return attention_mask [B,T] (1=real, 0=pad).
    """
    assert len(seqs) > 0, "Empty batch"
    T = max(int(s.numel()) for s in seqs)
    B = len(seqs)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((B, T), dtype=torch.long)
    for i, s in enumerate(seqs):
        n = int(s.numel())
        input_ids[i, :n] = s
        attn_mask[i, :n] = 1
    return input_ids, attn_mask

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
    if match:
        rdr = rdr.filter(match=match)  # e.g., child/folder name
    # words(..., by_utterances=True) returns List[List[str]]
    if by_utterance:
        utts = rdr.words(participants=participant, by_utterances=True)
        for i, words in enumerate(utts):
            if not words: continue
            text = " ".join(words)
            meta = {"participant": participant, "corpus": source, "utterance_index": i}
            yield to_record(text, source=f"{participant}_{text}#utt{i}", meta=meta)
    else:
        # one record per file: join all words
        all_words = rdr.words(participants=participant)
        text = " ".join(all_words)
        yield to_record(text, source=f"{source}", meta={"participant": participant, "corpus": source})

def split_data(
    records: Iterator[Record],
    save_path: str,
    test_ratio: float = 0.2,
    validation_ratio: float = 0.0,
    id_key: str = "id"
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(os.path.join(save_path, "train.jsonl"), "w", encoding="utf-8") as f_train, \
            open(os.path.join(save_path, "test.jsonl"), "w", encoding="utf-8") as f_test, \
            open(os.path.join(save_path, "validation.jsonl"), "w", encoding="utf-8") as f_validation:

            train_records = []
            test_records = []
            for rec in records:
                # pick something stable to hash
                sid = str(rec.id)

                # hash → float between 0 and 1
                h = hashlib.md5(sid.encode("utf-8")).hexdigest()
                p = int(h, 16) / 16**32  

                if p < test_ratio:
                    test_records.append(rec)
                    f_test.write(json.dumps(rec, ensure_ascii=False) + "\n")
                elif p < test_ratio + validation_ratio:
                    validation_records.append(rec)
                    f_validation.write(json.dumps(rec, ensure_ascii=False) + "\n")
                else:
                    train_records.append(rec)
                    f_train.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"✅ Wrote {len(train_records)} train records to {os.path.join(save_path, 'train.jsonl')}, {len(test_records)} test records to {os.path.join(save_path, 'test.jsonl')}")
        print(f"✅ Wrote {len(validation_records)} validation records to {os.path.join(save_path, 'validation.jsonl')}")
        return train_records, validation_records, test_records

def create_dataloader_hf(config: TrainingConfig,
                        model_manager: ModelManager) -> Tuple[DataLoader, DataLoader]:

    if model_manager.tokenizer.pad_token is None: model_manager.tokenizer.pad_token = model_manager.tokenizer.eos_token
    train_dataset = load_dataset("json", data_files=config.TRAIN_DATA_PATH, split="train")
    if config.EVAL_DATA_PATH is not None:
        eval_dataset = load_dataset("json", data_files=config.EVAL_DATA_PATH, split="train")
    else:
        eval_dataset = None
    
    def tokenize_function(examples):
        return model_manager.tokenizer(examples["text"], add_special_tokens=False)
    
    def create_inputs_and_labels(examples):
        #flatten the document 
        ids = list(itertools.chain.from_iterable(examples["input_ids"]))
        total = (len(ids) // (sequence_length + 1)) * (sequence_length + 1)
        ids = ids[:total]
        chunks = [ids[i:i+sequence_length+1] for i in range(0, total, sequence_length+1)]
        inputs  = [c[:-1] for c in chunks]
        labels  = [c[1:]  for c in chunks]
        return {"input_ids": inputs, "labels": labels}

    tok_ds = dataset.map(tokenize_function, batched=True, num_proc=1, remove_columns=dataset["train"].column_names)
    lm_ds  = tok_ds.map(create_inputs_and_labels, batched=True, num_proc=1)
    
    task = self.config.TASK
    if task == "causal_lm":
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif task == "masked_lm":
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    
    train_loader = DataLoader(lm_ds["train"], batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collator)
    val_loader   = DataLoader(lm_ds["test"],  batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collator)

    return train_loader, val_loader