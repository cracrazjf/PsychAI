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

PAD, BOS, EOS, UNK, CLS, SEP, MASK = "<pad>", "<bos>", "<eos>", "<unk>", "<cls>", "<sep>", "<mask>"

@dataclass
class Record:
    id: str
    text: str
    meta: Dict[str, Any]

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

def create_windows(
    ids: List[int],
    sequence_length: int,
    *,
    stride: Optional[int] = None,
    require_min_len: int = 2
) -> List[List[int]]:
    """
    Break one long stream into blocks of (<= block_size+1) so we can shift for LM.
    Returns a list of 1D int lists (each length >= require_min_len).
    """
    if stride is None: stride = sequence_length
    out: List[List[int]] = []
    i = 0
    while i + 1 < len(ids):
        blk = ids[i: i + sequence_length + 1]
        if len(blk) >= require_min_len:
            out.append(blk)
        i += stride
    return out

def create_windows_from_texts(texts: List[str], 
    vocab: VocabPack,
    *,
    level: str = "word",
    sequence_length: int = 256,
    add_eos: bool = True) -> List[List[int]]:
    if add_eos:
        eos = [vocab.eos_id]
        stream: List[int] = []
    for s in texts:
        ids = encode(tokenize(s, level), vocab)
        stream.extend(ids + eos)
    windows = create_windows(stream, sequence_length, stride=sequence_length, require_min_len=2)
    return [torch.tensor(w, dtype=torch.long) for w in windows]

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

def build_labels_causal_lm(batch_blocks: List[torch.Tensor], pad_id: int, ignore_index: int = -100) -> Dict[str, torch.Tensor]:
    """
    batch_blocks: list of 1D tensors length L_i (already (tokens + next) for each sample).
    Returns input_ids [B,T], attention_mask [B,T], labels [B,T] (shifted, with -100 on pads).
    """
    inputs = [b[:-1] for b in batch_blocks if b.numel() >= 2]
    labels = [b[1:]  for b in batch_blocks if b.numel() >= 2]
    assert inputs, "All sequences too short for next-token."
    x, attn = pad_sequences(inputs, pad_id)
    y, _    = pad_sequences(labels, pad_id=ignore_index)   # temporary; we’ll overwrite pads to -100
    y = torch.where(attn.bool(), y, torch.full_like(y, ignore_index))
    return {"input_ids": x, "attention_mask": attn, "labels": y}

def collate_causal_lm(batch_blocks: List[torch.Tensor], pad_id: int, ignore_index: int = -100) -> Dict[str, torch.Tensor]:
    return build_labels_causal_lm(batch_blocks, pad_id, ignore_index)

def decode(ids: Iterable[int], vocab: VocabPack, strip_specials: bool = True) -> List[str]:
    toks = [vocab.itos[i] if 0 <= i < len(vocab.itos) else UNK for i in ids]
    return [t for t in toks if not (strip_specials and t in {PAD,BOS,EOS,CLS,SEP,MASK})]
    
def to_record(text: str, *, source: str, meta: Optional[Dict[str, Any]] = None) -> Record:
    text = normalize_text(text)
    rid = stable_id(source, text)
    return Record(id=rid, text=text, meta=(meta or {"source": source}))

def read_txt_lines(path: str) -> Iterator[Record]:
    for i, line in enumerate(_open_text(path)):
        line = line.strip()
        if line:
            yield to_record(line, source=f"{path}#{i}")

def read_jsonl(path: str, text_key: str = "text") -> Iterator[Record]:
    for i, line in enumerate(_open_text(path)):
        if not line.strip(): continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        text = obj.get(text_key) or obj.get("content") or obj.get("body") or ""
        meta = {k: v for k, v in obj.items() if k != text_key}
        if text:
            yield to_record(text, source=f"{path}#{i}", meta=meta)

def read_csv(path: str, text_col: str = "text", delimiter: str = None) -> Iterator[Record]:
    sep = delimiter or ("\t" if path.endswith(".tsv") else ",")
    with open(path, "rt", encoding="utf-8", errors="ignore") as f:
        r = csv.DictReader(f, delimiter=sep)
        cols = r.fieldnames or []
        for i, row in enumerate(r):
            text = row.get(text_col, "")
            meta = {k: v for k in row.items() if k[0] != text_col}
            if text.strip():
                yield to_record(text, source=f"{path}#{i}", meta=meta)            

def read_childes(source: str, *, participant: str = "CHI", by_utterance: bool = True, match: Optional[str] = None) -> Iterator[Record]:
    """
    source: path to a local .zip/.cha directory or a TalkBank URL to a zip.
    participant: "CHI" for child (default), others like "MOT", "FAT"...
    """
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
            yield to_record(text, source=f"{source}#utt{i}", meta=meta)
    else:
        # one record per file: join all words
        all_words = rdr.words(participants=participant)
        text = " ".join(all_words)
        yield to_record(text, source=f"{source}", meta={"participant": participant, "corpus": source})

def train_test_split(
    records: Iterator[Record],
    train_path: str,
    test_path: str,
    test_ratio: float = 0.2,
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
    with open(train_path, "w", encoding="utf-8") as f_train, \
         open(test_path, "w", encoding="utf-8") as f_test:

        train_records = []
        test_records = []
        for rec in records:
            # pick something stable to hash
            sid = str(rec.get(id_key, json.dumps(rec, sort_keys=True)))

            # hash → float between 0 and 1
            h = hashlib.md5(sid.encode("utf-8")).hexdigest()
            p = int(h, 16) / 16**32  

            if p < test_ratio:
                test_records.append(rec)
                f_test.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                train_records.append(rec)
                f_train.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(train_records)} train records to {train_path}, {len(test_records)} test records to {test_path}")
    return train_records, test_records

def create_dataloader_hf(config: TrainingConfig, 
                        *,
                        tokenizer: Optional[AutoTokenizer] = None,
                        train_path: Optional[str] = None, 
                        test_path: Optional[str] = None,
                        train_data: Optional[List[Record]] = None,
                        test_data: Optional[List[Record]] = None) -> Tuple[DataLoader, DataLoader]:
    model_name = config.MODEL_NAME
    task_type = config.TASK_TYPE
    sequence_length = config.SEQUENCE_LENGTH
    if tokenizer is None:
        trust_remote_code = config.TRUST_REMOTE_CODE
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if train_path is not None and test_path is not None:
        dataset = load_dataset("json", data_files={"train": train_path, "test": test_path})
    elif train_path is not None and test_path is None:
        dataset = load_dataset("json", data_files={"train": train_path})
    elif train_data is not None and test_data is not None:
        dataset = DatasetDict({
            "train": Dataset.from_list(train_data),
            "test":  Dataset.from_list(test_data),
        })
    elif train_data is not None and test_data is None:
        dataset = DatasetDict({
            "train": Dataset.from_list(train_data),
        })
    else:
        raise ValueError("Either train_path or train_data must be provided")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], add_special_tokens=False)
    
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
    
    if task_type == "causal_lm":
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif task_type == "masked_lm":
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    
    train_loader = DataLoader(lm_ds["train"], batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collator)
    val_loader   = DataLoader(lm_ds["test"],  batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collator)

    return train_loader, val_loader

def create_dataloader_custom(config: TrainingConfig, 
                             *,
                             train_path: str, 
                             test_path: str) -> Tuple[DataLoader, DataLoader]:
    add_eos = config.ADD_EOS
    batch_size = config.BATCH_SIZE
    sequence_length = config.SEQUENCE_LENGTH
    def iter_texts(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                txt = obj.get("text", "").strip()
                if txt: yield txt
    train_texts = list(iter_texts(train_path))
    eval_texts  = list(iter_texts(test_path))
    vocab = build_vocab(train_texts, min_freq=1)
    train_windows = create_windows_from_texts(train_texts, vocab, sequence_length=sequence_length, add_eos=add_eos)
    eval_windows = create_windows_from_texts(eval_texts, vocab, sequence_length=sequence_length, add_eos=add_eos)

    class Blocks(Dataset):
        def __init__(self, blocks): self.blocks = blocks
        def __len__(self): return len(self.blocks)
        def __getitem__(self, i): return self.blocks[i]
    
    train_ds = Blocks(train_windows)
    eval_ds = Blocks(eval_windows)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_causal_lm(b, vocab.pad_id))
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_causal_lm(b, vocab.pad_id))

    return train_loader, eval_loader


