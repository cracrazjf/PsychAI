from __future__ import annotations

from .read_any import read_json, read_jsonl, read_csv
from typing import Dict, Optional, Sequence, Iterator, Union
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

def load_childes(source: str, 
                 *, 
                 participant: Union[str, Sequence[str]] = "CHI", 
                 by_utterance: bool = True, 
                 by_child: bool = False,
                 match: Optional[str] = None) -> Iterator[Dict]:
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
                yield {"text": text, "meta": meta}
    elif by_child:
        utts_in_files = rdr.words(participants=participant, by_utterances=True, by_files=True)
        file_paths   = [Path(p) for p in rdr.file_paths()] 
        by_child_map: Dict[str, list] = {}
        child_file_counts: Dict[str, int] = {}
        for utts, file_path in zip(utts_in_files, file_paths):
            child_name = file_path.parent.name
            child_file_counts[child_name] = child_file_counts.get(child_name, 0) + 1
            bucket = by_child_map.setdefault(child_name, [])
            for words in utts:
                if words:
                    bucket.append(words)
        for child_name, utterances in by_child_map.items():
            # Flatten utterances into one string per child
            # (join words within each utterance by spaces, then join utterances by spaces)
            text = " ".join(" ".join(words) for words in utterances if words)
            meta = {
                "participant": participant,
                "corpus": source,
                "child_name": child_name,
                "num_utterances": len(utterances),
                "num_files": child_file_counts.get(child_name, 0),
                "grouping": "child",
            }
            yield {"text": text, "meta": meta}
    else:
        meta = {"participant": participant, "corpus": source}
        all_words = rdr.words(participants=participant)
        text = " ".join(all_words)
        yield {"text": text, "meta": meta}