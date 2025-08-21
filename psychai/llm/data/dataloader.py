"""
Core data utilities

Essential functions for basic data processing and format handling.
Focus on simplicity and the 80% use case.
"""

import json
import os
import hashlib
import random
import pandas as pd
from typing import List, Dict, Any, Optional, Callable, Iterator

def load_json(filepath: str) -> Any:
    """Load data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_jsonl(p) -> Iterator[Any]:
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def save_json(data: Any, filepath: str, indent: int = 2):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)

def save_jsonl(data: List[Dict], filepath: str):
    """Save data to JSONL file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + '\n')

def validate_format(data: List[Any], format: str = "chat") -> bool:
    """
    Validate data format
    
    Args:
        data: Data to validate
        format: Format to validate against
        
    Returns:
        True if data is in valid format, False otherwise
    """
    if not isinstance(data, list):
        return False
    """
    Check if data is in valid chat format
    
    Expected format:
    [
        [  # Conversation
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        ...
    ]
    """
    if format == "chat":
        for conversation in data:
            if not isinstance(conversation, list):
                return False
            
            for message in conversation:
                if not isinstance(message, dict):
                    return False
                
                if 'role' not in message or 'content' not in message:
                    return False
                
                if message['role'] not in ['system', 'user', 'assistant']:
                    return False
        return True
    """
    Check if data is in valid instruction format
    
    Expected format:
    [
        {"instruction": "...", "output": "..."},
        ...
    ]
    """
    if format == "instruction":
        for conversation in data:
            if not isinstance(conversation, dict):
                return False
            
            if 'instruction' not in conversation or 'output' not in conversation:
                return False
        return True

def stable_id(*parts: Any) -> str:
    """Deterministic ID via BLAKE2b over input parts."""
    h = hashlib.blake2b(digest_size=16)   # 128-bit digest (shorter than full 512-bit BLAKE2b)
    for p in parts:
        if isinstance(p, (bytes, bytearray)):
            h.update(p)
        else:
            h.update(str(p).encode("utf-8", errors="ignore"))
            h.update(b"|")   # delimiter to avoid accidental merges
    return h.hexdigest()

def convert_to_chat_format(
    input_text: str, 
    output_text: str, 
    system_prompt: Optional[str] = None
) -> Dict:
    """
    Convert input/output pair to chat format
    
    Args:
        input_text: User input
        output_text: Expected output
        system_prompt: Optional system prompt
        
    Returns:
        Chat conversation format
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    id = stable_id(input_text, output_text, system_prompt)
    
    messages.extend([
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text}
    ])

    
    return {'id': id, 'messages': messages}

def convert_to_instruction_format(
    instruction: str,
    input_text: Optional[str] = None, 
    output_text: Optional[str] = None
) -> Dict:
    """
    Convert input/output pair to instruction format
    """
    return {
        "id": stable_id(instruction, input_text, output_text),
        "instruction": instruction,   
        "input": input_text,
        "output": output_text
    }

def load_csv_as_chat(
    filepath: str,
    input_column: str = "input",
    output_column: str = "output",
    system_prompt: Optional[str] = None,
    question: Optional[str] = None,
    constraint: Optional[str] = None,
    output_wrapper: Optional[Callable[[str], str]] = None,
    stype: str = "natural",
    clean_function: Optional[Callable[[str], str]] = None,
) -> Iterator[Dict]:
    df = pd.read_csv(filepath)

    if clean_function:
        df[input_column] = df[input_column].apply(clean_function)
        df = df[df[input_column].str.len() > 0].reset_index(drop=True)
    
    seen = set()
    for _, row in df.iterrows():
        if pd.notna(row[input_column]) and pd.notna(row[output_column]):
            if stype == "labeled":
                parts = [f"Input: {row[input_column]}"]
                if question: 
                    parts.append(f"Question: {question}")
                if constraint:
                    parts.append(f"Constraint: {constraint}")
                input_text = "\n".join(parts)
            elif stype == "natural":
                s = row[input_column].strip()
                if question: s += f"\n\n{question}"
                if constraint: s += f"\n{constraint}"
                input_text = s
            else:
                raise ValueError(f"Invalid stype: {stype}")
            if output_wrapper:
                output_text = output_wrapper(row[output_column])
            else:
                output_text = row[output_column]
            conversation = convert_to_chat_format(
                str(input_text),
                str(output_text),
                system_prompt
            )
            rid = conversation["id"]
            if rid in seen:
                continue
            seen.add(rid)
            yield conversation

def load_csv_as_instruction(
    filepath: str,
    instruction_column: str = "instruction",
    input_column: str = "input",
    output_column: str = "output",
    output_wrapper: Optional[Callable[[str], str]] = None,
    clean_function: Optional[Callable[[str], str]] = None,
) -> Iterator[Dict]:
    """
    Load CSV file and convert to instruction format
    """
    df = pd.read_csv(filepath)

    if clean_function:
        df[instruction_column] = df[instruction_column].apply(clean_function)
        df = df[df[instruction_column].str.len() > 0].reset_index(drop=True)

    seen = set()
    for _, row in df.iterrows():
        if pd.notna(row[instruction_column]) and pd.notna(row[output_column]):
            if output_wrapper:
                output_text = output_wrapper(row[output_column])
            else:
                output_text = row[output_column]
            conversation = convert_to_instruction_format(
                str(row[instruction_column]),
                str(row[input_column]),
                str(output_text)
            )
            rid = conversation["id"]
            if rid in seen:
                continue
            seen.add(rid)
            yield conversation

def merge_jsonl(files: List[str], out_path: str):
    def _content_fingerprint(ex: Dict) -> str:
        if "messages" in ex:
           return stable_id(json.dumps(ex["messages"], ensure_ascii=False, sort_keys=True))
        elif "instruction" in ex:
            return stable_id(json.dumps(
            {
                "instruction": ex.get("instruction",""),
                "input": ex.get("input",""),
                "output": ex.get("output","")
            }, ensure_ascii=False, sort_keys=True))
        else:
            return stable_id(json.dumps(ex, ensure_ascii=False, sort_keys=True))
            
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    seen_ids, seen_hash = set(), set()
    with open(out_path, "w", encoding="utf-8") as out:
        for f in files:
            with open(f, "r", encoding="utf-8") as fin:
                for line in fin:
                    ex = json.loads(line)
                    ch = _content_fingerprint(ex)
                    if ex["id"] in seen_ids or ch in seen_hash:
                        print(f"❌ Duplicate record found: {ex['id']}")
                        continue
                    seen_ids.add(ex["id"])
                    seen_hash.add(ch)
                    out.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"✅ Wrote {len(seen_ids)} unique records to {out_path}")

def sample_data(data: List[Any], n_samples: int, random_state: int = 42) -> List[Any]:
    """
    Sample n items from data
    
    Args:
        data: Input data
        n_samples: Number of samples to select
        random_state: Random seed
        
    Returns:
        Sampled data
    """
    random.seed(random_state)
    
    if n_samples >= len(data):
        return data.copy()
    
    return random.sample(data, n_samples)

def split_data(
    records: Iterator[Dict],
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    validation_path: Optional[str] = None,
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
    if train_path is None and test_path is None:
        for rec in records:
            sid = str(rec.get(id_key, json.dumps(rec, sort_keys=True)))

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
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_path), exist_ok=True)

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
                elif p < test_ratio + validation_ratio:
                    validation_records.append(rec)
                else:
                    train_records.append(rec)
                    f_train.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"✅ Wrote {len(train_records)} train records to {train_path}, {len(test_records)} test records to {test_path}")
        if validation_path:
            os.makedirs(os.path.dirname(validation_path), exist_ok=True)
            with open(validation_path, "w", encoding="utf-8") as f_validation:
                for rec in validation_records:
                    f_validation.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"✅ Wrote {len(validation_records)} validation records to {validation_path}")
        return train_records, validation_records, test_records

def print_data_stats(data: List[Any], name: str = "Dataset", format: str = "chat"):
    """
    Print basic statistics about the dataset
    
    Args:
        data: Dataset to analyze
        name: Name for display
    """
    print(f"\n📊 {name} Statistics")
    print("-" * 30)
    print(f"Total samples: {len(data)}")
    
    if format == "chat":
        if data and isinstance(data[0], dict):
            # Chat format
            total_messages = sum(len(conv['messages']) for conv in data)
            avg_messages = total_messages / len(data) if data else 0
            print(f"Total messages: {total_messages}")
            print(f"Avg messages per conversation: {avg_messages:.1f}")
            
            # Role distribution
            role_counts = {}
            for conversation in data:
                for message in conversation['messages']:
                    if isinstance(message, dict) and 'role' in message:
                        role = message['role']
                        role_counts[role] = role_counts.get(role, 0) + 1
            
            print("Role distribution:")
            for role, count in role_counts.items():
                print(f"  {role}: {count}")
    elif format == "instruction":
        if data and isinstance(data[0], dict):
            # Instruction format
            total_instructions = len(data)
            print(f"Total instructions: {total_instructions}")