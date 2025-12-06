from .utils import read_json, read_jsonl, read_csv, stable_id, infer_file_type
from typing import Dict, Optional, Iterator, Callable

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

def convert_to_chat_format(
    input_text: str, 
    output_text: str, 
    system_prompt: Optional[str] = None,
    add_id: bool = False
) -> Dict:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.extend([
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text}
    ])

    if add_id:
        id = stable_id(input_text, output_text, system_prompt)
        return {'id': id, 'messages': messages}
    else:
        return {'messages': messages}

def convert_to_instruction_format(
    instruction: str,
    input_text: Optional[str] = None, 
    output_text: Optional[str] = None,
    add_id: bool = False
) -> Dict:
    if add_id:
        id = stable_id(instruction, input_text, output_text)
        return {
            "id": id,
            "instruction": instruction,   
            "input": input_text,
            "output": output_text
        }
    else:
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        }

def load_any_as_chat(
    file_path: str,
    input_key: str = "input",
    *,
    output_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    question: Optional[str] = None,
    constraint: Optional[str] = None,
    output_wrapper: Optional[Callable[[str], str]] = None,
    stype: str = "natural",
    clean_function: Optional[Callable[[str], str]] = None,
) -> Iterator[Dict]:
    file_type = infer_file_type(file_path)
    if file_type == "csv":
        row_dicts = read_csv(file_path)
    elif file_type == "jsonl":
        row_dicts = read_jsonl(file_path)
    elif file_type == "json":
        row_dicts = read_json(file_path)
    else:
        raise ValueError(f"Invalid file type: {file_type}")

    for row in row_dicts:
        if row[input_key]:
            if clean_function:
                input_text = clean_function(row[input_key])
            else:
                input_text = row[input_key]
            if stype == "labeled":
                parts = [f"Input: {input_text}"]
                if question: 
                    parts.append(f"Question: {question}")
                if constraint:
                    parts.append(f"Constraint: {constraint}")
                input_text = "\n".join(parts)
            elif stype == "natural":
                s = input_text.strip()
                if question: s += f"\n\n{question}"
                if constraint: s += f"\n{constraint}"
                input_text = s
            else:
                raise ValueError(f"Invalid stype: {stype}, must be 'labeled' or 'natural'")

            if output_key:
                if output_wrapper:
                    output_text = output_wrapper(row[output_key])
                else:
                    output_text = row[output_key]
            else:
                output_text = ""

            conversation = convert_to_chat_format(
                str(input_text),
                str(output_text),
                system_prompt
            )
            yield conversation

def load_any_as_instruction(
    file_path: str,
    file_type: str = "csv",
    instruction_key: str = "instruction",
    output_key: str = "output",
    *,
    input_key: Optional[str] = "input",
    output_wrapper: Optional[Callable[[str], str]] = None,
    clean_function: Optional[Callable[[str], str]] = None,
) -> Iterator[Dict]:
    """
    Load CSV file and convert to instruction format
    """
    if file_type == "csv":
        row_dicts = read_csv(file_path)
    elif file_type == "jsonl":
        row_dicts = read_jsonl(file_path)
    elif file_type == "json":
        row_dicts = read_json(file_path)
    else:
        raise ValueError(f"Invalid file type: {file_type}")

    for row in row_dicts:
        if row[instruction_key] and row[output_key]:
            if clean_function:
                instruction_text = clean_function(row[instruction_key])
            else:
                instruction_text = row[instruction_key]
            if input_key:
                input_text = row[input_key]
            else:
                input_text = ""
            if output_wrapper:
                output_text = output_wrapper(row[output_key])
            else:
                output_text = row[output_key]
            conversation = convert_to_instruction_format(
                str(instruction_text),
                str(input_text),
                str(output_text)
            )
            yield conversation

