"""
Core data utilities

Essential functions for basic data processing and format handling.
Focus on simplicity and the 80% use case.
"""

import json
import os
import random
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Callable


def load_json(filepath: str) -> Any:
    """Load data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, filepath: str, indent: int = 2):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def train_test_split(
    data: List[Any], 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[List[Any], List[Any]]:
    """
    Simple train/test split
    
    Args:
        data: List of data samples
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        (train_data, test_data)
    """
    random.seed(random_state)
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    n_test = int(len(data_copy) * test_size)
    test_data = data_copy[:n_test]
    train_data = data_copy[n_test:]
    
    return train_data, test_data


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


def convert_to_chat_format(
    input_text: str, 
    output_text: str, 
    system_prompt: Optional[str] = None
) -> List[Dict]:
    """
    Convert input/output pair to chat format
    
    Args:
        input_text: User input
        output_text: Expected output
        system_prompt: Optional system prompt
        
    Returns:
        Chat conversation format
    """
    conversation = []
    
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    
    conversation.extend([
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text}
    ])
    
    return conversation


def convert_to_instruction_format(
    input_text: str, 
    output_text: str,
    instruction: Optional[str] = None
) -> Dict:
    """
    Convert input/output pair to instruction format
    """
    return {
        "instruction": f"{input_text}\n{instruction}" if instruction else input_text,
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
    verbose: bool = True
) -> List[List[Dict]]:
    """
    Load CSV file and convert to chat format
    
    Args:
        filepath: Path to CSV file
        input_column: Column containing input text
        output_column: Column containing output text
        system_prompt: Optional system prompt
        question: Optional question
        constraint: Optional constraint
        stype: "labeled" or "nat"
        
    Returns:
        List of chat conversations
    """

    df = pd.read_csv(filepath)

    if clean_function:
        df[input_column] = df[input_column].apply(clean_function)
        df = df[df[input_column].str.len() > 0].reset_index(drop=True)

    conversations = []
    
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
                if question: s += f"\n{question}"
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
            conversations.append(conversation)
    if verbose:
        print_data_stats(conversations, name=filepath, format="chat")
    
    return conversations


def load_csv_as_instruction(
    filepath: str,
    input_column: str = "input",
    output_column: str = "output",
    instruction: Optional[str] = None,
    output_wrapper: Optional[Callable[[str], str]] = None,
    clean_function: Optional[Callable[[str], str]] = None,
    verbose: bool = True
) -> List[Dict]:
    """
    Load CSV file and convert to instruction format
    """
    df = pd.read_csv(filepath)

    if clean_function:
        df[input_column] = df[input_column].apply(clean_function)
        df = df[df[input_column].str.len() > 0].reset_index(drop=True)

    conversations = []
    
    for _, row in df.iterrows():
        if pd.notna(row[input_column]) and pd.notna(row[output_column]):
            if output_wrapper:
                output_text = output_wrapper(row[output_column])
            else:
                output_text = row[output_column]
            conversation = convert_to_instruction_format(
                str(row[input_column]),
                str(output_text),
                instruction
            )
            conversations.append(conversation)
            
    if verbose:
        print_data_stats(conversations, name=filepath, format="instruction")
    
    return conversations


def combine_datasets(*datasets: List[Any]) -> List[Any]:
    """
    Simply combine multiple datasets
    
    Args:
        *datasets: Variable number of datasets to combine
        
    Returns:
        Combined dataset
    """
    combined = []
    for dataset in datasets:
        combined.extend(dataset)
    return combined


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
        if data and isinstance(data[0], list):
            # Chat format
            total_messages = sum(len(conv) for conv in data)
            avg_messages = total_messages / len(data) if data else 0
            print(f"Total messages: {total_messages}")
            print(f"Avg messages per conversation: {avg_messages:.1f}")
            
            # Role distribution
            role_counts = {}
            for conversation in data:
                for message in conversation:
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