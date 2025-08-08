"""
Multimodal data utilities

Functions for combining text, image, and audio data in training samples.
"""

from typing import List, Dict, Any, Optional
from .core import convert_to_chat_format
from .image import create_image_chat_sample
from .audio import create_audio_chat_sample


def create_multimodal_chat_sample(
    text: Optional[str] = None,
    image_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    question: str = "",
    answer: str = "",
    system_prompt: str = "You are a helpful multimodal assistant."
) -> List[Dict[str, Any]]:
    """
    Create a chat sample with multiple modalities
    
    Args:
        text: Text content
        image_path: Path to image file
        audio_path: Path to audio file
        question: Question about the content
        answer: Expected answer
        system_prompt: System prompt
        
    Returns:
        Chat conversation with multimodal content
    """
    user_message = {
        "role": "user",
        "content": question or text or ""
    }
    
    # Add multimodal content
    if image_path:
        user_message["image_path"] = image_path
    
    if audio_path:
        user_message["audio_path"] = audio_path
    
    if text and question:
        # If both text and question, combine them
        user_message["content"] = f"{text}\\n\\nQuestion: {question}"
    
    conversation = [
        {"role": "system", "content": system_prompt},
        user_message,
        {"role": "assistant", "content": answer}
    ]
    
    return conversation


def load_multimodal_dataset(
    data_dir: str,
    metadata_file: str,
    text_field: str = "text",
    image_field: str = "image",
    audio_field: str = "audio",
    label_field: str = "label"
) -> List[Dict[str, Any]]:
    """
    Load a multimodal dataset from directory with metadata
    
    Args:
        data_dir: Directory containing data files
        metadata_file: JSON/CSV file with metadata
        text_field: Field name for text content
        image_field: Field name for image filename
        audio_field: Field name for audio filename
        label_field: Field name for labels
        
    Returns:
        List of multimodal samples
    """
    import os
    from .core import load_json
    
    if metadata_file.endswith('.csv'):
        import pandas as pd
        metadata = pd.read_csv(metadata_file).to_dict('records')
    else:
        metadata = load_json(metadata_file)
    
    dataset = []
    
    for item in metadata:
        sample = {"label": item.get(label_field, "")}
        
        # Add text if present
        if text_field in item and item[text_field]:
            sample["text"] = item[text_field]
        
        # Add image path if present
        if image_field in item and item[image_field]:
            image_path = os.path.join(data_dir, item[image_field])
            if os.path.exists(image_path):
                sample["image_path"] = image_path
        
        # Add audio path if present
        if audio_field in item and item[audio_field]:
            audio_path = os.path.join(data_dir, item[audio_field])
            if os.path.exists(audio_path):
                sample["audio_path"] = audio_path
        
        # Only add if at least one modality is present
        if any(key in sample for key in ["text", "image_path", "audio_path"]):
            dataset.append(sample)
    
    return dataset


def convert_multimodal_to_chat(
    dataset: List[Dict[str, Any]],
    task_template: str = "Analyze this content and classify it as: {label}",
    system_prompt: str = "You are a helpful multimodal assistant."
) -> List[List[Dict[str, Any]]]:
    """
    Convert multimodal dataset to chat format
    
    Args:
        dataset: List of multimodal samples
        task_template: Template for creating questions (use {label} placeholder)
        system_prompt: System prompt
        
    Returns:
        List of chat conversations
    """
    conversations = []
    
    for sample in dataset:
        label = sample.get("label", "unknown")
        
        # Create question based on template
        question = "Analyze this content."
        answer = task_template.format(label=label)
        
        # Create user message
        user_message = {
            "role": "user",
            "content": question
        }
        
        # Add text content to question if present
        if "text" in sample:
            user_message["content"] = f"Text: {sample['text']}\\n\\n{question}"
        
        # Add multimodal paths
        if "image_path" in sample:
            user_message["image_path"] = sample["image_path"]
        
        if "audio_path" in sample:
            user_message["audio_path"] = sample["audio_path"]
        
        conversation = [
            {"role": "system", "content": system_prompt},
            user_message,
            {"role": "assistant", "content": answer}
        ]
        
        conversations.append(conversation)
    
    return conversations


def validate_multimodal_sample(sample: Dict[str, Any]) -> bool:
    """
    Validate that a multimodal sample has the required structure
    
    Args:
        sample: Multimodal sample to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Must have at least one modality
    modalities = ["text", "image_path", "audio_path"]
    has_modality = any(key in sample for key in modalities)
    
    if not has_modality:
        return False
    
    # Check file paths exist
    for path_key in ["image_path", "audio_path"]:
        if path_key in sample:
            import os
            if not os.path.exists(sample[path_key]):
                return False
    
    return True


def get_multimodal_stats(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about a multimodal dataset
    
    Args:
        dataset: Multimodal dataset
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        "total_samples": len(dataset),
        "text_samples": 0,
        "image_samples": 0,
        "audio_samples": 0,
        "multimodal_samples": 0
    }
    
    for sample in dataset:
        modalities = []
        
        if "text" in sample:
            stats["text_samples"] += 1
            modalities.append("text")
        
        if "image_path" in sample:
            stats["image_samples"] += 1
            modalities.append("image")
        
        if "audio_path" in sample:
            stats["audio_samples"] += 1
            modalities.append("audio")
        
        if len(modalities) > 1:
            stats["multimodal_samples"] += 1
    
    return stats