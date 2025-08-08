"""
Image data utilities

Simple functions for loading and processing image data for ML training.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


def load_image(filepath: str, return_path: bool = False):
    """
    Load image file
    
    Args:
        filepath: Path to image file
        return_path: If True, return (image, path) tuple
        
    Returns:
        PIL Image or (image, path) tuple
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL/Pillow required for image loading. Install with: pip install Pillow")
    
    image = Image.open(filepath)
    
    if return_path:
        return image, filepath
    return image


def load_image_dataset(
    data_dir: str,
    labels_file: Optional[str] = None,
    image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
) -> List[Dict[str, Any]]:
    """
    Load image dataset from directory
    
    Args:
        data_dir: Directory containing images
        labels_file: Optional CSV/JSON file with labels
        image_extensions: Valid image file extensions
        
    Returns:
        List of {"image_path": path, "label": label} dicts
    """
    dataset = []
    
    # Get all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(data_dir).glob(f"**/*{ext}"))
        image_files.extend(Path(data_dir).glob(f"**/*{ext.upper()}"))
    
    # If no labels file, try to infer from directory structure
    if labels_file is None:
        for image_path in image_files:
            # Assume parent directory is the label
            label = image_path.parent.name
            dataset.append({
                "image_path": str(image_path),
                "label": label
            })
    else:
        # Load labels from file
        if labels_file.endswith('.csv'):
            import pandas as pd
            labels_df = pd.read_csv(labels_file)
            
            # Assume columns are 'filename' and 'label'
            label_map = dict(zip(labels_df['filename'], labels_df['label']))
            
            for image_path in image_files:
                filename = image_path.name
                if filename in label_map:
                    dataset.append({
                        "image_path": str(image_path),
                        "label": label_map[filename]
                    })
        else:
            # Assume JSON format
            from .core import load_json
            label_data = load_json(labels_file)
            
            for item in label_data:
                image_path = os.path.join(data_dir, item['filename'])
                if os.path.exists(image_path):
                    dataset.append({
                        "image_path": image_path,
                        "label": item['label']
                    })
    
    return dataset


def create_image_chat_sample(
    image_path: str,
    question: str,
    answer: str,
    system_prompt: str = "You are a helpful assistant that can analyze images."
) -> List[Dict[str, Any]]:
    """
    Create a chat format sample with image
    
    Args:
        image_path: Path to image file
        question: Question about the image
        answer: Expected answer
        system_prompt: System prompt
        
    Returns:
        Chat conversation with image
    """
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user", 
            "content": question,
            "image_path": image_path  # Custom field for image
        },
        {"role": "assistant", "content": answer}
    ]


def resize_image(image, target_size: Tuple[int, int] = (224, 224)):
    """
    Resize image to target size
    
    Args:
        image: PIL Image
        target_size: (width, height) tuple
        
    Returns:
        Resized PIL Image
    """
    return image.resize(target_size)


def process_image_for_model(image, target_size: Tuple[int, int] = (224, 224)):
    """
    Process image for model input (basic preprocessing)
    
    Args:
        image: PIL Image
        target_size: Target size for resize
        
    Returns:
        Processed image tensor/array
    """
    try:
        import torch
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image)
        
    except ImportError:
        # Fallback to numpy if torch not available
        import numpy as np
        
        # Basic resize and normalize
        image = resize_image(image, target_size)
        img_array = np.array(image) / 255.0
        
        # Convert to CHW format if RGB
        if len(img_array.shape) == 3:
            img_array = np.transpose(img_array, (2, 0, 1))
        
        return img_array


def get_image_info(image_path: str) -> Dict[str, Any]:
    """
    Get basic information about an image
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with image info
    """
    image = load_image(image_path)
    
    return {
        "path": image_path,
        "size": image.size,  # (width, height)
        "mode": image.mode,  # RGB, RGBA, L, etc.
        "format": image.format,
        "filename": os.path.basename(image_path)
    }