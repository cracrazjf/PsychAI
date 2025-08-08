"""
Audio data utilities

Simple functions for loading and processing audio data for ML training.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


def load_audio(filepath: str, sr: Optional[int] = None, return_path: bool = False):
    """
    Load audio file
    
    Args:
        filepath: Path to audio file
        sr: Target sample rate (None for original)
        return_path: If True, return (audio, sr, path) tuple
        
    Returns:
        (audio_array, sample_rate) or (audio_array, sample_rate, path)
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa required for audio loading. Install with: pip install librosa")
    
    audio, sample_rate = librosa.load(filepath, sr=sr)
    
    if return_path:
        return audio, sample_rate, filepath
    return audio, sample_rate


def load_audio_dataset(
    data_dir: str,
    labels_file: Optional[str] = None,
    audio_extensions: Tuple[str, ...] = ('.wav', '.mp3', '.flac', '.m4a', '.ogg')
) -> List[Dict[str, Any]]:
    """
    Load audio dataset from directory
    
    Args:
        data_dir: Directory containing audio files
        labels_file: Optional CSV/JSON file with labels
        audio_extensions: Valid audio file extensions
        
    Returns:
        List of {"audio_path": path, "label": label} dicts
    """
    dataset = []
    
    # Get all audio files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(Path(data_dir).glob(f"**/*{ext}"))
        audio_files.extend(Path(data_dir).glob(f"**/*{ext.upper()}"))
    
    # If no labels file, try to infer from directory structure
    if labels_file is None:
        for audio_path in audio_files:
            # Assume parent directory is the label
            label = audio_path.parent.name
            dataset.append({
                "audio_path": str(audio_path),
                "label": label
            })
    else:
        # Load labels from file
        if labels_file.endswith('.csv'):
            import pandas as pd
            labels_df = pd.read_csv(labels_file)
            
            # Assume columns are 'filename' and 'label'
            label_map = dict(zip(labels_df['filename'], labels_df['label']))
            
            for audio_path in audio_files:
                filename = audio_path.name
                if filename in label_map:
                    dataset.append({
                        "audio_path": str(audio_path),
                        "label": label_map[filename]
                    })
        else:
            # Assume JSON format
            from .core import load_json
            label_data = load_json(labels_file)
            
            for item in label_data:
                audio_path = os.path.join(data_dir, item['filename'])
                if os.path.exists(audio_path):
                    dataset.append({
                        "audio_path": audio_path,
                        "label": item['label']
                    })
    
    return dataset


def create_audio_chat_sample(
    audio_path: str,
    question: str,
    answer: str,
    system_prompt: str = "You are a helpful assistant that can analyze audio."
) -> List[Dict[str, Any]]:
    """
    Create a chat format sample with audio
    
    Args:
        audio_path: Path to audio file
        question: Question about the audio
        answer: Expected answer
        system_prompt: System prompt
        
    Returns:
        Chat conversation with audio
    """
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user", 
            "content": question,
            "audio_path": audio_path  # Custom field for audio
        },
        {"role": "assistant", "content": answer}
    ]


def extract_audio_features(audio, sample_rate: int, feature_type: str = "mfcc") -> Any:
    """
    Extract basic audio features
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        feature_type: Type of features ("mfcc", "mel", "chroma")
        
    Returns:
        Feature array
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa required for feature extraction")
    
    if feature_type == "mfcc":
        return librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    elif feature_type == "mel":
        return librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    elif feature_type == "chroma":
        return librosa.feature.chroma(y=audio, sr=sample_rate)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


def process_audio_for_model(
    audio, 
    sample_rate: int,
    target_length: Optional[int] = None,
    feature_type: str = "mfcc"
) -> Any:
    """
    Process audio for model input
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        target_length: Target length in samples (None for no padding/truncation)
        feature_type: Feature extraction type
        
    Returns:
        Processed audio features
    """
    # Pad or truncate to target length
    if target_length is not None:
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            import numpy as np
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    
    # Extract features
    features = extract_audio_features(audio, sample_rate, feature_type)
    
    return features


def get_audio_info(audio_path: str) -> Dict[str, Any]:
    """
    Get basic information about an audio file
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with audio info
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa required for audio info")
    
    # Get duration without loading full file
    duration = librosa.get_duration(path=audio_path)
    
    # Load to get other info
    audio, sr = load_audio(audio_path)
    
    return {
        "path": audio_path,
        "duration": duration,
        "sample_rate": sr,
        "samples": len(audio),
        "filename": os.path.basename(audio_path)
    }


def trim_silence(audio, sample_rate: int, top_db: int = 20):
    """
    Trim silence from beginning and end of audio
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        top_db: The threshold (in decibels) below which to consider as silence
        
    Returns:
        Trimmed audio array
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa required for silence trimming")
    
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio