"""
Memory management utilities

Utilities for monitoring and managing GPU/CPU memory during training.
"""

import torch
import gc
import os
import psutil
from typing import Dict, Optional


def print_memory_usage(step_name: str = "Current"):
    """
    Print detailed memory usage information
    
    Args:
        step_name: Description of current step for logging
    """
    print(f"\n🔍 Memory Usage - {step_name}:")
    
    # GPU memory if available
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        print(f"  📊 GPU Allocated: {allocated:.2f} GB")
        print(f"  📦 GPU Reserved:  {reserved:.2f} GB")
        print(f"  📈 GPU Peak:      {max_allocated:.2f} GB")
        print(f"  🆓 GPU Free:      {(reserved - allocated):.2f} GB")
        
        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  💾 GPU Total:     {total_memory:.2f} GB")
        print(f"  📊 GPU Usage:     {(allocated / total_memory * 100):.1f}%")
    else:
        print("  🔍 CUDA not available")
    
    # CPU memory
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    cpu_memory_gb = memory_info.rss / 1024**3
    
    # System memory
    system_memory = psutil.virtual_memory()
    total_ram_gb = system_memory.total / 1024**3
    available_ram_gb = system_memory.available / 1024**3
    
    print(f"  🖥️  CPU Memory:    {cpu_memory_gb:.2f} GB")
    print(f"  💾 Total RAM:     {total_ram_gb:.2f} GB")
    print(f"  🆓 Available RAM: {available_ram_gb:.2f} GB")
    print(f"  📊 RAM Usage:     {(cpu_memory_gb / total_ram_gb * 100):.1f}%")


def get_memory_stats() -> Dict[str, float]:
    """
    Get current memory statistics as a dictionary
    
    Returns:
        Dictionary containing memory statistics
    """
    stats = {}
    
    # GPU memory
    if torch.cuda.is_available():
        stats['gpu_allocated_gb'] = torch.cuda.memory_allocated() / 1024**3
        stats['gpu_reserved_gb'] = torch.cuda.memory_reserved() / 1024**3
        stats['gpu_max_allocated_gb'] = torch.cuda.max_memory_allocated() / 1024**3
        stats['gpu_total_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        stats['gpu_usage_percent'] = (stats['gpu_allocated_gb'] / stats['gpu_total_gb']) * 100
    
    # CPU memory
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    system_memory = psutil.virtual_memory()
    
    stats['cpu_memory_gb'] = memory_info.rss / 1024**3
    stats['total_ram_gb'] = system_memory.total / 1024**3
    stats['available_ram_gb'] = system_memory.available / 1024**3
    stats['ram_usage_percent'] = (stats['cpu_memory_gb'] / stats['total_ram_gb']) * 100
    
    return stats
