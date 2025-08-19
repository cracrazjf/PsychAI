import torch

def pick_dtype():
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32

def is_all_int(x):
    if isinstance(x, list):
        return all(is_all_int(e) for e in x)
    try:
        int(x)
        return True
    except ValueError:
        return False

def cuda_memory_stats():
    mem = (
        torch.cuda.max_memory_allocated() / 1e9
        if torch.cuda.is_available() and torch.cuda.is_initialized()
        else None
    )
    return mem