import torch
import logging


def check_gpu_available() -> bool:
    """
    Check if a GPU is available.

    Returns:
    - available (bool): True if GPU is available, False otherwise.
    """
    return torch.cuda.is_available()


def set_device(use_gpu: bool) -> str:
    """
    Set the device to GPU or CPU.

    Parameters:
    - use_gpu (bool): Whether to use GPU if available.

    Returns:
    - device_string (str): Device string ('cuda' or 'cpu').
    """
    if use_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def memory_usage_report():
    """
    Print CUDA memory usage if GPU is available.
    """
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    else:
        print("CUDA is not available.")