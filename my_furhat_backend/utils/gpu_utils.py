"""
GPU Utilities Module

This module provides utilities for managing GPU resources, monitoring memory usage,
and handling model device placement in PyTorch applications.

Functions:
    setup_gpu: Configure and return GPU device information.
    move_model_to_device: Move a PyTorch model to the specified device.
    print_gpu_status: Display current GPU/CPU memory usage.
    clear_gpu_cache: Clear GPU memory cache.
"""

import torch
import psutil
from typing import Optional, Dict, Any

def setup_gpu() -> Dict[str, Any]:
    """
    Set up GPU configuration and return device information.
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - cuda_available (bool): Whether CUDA is available
            - device (torch.device): The device to use (CUDA or CPU)
            - device_name (str): Name of the device
            - memory_info (Dict): Memory usage information in MB
    """
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "device": None,
        "device_name": None,
        "memory_info": None
    }
    
    if device_info["cuda_available"]:
        device_info["device"] = torch.device("cuda")
        device_info["device_name"] = torch.cuda.get_device_name()
        device_info["memory_info"] = {
            "allocated": torch.cuda.memory_allocated() / 1024**2,  # MB
            "cached": torch.cuda.memory_reserved() / 1024**2,      # MB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**2  # MB
        }
    else:
        device_info["device"] = torch.device("cpu")
        device_info["device_name"] = "CPU"
        device_info["memory_info"] = {
            "total": psutil.virtual_memory().total / 1024**2,  # MB
            "available": psutil.virtual_memory().available / 1024**2,  # MB
            "used": psutil.virtual_memory().used / 1024**2  # MB
        }
    
    return device_info

def move_model_to_device(model: Any, device: Optional[torch.device] = None) -> Any:
    """
    Move a model to the specified device (GPU/CPU).
    
    Args:
        model (Any): The PyTorch model to move.
        device (Optional[torch.device]): Device to move the model to. If None,
            will use GPU if available, otherwise CPU.
        
    Returns:
        Any: The model moved to the specified device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if hasattr(model, "to"):
        return model.to(device)
    return model

def print_gpu_status() -> None:
    """
    Print current GPU/CPU status and memory usage.
    
    If CUDA is available, prints GPU information including:
    - CUDA device number
    - Device name
    - Memory allocated
    - Memory cached
    - Maximum memory allocated
    
    Otherwise, prints CPU memory information including:
    - Total memory
    - Available memory
    - Used memory
    """
    if torch.cuda.is_available():
        print("\n=== GPU Status ===")
        print(f"CUDA Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name()}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    else:
        print("\n=== CPU Status ===")
        memory = psutil.virtual_memory()
        print(f"Total Memory: {memory.total / 1024**2:.2f} MB")
        print(f"Available Memory: {memory.available / 1024**2:.2f} MB")
        print(f"Used Memory: {memory.used / 1024**2:.2f} MB")

def clear_gpu_cache() -> None:
    """
    Clear GPU memory cache to free up resources and prevent memory leaks.
    
    This function:
    1. Checks if CUDA is available on the system
    2. Calls torch.cuda.empty_cache() to release unused GPU memory
    3. Prints a confirmation message when the cache is cleared
    
    This is particularly useful when:
    - Switching between large models
    - After processing large batches of data
    - When experiencing out-of-memory errors
    - Before starting memory-intensive operations
    
    Note:
        This only clears the cache of unused memory. It does not free
        memory that is still in use by active tensors or models.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared") 