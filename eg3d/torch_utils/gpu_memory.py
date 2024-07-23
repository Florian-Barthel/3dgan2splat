import torch


def print_gpu_memory_stats(device_id=0):
    # Ensure that the specified device ID is valid and that PyTorch can use CUDA
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your installation and try again.")
        return
    if device_id >= torch.cuda.device_count():
        print(f"Device ID {device_id} is not valid. Please select a valid device ID.")
        return

    # Set the current device
    torch.cuda.set_device(device_id)

    # Get memory statistics
    total_memory = torch.cuda.get_device_properties(device_id).total_memory
    allocated_memory = torch.cuda.memory_allocated(device_id)
    cached_memory = torch.cuda.memory_reserved(device_id)

    # Convert bytes to gigabytes for readability
    total_memory_gb = total_memory / (1024 ** 3)
    allocated_memory_gb = allocated_memory / (1024 ** 3)
    cached_memory_gb = cached_memory / (1024 ** 3)
    free_memory_gb = total_memory_gb - allocated_memory_gb

    # Print memory statistics
    print(f"GPU {device_id} Memory Statistics:")
    print(f"Total Memory: {total_memory_gb:.2f} GB")
    print(f"Allocated Memory: {allocated_memory_gb:.2f} GB")
    print(f"Cached Memory: {cached_memory_gb:.2f} GB")
    print(f"Free Memory (Approx.): {free_memory_gb:.2f} GB")