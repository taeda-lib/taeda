import torch

def get_gpu_stats(device_num=0):
    stats = ""

    if not torch.cuda.is_available():
        return "CUDA is NOT available.\nGPU acceleration is disabled."

    device_count = torch.cuda.device_count()

    if device_num < 0 or device_num >= device_count:
        return f"Invalid GPU index: {device_num}. Available GPUs: {device_count}"

    stats += "CUDA is available"
    stats += f"\nDevice name: {torch.cuda.get_device_name(device_num)}"
    stats += f"\nDevice index: {device_num}"
    stats += f"\nCUDA version: {torch.version.cuda}"

    total_memory = torch.cuda.get_device_properties(device_num).total_memory
    free_memory, _ = torch.cuda.mem_get_info(device_num)
    used_memory = total_memory - free_memory

    stats += f"\nTotal VRAM: {total_memory / 1024**3:.2f} GB"
    stats += f"\nUsed VRAM: {used_memory / 1024**3:.2f} GB"
    stats += f"\nFree VRAM: {free_memory / 1024**3:.2f} GB"

    return stats