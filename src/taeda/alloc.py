
import torch

_allocations = {}

def alloc_vram(mb: int, device: int = 0):
    """Allocate MB of VRAM and return an allocation ID."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if mb <= 0:
        raise ValueError("MB must be greater than 0.")

    tensor = torch.empty(
        mb * 1024 * 1024,
        dtype=torch.uint8,
        device=f"cuda:{device}"
    )

    tensor.zero_()

    allocation_id = len(_allocations)
    _allocations[allocation_id] = tensor

    return allocation_id


def free_vram(allocation_id=None):
    """Free one allocation or all allocations."""
    if allocation_id is None:
        _allocations.clear()
    else:
        _allocations.pop(allocation_id, None)

    torch.cuda.empty_cache()


def get_allocated_vram():
    """Return total VRAM allocated by this library in MB."""
    return sum(
        tensor.numel() for tensor in _allocations.values()
    ) / (1024 ** 2)
