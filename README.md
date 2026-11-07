![Taeda Logo](https://raw.githubusercontent.com/taeda-lib/resc/main/taeda_transparent.png)

# Taeda

**A lightweight Python library for GPU utilities, performance estimation, and memory management.**

Taeda is a Python package that adds useful functions to PyTorch, including GPU monitoring, theoretical performance estimation, and CPU RAM-to-GPU VRAM tensor management.

## Features

- Detect available NVIDIA CUDA GPUs.
- Retrieve GPU clock speeds in real time.
- Calculate CUDA core counts.
- Estimate theoretical FP32 TFLOPS.
- Support multiple NVIDIA GPUs.
- Retrieve GPU information and memory statistics.
- **RAMVRAM:** Manage tensor storage between system RAM and GPU VRAM.
- Automatic VRAM usage monitoring.
- CPU RAM offloading when GPU memory is limited.
- Configurable VRAM usage limits.
- Simple Python API.

## Installation

```bash
pip install taeda
```

## Requirements

- Python 3.9+
- PyTorch
- NVIDIA GPU (for CUDA features)
- NVIDIA drivers
- `nvidia-ml-py`

## Usage

```python
import taeda

# Get theoretical GPU performance
tflops = taeda.get_tflops(0)

print(f"GPU Performance: {tflops:.3f} TFLOPS")

# Get GPU information and statistics
stats = taeda.get_gpu_stats(0)

print(stats)

# Initialize RAMVRAM
memory = taeda.RAMVRAM(max_vram_percent=80)

# Store tensors
a = memory.store(torch.randn(5000, 5000))
b = memory.store(torch.randn(5000, 5000))

# Retrieve a tensor
tensor = memory.get(a)

# Display memory information
memory.report()
```

## API

### `get_tflops(device_num=0)`

Estimates the theoretical FP32 performance of an NVIDIA GPU.

| Parameter | Description |
|---|---|
| `device_num` | CUDA GPU index (default: `0`) |

**Returns:** Estimated FP32 performance in TFLOPS (`float`), or `None` if CUDA is unavailable.

**Raises:** `ValueError` if the GPU index is invalid.

The clock speed is retrieved dynamically through NVML.

### `get_gpu_stats(device_num=0)`

Retrieves GPU information and memory statistics using PyTorch CUDA.

| Parameter | Description |
|---|---|
| `device_num` | CUDA GPU index (default: `0`) |

**Returns:** A formatted string containing:

- CUDA availability
- GPU device name
- GPU device index
- CUDA version
- Total VRAM
- Used VRAM
- Free VRAM

**Raises:** Returns an error message if CUDA is unavailable or the GPU index is invalid.

### `RAMVRAM(device="cuda", max_vram_percent=85, pin_memory=True)`

Creates a tensor memory manager that manages storage between system RAM and GPU VRAM.

| Parameter | Description |
|---|---|
| `device` | Target device (default: `"cuda"`) |
| `max_vram_percent` | Maximum target VRAM usage percentage (default: `85`) |
| `pin_memory` | Enable pinned CPU memory (default: `True`) |

**Methods:**

| Method | Description |
|---|---|
| `store(tensor)` | Stores a tensor and returns its ID |
| `get(tensor_id)` | Retrieves a tensor, restoring it to GPU if needed |
| `offload(tensor_id)` | Moves a tensor to system RAM |
| `restore(tensor_id)` | Restores a tensor to GPU |
| `offload_oldest()` | Offloads stored GPU tensors |
| `stats()` | Retrieves VRAM statistics |
| `usage()` | Returns GPU memory usage percentage |
| `report()` | Displays memory usage information |
| `remove(tensor_id)` | Removes a stored tensor |
| `clear()` | Clears stored tensors |

## Limitations

- TFLOPS values are theoretical estimates, not measured benchmark results.
- Clock speeds fluctuate depending on GPU load, temperature and power limits.
- CUDA core counts depend on GPU architecture.
- Currently intended for supported NVIDIA CUDA GPUs.
- Actual application performance depends on memory bandwidth, workload and other hardware limitations.
- GPU memory statistics depend on PyTorch CUDA availability.
- RAM is significantly slower than GPU VRAM.
- RAMVRAM requires explicit tensor retrieval through `memory.get()`.
- RAMVRAM does not currently provide transparent CUDA memory extension or seamless autograd-compatible model training.

## License

MIT License.