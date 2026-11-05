![logo](https://raw.githubusercontent.com/taeda-lib/resc/main/taeda_transparent.png)
# Taeda

**A lightweight Python library for GPU utilities and performance estimation.**

Taeda is a Python package that adds more functions to Pytorch.

## Features

- Detect available NVIDIA CUDA GPUs.
- Retrieve GPU clock speeds in real time.
- Calculate CUDA core counts.
- Estimate theoretical FP32 TFLOPS.
- Support multiple NVIDIA GPUs.
- Simple Python API.

## Installation

```bash
pip install taeda
```

## Requirements

- Python 3.9+
- PyTorch
- NVIDIA GPU
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
```

Example output:

```text
GPU Performance: 0.622 TFLOPS

CUDA is available
Device name: NVIDIA GeForce GTX 1070 Ti
Device index: 0
CUDA version: 12.1
Total VRAM: 8.00 GB
Used VRAM: 1.24 GB
Free VRAM: 6.76 GB
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

## Limitations

- TFLOPS values are theoretical estimates, not measured benchmark results.
- Clock speeds fluctuate depending on GPU load, temperature and power limits.
- CUDA core counts depend on GPU architecture.
- Currently intended for supported NVIDIA CUDA GPUs.
- Actual application performance depends on memory bandwidth, workload and other hardware limitations.
- GPU memory statistics depend on PyTorch CUDA availability.

## License

MIT License.