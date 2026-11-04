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

tflops = taeda.get_tflops(0)

print(f"GPU Performance: {tflops:.3f} TFLOPS")
```

Example output:

```text
GPU Performance: 0.622 TFLOPS
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

## Limitations

- TFLOPS values are theoretical estimates, not measured benchmark results.
- Clock speeds fluctuate depending on GPU load, temperature and power limits.
- CUDA core counts depend on GPU architecture.
- Currently intended for supported NVIDIA CUDA GPUs.
- Actual application performance depends on memory bandwidth, workload and other hardware limitations.

## License

MIT License.
