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
- **RAMVRAMModel:** Experimental model wrapper for layer-by-layer CPU/GPU transfers.
- Automatic VRAM usage monitoring.
- CPU RAM offloading when GPU memory is limited.
- Configurable VRAM usage limits.
- Experimental autograd-compatible activation offloading.
- Support for PyTorch backward passes and optimizer updates.
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

import torch
import taeda


# ============================================================
# GPU PERFORMANCE
# ============================================================

# Get theoretical GPU performance
tflops = taeda.get_tflops(0)
print(f"GPU Performance: {tflops:.3f} TFLOPS")


# ============================================================
# GPU STATISTICS
# ============================================================

# Get GPU information and statistics
stats = taeda.get_gpu_stats(0)
print(stats)


# ============================================================
# RAMVRAM
# ============================================================

# Initialize RAMVRAM
memory = taeda.RAMVRAM(max_vram_percent=80)

# Store tensors
a = memory.store(torch.randn(5000, 5000))
b = memory.store(torch.randn(5000, 5000))

# Retrieve a tensor
tensor = memory.get(a)

# Display RAMVRAM memory information
memory.report()

# ============================================================
# VRAM Allocation
# ============================================================

# Allocate 256 MB of VRAM
taeda.alloc_vram(256)

# Get allocated VRAM
allocated = taeda.get_allocated_vram()
print(f"Allocated VRAM: {allocated:.2f} MB")

# Free allocated VRAM
taeda.free_vram()

```

## RAMVRAMModel Usage

`RAMVRAMModel` is an experimental PyTorch model wrapper designed to keep master model parameters in system RAM and transfer layers to GPU for computation.

```python
import torch
import torch.nn as nn
import taeda

# Create a sequential model
model = nn.Sequential(
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)

# Wrap the model
model = taeda.RAMVRAMModel(
    model,
    max_vram_percent=80
)

# Optimizer operates on CPU master parameters
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    x = torch.randn(32, 512)
    y = torch.randint(0, 10, (32,))

    optimizer.zero_grad(set_to_none=True)

    output = model(x)
    loss = criterion(output, y.to(output.device))

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}: {loss.item():.5f}")

# Display memory information
model.report()
```

## Limitations

- TFLOPS values are theoretical estimates, not measured benchmark results.
- Clock speeds fluctuate depending on GPU load, temperature and power limits.
- CUDA core counts depend on GPU architecture.
- Currently intended for supported NVIDIA CUDA GPUs.
- Actual application performance depends on memory bandwidth, workload and other hardware limitations.
- GPU memory statistics depend on PyTorch CUDA availability.
- RAM is significantly slower than GPU VRAM.
- RAMVRAM requires explicit tensor retrieval through `memory.get()`.
- RAMVRAMModel currently supports `nn.Sequential` architectures.
- RAMVRAMModel is experimental and does not guarantee a strict VRAM usage limit.
- Individual layers must fit into available GPU memory.
- Optimizer states and gradients can consume substantial system RAM.
- Performance may decrease significantly due to CPU/GPU transfers.

## License

MIT License.