![Taeda Logo](https://raw.githubusercontent.com/taeda-lib/resc/main/taeda_transparent.png)

# Taeda

**A lightweight Python library for GPU utilities, performance estimation, and memory management.**

Taeda is a Python package that adds useful functions to PyTorch, including GPU monitoring, theoretical performance estimation, and CPU RAM-to-GPU VRAM tensor management.


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

# ============================================================
# RAM Eater
# ============================================================

taeda.rameat(10000, 100, 1, False)

# 10000 - lenght of list
# 100 - max value of random number
# 1 - min value of random number
# False - use verbose

# ============================================================
# Fake Nvidia SMI
# ============================================================

taeda.fake_nsmi()                                                  # defaults
taeda.fake_nsmi("572.83", "NVIDIA GeForce RTX 3060", 12288)        # driver, GPU name, total memory (MiB)
taeda.fake_nsmi("572.83", "NVIDIA GeForce RTX 3060", 12288, 3500)  # + fixed used memory

# Use :
#    python -c "import fake_nvidiasmi; print(fake_nvidiasmi.fake_nsmi('572.83', 'RTX PRO 6000 Blackwell', 97887, 3549))"
# For a better effect

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

## License

MIT License.