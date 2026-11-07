import torch
import torch.nn as nn
import gc
import threading

from torch.func import functional_call
from torch.autograd.graph import saved_tensors_hooks


# ============================================================
# RAMVRAM - Tensor memory manager
# ============================================================

class RAMVRAM:
    def __init__(
        self,
        device="cuda",
        max_vram_percent=85,
        pin_memory=True
    ):
        self.device = torch.device(device)
        self.max_vram_percent = max_vram_percent
        self.pin_memory = pin_memory

        self.tensors = {}
        self.lock = threading.RLock()
        self.next_id = 0

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        if not 0 < max_vram_percent <= 100:
            raise ValueError("max_vram_percent must be between 0 and 100")

    def stats(self):
        if self.device.type != "cuda":
            return {
                "allocated": 0,
                "reserved": 0,
                "total": 0,
                "free": 0
            }

        free, total = torch.cuda.mem_get_info(self.device)

        return {
            "allocated": torch.cuda.memory_allocated(self.device),
            "reserved": torch.cuda.memory_reserved(self.device),
            "total": total,
            "free": free
        }

    def usage(self):
        stats = self.stats()

        if not stats["total"]:
            return 0

        return (
            1 - stats["free"] / stats["total"]
        ) * 100

    def _needs_offload(self, required=0):
        stats = self.stats()

        if not stats["total"]:
            return False

        limit = stats["total"] * (
            self.max_vram_percent / 100
        )

        return stats["total"] - stats["free"] + required > limit

    def offload(self, tensor_id):
        with self.lock:
            entry = self.tensors[tensor_id]
            tensor = entry["tensor"]

            if tensor.device.type != "cuda":
                return

            entry["tensor"] = tensor.detach().to(
                "cpu",
                non_blocking=False
            )

            if self.pin_memory:
                entry["tensor"] = entry["tensor"].pin_memory()

            entry["on_gpu"] = False

            gc.collect()
            torch.cuda.empty_cache()

    def restore(self, tensor_id):
        with self.lock:
            entry = self.tensors[tensor_id]

            if entry["on_gpu"]:
                return entry["tensor"]

            tensor = entry["tensor"]

            if self._needs_offload(
                tensor.numel() * tensor.element_size()
            ):
                self.offload_oldest(exclude=tensor_id)

            entry["tensor"] = tensor.to(
                self.device,
                non_blocking=self.pin_memory
            )

            entry["on_gpu"] = True

            return entry["tensor"]

    def offload_oldest(self, exclude=None):
        for tensor_id, entry in list(self.tensors.items()):
            if tensor_id == exclude:
                continue

            if entry["on_gpu"]:
                self.offload(tensor_id)

                if not self._needs_offload():
                    return

    def store(self, tensor, keep_on_gpu=True):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Expected a PyTorch tensor")

        with self.lock:
            tensor_id = self.next_id
            self.next_id += 1

            if keep_on_gpu and self.device.type == "cuda":
                required = tensor.numel() * tensor.element_size()

                if self._needs_offload(required):
                    self.offload_oldest()

                try:
                    tensor = tensor.to(self.device)
                    on_gpu = True
                except torch.cuda.OutOfMemoryError:
                    tensor = tensor.to("cpu")
                    on_gpu = False
            else:
                tensor = tensor.to("cpu")
                on_gpu = False

            if not on_gpu and self.pin_memory:
                tensor = tensor.pin_memory()

            self.tensors[tensor_id] = {
                "tensor": tensor,
                "on_gpu": on_gpu
            }

            return tensor_id

    def get(self, tensor_id):
        return self.restore(tensor_id)

    def remove(self, tensor_id):
        with self.lock:
            self.tensors.pop(tensor_id, None)

    def clear(self):
        with self.lock:
            self.tensors.clear()
            gc.collect()

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def report(self):
        stats = self.stats()

        print("=== RAMVRAM ===")
        print(f"GPU usage: {self.usage():.2f}%")

        if stats["total"]:
            print(f"VRAM total: {stats['total'] / 1024**3:.2f} GB")
            print(f"VRAM free: {stats['free'] / 1024**3:.2f} GB")

        gpu = sum(
            1 for entry in self.tensors.values()
            if entry["on_gpu"]
        )

        ram = len(self.tensors) - gpu

        print(f"GPU tensors: {gpu}")
        print(f"RAM tensors: {ram}")

    def __len__(self):
        return len(self.tensors)


# ============================================================
# RAMVRAMModel - Autograd-aware sequential model offloading
# ============================================================

class RAMVRAMModel(nn.Module):
    def __init__(
        self,
        model,
        device="cuda",
        max_vram_percent=80,
        pin_memory=True
    ):
        super().__init__()

        if not isinstance(model, nn.Sequential):
            raise TypeError(
                "RAMVRAMModel currently supports nn.Sequential models."
            )

        self.device = torch.device(device)
        self.max_vram_percent = max_vram_percent
        self.pin_memory = pin_memory

        if not 0 < max_vram_percent <= 100:
            raise ValueError("max_vram_percent must be between 0 and 100")

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        # Master parameters remain in RAM.
        self.model = model.cpu()

        self.layers = list(self.model.children())

    def forward(self, x):
        for layer in self.layers:
            # FIX: activations are parked on the CPU between layers,
            # so move them back to the GPU at the start of EVERY layer.
            x = x.to(self.device)

            params = {
                name: param.to(self.device)
                for name, param in layer.named_parameters()
            }

            buffers = {
                name: buffer.to(self.device)
                for name, buffer in layer.named_buffers()
            }

            with saved_tensors_hooks(self._pack, self._unpack):
                output = functional_call(
                    layer,
                    (params, buffers),
                    (x,)
                )

            # Copy updated buffers (e.g. BatchNorm stats) back to the
            # original layer. Build the lookup once per layer.
            orig_buffers = dict(layer.named_buffers())
            for name, buffer in buffers.items():
                original = orig_buffers[name]
                original.copy_(buffer.detach().to(original.device))

            # Keep intermediate activations in system RAM
            x = output.to("cpu")

            del params, buffers, output

        return x.to(self.device)

    def _pack(self, tensor):
        if tensor.device.type != "cuda":
            return tensor

        cpu_tensor = tensor.detach().to("cpu")

        if self.pin_memory:
            cpu_tensor = cpu_tensor.pin_memory()

        return cpu_tensor, tensor.device

    def _unpack(self, packed):
        if isinstance(packed, torch.Tensor):
            return packed

        cpu_tensor, original_device = packed

        return cpu_tensor.to(
            original_device,
            non_blocking=self.pin_memory
        )

    def train(self, mode=True):
        # Sets self.training and recurses into self.model.
        super().train(mode)
        return self

    def eval(self):
        return self.train(False)

    def get_model(self):
        return self.model

    def report(self):
        print("=== RAMVRAMModel ===")
        print("Layers:", len(self.layers))
        print("Device:", self.device)
        print("VRAM budget:", self.max_vram_percent, "%")

        params = sum(
            p.numel() for p in self.model.parameters()
        )

        size = sum(
            p.numel() * p.element_size()
            for p in self.model.parameters()
        )

        print(f"Parameters: {params:,}")
        print(f"Master weights in RAM: {size / 1024**2:.2f} MB")

        if self.device.type == "cuda":
            free, total = torch.cuda.mem_get_info(self.device)

            print(f"VRAM total: {total / 1024**3:.2f} GB")
            print(f"VRAM free: {free / 1024**3:.2f} GB")