
import torch
import gc
import threading


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

            if self._needs_offload(tensor.numel() * tensor.element_size()):
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
