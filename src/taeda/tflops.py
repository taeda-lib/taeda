
import torch
import pynvml


def get_tflops(device_num=0):
    """
    Estimate theoretical FP32 GPU performance in TFLOPS.
    """

    if not torch.cuda.is_available():
        return None

    if not 0 <= device_num < torch.cuda.device_count():
        raise ValueError("Invalid GPU index")

    pynvml.nvmlInit()

    try:
        gpu = torch.cuda.get_device_properties(device_num)
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_num)

        # CUDA cores per SM depends on GPU architecture.
        cores_per_sm = {
            (5, 0): 128,
            (5, 2): 128,
            (5, 3): 128,
            (6, 0): 64,
            (6, 1): 128,
            (6, 2): 128,
            (7, 0): 64,
            (7, 5): 64,
            (8, 0): 64,
            (8, 6): 128,
            (8, 9): 128,
            (9, 0): 128,
        }

        capability = (gpu.major, gpu.minor)
        cores = cores_per_sm.get(capability)

        if cores is None:
            raise NotImplementedError(
                f"Unknown CUDA architecture: {capability}"
            )

        cuda_cores = gpu.multi_processor_count * cores

        clock_mhz = pynvml.nvmlDeviceGetClockInfo(
            handle, pynvml.NVML_CLOCK_SM
        )

        clock_ghz = clock_mhz / 1000

        return cuda_cores * clock_ghz * 2 / 1000

    finally:
        pynvml.nvmlShutdown()
