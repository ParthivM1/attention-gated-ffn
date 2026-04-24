import json
import os
import platform
import time

import torch


def main() -> None:
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
        "tf32_matmul": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)) if torch.cuda.is_available() else False,
        "tf32_cudnn": bool(getattr(torch.backends.cudnn, "allow_tf32", False)) if torch.cuda.is_available() else False,
        "env_gpu_request": os.environ.get("GEOVIT_PROPER_GPU", ""),
    }
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        payload.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "device_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": round(props.total_memory / (1024 ** 3), 3),
                "multi_processor_count": props.multi_processor_count,
            }
        )
        # Small warm benchmark to catch obvious throttling.
        x = torch.randn(4096, 4096, device=device, dtype=torch.float16)
        y = torch.randn(4096, 4096, device=device, dtype=torch.float16)
        for _ in range(3):
            z = x @ y
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            z = x @ y
        torch.cuda.synchronize()
        payload["matmul_ms_per_iter_fp16_4096"] = round((time.perf_counter() - start) * 1000 / 10.0, 3)
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
