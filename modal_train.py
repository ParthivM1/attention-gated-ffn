import modal
import os
import subprocess

app = modal.App("geo-vit-h100-run")

# Persist checkpoints
volume = modal.Volume.from_name("geo-vit-checkpoints", create_if_missing=True)

# Build a CUDA-capable image (so Torch can actually use the A100)
image = (
    modal.Image.from_registry(
        # CUDA 12.1 runtime image (good match for torch cu121 wheels)
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        # Common basics; add more if your stack needs them
        "git",
        "curl",
        "ca-certificates",
        "libgl1",
        "libglib2.0-0",
    )
    # Install CUDA-enabled PyTorch (cu121) + your deps
    .run_commands(
        "python -m pip install --upgrade pip",
        # PyTorch CUDA wheels
        "pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio",
        # Your other python deps (edit as needed)
        "pip install numpy scipy wandb geoopt tqdm pillow",
    )
    # Mount the entire repo so imports like `models.*` and `layers.*` work reliably
    .add_local_dir(".", remote_path="/root/src")
)

@app.function(
    image=image,
    gpu="A100",
    timeout=86400,  # 24 hours
    volumes={"/root/checkpoints": volume},
)
def train_remote():
    print("=== Modal remote training boot ===")
    print("CWD:", os.getcwd())
    print("Listing /root/src exists?", os.path.exists("/root/src"))
    if os.path.exists("/root/src"):
        print("Top-level /root/src:", os.listdir("/root/src")[:50])

    script_path = "/root/src/src/train.py"  # because your tree is ICML/src/train.py
    if not os.path.exists(script_path):
        # fallback if you ever move train.py to repo root
        alt = "/root/src/train.py"
        print(f"WARNING: {script_path} not found. Trying fallback: {alt}")
        script_path = alt

    print("Using script_path:", script_path)
    if not os.path.exists(script_path):
        raise FileNotFoundError(
            f"Training script not found. Tried: /root/src/src/train.py and /root/src/train.py. "
            f"Repo root listing: {os.listdir('/root/src') if os.path.exists('/root/src') else 'MISSING'}"
        )

    # Ensure your package imports resolve
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/src"
    env["WANDB_SILENT"] = "true"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print("PYTORCH_CUDA_ALLOC_CONF:", env.get("PYTORCH_CUDA_ALLOC_CONF"))

    # Run from repo root so any relative paths resolve consistently
    cmd = ["python", "-u", script_path]

    print("Launching:", " ".join(cmd))
    print("PYTHONPATH:", env["PYTHONPATH"])
    print("Checkpoints dir mounted at /root/checkpoints:", os.path.exists("/root/checkpoints"))
    if os.path.exists("/root/checkpoints"):
        print("Initial /root/checkpoints:", os.listdir("/root/checkpoints")[:50])

    # Stream logs live (so you see the real traceback)
    proc = subprocess.Popen(
        cmd,
        cwd="/root/src",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")

    rc = proc.wait()
    print(f"\n=== train.py exited with code {rc} ===")

    if rc != 0:
        # Extra diagnostics
        print("Diagnostics: /root/src listing:", os.listdir("/root/src")[:50])
        if os.path.exists("/root/src/src"):
            print("Diagnostics: /root/src/src listing:", os.listdir("/root/src/src")[:50])
        raise Exception(f"Training script failed with exit code {rc}")

    # Persist anything written to /root/checkpoints
    volume.commit()

if __name__ == "__main__":
    # This kicks off the remote function
    train_remote.remote()
