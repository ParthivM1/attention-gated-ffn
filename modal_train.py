import modal
import os
import subprocess
import time

APP_NAME = os.environ.get("GEOVIT_APP_NAME", "geo-vit-h100-run")
VOLUME_NAME = os.environ.get("GEOVIT_VOLUME_NAME", "geo-vit-checkpoints")
GPU_TYPE = os.environ.get("GEOVIT_GPU", "A100")
COMMIT_INTERVAL_SECS = int(os.environ.get("GEOVIT_COMMIT_INTERVAL_SECS", "1800"))
TRAIN_ARGS = os.environ.get("GEOVIT_TRAIN_ARGS", "").strip()

app = modal.App(APP_NAME)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "git",
        "curl",
        "ca-certificates",
        "libgl1",
        "libglib2.0-0",
    )
    .run_commands(
        "python -m pip install --upgrade pip",
        "pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio",
        "pip install numpy scipy wandb geoopt tqdm pillow six",
    )
    .env({"GEOVIT_TRAIN_ARGS": TRAIN_ARGS})
    .add_local_dir(".", remote_path="/root/src")
)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=86400,
    volumes={"/root/checkpoints": volume},
)
def train_remote():
    print("=== Modal remote training boot ===")
    print("CWD:", os.getcwd())

    script_path = "/root/src/src/train.py"
    if not os.path.exists(script_path):
        alt = "/root/src/train.py"
        print(f"WARNING: {script_path} not found. Trying fallback: {alt}")
        script_path = alt

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Training script not found at {script_path}")

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/src"
    env["WANDB_SILENT"] = "true"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    train_args = os.environ.get("GEOVIT_TRAIN_ARGS", "").strip()
    cmd = ["python", "-u", script_path]
    if train_args:
        cmd.extend(train_args.split())

    print("Launching:", " ".join(cmd))

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
    last_commit = time.time()
    for line in proc.stdout:
        print(line, end="")
        should_commit = "Epoch " in line and "Done |" in line
        if should_commit or (time.time() - last_commit > COMMIT_INTERVAL_SECS):
            volume.commit()
            print("[modal] Committed checkpoint volume snapshot.")
            last_commit = time.time()

    rc = proc.wait()
    print(f"\n=== train.py exited with code {rc} ===")

    if rc != 0:
        raise Exception(f"Training script failed with exit code {rc}")

    volume.commit()


if __name__ == "__main__":
    train_remote.remote()
