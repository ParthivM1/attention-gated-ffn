import os
import shlex
import subprocess
import time

import modal


APP_NAME = os.environ.get("TOKENSPAWN_APP_NAME", "tokenspawn-eval")
RESULTS_VOLUME_NAME = os.environ.get("TOKENSPAWN_RESULTS_VOLUME_NAME", "tokenspawn-results")
DATA_VOLUME_NAME = os.environ.get("TOKENSPAWN_DATA_VOLUME_NAME", "tokenspawn-data")
GPU_TYPE = os.environ.get("TOKENSPAWN_GPU", "A100")
COMMIT_INTERVAL_SECS = int(os.environ.get("TOKENSPAWN_COMMIT_INTERVAL_SECS", "600"))
EVAL_ARGS = os.environ.get("TOKENSPAWN_EVAL_ARGS", "").strip()
ENTRYPOINT = os.environ.get("TOKENSPAWN_ENTRYPOINT", "tokenspawn.eval.pilot_cifar").strip()

app = modal.App(APP_NAME)

results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)

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
        "pip install timm pyyaml numpy scipy pillow tqdm",
    )
    .env({"TOKENSPAWN_EVAL_ARGS": EVAL_ARGS, "TOKENSPAWN_ENTRYPOINT": ENTRYPOINT})
    .add_local_dir(".", remote_path="/root/src")
)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=21600,
    volumes={
        "/root/results": results_volume,
        "/root/data": data_volume,
    },
)
def run_remote():
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/src"
    env["TOKENSPAWN_RESULTS_ROOT"] = "/root/results"
    cmd = ["python", "-u", "-m", ENTRYPOINT]
    if EVAL_ARGS:
        cmd.extend(shlex.split(EVAL_ARGS))
    if "--data-root" not in cmd:
        cmd.extend(["--data-root", "/root/data"])

    print("=== Modal TokenSpawn eval boot ===")
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
        if time.time() - last_commit > COMMIT_INTERVAL_SECS or "\"best_acc\"" in line:
            results_volume.commit()
            data_volume.commit()
            print("[modal] committed volumes")
            last_commit = time.time()

    rc = proc.wait()
    print(f"\n=== pilot_cifar exited with code {rc} ===")
    results_volume.commit()
    data_volume.commit()
    if rc != 0:
        raise RuntimeError(f"TokenSpawn eval failed with exit code {rc}")


if __name__ == "__main__":
    run_remote.remote()
