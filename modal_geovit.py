import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import modal

WORKSPACE_ROOT = Path(__file__).resolve().parent
WORKSPACE_SRC = WORKSPACE_ROOT / "src"
if WORKSPACE_SRC.exists():
    sys.path.insert(0, str(WORKSPACE_SRC))

from src.geovit_presets import get_preset_args, list_presets


APP_NAME = os.environ.get("GEOVIT_PROPER_APP_NAME", "geovit-proper-train")
RESULTS_VOLUME_NAME = os.environ.get("GEOVIT_PROPER_RESULTS_VOLUME_NAME", "geovit-proper-results")
DATA_VOLUME_NAME = os.environ.get("GEOVIT_PROPER_DATA_VOLUME_NAME", "geovit-proper-data")
GPU_TYPE = os.environ.get("GEOVIT_PROPER_GPU", "A100")
COMMIT_INTERVAL_SECS = int(os.environ.get("GEOVIT_PROPER_COMMIT_INTERVAL_SECS", "600"))
TRAIN_ARGS = os.environ.get("GEOVIT_PROPER_TRAIN_ARGS", "").strip()
ENTRYPOINT = os.environ.get("GEOVIT_PROPER_ENTRYPOINT", "src.train_geovit").strip()
TRAIN_PRESET = os.environ.get("GEOVIT_PROPER_PRESET", "").strip()
RUN_ENV_PROBE = os.environ.get("GEOVIT_PROPER_RUN_ENV_PROBE", "1").strip().lower() not in {"0", "false", "no"}

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
        "pip install numpy scipy pillow tqdm transformers sentencepiece",
    )
    .env(
        {
            "GEOVIT_PROPER_TRAIN_ARGS": TRAIN_ARGS,
            "GEOVIT_PROPER_ENTRYPOINT": ENTRYPOINT,
            "GEOVIT_PROPER_PRESET": TRAIN_PRESET,
            "GEOVIT_PROPER_RUN_ENV_PROBE": "1" if RUN_ENV_PROBE else "0",
        }
    )
    .add_local_dir(
        ".",
        remote_path="/root/src",
        ignore=lambda p: (
            any(part in {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "tmp", "runs"} for part in Path(p).parts)
            and "checkpoints" not in Path(p).parts
            or Path(p).suffix in {".pyc", ".pyo", ".log"}
            or Path(p).name in {"FETCH_HEAD"}
        ),
    )
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
    cmd = ["python", "-u", "-m", ENTRYPOINT]
    if TRAIN_PRESET:
        cmd.extend(get_preset_args(TRAIN_PRESET))
    if TRAIN_ARGS:
        cmd.extend(shlex.split(TRAIN_ARGS))
    if "--data-root" not in cmd:
        cmd.extend(["--data-root", "/root/data"])
    if "--save-dir" not in cmd:
        cmd.extend(["--save-dir", "/root/results/geovit_proper"])

    print("=== Modal GeoViT-Proper boot ===")
    print("Launching:", " ".join(cmd))
    print(
        {
            "preset": TRAIN_PRESET,
            "known_presets": list_presets(),
            "entrypoint": ENTRYPOINT,
        }
    )

    run_name = _extract_flag_value(cmd, "--run-name") or "geovit_proper_modal"
    run_dir = Path("/root/results/geovit_proper") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if RUN_ENV_PROBE:
        probe_cmd = ["python", "-u", "-m", "src.modal_env_probe"]
        probe = subprocess.run(
            probe_cmd,
            cwd="/root/src",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        print("=== Modal GeoViT-Proper env probe ===")
        print(probe.stdout, end="" if probe.stdout.endswith("\n") else "\n")
        if probe.returncode == 0 and probe.stdout.strip():
            try:
                payload = probe.stdout.strip().splitlines()[-1]
                (run_dir / "env_probe.json").write_text(payload + "\n", encoding="utf-8")
            except Exception:
                pass

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
    print(f"\n=== GeoViT-Proper train exited with code {rc} ===")
    results_volume.commit()
    data_volume.commit()
    if rc != 0:
        raise RuntimeError(f"GeoViT-Proper run failed with exit code {rc}")


def _extract_flag_value(cmd: list[str], flag: str) -> str:
    for idx, token in enumerate(cmd[:-1]):
        if token == flag:
            return str(cmd[idx + 1])
    return ""


if __name__ == "__main__":
    run_remote.remote()
