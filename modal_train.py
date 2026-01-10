import modal
import sys
import os

app = modal.App("geo-vit-h100-run")


image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchvision", "geoopt", "numpy")
    .add_local_dir("./src", remote_path="/root/src")
)


volume = modal.Volume.from_name("geo-vit-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="a100", 
    volumes={"/root/checkpoints": volume},
    timeout=86400  # 24 hours
)
def train_remote():
    import subprocess
    import os
    
    print("🚀 Starting remote training on A100...")


    script_path = "/root/src/train.py"
    

    if not os.path.exists(script_path):
         print(f"Warning: {script_path} not found. checking current directory...")
         print(os.listdir("."))

    cmd = [
        "python",
        "-u",
        script_path,
        "--save_dir", "/root/checkpoints",
        "--dataset", "cifar100",
        "--epochs", "50",
        "--batch_size", "64"
    ]

    print(f"Executing: {' '.join(cmd)}")

 
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode != 0:
        raise Exception(f"Training script failed with exit code {process.returncode}")

    print("Remote training completed successfully.")

if __name__ == "__main__":
    # Local entry point to launch the remote function
    with app.run():
        train_remote.remote()
