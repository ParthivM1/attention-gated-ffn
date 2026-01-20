import modal
import sys

app = modal.App("geo-vit-eval")

# Define the image (same as training environment)
image = (
   modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04", add_python="3.11")
   .run_commands(
       "python -m pip install --upgrade pip",
       "pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio",
       "pip install numpy scipy geoopt wandb tqdm pillow"
   )
   .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
   .env({"PYTHONPATH": "/root/src"})
   .add_local_dir(".", remote_path="/root/src") 
)

# Attach the existing volume where your training saved weights
volume = modal.Volume.from_name("geo-vit-checkpoints")

@app.function(
   image=image,
   gpu="A100",  
   volumes={"/root/checkpoints": volume},
   timeout=1800 
)
def evaluate_remote(checkpoint_name: str = None):
   import subprocess
   import os
  
   print("📂 Checking /root/checkpoints contents:")
   if os.path.exists("/root/checkpoints"):
       files = os.listdir("/root/checkpoints")
       print(files)
       if not files:
           print("   (Empty directory)")
   else:
       print("   (Directory does not exist!)")

   # Build command
   cmd = [
       "python", "-u", "/root/src/src/eval.py",
       "--batch_size", "128"
   ]
   
   if checkpoint_name:
       cmd.extend(["--checkpoint_name", checkpoint_name])
  
   print("🚀 Launching:", " ".join(cmd))
  
   # Run the eval script
   try:
       subprocess.run(cmd, cwd="/root/src", check=True)
   except subprocess.CalledProcessError as e:
       print(f"❌ Evaluation failed with exit code {e.returncode}")
       raise e

if __name__ == "__main__":
    # How to run:
    # 1. Automatic (latest checkpoint): modal run modal_eval.py
    # 2. Specific checkpoint: modal run modal_eval.py --checkpoint-name geovit_e50.pth
    
    # Parse CLI args for modal run
    checkpoint_arg = None
    if "--checkpoint-name" in sys.argv:
        idx = sys.argv.index("--checkpoint-name")
        if idx + 1 < len(sys.argv):
            checkpoint_arg = sys.argv[idx + 1]
    
    with app.run():
        evaluate_remote.remote(checkpoint_name=checkpoint_arg)