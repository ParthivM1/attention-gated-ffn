import modal
import os
import shutil

app = modal.App("geo-vit-download")
volume = modal.Volume.from_name("geo-vit-checkpoints")

@app.function(volumes={"/root/checkpoints": volume})
def list_and_download():
    files = os.listdir("/root/checkpoints")
    print(f"📂 Found {len(files)} files in volume 'geo-vit-checkpoints':")
    for f in files:
        print(f" - {f}")
    return files

# Note: Modal Volumes also have a CLI: 'modal volume get geo-vit-checkpoints <remote_path> <local_dest>'
# But here is a script to verify what is there.

if __name__ == "__main__":
    with app.run():
        print("Checking remote volume...")
        files = list_and_download.remote()
        
        print("\n💡 To download these files to your local machine, run:")
        print("modal volume get geo-vit-checkpoints / .")
