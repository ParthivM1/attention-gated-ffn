import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import time
import os

# Import your modules
from models.vit import VisionTransformer
from layers.geodynamic_layer import GeoDynamicLayer
from trainer import HybridTrainer

def get_device():
    """Auto-detects the best available accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps") # Apple Silicon M-Series
    else:
        return torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(description="Train Geo-ViT")
    parser.add_argument("--batch_size", type=int, default=64, help="Reduce to 32 if OOM on M4")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr_base", type=float, default=1e-3, help="LR for Manifold U_0")
    parser.add_argument("--lr_controller", type=float, default=1e-4, help="LR for Controller")
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100", "fake"])
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()

    device = get_device()
    print(f"🚀 Training on device: {device}")

    # Ensure save_dir exists
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # 1. Data Setup (Using CIFAR for Dev because ImageNet is too big to download quickly)
    print("Preparing Data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    if args.dataset == "cifar100":
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        num_classes = 100
    elif args.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        num_classes = 10
    else:
        # Fake data for instant debugging if internet is slow
        train_dataset = datasets.FakeData(transform=transform, size=1000)
        num_classes = 10

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 2. Model Initialization (Injecting YOUR Geo-Layer)
    print("Initializing Geo-ViT...")
    model = VisionTransformer(
        img_size=32, 
        patch_size=8, 
        embed_dim=384,     # ViT-Small size (Friendly for M4 Pro)
        depth=12, 
        num_heads=6, 
        num_classes=num_classes,
        linear_layer=GeoDynamicLayer # <--- THE MAGIC SWAP
    )
    model.to(device)

    # 3. Trainer Setup (The Hybrid Manager)
    loss_fn = nn.CrossEntropyLoss()
    trainer = HybridTrainer(model, lr_base=args.lr_base, lr_controller=args.lr_controller)

    # 4. Training Loop
    print("🏁 Starting Training Loop...")
    model.train()
    
    for epoch in range(args.epochs):
        start_time = time.time()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # The Step (Forward + Backward + Hybrid Opt)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            trainer.step(loss) # Handles the split optimizers internally
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")
        
        epoch_time = time.time() - start_time
        print(f"✨ Epoch {epoch+1} Complete in {epoch_time:.1f}s | Avg Loss: {total_loss/len(train_loader):.4f} | Avg Acc: {100.*correct/total:.2f}%")
        
        # Save Checkpoint
        if args.save_dir:
            ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_manifold_state': trainer.opt_manifold.state_dict(),
                'optimizer_euclidean_state': trainer.opt_euclidean.state_dict(),
            }, ckpt_path)
            print(f"💾 Checkpoint saved: {ckpt_path}")

        print("-" * 50)

if __name__ == "__main__":
    main()