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
from trainer import HybridTrainer  # Ensure this path matches your folder structure

def get_device():
    """Auto-detects the best available accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps") # Apple Silicon M-Series
    else:
        return torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(description="Train Geo-ViT with Gradient Accumulation")
    # Reduced default batch size to 16 to fit in A100 memory with ODE Solver
    parser.add_argument("--batch_size", type=int, default=16, help="Physical batch size per step")
    # Accumulate gradients to simulate a larger batch (e.g., 16 * 8 = 128 effective batch)
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Steps to accumulate gradients before update")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr_base", type=float, default=1e-3, help="LR for Manifold U_0")
    parser.add_argument("--lr_controller", type=float, default=1e-4, help="LR for Controller")
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100"])
    parser.add_argument("--save_dir", type=str, default="/root/checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()

    device = get_device()
    effective_batch_size = args.batch_size * args.grad_accum_steps
    print(f"🚀 Training on device: {device}")
    print(f"📉 Config: Physical Batch={args.batch_size} | Accum={args.grad_accum_steps} | Effective Batch={effective_batch_size}")

    # Ensure save_dir exists
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Data Setup (CIFAR-100 Native Resolution 32x32)
    # No Resize operation ensures max speed and no CPU bottleneck
    print("Preparing Data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    if args.dataset == "cifar100":
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        num_classes = 100
    else:
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        num_classes = 10

    # Increased num_workers to 8 to feed the GPU faster
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True,
        prefetch_factor=2
    )

    # 2. Model Initialization
    # Using patch_size=8 for 32x32 images reduces sequence length to 16, saving massive memory
    print("Initializing Geo-ViT...")
    model = VisionTransformer(
        img_size=32, 
        patch_size=8, 
        embed_dim=384,     
        depth=12, 
        num_heads=6, 
        num_classes=num_classes,
        linear_layer=GeoDynamicLayer 
    )
    model.to(device)

    # 3. Trainer Setup
    loss_fn = nn.CrossEntropyLoss()
    trainer = HybridTrainer(model, lr_base=args.lr_base, lr_controller=args.lr_controller)
    
    # 4. Training Loop
    print("🏁 Starting Training Loop...")
    model.train()
    
    # Enable Automatic Mixed Precision (AMP) for A100 Tensor Cores
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    for epoch in range(args.epochs):
        start_time = time.time()
        total_loss = 0
        correct = 0
        total = 0
        
        # Zero gradients at the start of the epoch
        trainer.opt_manifold.zero_grad()
        trainer.opt_euclidean.zero_grad()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Use AMP for speed and memory savings
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    # Normalize loss by accumulation steps
                    loss = loss / args.grad_accum_steps
                
                # Backward pass with scaler
                scaler.scale(loss).backward()
            else:
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss = loss / args.grad_accum_steps
                loss.backward()
            
            # Step ONLY after accumulation period
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                if scaler:
                    scaler.step(trainer.opt_manifold)
                    scaler.step(trainer.opt_euclidean)
                    scaler.update()
                else:
                    trainer.opt_manifold.step()
                    trainer.opt_euclidean.step()
                
                # Zero gradients for next accumulation cycle
                trainer.opt_manifold.zero_grad()
                trainer.opt_euclidean.zero_grad()

            # Logging inputs
            total_loss += loss.item() * args.grad_accum_steps # Un-normalize for display
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # End of Epoch Metrics
        epoch_acc = 100. * correct / total
        epoch_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{args.epochs}] | "
              f"Time: {epoch_time:.1f}s | "
              f"Loss: {epoch_loss:.4f} | "
              f"Acc: {epoch_acc:.2f}%")
        
        # Checkpoint (Save every 5 epochs)
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(args.save_dir, f"geovit_e{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'opt_manifold': trainer.opt_manifold.state_dict(),
                'opt_euclidean': trainer.opt_euclidean.state_dict(),
                'acc': epoch_acc
            }, ckpt_path)
            print(f"✅ Saved checkpoint to {ckpt_path}")
            
    print("Training Complete.")


if __name__ == "__main__":
    main()