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
    parser.add_argument("--batch_size", type=int, default=32, help="Physical batch size per step")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Steps to accumulate gradients before update")


    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr_base", type=float, default=1e-4, help="LR for Manifold U_0")
    parser.add_argument("--lr_controller", type=float, default=3e-5, help="LR for Controller")
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
        persistent_workers=True,
        prefetch_factor=4
    )


    # 2. Model Initialization
    # Using patch_size=8 for 32x32 images reduces sequence length to 16, saving massive memory
    print("Initializing Geo-ViT...")
    model = VisionTransformer(
        img_size=32,
        patch_size=16,     # fewer tokens (32x32 with 16 => 4 tokens + cls)
        embed_dim=192,     # smaller matrices in Cayley map
        depth=6,           # fewer blocks
        num_heads=3,       # keep head_dim reasonable
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
            batch_start = time.time()

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Use AMP for speed and memory savings
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    # Normalize loss by accumulation steps
                    loss = loss / args.grad_accum_steps

                    # ---- Fix 4: debugging for instability (every 50 batches) ----
                    if (batch_idx % 50) == 0:
                        with torch.no_grad():
                            logits_abs_max = outputs.detach().abs().max().item()
                        print(f"[dbg] epoch={epoch+1} batch={batch_idx+1}/{len(train_loader)} logits_abs_max={logits_abs_max:.2f}")


                    if device.type == "cuda" and ((batch_idx + 1) % 200 == 0):
                        alloc = torch.cuda.memory_allocated() / 1024**3
                        reserv = torch.cuda.memory_reserved() / 1024**3
                        print(f"[mem-gpu] step {batch_idx+1}: allocated={alloc:.2f}GB reserved={reserv:.2f}GB")


                
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
                    # Unscale grads so clipping is meaningful
                    scaler.unscale_(trainer.opt_manifold)
                    scaler.unscale_(trainer.opt_euclidean)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    scaler.step(trainer.opt_manifold)
                    scaler.step(trainer.opt_euclidean)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    trainer.opt_manifold.step()
                    trainer.opt_euclidean.step()

                trainer.opt_manifold.zero_grad(set_to_none=True)
                trainer.opt_euclidean.zero_grad(set_to_none=True)

            # Logging inputs
            total_loss += loss.item() * args.grad_accum_steps # Un-normalize for display
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            batch_time = time.time() - batch_start

            if batch_idx % 5 == 0 or batch_idx == len(train_loader) - 1:
                print(
                    f"[Epoch {epoch+1}/{args.epochs}] "
                    f"Batch {batch_idx+1}/{len(train_loader)} | "
                    f"Loss: {loss.item() * args.grad_accum_steps:.4f} | "
                    f"Batch time: {batch_time:.3f}s"
                )

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