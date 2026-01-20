import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import time
import os
import re

from models.vit import VisionTransformer
from layers.geodynamic_layer import GeoDynamicLayer
from trainer import HybridTrainer

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    elif torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct, total, total_loss = 0, 0, 0
    loss_fn = nn.CrossEntropyLoss()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return 100. * correct / total, total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr_base", type=float, default=2e-4)       
    parser.add_argument("--lr_controller", type=float, default=1e-3) 
    parser.add_argument("--save_dir", type=str, default="/root/checkpoints")
    args = parser.parse_args()

    device = get_device()
    os.makedirs(args.save_dir, exist_ok=True)

    # CIFAR-100
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"🚀 Initializing Geo-ViT on {device}...")
    model = VisionTransformer(
        img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6, num_classes=100, 
        linear_layer=GeoDynamicLayer
    ).to(device)

    trainer = HybridTrainer(model, lr_base=args.lr_base, lr_controller=args.lr_controller)
    
    scheduler_euc = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.opt_euclidean, T_max=args.epochs)
    scheduler_man = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.opt_manifold, T_max=args.epochs)
    
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # --- RESUME LOGIC ---
    start_epoch = 0
    if os.path.exists(args.save_dir):
        files = [f for f in os.listdir(args.save_dir) if f.endswith(".pth")]
        if files:
            # Sort by epoch number: geovit_e10.pth
            try:
                files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
                latest = files[-1]
                ckpt_path = os.path.join(args.save_dir, latest)
                print(f"🔄 Found checkpoint: {latest}. Loading...")
                
                ckpt = torch.load(ckpt_path, map_location=device)
                
                # Load Weights
                if 'model_state_dict' in ckpt:
                    model.load_state_dict(ckpt['model_state_dict'])
                    start_epoch = ckpt.get('epoch', 0) + 1
                    # Attempt load optimizer if exists
                    if 'opt_manifold' in ckpt:
                        trainer.opt_manifold.load_state_dict(ckpt['opt_manifold'])
                        trainer.opt_euclidean.load_state_dict(ckpt['opt_euclidean'])
                else:
                    # Legacy load (just weights)
                    model.load_state_dict(ckpt)
                    # Infer epoch from filename
                    match = re.search(r'e(\d+)', latest)
                    if match:
                        start_epoch = int(match.group(1))
                
                print(f"✅ Resuming training from Epoch {start_epoch+1}")
                
                # Fast-forward schedulers to correct epoch
                for _ in range(start_epoch):
                    scheduler_euc.step()
                    scheduler_man.step()
                    
            except Exception as e:
                print(f"⚠️ Failed to resume from checkpoint: {e}")
                print("   Starting from scratch.")

    # --- TRAINING LOOP ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss, correct, total = 0, 0, 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            start = time.time()
            inputs, targets = inputs.to(device), targets.to(device)
            
            trainer.opt_manifold.zero_grad(set_to_none=True)
            trainer.opt_euclidean.zero_grad(set_to_none=True)

            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.unscale_(trainer.opt_euclidean)
                scaler.unscale_(trainer.opt_manifold)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(trainer.opt_manifold)
                scaler.step(trainer.opt_euclidean)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                trainer.opt_manifold.step()
                trainer.opt_euclidean.step()

            batch_time = time.time() - start
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}] Step [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}% | Time: {batch_time:.3f}s")

        val_acc, val_loss = validate(model, test_loader, device)
        scheduler_euc.step()
        scheduler_man.step()
        
        print(f"✨ Epoch {epoch+1} Done | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}")
        
        # Save Full State
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(args.save_dir, f"geovit_e{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'opt_manifold': trainer.opt_manifold.state_dict(),
                'opt_euclidean': trainer.opt_euclidean.state_dict(),
                'acc': val_acc
            }, save_path)
            print(f"💾 Checkpoint saved: {save_path}")

if __name__ == "__main__":
    main()