import argparse
import csv
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from layers.geodynamic_layer import GeoDynamicLayer
from models.vit import VisionTransformer
from trainer import HybridTrainer


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def append_metrics_row(csv_path, row):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        if not torch.isnan(loss):
            total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total, total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr_base", type=float, default=2e-4)
    parser.add_argument("--lr_controller", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--save_dir", type=str, default="/root/checkpoints")
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()

    device = get_device()
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    os.makedirs(args.save_dir, exist_ok=True)

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print("Initializing best Geo-ViT configuration...")
    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        embed_dim=192,
        depth=6,
        num_heads=6,
        num_classes=100,
        linear_layer=GeoDynamicLayer,
        drop_path_rate=0.1,
    ).to(device)

    trainer = HybridTrainer(
        model,
        lr_base=args.lr_base,
        lr_controller=args.lr_controller,
        weight_decay=args.weight_decay,
    )
    scheduler_euc = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.opt_euclidean, T_max=args.epochs)
    scheduler_man = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.opt_manifold, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    metrics_path = os.path.join(args.save_dir, "metrics.csv")

    start_epoch = 0
    if os.path.exists(args.save_dir):
        files = [f for f in os.listdir(args.save_dir) if f.endswith(".pth")]
        if files:
            try:
                latest_path = os.path.join(args.save_dir, "latest.pth")
                if os.path.exists(latest_path):
                    ckpt_path = latest_path
                else:
                    files.sort(key=lambda x: int(re.search(r"e(\d+)", x).group(1)) if re.search(r"e(\d+)", x) else -1)
                    ckpt_path = os.path.join(args.save_dir, files[-1])

                ckpt = torch.load(ckpt_path, map_location=device)
                if "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"])
                    start_epoch = ckpt.get("epoch", 0) + 1
                if "opt_manifold_state_dict" in ckpt:
                    trainer.opt_manifold.load_state_dict(ckpt["opt_manifold_state_dict"])
                if "opt_euclidean_state_dict" in ckpt:
                    trainer.opt_euclidean.load_state_dict(ckpt["opt_euclidean_state_dict"])
                if "scheduler_man_state_dict" in ckpt:
                    scheduler_man.load_state_dict(ckpt["scheduler_man_state_dict"])
                if "scheduler_euc_state_dict" in ckpt:
                    scheduler_euc.load_state_dict(ckpt["scheduler_euc_state_dict"])
                if scaler and "scaler_state_dict" in ckpt:
                    scaler.load_state_dict(ckpt["scaler_state_dict"])
                print(f"Resuming from Epoch {start_epoch + 1}")
            except Exception as exc:
                print(f"Resume skipped due to checkpoint load issue: {exc}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        num_steps = len(train_loader)

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            step_start = time.time()
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=args.mixup_alpha)
            inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs, targets_a, targets_b))

            if batch_idx % args.grad_accum_steps == 0:
                trainer.opt_manifold.zero_grad(set_to_none=True)
                trainer.opt_euclidean.zero_grad(set_to_none=True)

            if scaler:
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs)
                    loss = mixup_criterion(loss_fn, outputs, targets_a, targets_b, lam) / args.grad_accum_steps

                if torch.isnan(loss):
                    scaler.update()
                    continue

                scaler.scale(loss).backward()
                should_step = ((batch_idx + 1) % args.grad_accum_steps == 0) or ((batch_idx + 1) == num_steps)
                if should_step:
                    scaler.unscale_(trainer.opt_euclidean)
                    scaler.unscale_(trainer.opt_manifold)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    if torch.isnan(grad_norm):
                        scaler.update()
                        trainer.opt_manifold.zero_grad(set_to_none=True)
                        trainer.opt_euclidean.zero_grad(set_to_none=True)
                        continue

                    scaler.step(trainer.opt_manifold)
                    scaler.step(trainer.opt_euclidean)
                    scaler.update()
            else:
                outputs = model(inputs)
                loss = mixup_criterion(loss_fn, outputs, targets_a, targets_b, lam) / args.grad_accum_steps
                loss.backward()
                should_step = ((batch_idx + 1) % args.grad_accum_steps == 0) or ((batch_idx + 1) == num_steps)
                if should_step:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    trainer.opt_manifold.step()
                    trainer.opt_euclidean.step()

            epoch_loss += loss.item() * args.grad_accum_steps

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}] Step [{batch_idx + 1}] "
                    f"Loss: {loss.item() * args.grad_accum_steps:.4f} | "
                    f"Time: {time.time() - step_start:.3f}s"
                )

        val_acc, val_loss = validate(model, test_loader, device)
        scheduler_euc.step()
        scheduler_man.step()

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1} Done | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f} | Epoch Time: {epoch_time:.1f}s")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "opt_manifold_state_dict": trainer.opt_manifold.state_dict(),
            "opt_euclidean_state_dict": trainer.opt_euclidean.state_dict(),
            "scheduler_man_state_dict": scheduler_man.state_dict(),
            "scheduler_euc_state_dict": scheduler_euc.state_dict(),
            "acc": val_acc,
            "val_loss": val_loss,
        }
        if scaler:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        torch.save(checkpoint, os.path.join(args.save_dir, "latest.pth"))
        append_metrics_row(
            metrics_path,
            {
                "epoch": epoch + 1,
                "train_loss": epoch_loss / num_steps,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "lr_manifold": scheduler_man.get_last_lr()[0],
                "lr_euclidean": scheduler_euc.get_last_lr()[0],
                "epoch_time_sec": round(epoch_time, 3),
            },
        )

        if (epoch + 1) % args.save_every == 0:
            torch.save(checkpoint, os.path.join(args.save_dir, f"geovit_e{epoch + 1}.pth"))


if __name__ == "__main__":
    main()
