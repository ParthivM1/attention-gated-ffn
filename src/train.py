import argparse
import copy
import csv
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_factory import add_model_args, build_model, config_from_args, config_from_checkpoint, count_parameters
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


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size, _, height, width = x.shape
    index = torch.randperm(batch_size, device=x.device)
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, height)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1.0 - float((x2 - x1) * (y2 - y1)) / float(width * height)
    return mixed_x, y, y[index], lam


def apply_batch_mix(x, y, mixup_alpha=0.0, cutmix_alpha=0.0, mix_prob=1.0, switch_prob=0.5):
    if mix_prob <= 0.0 or np.random.rand() > mix_prob:
        return x, y, y, 1.0

    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or np.random.rand() < switch_prob)
    if use_cutmix:
        return cutmix_data(x, y, alpha=cutmix_alpha)
    if mixup_alpha > 0:
        return mixup_data(x, y, alpha=mixup_alpha)
    return x, y, y, 1.0


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def append_metrics_row(csv_path, row):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


class ModelEma:
    def __init__(self, model, decay=0.9998):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        self.parameter_names = {name for name, _ in self.module.named_parameters()}
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def set(self, model):
        self.module.load_state_dict(model.state_dict())

    @torch.no_grad()
    def update(self, model):
        model_state = model.state_dict()
        ema_state = self.module.state_dict()
        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach()
            if name not in self.parameter_names or not torch.is_floating_point(ema_value):
                ema_value.copy_(model_value)
            else:
                ema_value.lerp_(model_value, 1.0 - self.decay)


def build_scheduler(optimizer, total_epochs, warmup_epochs, warmup_start_factor, min_lr_ratio):
    if optimizer is None:
        return None

    base_lr = optimizer.param_groups[0]["lr"]
    eta_min = base_lr * min_lr_ratio
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_epochs - warmup_epochs, 1),
            eta_min=eta_min,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs, 1),
        eta_min=eta_min,
    )


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


def set_geo_adapter_scale(model, scale):
    for module in model.modules():
        setter = getattr(module, "set_adapter_scale", None)
        if callable(setter):
            setter(scale)


def collect_geo_diagnostics(model):
    aggregates = {}
    count = 0
    for module in model.modules():
        getter = getattr(module, "get_diagnostics", None)
        if not callable(getter):
            continue
        stats = getter()
        if not stats:
            continue
        count += 1
        for key, value in stats.items():
            aggregates[key] = aggregates.get(key, 0.0) + float(value)

    if count == 0:
        return {}
    return {key: value / count for key, value in aggregates.items()}


def compute_geo_adapter_scale(epoch_index, geo_start_epoch, geo_ramp_epochs, geo_target_scale):
    current_epoch = epoch_index + 1
    if current_epoch <= geo_start_epoch:
        return 0.0
    if geo_ramp_epochs <= 0:
        return float(geo_target_scale)
    progress = (current_epoch - geo_start_epoch) / float(geo_ramp_epochs)
    return float(min(max(progress, 0.0), 1.0) * geo_target_scale)


def parse_block_indices(spec):
    if not spec:
        return []
    parts = [part.strip() for part in spec.split(",")]
    return [int(part) for part in parts if part]


def locality_prior_weight(epoch_index, base_lambda, warm_epochs, decay_epochs):
    if base_lambda <= 0.0:
        return 0.0
    current_epoch = epoch_index + 1
    if current_epoch <= warm_epochs:
        return float(base_lambda)
    if decay_epochs <= warm_epochs:
        return 0.0
    progress = (current_epoch - warm_epochs) / float(max(decay_epochs - warm_epochs, 1))
    return float(base_lambda * max(0.0, 1.0 - progress))


def build_locality_prior(num_patches, sigma, device, include_cls):
    grid_size = int(round(math.sqrt(num_patches)))
    if grid_size * grid_size != num_patches:
        return None

    coords = torch.stack(
        torch.meshgrid(
            torch.arange(grid_size, device=device, dtype=torch.float32),
            torch.arange(grid_size, device=device, dtype=torch.float32),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    dist2 = torch.cdist(coords, coords, p=2.0).pow(2)
    patch_prior = torch.exp(-dist2 / max(2.0 * sigma * sigma, 1e-6))
    patch_prior = patch_prior / patch_prior.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    if not include_cls:
        return patch_prior

    total_tokens = num_patches + 1
    full_prior = torch.full((total_tokens, total_tokens), 1e-6, device=device, dtype=torch.float32)
    full_prior[0].fill_(1.0 / float(total_tokens))
    full_prior[1:, 1:] = patch_prior
    full_prior = full_prior / full_prior.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return full_prior


def compute_locality_prior_loss(model, block_indices, sigma=1.0, exclude_cls=True):
    get_maps = getattr(model, "get_attention_maps", None)
    if not callable(get_maps):
        return None

    maps = get_maps(block_indices)
    num_patches = getattr(model, "num_patches", None)
    if not maps or not num_patches:
        return None

    prior = build_locality_prior(num_patches, sigma=sigma, device=maps[0][1].device, include_cls=not exclude_cls)
    if prior is None:
        return None

    losses = []
    for _, attn in maps:
        attn_f32 = attn.to(torch.float32)
        if exclude_cls:
            attn_used = attn_f32[:, :, 1 : 1 + num_patches, 1 : 1 + num_patches]
        else:
            attn_used = attn_f32[:, :, : 1 + num_patches, : 1 + num_patches]

        log_attn = attn_used.clamp_min(1e-8).log()
        log_prior = prior.clamp_min(1e-8).log().view(1, 1, prior.shape[0], prior.shape[1])
        kl = torch.sum(attn_used * (log_attn - log_prior), dim=-1).mean()
        losses.append(kl)

    if not losses:
        return None
    return torch.stack(losses).mean()


def compute_locality_prior_diagnostics(model, block_indices, exclude_cls=True):
    get_maps = getattr(model, "get_attention_maps", None)
    if not callable(get_maps):
        return {}

    maps = get_maps(block_indices)
    num_patches = getattr(model, "num_patches", None)
    if not maps or not num_patches:
        return {}

    grid_size = int(round(math.sqrt(num_patches)))
    if grid_size * grid_size != num_patches:
        return {}

    coords = torch.stack(
        torch.meshgrid(
            torch.arange(grid_size, device=maps[0][1].device, dtype=torch.float32),
            torch.arange(grid_size, device=maps[0][1].device, dtype=torch.float32),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    distances = torch.cdist(coords, coords, p=2.0)

    mean_distances = []
    entropies = []
    for _, attn in maps:
        attn_f32 = attn.to(torch.float32)
        if exclude_cls:
            attn_used = attn_f32[:, :, 1 : 1 + num_patches, 1 : 1 + num_patches]
        else:
            attn_used = attn_f32[:, :, : 1 + num_patches, : 1 + num_patches]
            distances = torch.zeros_like(attn_used[0, 0])
        mean_distances.append((attn_used * distances.view(1, 1, distances.shape[0], distances.shape[1])).sum(dim=-1).mean())
        entropies.append((-(attn_used.clamp_min(1e-8) * attn_used.clamp_min(1e-8).log()).sum(dim=-1)).mean())

    return {
        "attn_mean_distance": float(torch.stack(mean_distances).mean().item()),
        "attn_entropy": float(torch.stack(entropies).mean().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--warmup_start_factor", type=float, default=0.1)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--lr_base", type=float, default=1.5e-4)
    parser.add_argument("--lr_controller", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--mix_prob", type=float, default=1.0)
    parser.add_argument("--cutmix_switch_prob", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--randaugment_num_ops", type=int, default=2)
    parser.add_argument("--randaugment_magnitude", type=int, default=9)
    parser.add_argument("--random_erasing_prob", type=float, default=0.25)
    parser.add_argument("--ema_decay", type=float, default=0.9998)
    parser.add_argument("--ema_start_epoch", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="/root/checkpoints")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--geo_start_epoch", type=int, default=10)
    parser.add_argument("--geo_ramp_epochs", type=int, default=10)
    parser.add_argument("--geo_target_scale", type=float, default=1.0)
    parser.add_argument("--drop_epoch", type=int, default=50)
    parser.add_argument("--min_best_acc_by_drop_epoch", type=float, default=60.0)
    parser.add_argument("--plateau_start_epoch", type=int, default=35)
    parser.add_argument("--plateau_patience", type=int, default=10)
    parser.add_argument("--plateau_min_delta", type=float, default=0.2)
    parser.add_argument("--use_locality_prior", action="store_true")
    parser.add_argument("--prior_blocks", type=str, default="0,1")
    parser.add_argument("--prior_lambda", type=float, default=0.1)
    parser.add_argument("--prior_warm_epochs", type=int, default=0)
    parser.add_argument("--prior_decay_epochs", type=int, default=80)
    parser.add_argument("--prior_sigma", type=float, default=1.0)
    parser.add_argument("--prior_exclude_cls", action="store_true", default=True)
    parser.add_argument("--include_cls_in_prior", action="store_false", dest="prior_exclude_cls")
    add_model_args(parser)
    args = parser.parse_args()

    device = get_device()
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    os.makedirs(args.save_dir, exist_ok=True)

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=args.randaugment_num_ops, magnitude=args.randaugment_magnitude),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            transforms.RandomErasing(p=args.random_erasing_prob, value="random"),
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

    resume_checkpoint_path = None
    if os.path.exists(args.save_dir):
        files = [f for f in os.listdir(args.save_dir) if f.endswith(".pth")]
        if files:
            latest_path = os.path.join(args.save_dir, "latest.pth")
            if os.path.exists(latest_path):
                resume_checkpoint_path = latest_path
            else:
                files.sort(key=lambda x: int(re.search(r"e(\d+)", x).group(1)) if re.search(r"e(\d+)", x) else -1)
                resume_checkpoint_path = os.path.join(args.save_dir, files[-1])

    model_config = config_from_args(args, num_classes=100)
    if resume_checkpoint_path is not None:
        try:
            resume_meta = torch.load(resume_checkpoint_path, map_location="cpu")
            checkpoint_model_config = resume_meta.get("model_config")
            if checkpoint_model_config:
                model_config = config_from_checkpoint(checkpoint_model_config, num_classes=100)
                print(f"Found checkpoint config in save_dir. Resuming with variant: {model_config['model_variant']}")
        except Exception as exc:
            print(f"Checkpoint config probe skipped: {exc}")

    print(f"Initializing model variant: {model_config['model_variant']}")
    model = build_model(model_config).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Model parameters: total={total_params:,} trainable={trainable_params:,}")

    trainer = HybridTrainer(
        model,
        lr_base=args.lr_base,
        lr_controller=args.lr_controller,
        weight_decay=args.weight_decay,
    )
    scheduler_euc = build_scheduler(
        trainer.opt_euclidean,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        warmup_start_factor=args.warmup_start_factor,
        min_lr_ratio=args.min_lr_ratio,
    )
    scheduler_man = build_scheduler(
        trainer.opt_manifold,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        warmup_start_factor=args.warmup_start_factor,
        min_lr_ratio=args.min_lr_ratio,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    ema = ModelEma(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    ema_active = False
    metrics_path = os.path.join(args.save_dir, "metrics.csv")
    optimizers = [opt for opt in (trainer.opt_manifold, trainer.opt_euclidean) if opt is not None]
    prior_block_indices = parse_block_indices(args.prior_blocks) if args.use_locality_prior else []

    start_epoch = 0
    best_acc = float("-inf")
    best_epoch = -1
    if resume_checkpoint_path is not None:
        ckpt_path = resume_checkpoint_path
        if os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location=device)
            except Exception as exc:
                print(f"Resume skipped due to checkpoint load issue: {exc}")
                ckpt = None

            if ckpt is not None:
                try:
                    if "model_state_dict" in ckpt:
                        model.load_state_dict(ckpt["model_state_dict"])
                        start_epoch = ckpt.get("epoch", 0) + 1
                        best_acc = ckpt.get("best_acc", ckpt.get("acc", best_acc))
                        best_epoch = ckpt.get("best_epoch", ckpt.get("epoch", best_epoch))
                except Exception as exc:
                    print(f"Model state resume skipped: {exc}")
                    ckpt = None

            if ckpt is not None:
                if trainer.opt_manifold is not None and "opt_manifold_state_dict" in ckpt:
                    try:
                        trainer.opt_manifold.load_state_dict(ckpt["opt_manifold_state_dict"])
                    except Exception as exc:
                        print(f"Manifold optimizer resume skipped: {exc}")
                if "opt_euclidean_state_dict" in ckpt:
                    try:
                        trainer.opt_euclidean.load_state_dict(ckpt["opt_euclidean_state_dict"])
                    except Exception as exc:
                        print(f"Euclidean optimizer resume skipped: {exc}")
                if scheduler_man is not None and "scheduler_man_state_dict" in ckpt:
                    try:
                        scheduler_man.load_state_dict(ckpt["scheduler_man_state_dict"])
                    except Exception as exc:
                        print(f"Manifold scheduler resume skipped: {exc}")
                if scheduler_euc is not None and "scheduler_euc_state_dict" in ckpt:
                    try:
                        scheduler_euc.load_state_dict(ckpt["scheduler_euc_state_dict"])
                    except Exception as exc:
                        print(f"Euclidean scheduler resume skipped: {exc}")
                if scaler and "scaler_state_dict" in ckpt:
                    try:
                        scaler.load_state_dict(ckpt["scaler_state_dict"])
                    except Exception as exc:
                        print(f"GradScaler resume skipped: {exc}")
                if ema is not None:
                    ema_active = ckpt.get("ema_active", start_epoch >= args.ema_start_epoch)
                    if ema_active and "ema_state_dict" in ckpt:
                        try:
                            ema.module.load_state_dict(ckpt["ema_state_dict"])
                        except Exception as exc:
                            print(f"EMA resume skipped: {exc}")
                            ema.set(model)
                            ema_active = False
                    else:
                        ema.set(model)
                print(f"Resuming from Epoch {start_epoch + 1}")
    elif ema is not None:
        ema.set(model)

    for epoch in range(start_epoch, args.epochs):
        geo_adapter_scale = compute_geo_adapter_scale(
            epoch,
            args.geo_start_epoch,
            args.geo_ramp_epochs,
            args.geo_target_scale,
        )
        prior_lambda = locality_prior_weight(
            epoch,
            args.prior_lambda,
            args.prior_warm_epochs,
            args.prior_decay_epochs,
        ) if args.use_locality_prior else 0.0
        set_geo_adapter_scale(model, geo_adapter_scale)
        if ema is not None:
            set_geo_adapter_scale(ema.module, geo_adapter_scale)

        model.train()
        epoch_loss = 0.0
        epoch_task_loss = 0.0
        epoch_prior_loss = 0.0
        epoch_start = time.time()
        num_steps = len(train_loader)

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            step_start = time.time()
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            inputs, targets_a, targets_b, lam = apply_batch_mix(
                inputs,
                targets,
                mixup_alpha=args.mixup_alpha,
                cutmix_alpha=args.cutmix_alpha,
                mix_prob=args.mix_prob,
                switch_prob=args.cutmix_switch_prob,
            )
            inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs, targets_a, targets_b))

            if batch_idx % args.grad_accum_steps == 0:
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)

            if scaler:
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs)
                    task_loss = mixup_criterion(loss_fn, outputs, targets_a, targets_b, lam)
                    prior_loss = compute_locality_prior_loss(
                        model,
                        prior_block_indices,
                        sigma=args.prior_sigma,
                        exclude_cls=args.prior_exclude_cls,
                    ) if args.use_locality_prior and prior_lambda > 0.0 else None
                    total_loss = task_loss if prior_loss is None else task_loss + (prior_lambda * prior_loss)
                    loss = total_loss / args.grad_accum_steps

                if torch.isnan(loss):
                    scaler.update()
                    continue

                scaler.scale(loss).backward()
                should_step = ((batch_idx + 1) % args.grad_accum_steps == 0) or ((batch_idx + 1) == num_steps)
                if should_step:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    if torch.isnan(grad_norm):
                        scaler.update()
                        for optimizer in optimizers:
                            optimizer.zero_grad(set_to_none=True)
                        continue

                    for optimizer in optimizers:
                        scaler.step(optimizer)
                    scaler.update()
                    if ema is not None and ema_active:
                        ema.update(model)
            else:
                outputs = model(inputs)
                task_loss = mixup_criterion(loss_fn, outputs, targets_a, targets_b, lam)
                prior_loss = compute_locality_prior_loss(
                    model,
                    prior_block_indices,
                    sigma=args.prior_sigma,
                    exclude_cls=args.prior_exclude_cls,
                ) if args.use_locality_prior and prior_lambda > 0.0 else None
                total_loss = task_loss if prior_loss is None else task_loss + (prior_lambda * prior_loss)
                loss = total_loss / args.grad_accum_steps
                loss.backward()
                should_step = ((batch_idx + 1) % args.grad_accum_steps == 0) or ((batch_idx + 1) == num_steps)
                if should_step:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    for optimizer in optimizers:
                        optimizer.step()
                    if ema is not None and ema_active:
                        ema.update(model)

            epoch_loss += loss.item() * args.grad_accum_steps
            epoch_task_loss += task_loss.item()
            if prior_loss is not None:
                epoch_prior_loss += prior_loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}] Step [{batch_idx + 1}] "
                    f"Loss: {loss.item() * args.grad_accum_steps:.4f} | "
                    f"Time: {time.time() - step_start:.3f}s"
                )

        eval_model = ema.module if (ema is not None and ema_active) else model
        val_acc, val_loss = validate(eval_model, test_loader, device)
        if scheduler_euc is not None:
            scheduler_euc.step()
        if scheduler_man is not None:
            scheduler_man.step()

        epoch_time = time.time() - epoch_start
        eval_source = "ema" if (ema is not None and ema_active) else "model"
        geo_diagnostics = collect_geo_diagnostics(model)
        prior_diagnostics = compute_locality_prior_diagnostics(
            model,
            prior_block_indices,
            exclude_cls=args.prior_exclude_cls,
        ) if args.use_locality_prior else {}
        print(
            f"Epoch {epoch + 1} Done | Eval: {eval_source} | "
            f"Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f} | "
            f"Geo Scale: {geo_adapter_scale:.2f} | Prior Lambda: {prior_lambda:.4f} | Epoch Time: {epoch_time:.1f}s"
        )

        if val_acc > best_acc + 1e-12:
            best_acc = val_acc
            best_epoch = epoch

        checkpoint = {
            "epoch": epoch,
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "opt_euclidean_state_dict": trainer.opt_euclidean.state_dict(),
            "eval_weights": eval_source,
            "ema_active": ema_active,
            "acc": val_acc,
            "val_loss": val_loss,
            "best_acc": best_acc,
            "best_epoch": best_epoch,
        }
        if trainer.opt_manifold is not None:
            checkpoint["opt_manifold_state_dict"] = trainer.opt_manifold.state_dict()
        if scheduler_man is not None:
            checkpoint["scheduler_man_state_dict"] = scheduler_man.state_dict()
        if scheduler_euc is not None:
            checkpoint["scheduler_euc_state_dict"] = scheduler_euc.state_dict()
        if scaler:
            checkpoint["scaler_state_dict"] = scaler.state_dict()
        if ema is not None:
            checkpoint["ema_state_dict"] = ema.module.state_dict()

        torch.save(checkpoint, os.path.join(args.save_dir, "latest.pth"))
        if best_epoch == epoch:
            torch.save(checkpoint, os.path.join(args.save_dir, "best.pth"))
        metrics_row = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss / num_steps,
            "train_task_loss": epoch_task_loss / num_steps,
            "train_prior_loss": epoch_prior_loss / max(num_steps, 1),
            "val_acc": val_acc,
            "val_loss": val_loss,
            "best_acc": best_acc,
            "best_epoch": best_epoch + 1,
            "eval_weights": eval_source,
            "model_variant": model_config["model_variant"],
            "total_params": total_params,
            "trainable_params": trainable_params,
            "geo_adapter_scale": round(geo_adapter_scale, 4),
            "prior_lambda": round(prior_lambda, 6),
            "lr_manifold": scheduler_man.get_last_lr()[0] if scheduler_man is not None else 0.0,
            "lr_euclidean": scheduler_euc.get_last_lr()[0] if scheduler_euc is not None else 0.0,
            "epoch_time_sec": round(epoch_time, 3),
        }
        metrics_row.update({f"geo_{key}": round(value, 6) for key, value in geo_diagnostics.items()})
        metrics_row.update({f"prior_{key}": round(value, 6) for key, value in prior_diagnostics.items()})
        append_metrics_row(metrics_path, metrics_row)

        if (epoch + 1) % args.save_every == 0:
            torch.save(checkpoint, os.path.join(args.save_dir, f"geovit_e{epoch + 1}.pth"))

        if ema is not None and not ema_active and (epoch + 1) >= args.ema_start_epoch:
            ema.set(model)
            ema_active = True
            print(f"EMA activated after epoch {epoch + 1}.")

        if (epoch + 1) >= args.drop_epoch and best_acc < args.min_best_acc_by_drop_epoch:
            print(
                f"EARLY STOP: best val acc {best_acc:.2f}% by epoch {epoch + 1} "
                f"is below required {args.min_best_acc_by_drop_epoch:.2f}%."
            )
            break

        if (
            (epoch + 1) >= args.plateau_start_epoch
            and best_epoch >= 0
            and (epoch - best_epoch) >= args.plateau_patience
            and (best_acc - val_acc) <= args.plateau_min_delta
        ):
            print(
                f"EARLY STOP: plateau detected. No new best for {epoch - best_epoch} epochs, "
                f"current {val_acc:.2f}% vs best {best_acc:.2f}%."
            )
            break


if __name__ == "__main__":
    main()
