from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.train_geovit as tg


def log_step(log_path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def main() -> None:
    parser = tg.add_args(argparse.ArgumentParser(description="Local smoke runner for GeoViT architecture screens"))
    args = parser.parse_args()
    tg.apply_locked_smoke_protocol(args)
    tg.set_seed(args.seed)
    run_dir = Path(args.save_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    step_log = run_dir / "debug_steps.log"
    if step_log.exists():
        step_log.unlink()
    log_step(step_log, "parsed args")

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    log_step(step_log, f"device setup complete ({args.device})")

    train_transform, test_transform = tg.build_transforms(args.img_size, strong_aug=not args.disable_strong_aug)
    log_step(step_log, "built transforms")
    train_ds = tg.maybe_subset(tg.build_cifar100(args.data_root, train=True, transform=train_transform), args.limit_train)
    test_ds = tg.maybe_subset(tg.build_cifar100(args.data_root, train=False, transform=test_transform), args.limit_test)
    log_step(step_log, f"loaded datasets train={len(train_ds)} test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=False,
    )
    log_step(step_log, "built dataloaders")

    model = tg.build_model(args).to(args.device)
    log_step(step_log, "built model")
    total_params, trainable_params = tg.count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = tg.build_scheduler(
        optimizer,
        total_epochs=args.epochs,
        warmup_epochs=min(args.warmup_epochs, max(args.epochs - 1, 0)),
        warmup_start_factor=args.warmup_start_factor,
        min_lr_ratio=args.min_lr_ratio,
    )
    scaler = None
    if args.device.startswith("cuda") and not args.disable_amp:
        amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    condition_table = None
    if args.use_conditioner and not args.use_class_text_router:
        condition_table = tg.build_condition_labels(100, args.condition_dim, torch.device(args.device))
    log_step(step_log, "optimizer/scheduler ready")

    history: list[dict[str, float | int | str]] = []
    best_acc = 0.0
    best_epoch = 0
    best_state = None
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        adapter_scale = tg.compute_geo_adapter_scale(
            epoch,
            start_epoch=args.geo_start_epoch,
            ramp_epochs=args.geo_ramp_epochs,
            target_scale=args.geo_target_scale,
            end_scale=args.geo_end_scale,
            decay_start_epoch=args.geo_decay_start_epoch,
        )
        spectral_scale_mult = tg.compute_geo_adapter_scale(
            epoch,
            start_epoch=args.geo_start_epoch,
            ramp_epochs=args.geo_ramp_epochs,
            target_scale=1.0,
            end_scale=args.geo_spectral_end_mult if args.geo_spectral_end_mult >= 0 else 1.0,
            decay_start_epoch=args.geo_spectral_decay_start_epoch if args.geo_spectral_decay_start_epoch > 0 else args.epochs + 1,
        )
        low_rank_scale_mult = tg.compute_geo_adapter_scale(
            epoch,
            start_epoch=args.geo_start_epoch,
            ramp_epochs=args.geo_ramp_epochs,
            target_scale=1.0,
            end_scale=args.geo_low_rank_end_mult,
            decay_start_epoch=args.geo_low_rank_decay_start_epoch,
        )
        rotation_scale_mult = tg.compute_geo_adapter_scale(
            epoch,
            start_epoch=args.geo_start_epoch,
            ramp_epochs=args.geo_ramp_epochs,
            target_scale=1.0,
            end_scale=args.geo_rotation_end_mult if args.geo_rotation_end_mult >= 0 else 1.0,
            decay_start_epoch=args.geo_rotation_decay_start_epoch if args.geo_rotation_decay_start_epoch > 0 else args.epochs + 1,
        )
        tg.set_geo_adapter_scale(model, adapter_scale)
        tg.set_geo_component_scale_multipliers(
            model,
            spectral=spectral_scale_mult,
            low_rank=low_rank_scale_mult,
            rotation=rotation_scale_mult,
        )

        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for images, labels in train_loader:
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            condition = tg.build_condition_vectors(labels, condition_table)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = model(images, condition)
                    loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
                scaler.scale(loss).backward()
                if args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images, condition)
                loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
                loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()

            total_loss += float(loss.item()) * int(labels.shape[0])
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_count += int(labels.shape[0])

        train_loss = total_loss / max(total_count, 1)
        train_acc = 100.0 * total_correct / max(total_count, 1)
        test_loss, test_acc = tg.evaluate(model, test_loader, args.device)
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        diagnostics = model.get_diagnostics()
        scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_start
        log_step(step_log, f"finished epoch {epoch} test_acc={test_acc:.4f}")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "best_test_acc": best_acc,
                "epoch_seconds": epoch_seconds,
                "adapter_scale": adapter_scale,
                **diagnostics,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "epoch",
                    "epoch": epoch,
                    "train_acc": round(train_acc, 4),
                    "test_acc": round(test_acc, 4),
                    "best_acc": round(best_acc, 4),
                    "adapter_scale": round(adapter_scale, 4),
                    "diagnostics": diagnostics,
                }
            ),
            flush=True,
        )

    log_step(step_log, "writing outputs")
    comparison_protocol = tg.build_comparison_protocol(args)
    architecture_signature = tg.build_architecture_signature(args)
    payload = {
        "run_name": args.run_name,
        "seed": args.seed,
        "variant": tg.build_variant_name(args),
        "locked_smoke_protocol": str(args.locked_smoke_protocol).strip(),
        "comparison_protocol": comparison_protocol,
        "comparison_protocol_hash": tg.stable_hash(comparison_protocol),
        "architecture_signature": architecture_signature,
        "architecture_signature_hash": tg.stable_hash(architecture_signature),
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "best_acc": round(best_acc, 4),
        "best_epoch": best_epoch,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_config": {
            "img_size": args.img_size,
            "patch_size": args.patch_size,
            "embed_dim": args.embed_dim,
            "depth": args.depth,
            "num_heads": args.num_heads,
            "fc1_base_rank": args.fc1_base_rank,
            "spectral_scale": args.spectral_scale,
            "low_rank_scale": args.low_rank_scale,
            "rotation_scale": args.rotation_scale,
            "enable_local_geo": args.enable_local_geo,
            "coupled_spectral_low_rank": bool(getattr(args, "coupled_spectral_low_rank", False)),
            "coupled_learnable_input_basis": bool(getattr(args, "coupled_learnable_input_basis", False)),
            "coupled_shared_gate": bool(getattr(args, "coupled_shared_gate", False)),
        },
        "history": history,
        "last_diagnostics": model.get_diagnostics(),
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(run_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    if best_state is not None:
        torch.save(
            {
                "model_state_dict": best_state,
                "summary": payload,
            },
            run_dir / "best.pth",
        )
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
