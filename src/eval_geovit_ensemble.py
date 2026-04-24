import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from train_geovit import add_args, build_cifar100, build_model, build_transforms, get_class_texts, load_checkpoint_state, maybe_subset
except ImportError:
    from src.train_geovit import add_args, build_cifar100, build_model, build_transforms, get_class_texts, load_checkpoint_state, maybe_subset


def add_eval_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = add_args(parser)
    parser.add_argument("--student-checkpoints", nargs="+", required=True)
    return parser


def build_eval_views(images: torch.Tensor, *, hflip_tta: bool = False, shift_tta: int = 0) -> list[torch.Tensor]:
    views = [images]
    base_views = [images]
    if hflip_tta:
        flipped = torch.flip(images, dims=[-1])
        views.append(flipped)
        base_views.append(flipped)
    if shift_tta > 0:
        pad = int(shift_tta)
        _, _, height, width = images.shape
        offsets = [(-pad, 0), (pad, 0), (0, -pad), (0, pad)]
        for base in base_views:
            padded = F.pad(base, (pad, pad, pad, pad), mode="reflect")
            for dx, dy in offsets:
                x0 = pad + dx
                y0 = pad + dy
                views.append(padded[:, :, y0 : y0 + height, x0 : x0 + width])
    return views


def infer_model_args_from_checkpoint(args, checkpoint_path: str):
    ckpt = load_checkpoint_state(checkpoint_path, map_location="cpu")
    summary = ckpt.get("summary", {})
    model_config = summary.get("model_config", ckpt.get("model_config", {}))
    for key, value in model_config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return ckpt


def load_model_from_checkpoint(args, checkpoint_path: str, class_texts: list[str] | None):
    ckpt = load_checkpoint_state(checkpoint_path, map_location="cpu")
    state = ckpt.get("ema_state_dict") or ckpt.get("model_state_dict")
    model = build_model(args, class_texts=class_texts).to(args.device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def evaluate_ensemble(models, loader, device, *, hflip_tta: bool = False, shift_tta: int = 0):
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            views = build_eval_views(images, hflip_tta=hflip_tta, shift_tta=shift_tta)
            model_logits = []
            for model in models:
                logits = torch.stack([model(view) for view in views], dim=0).mean(dim=0)
                model_logits.append(logits)
            ensemble_logits = torch.stack(model_logits, dim=0).mean(dim=0)
            loss = F.cross_entropy(ensemble_logits, labels)
            total_loss += float(loss.item()) * int(labels.shape[0])
            total_correct += int((ensemble_logits.argmax(dim=1) == labels).sum().item())
            total_count += int(labels.shape[0])
    elapsed = time.perf_counter() - start
    return (
        total_loss / max(total_count, 1),
        100.0 * total_correct / max(total_count, 1),
        elapsed,
    )


def main():
    parser = add_eval_args(argparse.ArgumentParser(description="Evaluate a GeoViT ensemble on CIFAR-100."))
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    infer_model_args_from_checkpoint(args, args.student_checkpoints[0])

    _, test_transform = build_transforms(args.img_size, strong_aug=not args.disable_strong_aug)
    test_ds = maybe_subset(build_cifar100(args.data_root, train=False, transform=test_transform), args.limit_test)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=(0 if args.num_workers <= 0 else max(1, args.num_workers // 2)),
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 1,
        prefetch_factor=(
            args.eval_prefetch_factor
            if (args.num_workers > 1 and max(1, args.num_workers // 2) > 0)
            else None
        ),
    )
    class_texts = get_class_texts(test_ds)
    models = [load_model_from_checkpoint(args, path, class_texts) for path in args.student_checkpoints]
    test_loss, test_acc, elapsed = evaluate_ensemble(
        models,
        test_loader,
        args.device,
        hflip_tta=args.eval_hflip_tta,
        shift_tta=args.eval_shift_tta,
    )

    run_dir = Path(args.save_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": args.run_name,
        "student_checkpoints": list(args.student_checkpoints),
        "num_models": len(models),
        "test_loss": round(test_loss, 6),
        "best_acc": round(test_acc, 4),
        "elapsed_seconds": round(elapsed, 3),
        "eval_hflip_tta": args.eval_hflip_tta,
        "eval_shift_tta": args.eval_shift_tta,
        "device": args.device,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
