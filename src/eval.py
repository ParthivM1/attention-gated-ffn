import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_factory import add_model_args, build_model, config_from_args, config_from_checkpoint, count_parameters


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--checkpoint_dir", type=str, default="/root/checkpoints")
    parser.add_argument("--checkpoint_name", type=str, default=None, help="Specific .pth file to load")
    parser.add_argument("--eval_weights", type=str, choices=("auto", "model", "ema"), default="auto")
    add_model_args(parser)
    args = parser.parse_args()

    device = get_device()
    print(f"Evaluating on device: {device}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ]
    )

    if args.dataset == "cifar100":
        test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
        num_classes = 100
    else:
        test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        num_classes = 10

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.checkpoint_name:
        ckpt_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    else:
        print(f"Searching for checkpoints in {args.checkpoint_dir}...")
        if not os.path.exists(args.checkpoint_dir):
            print("Checkpoint directory not found.")
            return

        files = [f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pth")]
        if not files:
            print("No checkpoints found in volume.")
            return

        try:
            files.sort(key=lambda x: int(x.split("_e")[-1].split(".")[0]))
        except Exception:
            files.sort()

        ckpt_path = os.path.join(args.checkpoint_dir, files[-1])

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_config = checkpoint.get("model_config", {})
    if checkpoint_config:
        model_config = config_from_checkpoint(checkpoint_config, num_classes=num_classes)
        print(f"Restoring model from checkpoint config: {model_config['model_variant']}")
    else:
        model_config = config_from_args(args, num_classes=num_classes)
        print(f"Checkpoint has no model config. Using CLI/default config: {model_config['model_variant']}")

    model = build_model(model_config).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Model parameters: total={total_params:,} trainable={trainable_params:,}")

    if "model_state_dict" in checkpoint:
        if args.eval_weights == "model":
            state_key = "model_state_dict"
        elif args.eval_weights == "ema":
            if "ema_state_dict" not in checkpoint:
                raise ValueError("Checkpoint does not contain EMA weights. Re-run with --eval_weights model or auto.")
            state_key = "ema_state_dict"
        else:
            state_key = "ema_state_dict" if "ema_state_dict" in checkpoint else "model_state_dict"
        model.load_state_dict(checkpoint[state_key])
        epoch = checkpoint.get("epoch", -1)
        train_acc = checkpoint.get("acc", 0.0)
        print(f"   -> Loaded {state_key} from Epoch {epoch + 1} (val acc at save: {train_acc:.2f}%)")
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 20 == 0:
                print(f"Batch {batch_idx + 1}/{len(test_loader)}: Acc so far: {100.0 * correct / total:.2f}%")

    acc = 100.0 * correct / total
    avg_loss = total_loss / len(test_loader)

    print("=" * 40)
    print(f"Results for checkpoint: {os.path.basename(ckpt_path)}")
    print(f"Test Accuracy: {acc:.2f}%")
    print(f"Test Loss:     {avg_loss:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
