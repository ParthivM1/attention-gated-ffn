import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100

try:
    from layers.geo_operator_linear import GeoOperatorLinear
    from layers.operator_bank import FactorizedLinear
    from layers.flow_ffn import compute_flow_ffn_strength, set_flow_ffn_strength
    from models.geovit import GeoVisionTransformer
    from models.vit import VisionTransformer
    from models.flow_depth_vit import FlowDepthViT
except ImportError:
    from src.layers.geo_operator_linear import GeoOperatorLinear
    from src.layers.operator_bank import FactorizedLinear
    from src.layers.flow_ffn import compute_flow_ffn_strength, set_flow_ffn_strength
    from src.models.geovit import GeoVisionTransformer
    from src.models.vit import VisionTransformer
    from src.models.flow_depth_vit import FlowDepthViT


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def build_transforms(image_size: int, *, strong_aug: bool = True):
    cifar_mean = (0.5071, 0.4867, 0.4408)
    cifar_std = (0.2675, 0.2565, 0.2761)
    train_ops = [
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if strong_aug:
        train_ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    train_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar_mean, std=cifar_std),
        ]
    )
    if strong_aug:
        train_ops.append(transforms.RandomErasing(p=0.25, value="random"))
    train_transform = transforms.Compose(train_ops)
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar_mean, std=cifar_std),
        ]
    )
    return train_transform, test_transform


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    return -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    temp = max(float(temperature), 1e-6)
    student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temp * temp)


def feature_distill_loss(student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
    student_norm = F.normalize(student_features, dim=-1)
    teacher_norm = F.normalize(teacher_features, dim=-1)
    return 1.0 - F.cosine_similarity(student_norm, teacher_norm, dim=-1).mean()


def block_feature_distill_loss(
    student_blocks: list[torch.Tensor],
    teacher_blocks: list[torch.Tensor],
    *,
    last_k: int = 0,
) -> torch.Tensor:
    if not student_blocks or not teacher_blocks:
        raise ValueError("Block feature lists must be non-empty for block distillation.")
    count = min(len(student_blocks), len(teacher_blocks))
    if last_k > 0:
        count = min(count, last_k)
        student_seq = student_blocks[-count:]
        teacher_seq = teacher_blocks[-count:]
    else:
        student_seq = student_blocks[:count]
        teacher_seq = teacher_blocks[:count]
    losses = [feature_distill_loss(student_feat, teacher_feat) for student_feat, teacher_feat in zip(student_seq, teacher_seq)]
    return torch.stack(losses).mean()


def load_checkpoint_state(path: str, map_location="cpu") -> dict:
    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict) and ("model_state_dict" in checkpoint or "ema_state_dict" in checkpoint):
        return checkpoint
    if isinstance(checkpoint, dict):
        return {"model_state_dict": checkpoint}
    raise ValueError(f"Unsupported checkpoint format at {path}")


@torch.no_grad()
def load_student_checkpoint_partial(model: nn.Module, path: str, map_location="cpu") -> dict[str, int | str]:
    checkpoint = load_checkpoint_state(path, map_location=map_location)
    source_state = checkpoint.get("ema_state_dict") or checkpoint.get("model_state_dict") or {}
    model_state = model.state_dict()
    loaded = 0
    missing_in_model = 0
    shape_mismatch = 0
    for name, value in source_state.items():
        if name not in model_state:
            missing_in_model += 1
            continue
        if model_state[name].shape != value.shape:
            shape_mismatch += 1
            continue
        model_state[name].copy_(value)
        loaded += 1
    missing_in_checkpoint = sum(1 for name in model_state if name not in source_state)
    return {
        "path": path,
        "loaded_tensors": loaded,
        "missing_in_model": missing_in_model,
        "shape_mismatch": shape_mismatch,
        "missing_in_checkpoint": missing_in_checkpoint,
    }


@torch.no_grad()
def _factorized_copy_from_dense(factorized: FactorizedLinear, dense: nn.Linear) -> None:
    weight = dense.weight.detach().to(dtype=torch.float32)
    rank = factorized.rank
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    u = u[:, :rank]
    s = s[:rank]
    vh = vh[:rank, :]
    factorized.left.weight.copy_(vh.to(dtype=factorized.left.weight.dtype))
    factorized.right.weight.copy_((u * s.unsqueeze(0)).to(dtype=factorized.right.weight.dtype))
    if factorized.right.bias is not None:
        if dense.bias is not None:
            factorized.right.bias.copy_(dense.bias.detach().to(dtype=factorized.right.bias.dtype))
        else:
            factorized.right.bias.zero_()


@torch.no_grad()
def initialize_geo_linear_from_dense(module: GeoOperatorLinear, dense: nn.Linear, gate_bias: float = -4.0) -> None:
    if isinstance(module.base, FactorizedLinear):
        _factorized_copy_from_dense(module.base, dense)
    elif isinstance(module.base, nn.Linear):
        module.base.weight.copy_(dense.weight.detach().to(dtype=module.base.weight.dtype))
        if module.base.bias is not None:
            if dense.bias is not None:
                module.base.bias.copy_(dense.bias.detach().to(dtype=module.base.bias.dtype))
            else:
                module.base.bias.zero_()

    module.local_gate_head.weight.zero_()
    module.local_gate_head.bias.fill_(gate_bias)
    module.rotation_gate_head.weight.zero_()
    module.rotation_gate_head.bias.fill_(gate_bias)
    module.rotation_coeff_head.weight.zero_()
    module.rotation_coeff_head.bias.zero_()
    module.spectral_gate_head.weight.zero_()
    module.spectral_gate_head.bias.zero_()
    module.low_rank_head.weight.zero_()
    module.low_rank_head.bias.zero_()
    if getattr(module, "flow_gate_head", None) is not None:
        module.flow_gate_head.weight.zero_()
        flow_gate_bias = gate_bias + 2.0
        if getattr(module, "manifold_alignment_mode", False):
            flow_gate_bias = gate_bias + 3.0
        module.flow_gate_head.bias.fill_(flow_gate_bias)
    if getattr(module, "flow_head", None) is not None:
        module.flow_head.weight.zero_()
        module.flow_head.bias.zero_()
    if getattr(module, "semantic_expert_head", None) is not None:
        nn.init.trunc_normal_(module.semantic_expert_head.weight, std=0.02)
        module.semantic_expert_head.bias.copy_(
            torch.linspace(
                -0.15,
                0.15,
                steps=module.semantic_expert_head.bias.numel(),
                device=module.semantic_expert_head.bias.device,
                dtype=module.semantic_expert_head.bias.dtype,
            )
        )
    if getattr(module, "semantic_mode_head", None) is not None:
        module.semantic_mode_head.weight.zero_()
        module.semantic_mode_head.bias.fill_(0.25)
    if getattr(module, "semantic_gain_head", None) is not None:
        nn.init.trunc_normal_(module.semantic_gain_head.weight, std=0.02)
        module.semantic_gain_head.bias.zero_()
    if getattr(module, "semantic_gate_head", None) is not None:
        module.semantic_gate_head.weight.zero_()
        module.semantic_gate_head.bias.fill_(gate_bias + 5.0)
    if getattr(module, "flow_operator", None) is not None:
        warm_modes = min(
            int(getattr(module, "flow_rank", 0)),
            int(getattr(module, "num_spectral_bases", 0)),
            int(module.spectral_gate_head.bias.numel()),
        )
        if warm_modes > 0:
            module.spectral_gate_head.bias[:warm_modes].fill_(0.1)
    if getattr(module, "magnus_operator", None) is not None:
        module.magnus_operator.initialize_from_weight(dense.weight)


@torch.no_grad()
def initialize_shared_spectral_bank_from_teacher(student: nn.Module, teacher: nn.Module) -> None:
    try:
        shared_bank = student.blocks[0].mlp.fc1.shared_bank
    except Exception:
        return
    if shared_bank is None or not hasattr(shared_bank, "spectral_operator"):
        return

    spectral_operator = shared_bank.spectral_operator
    input_basis = spectral_operator.input_basis.to(dtype=torch.float32)
    teacher_weights = []
    for teacher_block in teacher.blocks:
        teacher_fc1 = teacher_block.mlp.fc1
        if isinstance(teacher_fc1, nn.Linear):
            teacher_weights.append(teacher_fc1.weight.detach().to(dtype=torch.float32))
    if not teacher_weights:
        return

    projected = torch.stack([input_basis @ weight.T for weight in teacher_weights], dim=0).mean(dim=0)
    spectral_operator.output_basis.copy_(projected.to(dtype=spectral_operator.output_basis.dtype))


@torch.no_grad()
def initialize_student_from_teacher(
    student: nn.Module,
    teacher: nn.Module,
    *,
    gate_bias: float = -4.0,
) -> None:
    student_state = student.state_dict()
    teacher_state = teacher.state_dict()
    copied = 0
    for name, value in teacher_state.items():
        if name in student_state and student_state[name].shape == value.shape:
            student_state[name].copy_(value)
            copied += 1
    for student_block, teacher_block in zip(student.blocks, teacher.blocks):
        student_fc1 = student_block.mlp.fc1
        teacher_fc1 = teacher_block.mlp.fc1
        if isinstance(student_fc1, GeoOperatorLinear) and isinstance(teacher_fc1, nn.Linear):
            initialize_geo_linear_from_dense(student_fc1, teacher_fc1, gate_bias=gate_bias)
    initialize_shared_spectral_bank_from_teacher(student, teacher)
    print(json.dumps({"stage": "teacher_init", "copied_tensors": copied, "gate_bias": gate_bias}), flush=True)


@torch.no_grad()
def warm_start_manifold_from_anchor(
    model: nn.Module,
    *,
    gate_bias: float = -1.25,
    flow_scale: float = 1.0,
    warm_flow: bool = True,
    warm_semantic: bool = False,
) -> dict[str, float | int]:
    updated_blocks = 0
    copied_flow_heads = 0
    copied_gate_heads = 0
    copied_semantic_modes = 0
    copied_semantic_gates = 0
    copied_semantic_experts = 0
    flow_scale = float(flow_scale)
    for block in getattr(model, "blocks", []):
        fc1 = getattr(getattr(block, "mlp", None), "fc1", None)
        if fc1 is None:
            continue
        spectral_head = getattr(fc1, "spectral_gate_head", None)
        flow_head = getattr(fc1, "flow_head", None)
        flow_gate_head = getattr(fc1, "flow_gate_head", None)
        semantic_mode_head = getattr(fc1, "semantic_mode_head", None)
        semantic_gate_head = getattr(fc1, "semantic_gate_head", None)
        semantic_expert_head = getattr(fc1, "semantic_expert_head", None)
        semantic_gain_head = getattr(fc1, "semantic_gain_head", None)
        if warm_flow and spectral_head is not None and flow_head is not None:
            rows = min(flow_head.weight.shape[0], spectral_head.weight.shape[0])
            flow_head.weight.zero_()
            flow_head.bias.zero_()
            flow_head.weight[:rows].copy_((flow_scale * spectral_head.weight[:rows].detach()).to(dtype=flow_head.weight.dtype))
            flow_head.bias[:rows].copy_((flow_scale * spectral_head.bias[:rows].detach()).to(dtype=flow_head.bias.dtype))
            copied_flow_heads += 1
        if warm_flow and spectral_head is not None and flow_gate_head is not None:
            mean_weight = flow_scale * spectral_head.weight.detach().mean(dim=0, keepdim=True)
            mean_bias = float(flow_scale * spectral_head.bias.detach().mean().item())
            flow_gate_head.weight.copy_(mean_weight.to(dtype=flow_gate_head.weight.dtype))
            flow_gate_head.bias.fill_(max(mean_bias, gate_bias))
            copied_gate_heads += 1
        if warm_semantic and spectral_head is not None and semantic_mode_head is not None:
            rows = min(semantic_mode_head.weight.shape[0], spectral_head.weight.shape[0])
            semantic_mode_head.weight.zero_()
            semantic_mode_head.bias.zero_()
            semantic_mode_head.weight[:rows].copy_((flow_scale * spectral_head.weight[:rows].detach()).to(dtype=semantic_mode_head.weight.dtype))
            semantic_mode_head.bias[:rows].copy_((flow_scale * spectral_head.bias[:rows].detach() + 0.25).to(dtype=semantic_mode_head.bias.dtype))
            copied_semantic_modes += 1
        if warm_semantic and spectral_head is not None and semantic_gate_head is not None:
            mean_weight = flow_scale * spectral_head.weight.detach().mean(dim=0, keepdim=True)
            mean_bias = float(flow_scale * spectral_head.bias.detach().mean().item())
            semantic_gate_head.weight.copy_(mean_weight.to(dtype=semantic_gate_head.weight.dtype))
            semantic_gate_head.bias.fill_(max(mean_bias, -0.25))
            copied_semantic_gates += 1
        if warm_semantic and semantic_expert_head is not None:
            semantic_expert_head.weight.zero_()
            semantic_expert_head.bias.copy_(
                torch.linspace(
                    0.35,
                    -0.35,
                    steps=semantic_expert_head.bias.numel(),
                    device=semantic_expert_head.bias.device,
                    dtype=semantic_expert_head.bias.dtype,
                )
            )
            copied_semantic_experts += 1
        if warm_semantic and semantic_gain_head is not None:
            semantic_gain_head.weight.zero_()
            semantic_gain_head.bias.zero_()
        updated_blocks += 1
    return {
        "updated_blocks": updated_blocks,
        "copied_flow_heads": copied_flow_heads,
        "copied_gate_heads": copied_gate_heads,
        "copied_semantic_modes": copied_semantic_modes,
        "copied_semantic_gates": copied_semantic_gates,
        "copied_semantic_experts": copied_semantic_experts,
        "flow_gate_bias": gate_bias,
        "flow_scale": flow_scale,
    }


def configure_trainable_parameters(
    model: nn.Module,
    *,
    head_probe_only: bool = False,
    manifold_refine_only: bool = False,
    manifold_train_head: bool = False,
    manifold_train_output_basis: bool = False,
    manifold_train_text_router: bool = False,
    geo_refine_only: bool = False,
    geo_train_head: bool = False,
    geo_train_output_basis: bool = False,
) -> dict[str, object]:
    if not head_probe_only and not manifold_refine_only and not geo_refine_only:
        total_params, trainable_params = count_parameters(model)
        return {
            "mode": "full",
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_tensors": sum(1 for _, param in model.named_parameters() if param.requires_grad),
        }

    trainable_names = []
    for _, param in model.named_parameters():
        param.requires_grad_(False)
    for name, param in model.named_parameters():
        allow = False
        if head_probe_only:
            allow = name.startswith("head.") or name.startswith("norm.")
        if manifold_refine_only:
            allow = any(
                pattern in name
                for pattern in [
                    "magnus_operator.",
                    "flow_operator.",
                    "flow_gate_head.",
                    "flow_head.",
                    "semantic_operator.",
                    "semantic_expert_head.",
                    "semantic_mode_head.",
                    "semantic_gain_head.",
                    "semantic_gate_head.",
                ]
            )
            if not allow and manifold_train_head:
                allow = name.startswith("head.") or name.startswith("norm.")
            if not allow and manifold_train_output_basis:
                allow = (
                    "spectral_operator.output_basis" in name
                    or "magnus_operator.output_basis" in name
                    or "flow_operator.output_basis" in name
                )
            if not allow and manifold_train_text_router:
                allow = name.startswith("class_text_router.")
        if geo_refine_only:
            if not allow:
                allow = any(
                    pattern in name
                    for pattern in [
                        "magnus_operator.",
                        "local_gate_head.",
                        "rotation_gate_head.",
                        "rotation_coeff_head.",
                        "spectral_gate_head.",
                        "low_rank_head.",
                    ]
                )
            if not allow and geo_train_head:
                allow = name.startswith("head.") or name.startswith("norm.")
            if not allow and geo_train_output_basis:
                allow = "spectral_operator.output_basis" in name or "magnus_operator.output_basis" in name
        if allow:
            param.requires_grad_(True)
            trainable_names.append(name)

    total_params, trainable_params = count_parameters(model)
    return {
        "mode": (
            "head_probe_only"
            if head_probe_only and not manifold_refine_only and not geo_refine_only
            else "manifold_refine_only"
            if manifold_refine_only and not geo_refine_only
            else "geo_refine_only"
            if geo_refine_only and not manifold_refine_only
            else "hybrid_refine_only"
        ),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_tensors": len(trainable_names),
        "trainable_names": trainable_names,
    }


def resolve_training_phase(args, epoch: int) -> dict[str, object]:
    head_warmup_epochs = max(int(args.geo_head_warmup_epochs), 0)
    basis_last_epochs = max(int(args.geo_basis_last_epochs), 0) if args.geo_refine_only else 0
    basis_start_epoch = max(head_warmup_epochs + 1, args.epochs - basis_last_epochs + 1) if basis_last_epochs > 0 else args.epochs + 1

    phase = {
        "head_probe_only": False,
        "manifold_refine_only": args.manifold_refine_only,
        "manifold_train_head": args.manifold_train_head,
        "manifold_train_output_basis": args.manifold_train_output_basis,
        "manifold_train_text_router": args.manifold_train_text_router,
        "geo_refine_only": args.geo_refine_only,
        "geo_train_head": args.geo_train_head,
        "geo_train_output_basis": args.geo_train_output_basis,
        "name": "configured",
        "start_epoch": 1,
        "end_epoch": args.epochs,
        "lr": args.lr,
    }

    if head_warmup_epochs > 0 and epoch <= head_warmup_epochs:
        phase.update(
            {
                "head_probe_only": True,
                "manifold_refine_only": False,
                "manifold_train_head": False,
                "manifold_train_output_basis": False,
                "manifold_train_text_router": False,
                "geo_refine_only": False,
                "geo_train_head": False,
                "geo_train_output_basis": False,
                "name": "head_probe_warmup",
                "start_epoch": 1,
                "end_epoch": head_warmup_epochs,
                "lr": args.head_probe_lr if args.head_probe_lr > 0 else args.lr,
            }
        )
        return phase

    if args.geo_refine_only:
        phase["start_epoch"] = head_warmup_epochs + 1
        phase["name"] = "geo_refine"
        if basis_last_epochs > 0 and epoch >= basis_start_epoch:
            phase["geo_train_output_basis"] = True
            phase["name"] = "geo_refine_basis"
            phase["start_epoch"] = basis_start_epoch
            phase["end_epoch"] = args.epochs
        else:
            phase["end_epoch"] = basis_start_epoch - 1 if basis_last_epochs > 0 else args.epochs
        return phase

    if args.manifold_refine_only:
        phase["start_epoch"] = head_warmup_epochs + 1
        phase["name"] = "manifold_refine"
        return phase

    if head_warmup_epochs > 0:
        phase["start_epoch"] = head_warmup_epochs + 1
        phase["name"] = "full_after_warmup"
    else:
        phase["name"] = "full"
    return phase


def build_optimizer_and_scheduler(
    model: nn.Module,
    args,
    *,
    phase: dict[str, object],
) -> tuple[dict[str, object], list[torch.nn.Parameter], torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    trainable_info = configure_trainable_parameters(
        model,
        head_probe_only=bool(phase["head_probe_only"]),
        manifold_refine_only=bool(phase["manifold_refine_only"]),
        manifold_train_head=bool(phase["manifold_train_head"]),
        manifold_train_output_basis=bool(phase["manifold_train_output_basis"]),
        manifold_train_text_router=bool(phase["manifold_train_text_router"]),
        geo_refine_only=bool(phase["geo_refine_only"]),
        geo_train_head=bool(phase["geo_train_head"]),
        geo_train_output_basis=bool(phase["geo_train_output_basis"]),
    )
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters remain after applying the training configuration.")
    phase_epochs = max(int(phase["end_epoch"]) - int(phase["start_epoch"]) + 1, 1)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(phase["lr"]),
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        total_epochs=phase_epochs,
        warmup_epochs=min(args.warmup_epochs, max(phase_epochs - 1, 0)),
        warmup_start_factor=args.warmup_start_factor,
        min_lr_ratio=args.min_lr_ratio,
    )
    return trainable_info, trainable_parameters, optimizer, scheduler


def build_condition_labels(num_classes: int, condition_dim: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    table = torch.randn(num_classes, condition_dim, generator=generator)
    table = F.normalize(table, dim=-1)
    return table.to(device)


def build_condition_vectors(labels: torch.Tensor, condition_table: torch.Tensor | None) -> torch.Tensor | None:
    if condition_table is None:
        return None
    return condition_table[labels]


def build_variant_name(args) -> str:
    tokenizer_suffix = "_detail" if getattr(args, "tokenizer_type", "standard") == "budgeted_detail" else ""
    has_token_flow = bool(getattr(args, "token_flow_input", False) or int(getattr(args, "token_flow_last_k_blocks", 0)) > 0)
    token_flow_suffix = ""
    if has_token_flow:
        token_flow_suffix = "_tokflow" if bool(getattr(args, "share_token_flow", True)) else "_tokflowns"
    hidden_grid_suffix = ""
    if bool(getattr(args, "hidden_grid_refiner", False)) and int(getattr(args, "hidden_grid_refiner_last_k_blocks", 0)) > 0:
        hidden_grid_suffix = "_hgrid" if bool(getattr(args, "share_hidden_grid_refiner", False)) else "_hgridns"
    hidden_cls_suffix = ""
    if bool(getattr(args, "hidden_cls_bridge", False)) and int(getattr(args, "hidden_cls_bridge_last_k_blocks", 0)) > 0:
        hidden_cls_suffix = "_hcls" if bool(getattr(args, "share_hidden_cls_bridge", False)) else "_hclsns"
    hidden_channel_suffix = ""
    if bool(getattr(args, "hidden_channel_flow", False)) and int(getattr(args, "hidden_channel_flow_last_k_blocks", 0)) > 0:
        hidden_channel_suffix = "_hcflow" if bool(getattr(args, "share_hidden_channel_flow", False)) else "_hcflowns"
    response_flow_suffix = ""
    if bool(getattr(args, "response_flow_norm", False)) and int(getattr(args, "response_flow_last_k_blocks", 0)) > 0:
        response_flow_mode = str(getattr(args, "response_flow_mode", "simple")).strip().lower()
        response_flow_suffix = "_rflowbx" if response_flow_mode == "biaxial" else "_rflow"
        if bool(getattr(args, "share_response_flow", False)):
            response_flow_suffix = f"{response_flow_suffix}s"
        if bool(getattr(args, "response_flow_pre_act", False)):
            response_flow_suffix = f"{response_flow_suffix}pre"
    hidden_group_router_suffix = ""
    if bool(getattr(args, "hidden_group_router", False)) and int(getattr(args, "hidden_group_router_last_k_blocks", 0)) > 0:
        hidden_group_router_suffix = f"_hgr{int(getattr(args, 'hidden_group_router_last_k_blocks', 0))}"
    attention_hidden_fusion_suffix = ""
    if bool(getattr(args, "attention_hidden_fusion", False)) and int(getattr(args, "attention_hidden_fusion_last_k_blocks", 0)) > 0:
        attention_hidden_fusion_suffix = "_ahfs" if bool(getattr(args, "share_attention_hidden_fusion", True)) else "_ahf"
        attention_hidden_fusion_suffix = f"{attention_hidden_fusion_suffix}{int(getattr(args, 'attention_hidden_fusion_last_k_blocks', 0))}"
    hidden_token_mixer_suffix = ""
    if bool(getattr(args, "hidden_token_mixer", False)) and int(getattr(args, "hidden_token_mixer_last_k_blocks", 0)) > 0:
        hidden_token_mixer_suffix = "_htms" if bool(getattr(args, "share_hidden_token_mixer", True)) else "_htm"
        if str(getattr(args, "hidden_token_mixer_mode", "conv")).strip().lower() == "sparse":
            hidden_token_mixer_suffix = f"{hidden_token_mixer_suffix}sp"
        hidden_token_mixer_suffix = f"{hidden_token_mixer_suffix}{int(getattr(args, 'hidden_token_mixer_last_k_blocks', 0))}"
    competitive_residual_suffix = ""
    if bool(getattr(args, "competitive_residual", False)) and int(getattr(args, "competitive_residual_last_k_blocks", 0)) > 0:
        competitive_residual_suffix = f"_crm{int(getattr(args, 'competitive_residual_last_k_blocks', 0))}"
    parallel_block_suffix = ""
    if bool(getattr(args, "parallel_block_update", False)) and int(getattr(args, "parallel_block_last_k_blocks", 0)) > 0:
        parallel_block_suffix = f"_par{int(getattr(args, 'parallel_block_last_k_blocks', 0))}"
    mlp_first_suffix = ""
    if bool(getattr(args, "mlp_first_update", False)) and int(getattr(args, "mlp_first_last_k_blocks", 0)) > 0:
        mlp_first_suffix = f"_mf{int(getattr(args, 'mlp_first_last_k_blocks', 0))}"
    tail_token_suffix = ""
    if bool(getattr(args, "tail_token_mixer", False)) and int(getattr(args, "tail_token_mixer_last_k_blocks", 0)) > 0:
        tail_token_suffix = f"_ttm{int(getattr(args, 'tail_token_mixer_last_k_blocks', 0))}"
    activation_flow_suffix = ""
    if bool(getattr(args, "activation_flow", False)) and int(getattr(args, "activation_flow_last_k_blocks", 0)) > 0:
        activation_flow_suffix = "_aflow" if bool(getattr(args, "share_activation_flow", False)) else "_aflowns"
    inter_layer_flow_suffix = ""
    if bool(getattr(args, "inter_layer_flow", False)) and int(getattr(args, "inter_layer_flow_last_k_blocks", 0)) > 0:
        flow_mode = str(getattr(args, "inter_layer_flow_mode", "transport")).strip().lower()
        flow_tag = "ilsum" if flow_mode == "summary" else "ilflow"
        inter_layer_flow_suffix = f"_{flow_tag}" if bool(getattr(args, "share_inter_layer_flow", True)) else f"_{flow_tag}ns"
    flow_state_suffix = ""
    if bool(getattr(args, "flow_state_carrier", False)) and int(getattr(args, "flow_state_last_k_blocks", 0)) > 0:
        flow_state_suffix = "_fstate" if bool(getattr(args, "share_flow_state_carrier", True)) else "_fstates"
    grid_refine_suffix = "_grid" if bool(getattr(args, "patch_grid_refiner", False)) else ""
    attn_metric_suffix = ""
    if bool(getattr(args, "attention_metric_adapter", False)):
        attn_metric_type = str(getattr(args, "attention_metric_type", "linear")).strip().lower()
        attn_metric_suffix = "_attngrid" if attn_metric_type == "grid" else "_attnmetric"
    fc1_late_suffix = "_fc1late" if int(getattr(args, "geo_fc1_last_k_blocks", 0)) > 0 else ""
    local_geo_suffix = ""
    if int(getattr(args, "local_geo_last_k_blocks", 0)) > 0:
        local_geo_suffix = f"_lg{int(getattr(args, 'local_geo_last_k_blocks', 0))}"
    low_rank_tail_suffix = ""
    if int(getattr(args, "geo_low_rank_last_k_blocks", 0)) > 0:
        low_rank_tail_suffix = f"_lrk{int(getattr(args, 'geo_low_rank_last_k_blocks', 0))}"
    dual_path_suffix = ""
    if bool(getattr(args, "dual_path_mlp", False)) and int(getattr(args, "dual_path_last_k_blocks", 0)) > 0:
        dual_path_suffix = f"_dual{int(getattr(args, 'dual_path_last_k_blocks', 0))}"
    block_profile_suffix = ""
    geo_block_profile = str(getattr(args, "geo_block_profile", "uniform")).strip().lower()
    if geo_block_profile and geo_block_profile != "uniform":
        block_profile_suffix = f"_gb{geo_block_profile}"
    if bool(getattr(args, "geo_learnable_block_scale", False)):
        block_profile_suffix = f"{block_profile_suffix}_gblearn"
    bank_group_suffix = ""
    if bool(getattr(args, "share_fc1_bank", True)) and int(getattr(args, "fc1_bank_groups", 1)) > 1:
        bank_group_suffix = f"_bg{int(getattr(args, 'fc1_bank_groups', 1))}"
    conditioner_group_suffix = ""
    if bool(getattr(args, "share_fc1_conditioner", True)) and int(getattr(args, "fc1_conditioner_groups", 1)) > 1:
        conditioner_group_suffix = f"_cg{int(getattr(args, 'fc1_conditioner_groups', 1))}"
    group_suffix = f"{bank_group_suffix}{conditioner_group_suffix}"
    coupled_suffix = ""
    if bool(getattr(args, "coupled_spectral_low_rank", False)):
        coupled_suffix = "_cpl"
        if bool(getattr(args, "coupled_learnable_input_basis", False)):
            coupled_suffix = "_cpll"
        if bool(getattr(args, "coupled_shared_gate", False)):
            coupled_suffix = f"{coupled_suffix}sg"
    if getattr(args, "magnus_single_operator", False):
        return f"geovit_magnus_single_fc1{fc1_late_suffix}{local_geo_suffix}{low_rank_tail_suffix}{dual_path_suffix}{hidden_group_router_suffix}{attention_hidden_fusion_suffix}{hidden_token_mixer_suffix}{competitive_residual_suffix}{parallel_block_suffix}{mlp_first_suffix}{tail_token_suffix}{block_profile_suffix}{group_suffix}{coupled_suffix}{token_flow_suffix}{hidden_grid_suffix}{hidden_cls_suffix}{hidden_channel_suffix}{response_flow_suffix}{activation_flow_suffix}{inter_layer_flow_suffix}{flow_state_suffix}{grid_refine_suffix}{attn_metric_suffix}{tokenizer_suffix}"
    if getattr(args, "magnus_semantic_mode", False):
        return f"geovit_magnus_fc1{fc1_late_suffix}{local_geo_suffix}{low_rank_tail_suffix}{dual_path_suffix}{hidden_group_router_suffix}{attention_hidden_fusion_suffix}{hidden_token_mixer_suffix}{competitive_residual_suffix}{parallel_block_suffix}{mlp_first_suffix}{tail_token_suffix}{block_profile_suffix}{group_suffix}{coupled_suffix}{token_flow_suffix}{hidden_grid_suffix}{hidden_cls_suffix}{hidden_channel_suffix}{response_flow_suffix}{activation_flow_suffix}{inter_layer_flow_suffix}{flow_state_suffix}{grid_refine_suffix}{attn_metric_suffix}{tokenizer_suffix}"
    if getattr(args, "geo_attention_last_k_blocks", 0) > 0:
        return f"geovit_fc1{fc1_late_suffix}{local_geo_suffix}{low_rank_tail_suffix}{dual_path_suffix}{hidden_group_router_suffix}{attention_hidden_fusion_suffix}{hidden_token_mixer_suffix}{competitive_residual_suffix}{parallel_block_suffix}{mlp_first_suffix}{tail_token_suffix}{block_profile_suffix}{group_suffix}{coupled_suffix}_attnlate{token_flow_suffix}{hidden_grid_suffix}{hidden_cls_suffix}{hidden_channel_suffix}{response_flow_suffix}{activation_flow_suffix}{inter_layer_flow_suffix}{flow_state_suffix}{grid_refine_suffix}{attn_metric_suffix}{tokenizer_suffix}"
    if args.geo_on_attention:
        return f"geovit_attn{fc1_late_suffix}{local_geo_suffix}{low_rank_tail_suffix}{dual_path_suffix}{hidden_group_router_suffix}{attention_hidden_fusion_suffix}{hidden_token_mixer_suffix}{competitive_residual_suffix}{parallel_block_suffix}{mlp_first_suffix}{tail_token_suffix}{block_profile_suffix}{group_suffix}{coupled_suffix}{token_flow_suffix}{hidden_grid_suffix}{hidden_cls_suffix}{hidden_channel_suffix}{response_flow_suffix}{activation_flow_suffix}{inter_layer_flow_suffix}{flow_state_suffix}{grid_refine_suffix}{attn_metric_suffix}{tokenizer_suffix}"
    if args.geo_on_fc1 and args.geo_on_fc2:
        return f"geovit_fc1{fc1_late_suffix}{local_geo_suffix}{low_rank_tail_suffix}{dual_path_suffix}{hidden_group_router_suffix}{attention_hidden_fusion_suffix}{hidden_token_mixer_suffix}{competitive_residual_suffix}{parallel_block_suffix}{mlp_first_suffix}{tail_token_suffix}{block_profile_suffix}{group_suffix}{coupled_suffix}_fc2{token_flow_suffix}{hidden_grid_suffix}{hidden_cls_suffix}{hidden_channel_suffix}{response_flow_suffix}{activation_flow_suffix}{inter_layer_flow_suffix}{flow_state_suffix}{grid_refine_suffix}{attn_metric_suffix}{tokenizer_suffix}"
    if args.geo_on_fc1:
        return f"geovit_fc1{fc1_late_suffix}{local_geo_suffix}{low_rank_tail_suffix}{dual_path_suffix}{hidden_group_router_suffix}{attention_hidden_fusion_suffix}{hidden_token_mixer_suffix}{competitive_residual_suffix}{parallel_block_suffix}{mlp_first_suffix}{tail_token_suffix}{block_profile_suffix}{group_suffix}{coupled_suffix}{token_flow_suffix}{hidden_grid_suffix}{hidden_cls_suffix}{hidden_channel_suffix}{response_flow_suffix}{activation_flow_suffix}{inter_layer_flow_suffix}{flow_state_suffix}{grid_refine_suffix}{attn_metric_suffix}{tokenizer_suffix}"
    if token_flow_suffix:
        return f"vit{token_flow_suffix}{hidden_grid_suffix}{hidden_cls_suffix}{hidden_channel_suffix}{response_flow_suffix}{activation_flow_suffix}{inter_layer_flow_suffix}{flow_state_suffix}{tokenizer_suffix}"
    if hidden_grid_suffix:
        return f"vit{hidden_grid_suffix}{hidden_cls_suffix}{hidden_channel_suffix}{response_flow_suffix}{activation_flow_suffix}{inter_layer_flow_suffix}{flow_state_suffix}{tokenizer_suffix}"
    if hidden_cls_suffix:
        return f"vit{hidden_cls_suffix}{hidden_channel_suffix}{response_flow_suffix}{activation_flow_suffix}{inter_layer_flow_suffix}{flow_state_suffix}{tokenizer_suffix}"
    if hidden_channel_suffix:
        return f"vit{hidden_channel_suffix}{response_flow_suffix}{activation_flow_suffix}{inter_layer_flow_suffix}{flow_state_suffix}{tokenizer_suffix}"
    if response_flow_suffix:
        return f"vit{response_flow_suffix}{activation_flow_suffix}{inter_layer_flow_suffix}{flow_state_suffix}{tokenizer_suffix}"
    if activation_flow_suffix:
        return f"vit{activation_flow_suffix}{inter_layer_flow_suffix}{flow_state_suffix}{tokenizer_suffix}"
    if inter_layer_flow_suffix:
        return f"vit{inter_layer_flow_suffix}{flow_state_suffix}{tokenizer_suffix}"
    if flow_state_suffix:
        return f"vit{flow_state_suffix}{tokenizer_suffix}"
    if attn_metric_suffix:
        return f"vit{attn_metric_suffix}{tokenizer_suffix}"
    return f"plain_vit{tokenizer_suffix}"


def set_geo_adapter_scale(model: nn.Module, scale: float) -> None:
    for module in model.modules():
        setter = getattr(module, "set_adapter_scale", None)
        if callable(setter):
            setter(scale)


def set_geo_component_scale_multipliers(
    model: nn.Module,
    *,
    spectral: float = 1.0,
    low_rank: float = 1.0,
    rotation: float = 1.0,
) -> None:
    for module in model.modules():
        setter = getattr(module, "set_component_scale_multipliers", None)
        if callable(setter):
            setter(spectral=spectral, low_rank=low_rank, rotation=rotation)


def compute_geo_adapter_scale(
    epoch: int,
    start_epoch: int,
    ramp_epochs: int,
    target_scale: float,
    *,
    total_epochs: int | None = None,
    end_scale: float = -1.0,
    decay_start_epoch: int = 0,
) -> float:
    current_epoch = int(epoch)
    if current_epoch <= int(start_epoch):
        return 0.0
    ramp_target = float(target_scale)
    if int(ramp_epochs) <= 0:
        scale = ramp_target
    else:
        progress = (current_epoch - int(start_epoch)) / float(max(int(ramp_epochs), 1))
        scale = float(min(max(progress, 0.0), 1.0) * ramp_target)
    if end_scale < 0 or total_epochs is None or int(decay_start_epoch) <= 0:
        return float(scale)
    effective_decay_start = max(int(decay_start_epoch), int(start_epoch) + int(ramp_epochs))
    if current_epoch <= effective_decay_start:
        return float(scale)
    if int(total_epochs) <= effective_decay_start:
        return float(end_scale)
    decay_progress = (current_epoch - effective_decay_start) / float(max(int(total_epochs) - effective_decay_start, 1))
    decay_progress = min(max(decay_progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return float(end_scale + (ramp_target - end_scale) * cosine)


def compute_scheduled_weight(epoch: int, start_epoch: int, ramp_epochs: int, target_weight: float) -> float:
    return compute_geo_adapter_scale(
        epoch,
        start_epoch=start_epoch,
        ramp_epochs=ramp_epochs,
        target_scale=target_weight,
    )


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def maybe_subset(dataset, limit: int):
    if limit and limit < len(dataset):
        return Subset(dataset, list(range(limit)))
    return dataset


def build_cifar100(root: str, *, train: bool, transform):
    try:
        return CIFAR100(root=root, train=train, transform=transform, download=False)
    except RuntimeError:
        return CIFAR100(root=root, train=train, transform=transform, download=True)


def unwrap_dataset(dataset):
    base = dataset
    while isinstance(base, Subset):
        base = base.dataset
    return base


def get_class_texts(dataset) -> list[str] | None:
    base = unwrap_dataset(dataset)
    classes = getattr(base, "classes", None)
    if not classes:
        return None
    return [f"a photo of a {str(name).replace('_', ' ').replace('-', ' ')}" for name in classes]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


LOCKED_SMOKE_PROTOCOLS: dict[str, dict[str, object]] = {
    "mid192d6_e40": {
        "img_size": 32,
        "patch_size": 4,
        "embed_dim": 192,
        "depth": 6,
        "num_heads": 6,
        "drop_path_rate": 0.1,
        "limit_train": 2000,
        "limit_test": 500,
        "epochs": 40,
        "batch_size": 96,
        "eval_batch_size": 192,
        "num_workers": 4,
        "prefetch_factor": 4,
        "eval_prefetch_factor": 2,
        "lr": 3e-4,
        "warmup_epochs": 5,
        "min_lr_ratio": 0.05,
        "weight_decay": 0.05,
        "label_smoothing": 0.05,
        "disable_strong_aug": True,
        "mixup_alpha": 0.0,
        "cutmix_alpha": 0.0,
        "mix_prob": 0.0,
        "cutmix_switch_prob": 0.5,
        "ema_decay": 0.0,
        "teacher_checkpoint": "",
        "anchor_checkpoint": "",
        "student_checkpoint": "",
        "teacher_init_student": False,
        "teacher_logit_weight": 0.0,
        "teacher_feature_weight": 0.0,
        "teacher_block_feature_weight": 0.0,
        "teacher_block_feature_layers": 0,
        "anchor_logit_weight": 0.0,
        "anchor_feature_weight": 0.0,
        "anchor_block_feature_weight": 0.0,
        "anchor_block_feature_layers": 0,
        "semantic_anchor_weight": 0.0,
        "semantic_expert_weight": 0.0,
        "semantic_mode_weight": 0.0,
        "geo_structure_weight": 0.0,
        "flow_anchor_weight": 0.0,
        "flow_energy_weight": 0.0,
        "magnus_anchor_weight": 0.0,
        "magnus_motion_weight": 0.0,
        "magnus_comm_weight": 0.0,
        "magnus_mode_weight": 0.0,
        "use_conditioner": False,
        "use_internal_conditioner": True,
        "use_class_text_router": False,
        "attention_metric_adapter": False,
        "attention_metric_type": "linear",
        "attention_metric_scale": 1.0,
        "attention_metric_last_k_blocks": 0,
        "share_attention_metric": True,
        "attention_metric_bottleneck": 8,
        "attention_metric_patch_only": True,
        "attention_metric_gate_bias": -3.0,
        "attention_metric_init_scale": 0.01,
        "attention_metric_cls_context_scale": 0.25,
        "eval_only": False,
        "eval_hflip_tta": False,
        "eval_shift_tta": 0,
        "eval_diagonal_shift_tta": False,
        "disable_amp": False,
        "amp_dtype": "float16",
        "grad_clip_norm": 0.0,
        "tokenizer_type": "standard",
        "detail_tokens": 0,
        "detail_score_type": "variance",
        "geo_on_fc2": False,
        "geo_on_attention": False,
        "geo_fc1_last_k_blocks": 0,
        "fc1_bank_groups": 1,
        "fc1_conditioner_groups": 1,
        "geo_attention_last_k_blocks": 0,
    },
}


def apply_locked_smoke_protocol(args) -> None:
    protocol_name = str(getattr(args, "locked_smoke_protocol", "")).strip()
    if not protocol_name:
        return
    if protocol_name not in LOCKED_SMOKE_PROTOCOLS:
        known = ", ".join(sorted(LOCKED_SMOKE_PROTOCOLS))
        raise ValueError(f"Unknown locked smoke protocol '{protocol_name}'. Known: {known}")
    raw_argv = set(sys.argv[1:])
    preserve_flags = {
        "img_size": {"--img-size"},
        "patch_size": {"--patch-size"},
        "embed_dim": {"--embed-dim"},
        "depth": {"--depth"},
        "num_heads": {"--num-heads"},
        "mlp_ratio": {"--mlp-ratio"},
        "drop_path_rate": {"--drop-path-rate"},
        "use_conv_stem": {"--use-conv-stem", "--no-conv-stem"},
        "stem_channels": {"--stem-channels"},
        "stem_flow_refiner": {"--stem-flow-refiner"},
        "stem_flow_scale": {"--stem-flow-scale"},
        "stem_flow_bottleneck": {"--stem-flow-bottleneck"},
        "stem_flow_gate_bias": {"--stem-flow-gate-bias"},
        "stem_flow_init_scale": {"--stem-flow-init-scale"},
        "stem_flow_detail_scale": {"--stem-flow-detail-scale"},
        "stem_flow_context_scale": {"--stem-flow-context-scale"},
        "summary_token": {"--summary-token"},
        "summary_token_scale": {"--summary-token-scale"},
        "summary_head_fusion": {"--summary-head-fusion"},
        "use_conditioner": {"--use-conditioner"},
        "use_internal_conditioner": {"--use-internal-conditioner", "--no-internal-conditioner"},
        "use_class_text_router": {"--use-class-text-router"},
        "tokenizer_type": {"--tokenizer-type"},
        "detail_tokens": {"--detail-tokens"},
        "detail_score_type": {"--detail-score-type"},
        "batch_size": {"--batch-size"},
        "eval_batch_size": {"--eval-batch-size"},
        "num_workers": {"--num-workers"},
        "prefetch_factor": {"--prefetch-factor"},
        "eval_prefetch_factor": {"--eval-prefetch-factor"},
        "geo_on_fc1": {"--geo-on-fc1", "--no-geo-on-fc1"},
        "geo_fc1_last_k_blocks": {"--geo-fc1-last-k-blocks"},
        "flow_depth_mode": {"--flow-depth-mode"},
        "flow_depth_iterations": {"--flow-depth-iterations"},
        "flow_depth_inject": {"--flow-depth-inject"},
        "flow_depth_momentum": {"--flow-depth-momentum"},
        "flow_depth_momentum_beta": {"--flow-depth-momentum-beta"},
        "flow_depth_independent_blocks": {"--flow-depth-independent-blocks"},
        "flow_depth_time_conditioning": {"--flow-depth-time-conditioning"},
        "flow_depth_attn_pool": {"--flow-depth-attn-pool"},
        "agff_last_k_blocks": {"--agff-last-k-blocks"},
        "agff_gate_mode": {"--agff-gate-mode"},
        "agff_gate_ln": {"--agff-gate-ln", "--no-agff-gate-ln"},
        "agff_gate_init_scale": {"--agff-gate-init-scale"},
        "fnfl_last_k_blocks": {"--fnfl-last-k-blocks"},
        "fnfl_num_steps": {"--fnfl-num-steps"},
        "fnfl_rank": {"--fnfl-rank"},
        "fnfl_num_spectral_bases": {"--fnfl-num-spectral-bases"},
        "fnfl_low_rank": {"--fnfl-low-rank"},
        "fnfl_controller_hidden_dim": {"--fnfl-controller-hidden-dim"},
        "fnfl_strength_init": {"--fnfl-strength-init"},
        "fnfl_strength_anneal_epochs": {"--fnfl-strength-anneal-epochs"},
        "fnfl_strength_max": {"--fnfl-strength-max"},
        "gfn_last_k_blocks": {"--gfn-last-k-blocks"},
        "gfn_corr_bottleneck": {"--gfn-corr-bottleneck"},
        "gfn_n_train_iters": {"--gfn-n-train-iters"},
        "gfn_gate_init": {"--gfn-gate-init"},
        "geo_low_rank_last_k_blocks": {"--geo-low-rank-last-k-blocks"},
        "local_geo_last_k_blocks": {"--local-geo-last-k-blocks"},
        "fc1_bank_groups": {"--fc1-bank-groups"},
        "fc1_conditioner_groups": {"--fc1-conditioner-groups"},
        "coupled_spectral_low_rank": {"--coupled-spectral-low-rank"},
        "coupled_learnable_input_basis": {"--coupled-learnable-input-basis"},
        "coupled_shared_gate": {"--coupled-shared-gate"},
        "geo_on_fc2": {"--geo-on-fc2"},
        "geo_on_attention": {"--geo-on-attention"},
        "geo_attention_last_k_blocks": {"--geo-attention-last-k-blocks"},
        "hidden_diffusion": {"--hidden-diffusion"},
        "hidden_diffusion_last_k_blocks": {"--hidden-diffusion-last-k-blocks"},
        "share_hidden_diffusion": {"--share-hidden-diffusion"},
        "hidden_diffusion_bottleneck": {"--hidden-diffusion-bottleneck"},
        "hidden_diffusion_gate_bias": {"--hidden-diffusion-gate-bias"},
        "hidden_diffusion_init_scale": {"--hidden-diffusion-init-scale"},
        "hidden_diffusion_cls_context_scale": {"--hidden-diffusion-cls-context-scale"},
        "hidden_grid_refiner": {"--hidden-grid-refiner"},
        "hidden_grid_refiner_scale": {"--hidden-grid-refiner-scale"},
        "hidden_grid_refiner_last_k_blocks": {"--hidden-grid-refiner-last-k-blocks"},
        "share_hidden_grid_refiner": {"--share-hidden-grid-refiner", "--no-share-hidden-grid-refiner"},
        "hidden_grid_refiner_bottleneck": {"--hidden-grid-refiner-bottleneck"},
        "hidden_grid_refiner_gate_bias": {"--hidden-grid-refiner-gate-bias"},
        "hidden_grid_refiner_init_scale": {"--hidden-grid-refiner-init-scale"},
        "hidden_grid_refiner_cls_context_scale": {"--hidden-grid-refiner-cls-context-scale"},
        "hidden_cls_bridge": {"--hidden-cls-bridge"},
        "hidden_cls_bridge_scale": {"--hidden-cls-bridge-scale"},
        "hidden_cls_bridge_last_k_blocks": {"--hidden-cls-bridge-last-k-blocks"},
        "share_hidden_cls_bridge": {"--share-hidden-cls-bridge", "--no-share-hidden-cls-bridge"},
        "hidden_cls_bridge_bottleneck": {"--hidden-cls-bridge-bottleneck"},
        "hidden_cls_bridge_gate_bias": {"--hidden-cls-bridge-gate-bias"},
        "hidden_cls_bridge_init_scale": {"--hidden-cls-bridge-init-scale"},
        "hidden_cls_bridge_patch_feedback_scale": {"--hidden-cls-bridge-patch-feedback-scale"},
        "response_flow_norm": {"--response-flow-norm"},
        "response_flow_scale": {"--response-flow-scale"},
        "response_flow_last_k_blocks": {"--response-flow-last-k-blocks"},
        "response_flow_init_scale": {"--response-flow-init-scale"},
        "response_flow_mode": {"--response-flow-mode"},
        "share_response_flow": {"--share-response-flow", "--no-share-response-flow"},
        "response_flow_bottleneck": {"--response-flow-bottleneck"},
        "response_flow_gate_bias": {"--response-flow-gate-bias"},
        "response_flow_patch_only": {"--response-flow-patch-only", "--response-flow-all-tokens"},
        "response_flow_cls_mix_scale": {"--response-flow-cls-mix-scale"},
        "response_flow_mean_mix_scale": {"--response-flow-mean-mix-scale"},
        "response_flow_token_exponent": {"--response-flow-token-exponent"},
        "response_flow_channel_exponent": {"--response-flow-channel-exponent"},
        "response_flow_pre_act": {"--response-flow-pre-act"},
        "dual_path_mlp": {"--dual-path-mlp"},
        "dual_path_last_k_blocks": {"--dual-path-last-k-blocks"},
        "dual_path_refine_ratio": {"--dual-path-refine-ratio"},
        "dual_path_cross_scale": {"--dual-path-cross-scale"},
        "dual_path_gate_bias": {"--dual-path-gate-bias"},
        "hidden_group_router": {"--hidden-group-router"},
        "hidden_group_router_last_k_blocks": {"--hidden-group-router-last-k-blocks"},
        "hidden_group_router_groups": {"--hidden-group-router-groups"},
        "hidden_group_router_scale": {"--hidden-group-router-scale"},
        "hidden_group_router_gate_bias": {"--hidden-group-router-gate-bias"},
        "hidden_group_router_init_scale": {"--hidden-group-router-init-scale"},
        "hidden_group_router_cls_mix_scale": {"--hidden-group-router-cls-mix-scale"},
        "hidden_group_router_mean_mix_scale": {"--hidden-group-router-mean-mix-scale"},
        "attention_hidden_fusion": {"--attention-hidden-fusion"},
        "attention_hidden_fusion_last_k_blocks": {"--attention-hidden-fusion-last-k-blocks"},
        "share_attention_hidden_fusion": {"--share-attention-hidden-fusion", "--no-share-attention-hidden-fusion"},
        "attention_hidden_fusion_scale": {"--attention-hidden-fusion-scale"},
        "attention_hidden_fusion_bottleneck": {"--attention-hidden-fusion-bottleneck"},
        "attention_hidden_fusion_gate_bias": {"--attention-hidden-fusion-gate-bias"},
        "attention_hidden_fusion_init_scale": {"--attention-hidden-fusion-init-scale"},
        "attention_hidden_fusion_patch_only": {"--attention-hidden-fusion-patch-only", "--attention-hidden-fusion-all-tokens"},
        "attention_hidden_fusion_cls_context_scale": {"--attention-hidden-fusion-cls-context-scale"},
        "hidden_token_mixer": {"--hidden-token-mixer"},
        "hidden_token_mixer_last_k_blocks": {"--hidden-token-mixer-last-k-blocks"},
        "share_hidden_token_mixer": {"--share-hidden-token-mixer", "--no-share-hidden-token-mixer"},
        "hidden_token_mixer_scale": {"--hidden-token-mixer-scale"},
        "hidden_token_mixer_gate_bias": {"--hidden-token-mixer-gate-bias"},
        "hidden_token_mixer_init_scale": {"--hidden-token-mixer-init-scale"},
        "hidden_token_mixer_patch_only": {"--hidden-token-mixer-patch-only", "--hidden-token-mixer-all-tokens"},
        "hidden_token_mixer_mode": {"--hidden-token-mixer-mode"},
        "hidden_token_mixer_topk": {"--hidden-token-mixer-topk"},
        "competitive_residual": {"--competitive-residual"},
        "competitive_residual_last_k_blocks": {"--competitive-residual-last-k-blocks"},
        "competitive_residual_scale": {"--competitive-residual-scale"},
        "competitive_residual_gate_bias": {"--competitive-residual-gate-bias"},
        "competitive_residual_init_scale": {"--competitive-residual-init-scale"},
        "competitive_residual_cls_mix_scale": {"--competitive-residual-cls-mix-scale"},
        "competitive_residual_mean_mix_scale": {"--competitive-residual-mean-mix-scale"},
        "competitive_residual_patch_only": {"--competitive-residual-patch-only", "--competitive-residual-all-tokens"},
        "parallel_block_update": {"--parallel-block-update"},
        "parallel_block_last_k_blocks": {"--parallel-block-last-k-blocks"},
        "mlp_first_update": {"--mlp-first-update"},
        "mlp_first_last_k_blocks": {"--mlp-first-last-k-blocks"},
        "tail_token_mixer": {"--tail-token-mixer"},
        "tail_token_mixer_last_k_blocks": {"--tail-token-mixer-last-k-blocks"},
        "tail_token_mixer_scale": {"--tail-token-mixer-scale"},
        "tail_token_mixer_gate_bias": {"--tail-token-mixer-gate-bias"},
        "tail_token_mixer_init_scale": {"--tail-token-mixer-init-scale"},
        "tail_token_mixer_patch_only": {"--tail-token-mixer-patch-only", "--tail-token-mixer-all-tokens"},
        "activation_flow": {"--activation-flow"},
        "activation_flow_scale": {"--activation-flow-scale"},
        "activation_flow_last_k_blocks": {"--activation-flow-last-k-blocks"},
        "share_activation_flow": {"--share-activation-flow", "--no-share-activation-flow"},
        "activation_flow_bottleneck": {"--activation-flow-bottleneck"},
        "activation_flow_gate_bias": {"--activation-flow-gate-bias"},
        "activation_flow_init_scale": {"--activation-flow-init-scale"},
        "activation_flow_patch_only": {"--activation-flow-patch-only", "--activation-flow-all-tokens"},
        "activation_flow_cls_mix_scale": {"--activation-flow-cls-mix-scale"},
        "activation_flow_mean_mix_scale": {"--activation-flow-mean-mix-scale"},
        "activation_flow_std_mix_scale": {"--activation-flow-std-mix-scale"},
        "activation_flow_cls_token_scale": {"--activation-flow-cls-token-scale"},
        "attention_metric_adapter": {"--attention-metric-adapter"},
        "attention_metric_type": {"--attention-metric-type"},
        "attention_metric_scale": {"--attention-metric-scale"},
        "attention_metric_last_k_blocks": {"--attention-metric-last-k-blocks"},
        "share_attention_metric": {"--share-attention-metric", "--no-share-attention-metric"},
        "attention_metric_bottleneck": {"--attention-metric-bottleneck"},
        "attention_metric_patch_only": {"--attention-metric-patch-only", "--attention-metric-all-tokens"},
        "attention_metric_gate_bias": {"--attention-metric-gate-bias"},
        "attention_metric_init_scale": {"--attention-metric-init-scale"},
        "attention_metric_cls_context_scale": {"--attention-metric-cls-context-scale"},
        "geo_layer_scale_init": {"--geo-layer-scale-init"},
        "geo_block_profile": {"--geo-block-profile"},
        "geo_learnable_block_scale": {"--geo-learnable-block-scale"},
        "token_flow_input": {"--token-flow-input"},
        "token_flow_last_k_blocks": {"--token-flow-last-k-blocks"},
        "share_token_flow": {"--share-token-flow", "--no-share-token-flow"},
        "token_flow_scale": {"--token-flow-scale"},
        "token_flow_input_scale": {"--token-flow-input-scale"},
        "token_flow_block_scale": {"--token-flow-block-scale"},
        "token_flow_bottleneck": {"--token-flow-bottleneck"},
        "token_flow_patch_only": {"--token-flow-patch-only", "--token-flow-all-tokens"},
        "token_flow_detail_topk": {"--token-flow-detail-topk"},
        "token_flow_detail_boost_scale": {"--token-flow-detail-boost-scale"},
        "inter_layer_flow": {"--inter-layer-flow"},
        "inter_layer_flow_last_k_blocks": {"--inter-layer-flow-last-k-blocks"},
        "share_inter_layer_flow": {"--share-inter-layer-flow", "--no-share-inter-layer-flow"},
        "inter_layer_flow_mode": {"--inter-layer-flow-mode"},
        "inter_layer_flow_scale": {"--inter-layer-flow-scale"},
        "inter_layer_flow_bottleneck": {"--inter-layer-flow-bottleneck"},
        "inter_layer_flow_patch_only": {"--inter-layer-flow-patch-only", "--inter-layer-flow-all-tokens"},
        "inter_layer_flow_gate_bias": {"--inter-layer-flow-gate-bias"},
        "inter_layer_flow_init_scale": {"--inter-layer-flow-init-scale"},
        "inter_layer_flow_cls_context_scale": {"--inter-layer-flow-cls-context-scale"},
        "inter_layer_flow_delta_scale": {"--inter-layer-flow-delta-scale"},
        "flow_state_carrier": {"--flow-state-carrier"},
        "flow_state_last_k_blocks": {"--flow-state-last-k-blocks"},
        "share_flow_state_carrier": {"--share-flow-state-carrier", "--no-share-flow-state-carrier"},
        "flow_state_dim": {"--flow-state-dim"},
        "flow_state_scale": {"--flow-state-scale"},
        "flow_state_gate_bias": {"--flow-state-gate-bias"},
        "flow_state_init_scale": {"--flow-state-init-scale"},
        "flow_state_cls_scale": {"--flow-state-cls-scale"},
        "flow_state_patch_scale": {"--flow-state-patch-scale"},
    }
    for key, value in LOCKED_SMOKE_PROTOCOLS[protocol_name].items():
        if key in preserve_flags and any(flag in raw_argv for flag in preserve_flags[key]):
            continue
        setattr(args, key, value)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_comparison_protocol(args) -> dict[str, object]:
    return {
        "locked_smoke_protocol": str(getattr(args, "locked_smoke_protocol", "")).strip(),
        "img_size": int(args.img_size),
        "patch_size": int(args.patch_size),
        "embed_dim": int(args.embed_dim),
        "depth": int(args.depth),
        "num_heads": int(args.num_heads),
        "drop_path_rate": float(args.drop_path_rate),
        "limit_train": int(args.limit_train),
        "limit_test": int(args.limit_test),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "eval_prefetch_factor": int(args.eval_prefetch_factor),
        "lr": float(args.lr),
        "warmup_epochs": int(args.warmup_epochs),
        "warmup_start_factor": float(args.warmup_start_factor),
        "min_lr_ratio": float(args.min_lr_ratio),
        "weight_decay": float(args.weight_decay),
        "label_smoothing": float(args.label_smoothing),
        "disable_strong_aug": bool(args.disable_strong_aug),
        "mixup_alpha": float(args.mixup_alpha),
        "cutmix_alpha": float(args.cutmix_alpha),
        "mix_prob": float(args.mix_prob),
        "cutmix_switch_prob": float(args.cutmix_switch_prob),
        "ema_decay": float(args.ema_decay),
        "ema_start_epoch": int(args.ema_start_epoch),
        "grad_clip_norm": float(args.grad_clip_norm),
        "disable_amp": bool(args.disable_amp),
        "amp_dtype": str(args.amp_dtype),
        "eval_hflip_tta": bool(args.eval_hflip_tta),
        "eval_shift_tta": int(args.eval_shift_tta),
        "eval_diagonal_shift_tta": bool(args.eval_diagonal_shift_tta),
        "tokenizer_type": str(args.tokenizer_type),
        "detail_tokens": int(args.detail_tokens),
        "detail_score_type": str(args.detail_score_type),
        "teacher_init_student": bool(args.teacher_init_student),
        "teacher_logit_weight": float(args.teacher_logit_weight),
        "teacher_feature_weight": float(args.teacher_feature_weight),
        "teacher_block_feature_weight": float(args.teacher_block_feature_weight),
        "anchor_logit_weight": float(args.anchor_logit_weight),
        "anchor_feature_weight": float(args.anchor_feature_weight),
        "anchor_block_feature_weight": float(args.anchor_block_feature_weight),
        "use_conditioner": bool(args.use_conditioner),
        "use_internal_conditioner": bool(getattr(args, "use_internal_conditioner", True)),
        "use_class_text_router": bool(args.use_class_text_router),
        "seed": int(args.seed),
    }


def build_architecture_signature(args) -> dict[str, object]:
    return {
        "variant": build_variant_name(args),
        "fc1_base_rank": int(args.fc1_base_rank),
        "stem_flow_refiner": bool(getattr(args, "stem_flow_refiner", False)),
        "stem_flow_scale": float(getattr(args, "stem_flow_scale", 1.0)),
        "stem_flow_bottleneck": int(getattr(args, "stem_flow_bottleneck", 16)),
        "stem_flow_gate_bias": float(getattr(args, "stem_flow_gate_bias", -3.0)),
        "stem_flow_init_scale": float(getattr(args, "stem_flow_init_scale", 0.02)),
        "stem_flow_detail_scale": float(getattr(args, "stem_flow_detail_scale", 0.25)),
        "stem_flow_context_scale": float(getattr(args, "stem_flow_context_scale", 0.5)),
        "summary_token": bool(getattr(args, "summary_token", False)),
        "summary_token_scale": float(getattr(args, "summary_token_scale", 1.0)),
        "summary_head_fusion": float(getattr(args, "summary_head_fusion", 0.5)),
        "share_fc1_bank": bool(args.share_fc1_bank),
        "fc1_bank_groups": int(getattr(args, "fc1_bank_groups", 1)),
        "share_fc1_conditioner": bool(args.share_fc1_conditioner),
        "fc1_conditioner_groups": int(getattr(args, "fc1_conditioner_groups", 1)),
        "coupled_spectral_low_rank": bool(getattr(args, "coupled_spectral_low_rank", False)),
        "coupled_learnable_input_basis": bool(getattr(args, "coupled_learnable_input_basis", False)),
        "coupled_shared_gate": bool(getattr(args, "coupled_shared_gate", False)),
        "geo_on_fc1": bool(args.geo_on_fc1),
        "geo_fc1_last_k_blocks": int(getattr(args, "geo_fc1_last_k_blocks", 0)),
        "geo_low_rank_last_k_blocks": int(getattr(args, "geo_low_rank_last_k_blocks", 0)),
        "enable_local_geo": bool(args.enable_local_geo),
        "local_geo_last_k_blocks": int(getattr(args, "local_geo_last_k_blocks", 0)),
        "geo_on_fc2": bool(args.geo_on_fc2),
        "geo_on_attention": bool(args.geo_on_attention),
        "geo_attention_last_k_blocks": int(args.geo_attention_last_k_blocks),
        "residual_scale": float(args.residual_scale),
        "geo_residual_budget": float(args.geo_residual_budget),
        "spectral_scale": float(args.spectral_scale),
        "low_rank_scale": float(args.low_rank_scale),
        "rotation_scale": float(args.rotation_scale),
        "geo_start_epoch": int(args.geo_start_epoch),
        "geo_ramp_epochs": int(args.geo_ramp_epochs),
        "geo_target_scale": float(args.geo_target_scale),
        "geo_end_scale": float(args.geo_end_scale),
        "geo_decay_start_epoch": int(args.geo_decay_start_epoch),
        "geo_low_rank_end_mult": float(args.geo_low_rank_end_mult),
        "geo_low_rank_decay_start_epoch": int(args.geo_low_rank_decay_start_epoch),
        "hidden_diffusion": bool(args.hidden_diffusion),
        "hidden_diffusion_scale": float(args.hidden_diffusion_scale),
        "hidden_diffusion_last_k_blocks": int(args.hidden_diffusion_last_k_blocks),
        "share_hidden_diffusion": bool(getattr(args, "share_hidden_diffusion", False)),
        "hidden_diffusion_bottleneck": int(getattr(args, "hidden_diffusion_bottleneck", 32)),
        "hidden_diffusion_gate_bias": float(getattr(args, "hidden_diffusion_gate_bias", -4.0)),
        "hidden_diffusion_init_scale": float(getattr(args, "hidden_diffusion_init_scale", 0.005)),
        "hidden_diffusion_cls_context_scale": float(getattr(args, "hidden_diffusion_cls_context_scale", 0.1)),
        "hidden_grid_refiner": bool(getattr(args, "hidden_grid_refiner", False)),
        "hidden_grid_refiner_scale": float(getattr(args, "hidden_grid_refiner_scale", 1.0)),
        "hidden_grid_refiner_last_k_blocks": int(getattr(args, "hidden_grid_refiner_last_k_blocks", 0)),
        "share_hidden_grid_refiner": bool(getattr(args, "share_hidden_grid_refiner", False)),
        "hidden_grid_refiner_bottleneck": int(getattr(args, "hidden_grid_refiner_bottleneck", 16)),
        "hidden_grid_refiner_gate_bias": float(getattr(args, "hidden_grid_refiner_gate_bias", -5.0)),
        "hidden_grid_refiner_init_scale": float(getattr(args, "hidden_grid_refiner_init_scale", 0.002)),
        "hidden_grid_refiner_cls_context_scale": float(getattr(args, "hidden_grid_refiner_cls_context_scale", 0.05)),
        "hidden_cls_bridge": bool(getattr(args, "hidden_cls_bridge", False)),
        "hidden_cls_bridge_scale": float(getattr(args, "hidden_cls_bridge_scale", 1.0)),
        "hidden_cls_bridge_last_k_blocks": int(getattr(args, "hidden_cls_bridge_last_k_blocks", 0)),
        "share_hidden_cls_bridge": bool(getattr(args, "share_hidden_cls_bridge", False)),
        "hidden_cls_bridge_bottleneck": int(getattr(args, "hidden_cls_bridge_bottleneck", 16)),
        "hidden_cls_bridge_gate_bias": float(getattr(args, "hidden_cls_bridge_gate_bias", -4.0)),
        "hidden_cls_bridge_init_scale": float(getattr(args, "hidden_cls_bridge_init_scale", 0.01)),
        "hidden_cls_bridge_patch_feedback_scale": float(getattr(args, "hidden_cls_bridge_patch_feedback_scale", 0.0)),
        "hidden_channel_flow": bool(getattr(args, "hidden_channel_flow", False)),
        "hidden_channel_flow_scale": float(getattr(args, "hidden_channel_flow_scale", 1.0)),
        "hidden_channel_flow_last_k_blocks": int(getattr(args, "hidden_channel_flow_last_k_blocks", 0)),
        "share_hidden_channel_flow": bool(getattr(args, "share_hidden_channel_flow", False)),
        "hidden_channel_flow_bottleneck": int(getattr(args, "hidden_channel_flow_bottleneck", 16)),
        "hidden_channel_flow_rank": int(getattr(args, "hidden_channel_flow_rank", 16)),
        "hidden_channel_flow_gate_bias": float(getattr(args, "hidden_channel_flow_gate_bias", -3.5)),
        "hidden_channel_flow_init_scale": float(getattr(args, "hidden_channel_flow_init_scale", 0.01)),
        "hidden_channel_flow_patch_only": bool(getattr(args, "hidden_channel_flow_patch_only", False)),
        "hidden_channel_flow_cls_mix_scale": float(getattr(args, "hidden_channel_flow_cls_mix_scale", 1.0)),
        "hidden_channel_flow_mean_mix_scale": float(getattr(args, "hidden_channel_flow_mean_mix_scale", 0.5)),
        "response_flow_norm": bool(getattr(args, "response_flow_norm", False)),
        "response_flow_scale": float(getattr(args, "response_flow_scale", 1.0)),
        "response_flow_last_k_blocks": int(getattr(args, "response_flow_last_k_blocks", 0)),
        "response_flow_init_scale": float(getattr(args, "response_flow_init_scale", 0.01)),
        "response_flow_pre_act": bool(getattr(args, "response_flow_pre_act", False)),
        "dual_path_mlp": bool(getattr(args, "dual_path_mlp", False)),
        "dual_path_last_k_blocks": int(getattr(args, "dual_path_last_k_blocks", 0)),
        "dual_path_refine_ratio": float(getattr(args, "dual_path_refine_ratio", 0.25)),
        "dual_path_cross_scale": float(getattr(args, "dual_path_cross_scale", 1.0)),
        "dual_path_gate_bias": float(getattr(args, "dual_path_gate_bias", -2.0)),
        "hidden_group_router": bool(getattr(args, "hidden_group_router", False)),
        "hidden_group_router_last_k_blocks": int(getattr(args, "hidden_group_router_last_k_blocks", 0)),
        "hidden_group_router_groups": int(getattr(args, "hidden_group_router_groups", 0)),
        "hidden_group_router_scale": float(getattr(args, "hidden_group_router_scale", 1.0)),
        "hidden_group_router_gate_bias": float(getattr(args, "hidden_group_router_gate_bias", -2.0)),
        "hidden_group_router_init_scale": float(getattr(args, "hidden_group_router_init_scale", 0.1)),
        "hidden_group_router_cls_mix_scale": float(getattr(args, "hidden_group_router_cls_mix_scale", 1.0)),
        "hidden_group_router_mean_mix_scale": float(getattr(args, "hidden_group_router_mean_mix_scale", 0.5)),
        "attention_hidden_fusion": bool(getattr(args, "attention_hidden_fusion", False)),
        "attention_hidden_fusion_last_k_blocks": int(getattr(args, "attention_hidden_fusion_last_k_blocks", 0)),
        "share_attention_hidden_fusion": bool(getattr(args, "share_attention_hidden_fusion", True)),
        "attention_hidden_fusion_scale": float(getattr(args, "attention_hidden_fusion_scale", 1.0)),
        "attention_hidden_fusion_bottleneck": int(getattr(args, "attention_hidden_fusion_bottleneck", 6)),
        "attention_hidden_fusion_gate_bias": float(getattr(args, "attention_hidden_fusion_gate_bias", -2.5)),
        "attention_hidden_fusion_init_scale": float(getattr(args, "attention_hidden_fusion_init_scale", 0.02)),
        "attention_hidden_fusion_patch_only": bool(getattr(args, "attention_hidden_fusion_patch_only", False)),
        "attention_hidden_fusion_cls_context_scale": float(getattr(args, "attention_hidden_fusion_cls_context_scale", 1.0)),
        "hidden_token_mixer": bool(getattr(args, "hidden_token_mixer", False)),
        "hidden_token_mixer_last_k_blocks": int(getattr(args, "hidden_token_mixer_last_k_blocks", 0)),
        "share_hidden_token_mixer": bool(getattr(args, "share_hidden_token_mixer", True)),
        "hidden_token_mixer_scale": float(getattr(args, "hidden_token_mixer_scale", 1.0)),
        "hidden_token_mixer_gate_bias": float(getattr(args, "hidden_token_mixer_gate_bias", -3.0)),
        "hidden_token_mixer_init_scale": float(getattr(args, "hidden_token_mixer_init_scale", 0.02)),
        "hidden_token_mixer_patch_only": bool(getattr(args, "hidden_token_mixer_patch_only", True)),
        "hidden_token_mixer_mode": str(getattr(args, "hidden_token_mixer_mode", "conv")),
        "hidden_token_mixer_topk": int(getattr(args, "hidden_token_mixer_topk", 8)),
        "competitive_residual": bool(getattr(args, "competitive_residual", False)),
        "competitive_residual_last_k_blocks": int(getattr(args, "competitive_residual_last_k_blocks", 0)),
        "competitive_residual_scale": float(getattr(args, "competitive_residual_scale", 1.0)),
        "competitive_residual_gate_bias": float(getattr(args, "competitive_residual_gate_bias", 0.0)),
        "competitive_residual_init_scale": float(getattr(args, "competitive_residual_init_scale", 1.0)),
        "competitive_residual_cls_mix_scale": float(getattr(args, "competitive_residual_cls_mix_scale", 1.0)),
        "competitive_residual_mean_mix_scale": float(getattr(args, "competitive_residual_mean_mix_scale", 0.5)),
        "competitive_residual_patch_only": bool(getattr(args, "competitive_residual_patch_only", False)),
        "parallel_block_update": bool(getattr(args, "parallel_block_update", False)),
        "parallel_block_last_k_blocks": int(getattr(args, "parallel_block_last_k_blocks", 0)),
        "mlp_first_update": bool(getattr(args, "mlp_first_update", False)),
        "mlp_first_last_k_blocks": int(getattr(args, "mlp_first_last_k_blocks", 0)),
        "tail_token_mixer": bool(getattr(args, "tail_token_mixer", False)),
        "tail_token_mixer_last_k_blocks": int(getattr(args, "tail_token_mixer_last_k_blocks", 0)),
        "tail_token_mixer_scale": float(getattr(args, "tail_token_mixer_scale", 1.0)),
        "tail_token_mixer_gate_bias": float(getattr(args, "tail_token_mixer_gate_bias", -3.0)),
        "tail_token_mixer_init_scale": float(getattr(args, "tail_token_mixer_init_scale", 0.02)),
        "tail_token_mixer_patch_only": bool(getattr(args, "tail_token_mixer_patch_only", True)),
        "activation_flow": bool(getattr(args, "activation_flow", False)),
        "activation_flow_scale": float(getattr(args, "activation_flow_scale", 1.0)),
        "activation_flow_last_k_blocks": int(getattr(args, "activation_flow_last_k_blocks", 0)),
        "share_activation_flow": bool(getattr(args, "share_activation_flow", False)),
        "activation_flow_bottleneck": int(getattr(args, "activation_flow_bottleneck", 16)),
        "activation_flow_gate_bias": float(getattr(args, "activation_flow_gate_bias", -4.0)),
        "activation_flow_init_scale": float(getattr(args, "activation_flow_init_scale", 0.01)),
        "activation_flow_patch_only": bool(getattr(args, "activation_flow_patch_only", False)),
        "activation_flow_cls_mix_scale": float(getattr(args, "activation_flow_cls_mix_scale", 1.0)),
        "activation_flow_mean_mix_scale": float(getattr(args, "activation_flow_mean_mix_scale", 0.5)),
        "activation_flow_std_mix_scale": float(getattr(args, "activation_flow_std_mix_scale", 0.25)),
        "activation_flow_cls_token_scale": float(getattr(args, "activation_flow_cls_token_scale", 1.0)),
        "attn_flow_modulator": bool(getattr(args, "attn_flow_modulator", False)),
        "attn_flow_scale": float(getattr(args, "attn_flow_scale", 1.0)),
        "attn_flow_last_k_blocks": int(getattr(args, "attn_flow_last_k_blocks", 0)),
        "share_attn_flow_modulator": bool(getattr(args, "share_attn_flow_modulator", False)),
        "attn_flow_bottleneck": int(getattr(args, "attn_flow_bottleneck", 24)),
        "attn_flow_gate_bias": float(getattr(args, "attn_flow_gate_bias", -2.5)),
        "attn_flow_init_scale": float(getattr(args, "attn_flow_init_scale", 0.02)),
        "attn_flow_detail_topk": int(getattr(args, "attn_flow_detail_topk", 8)),
        "attn_flow_patch_only": bool(getattr(args, "attn_flow_patch_only", False)),
        "patch_grid_refiner": bool(getattr(args, "patch_grid_refiner", False)),
        "patch_grid_refiner_scale": float(getattr(args, "patch_grid_refiner_scale", 1.0)),
        "patch_grid_refiner_last_k_blocks": int(getattr(args, "patch_grid_refiner_last_k_blocks", 0)),
        "share_patch_grid_refiner": bool(getattr(args, "share_patch_grid_refiner", False)),
        "patch_grid_refiner_bottleneck": int(getattr(args, "patch_grid_refiner_bottleneck", 16)),
        "patch_grid_refiner_gate_bias": float(getattr(args, "patch_grid_refiner_gate_bias", -5.0)),
        "patch_grid_refiner_init_scale": float(getattr(args, "patch_grid_refiner_init_scale", 0.002)),
        "patch_grid_refiner_cls_context_scale": float(getattr(args, "patch_grid_refiner_cls_context_scale", 0.05)),
        "attention_metric_adapter": bool(getattr(args, "attention_metric_adapter", False)),
        "attention_metric_type": str(getattr(args, "attention_metric_type", "linear")),
        "attention_metric_scale": float(getattr(args, "attention_metric_scale", 1.0)),
        "attention_metric_last_k_blocks": int(getattr(args, "attention_metric_last_k_blocks", 0)),
        "share_attention_metric": bool(getattr(args, "share_attention_metric", True)),
        "attention_metric_bottleneck": int(getattr(args, "attention_metric_bottleneck", 8)),
        "attention_metric_patch_only": bool(getattr(args, "attention_metric_patch_only", True)),
        "attention_metric_gate_bias": float(getattr(args, "attention_metric_gate_bias", -3.0)),
        "attention_metric_init_scale": float(getattr(args, "attention_metric_init_scale", 0.01)),
        "attention_metric_cls_context_scale": float(getattr(args, "attention_metric_cls_context_scale", 0.25)),
        "geo_layer_scale_init": float(getattr(args, "geo_layer_scale_init", 0.0)),
        "geo_block_profile": str(getattr(args, "geo_block_profile", "uniform")),
        "geo_learnable_block_scale": bool(getattr(args, "geo_learnable_block_scale", False)),
        "token_flow_input": bool(getattr(args, "token_flow_input", False)),
        "token_flow_last_k_blocks": int(getattr(args, "token_flow_last_k_blocks", 0)),
        "share_token_flow": bool(getattr(args, "share_token_flow", True)),
        "token_flow_scale": float(getattr(args, "token_flow_scale", 1.0)),
        "token_flow_input_scale": float(getattr(args, "token_flow_input_scale", -1.0)),
        "token_flow_block_scale": float(getattr(args, "token_flow_block_scale", -1.0)),
        "token_flow_bottleneck": int(getattr(args, "token_flow_bottleneck", 32)),
        "token_flow_patch_only": bool(getattr(args, "token_flow_patch_only", True)),
        "token_flow_gate_bias": float(getattr(args, "token_flow_gate_bias", -3.0)),
        "token_flow_init_scale": float(getattr(args, "token_flow_init_scale", 0.01)),
        "token_flow_cls_context_scale": float(getattr(args, "token_flow_cls_context_scale", 0.25)),
        "token_flow_detail_topk": int(getattr(args, "token_flow_detail_topk", 0)),
        "token_flow_detail_boost_scale": float(getattr(args, "token_flow_detail_boost_scale", 0.0)),
        "inter_layer_flow": bool(getattr(args, "inter_layer_flow", False)),
        "inter_layer_flow_last_k_blocks": int(getattr(args, "inter_layer_flow_last_k_blocks", 0)),
        "share_inter_layer_flow": bool(getattr(args, "share_inter_layer_flow", True)),
        "inter_layer_flow_mode": str(getattr(args, "inter_layer_flow_mode", "transport")),
        "inter_layer_flow_scale": float(getattr(args, "inter_layer_flow_scale", 1.0)),
        "inter_layer_flow_bottleneck": int(getattr(args, "inter_layer_flow_bottleneck", 16)),
        "inter_layer_flow_patch_only": bool(getattr(args, "inter_layer_flow_patch_only", True)),
        "inter_layer_flow_gate_bias": float(getattr(args, "inter_layer_flow_gate_bias", -4.0)),
        "inter_layer_flow_init_scale": float(getattr(args, "inter_layer_flow_init_scale", 0.005)),
        "inter_layer_flow_cls_context_scale": float(getattr(args, "inter_layer_flow_cls_context_scale", 0.15)),
        "inter_layer_flow_delta_scale": float(getattr(args, "inter_layer_flow_delta_scale", 0.5)),
        "flow_state_carrier": bool(getattr(args, "flow_state_carrier", False)),
        "flow_state_last_k_blocks": int(getattr(args, "flow_state_last_k_blocks", 0)),
        "share_flow_state_carrier": bool(getattr(args, "share_flow_state_carrier", True)),
        "flow_state_dim": int(getattr(args, "flow_state_dim", 24)),
        "flow_state_scale": float(getattr(args, "flow_state_scale", 1.0)),
        "flow_state_gate_bias": float(getattr(args, "flow_state_gate_bias", -5.0)),
        "flow_state_init_scale": float(getattr(args, "flow_state_init_scale", 0.0025)),
        "flow_state_cls_scale": float(getattr(args, "flow_state_cls_scale", 1.0)),
        "flow_state_patch_scale": float(getattr(args, "flow_state_patch_scale", 0.1)),
        "flow_rank": int(args.flow_rank),
        "flow_steps": int(args.flow_steps),
        "flow_step_size": float(args.flow_step_size),
        "semantic_manifold_mode": bool(args.semantic_manifold_mode),
        "magnus_semantic_mode": bool(args.magnus_semantic_mode),
        "magnus_single_operator": bool(getattr(args, "magnus_single_operator", False)),
        "magnus_detail_topk": int(args.magnus_detail_topk),
        "strict_semantic_operator": bool(args.strict_semantic_operator),
    }


def stable_hash(payload: dict[str, object]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--save-dir", type=str, default="runs/geovit_proper")
    parser.add_argument("--run-name", type=str, default="geovit_proper_smoke")
    parser.add_argument("--locked-smoke-protocol", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--eval-prefetch-factor", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--eval-only", action="store_true", default=False)
    parser.add_argument("--disable-amp", action="store_true", default=False)
    parser.add_argument("--amp-dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--eval-hflip-tta", action="store_true", default=False)
    parser.add_argument("--eval-shift-tta", type=int, default=0)
    parser.add_argument("--eval-diagonal-shift-tta", action="store_true", default=False)
    parser.add_argument("--disable-strong-aug", action="store_true", default=False)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--mix-prob", type=float, default=1.0)
    parser.add_argument("--cutmix-switch-prob", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.9998)
    parser.add_argument("--ema-start-epoch", type=int, default=5)
    parser.add_argument("--use-conditioner", action="store_true", default=False)
    parser.add_argument("--use-internal-conditioner", action="store_true", default=True)
    parser.add_argument("--no-internal-conditioner", action="store_false", dest="use_internal_conditioner")
    parser.add_argument("--save-best-checkpoint", action="store_true", default=True)
    parser.add_argument("--no-save-best-checkpoint", action="store_false", dest="save_best_checkpoint")
    parser.add_argument("--save-last-checkpoint", action="store_true", default=False)
    parser.add_argument("--teacher-checkpoint", type=str, default="")
    parser.add_argument("--anchor-checkpoint", type=str, default="")
    parser.add_argument("--student-checkpoint", type=str, default="")
    parser.add_argument("--teacher-init-student", action="store_true", default=False)
    parser.add_argument("--teacher-init-gate-bias", type=float, default=-4.0)
    parser.add_argument("--teacher-temperature", type=float, default=2.0)
    parser.add_argument("--teacher-logit-weight", type=float, default=0.0)
    parser.add_argument("--teacher-feature-weight", type=float, default=0.0)
    parser.add_argument("--teacher-block-feature-weight", type=float, default=0.0)
    parser.add_argument("--teacher-block-feature-layers", type=int, default=0)
    parser.add_argument("--anchor-logit-weight", type=float, default=0.0)
    parser.add_argument("--anchor-feature-weight", type=float, default=0.0)
    parser.add_argument("--anchor-block-feature-weight", type=float, default=0.0)
    parser.add_argument("--anchor-block-feature-layers", type=int, default=0)
    parser.add_argument("--semantic-anchor-weight", type=float, default=0.0)
    parser.add_argument("--semantic-expert-weight", type=float, default=0.0)
    parser.add_argument("--semantic-mode-weight", type=float, default=0.0)
    parser.add_argument("--geo-structure-weight", type=float, default=0.0)
    parser.add_argument("--geo-structure-start-epoch", type=int, default=0)
    parser.add_argument("--geo-structure-ramp-epochs", type=int, default=0)
    parser.add_argument("--flow-anchor-weight", type=float, default=0.0)
    parser.add_argument("--flow-energy-weight", type=float, default=0.0)
    parser.add_argument("--use-class-text-router", action="store_true", default=False)
    parser.add_argument("--text-router-temperature", type=float, default=1.0)
    parser.add_argument("--text-embedding-source", type=str, default="hashed", choices=["hashed", "clip"])
    parser.add_argument("--text-embedding-model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--manifold-refine-only", action="store_true", default=False)
    parser.add_argument("--manifold-train-head", action="store_true", default=False)
    parser.add_argument("--manifold-train-output-basis", action="store_true", default=False)
    parser.add_argument("--manifold-train-text-router", action="store_true", default=False)
    parser.add_argument("--geo-refine-only", action="store_true", default=False)
    parser.add_argument("--geo-train-head", action="store_true", default=False)
    parser.add_argument("--geo-train-output-basis", action="store_true", default=False)
    parser.add_argument("--geo-head-warmup-epochs", type=int, default=0)
    parser.add_argument("--geo-basis-last-epochs", type=int, default=0)
    parser.add_argument("--head-probe-lr", type=float, default=0.0)
    parser.add_argument("--geo-start-epoch", type=int, default=0)
    parser.add_argument("--geo-ramp-epochs", type=int, default=0)
    parser.add_argument("--geo-target-scale", type=float, default=1.0)
    parser.add_argument("--geo-end-scale", type=float, default=-1.0)
    parser.add_argument("--geo-decay-start-epoch", type=int, default=0)
    parser.add_argument("--geo-spectral-end-mult", type=float, default=-1.0)
    parser.add_argument("--geo-spectral-decay-start-epoch", type=int, default=0)
    parser.add_argument("--geo-low-rank-end-mult", type=float, default=-1.0)
    parser.add_argument("--geo-low-rank-decay-start-epoch", type=int, default=0)
    parser.add_argument("--geo-rotation-end-mult", type=float, default=-1.0)
    parser.add_argument("--geo-rotation-decay-start-epoch", type=int, default=0)
    parser.add_argument("--warm-start-manifold-from-anchor", action="store_true", default=False)
    parser.add_argument("--warm-start-semantic-from-anchor", action="store_true", default=False)
    parser.add_argument("--warm-start-flow-scale", type=float, default=1.0)

    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--tokenizer-type", type=str, default="standard", choices=["standard", "budgeted_detail"])
    parser.add_argument("--detail-tokens", type=int, default=0)
    parser.add_argument("--detail-score-type", type=str, default="variance", choices=["variance", "learned"])
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--condition-dim", type=int, default=512)
    parser.add_argument("--controller-hidden-dim", type=int, default=128)
    parser.add_argument("--num-spectral-bases", type=int, default=8)
    parser.add_argument("--low-rank-rank", type=int, default=8)
    parser.add_argument("--orthogonal-rank", type=int, default=8)
    parser.add_argument("--flow-rank", type=int, default=0)
    parser.add_argument("--flow-steps", type=int, default=1)
    parser.add_argument("--flow-step-size", type=float, default=0.5)
    parser.add_argument("--semantic-manifold-mode", action="store_true", default=False)
    parser.add_argument("--semantic-num-experts", type=int, default=2)
    parser.add_argument("--semantic-expert-temperature", type=float, default=0.5)
    parser.add_argument("--magnus-semantic-mode", action="store_true", default=False)
    parser.add_argument("--magnus-single-operator", action="store_true", default=False)
    parser.add_argument("--magnus-detail-topk", type=int, default=4)
    parser.add_argument("--magnus-rotation-mode", action="store_true", default=False)
    parser.add_argument("--magnus-rotation-last-k-blocks", type=int, default=0)
    parser.add_argument("--magnus-rotation-strength", type=float, default=1.0)
    parser.add_argument("--magnus-anchor-weight", type=float, default=0.0)
    parser.add_argument("--magnus-motion-weight", type=float, default=0.0)
    parser.add_argument("--magnus-comm-weight", type=float, default=0.0)
    parser.add_argument("--magnus-mode-weight", type=float, default=0.0)
    parser.add_argument("--strict-semantic-operator", action="store_true", default=False)
    parser.add_argument("--manifold-alignment-mode", action="store_true", default=False)
    parser.add_argument("--fc1-base-rank", type=int, default=64)
    parser.add_argument("--share-fc1-bank", action="store_true", default=True)
    parser.add_argument("--no-share-fc1-bank", action="store_false", dest="share_fc1_bank")
    parser.add_argument("--fc1-bank-groups", type=int, default=1)
    parser.add_argument("--share-fc1-conditioner", action="store_true", default=True)
    parser.add_argument("--no-share-fc1-conditioner", action="store_false", dest="share_fc1_conditioner")
    parser.add_argument("--fc1-conditioner-groups", type=int, default=1)
    parser.add_argument("--coupled-spectral-low-rank", action="store_true", default=False)
    parser.add_argument("--coupled-learnable-input-basis", action="store_true", default=False)
    parser.add_argument("--coupled-shared-gate", action="store_true", default=False)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--geo-residual-budget", type=float, default=0.0)
    parser.add_argument("--spectral-scale", type=float, default=1.0)
    parser.add_argument("--low-rank-scale", type=float, default=1.0)
    parser.add_argument("--rotation-scale", type=float, default=1.0)
    parser.add_argument("--enable-local-geo", action="store_true", default=True)
    parser.add_argument("--disable-local-geo", action="store_false", dest="enable_local_geo")
    parser.add_argument("--use-conv-stem", action="store_true", default=True)
    parser.add_argument("--no-conv-stem", action="store_false", dest="use_conv_stem")
    parser.add_argument("--stem-channels", type=int, default=64)
    parser.add_argument("--stem-flow-refiner", action="store_true", default=False)
    parser.add_argument("--stem-flow-scale", type=float, default=1.0)
    parser.add_argument("--stem-flow-bottleneck", type=int, default=16)
    parser.add_argument("--stem-flow-gate-bias", type=float, default=-3.0)
    parser.add_argument("--stem-flow-init-scale", type=float, default=0.02)
    parser.add_argument("--stem-flow-detail-scale", type=float, default=0.25)
    parser.add_argument("--stem-flow-context-scale", type=float, default=0.5)
    parser.add_argument("--summary-token", action="store_true", default=False)
    parser.add_argument("--summary-token-scale", type=float, default=1.0)
    parser.add_argument("--summary-head-fusion", type=float, default=0.5)
    parser.add_argument("--geo-on-fc1", action="store_true", default=True)
    parser.add_argument("--no-geo-on-fc1", action="store_false", dest="geo_on_fc1")
    parser.add_argument("--geo-fc1-last-k-blocks", type=int, default=0)
    parser.add_argument("--flow-depth-mode", action="store_true", default=False,
                        help="Use FlowDepthViT: weight-tied iterative transformer (continuous-depth flow)")
    parser.add_argument("--flow-depth-iterations", type=int, default=12,
                        help="Number of iterations for FlowDepthViT shared block")
    parser.add_argument("--flow-depth-inject", type=float, default=0.1,
                        help="Input injection strength for FlowDepthViT")
    parser.add_argument("--flow-depth-momentum", action="store_true", default=False,
                        help="Use Hamiltonian momentum dynamics in FlowDepthViT")
    parser.add_argument("--flow-depth-momentum-beta", type=float, default=0.9)
    parser.add_argument("--flow-depth-independent-blocks", type=int, default=0,
                        help="Number of independent (non-shared) blocks before weight-tied iterations")
    parser.add_argument("--flow-depth-time-conditioning", action="store_true", default=False,
                        help="Add per-step time embeddings for time-dependent ODE f(h,t)")
    parser.add_argument("--flow-depth-attn-pool", action="store_true", default=False,
                        help="Use attention pooling head instead of CLS token")
    parser.add_argument("--agff-last-k-blocks", type=int, default=0,
                        help="Replace MLP with Attention-Gated FFN in last K blocks (0 disables)")
    parser.add_argument("--agff-gate-mode", type=str, default="attn",
                        choices=["attn", "dual", "scale"],
                        help="Gate formulation: attn=sigmoid(W@LN(a)), dual=sigmoid(W@LN(a)+V@x), scale=1+tanh(W@LN(a))")
    parser.add_argument("--agff-gate-ln", action="store_true", default=True,
                        help="LayerNorm attn_out before gate (prevents scale collapse)")
    parser.add_argument("--no-agff-gate-ln", action="store_false", dest="agff_gate_ln")
    parser.add_argument("--agff-gate-init-scale", type=float, default=-1.0,
                        help="Init std for gate weights; -1 = auto 1/sqrt(hidden)")
    parser.add_argument("--fnfl-last-k-blocks", type=int, default=0,
                        help="Replace MLP with Flow-Native FFN in this many last blocks (0 disables)")
    parser.add_argument("--fnfl-num-steps", type=int, default=2)
    parser.add_argument("--fnfl-rank", type=int, default=64)
    parser.add_argument("--fnfl-num-spectral-bases", type=int, default=8)
    parser.add_argument("--fnfl-low-rank", type=int, default=4)
    parser.add_argument("--fnfl-controller-hidden-dim", type=int, default=128)
    parser.add_argument("--fnfl-strength-init", type=float, default=0.0,
                        help="Initial flow strength (0 = block reduces to low-rank linear MLP)")
    parser.add_argument("--fnfl-strength-anneal-epochs", type=int, default=20,
                        help="Linearly ramp flow strength from init to max over this many epochs")
    parser.add_argument("--fnfl-strength-max", type=float, default=1.0)
    parser.add_argument("--gfn-last-k-blocks", type=int, default=0,
                        help="Apply GradFlow FFN to the last K blocks (0 = disabled)")
    parser.add_argument("--gfn-corr-bottleneck", type=int, default=32,
                        help="Bottleneck dim for the GradFlow correction branch")
    parser.add_argument("--gfn-n-train-iters", type=int, default=1,
                        help="Correction iterations during training (>1 = multi-step)")
    parser.add_argument("--gfn-gate-init", type=float, default=-4.0,
                        help="Initial gate logit; sigmoid(-4)≈0.018 keeps corrections near-zero at start")
    parser.add_argument("--geo-low-rank-last-k-blocks", type=int, default=0)
    parser.add_argument("--local-geo-last-k-blocks", type=int, default=0)
    parser.add_argument("--geo-on-fc2", action="store_true", default=False)
    parser.add_argument("--geo-on-attention", action="store_true", default=False)
    parser.add_argument("--geo-attention-last-k-blocks", type=int, default=0)
    parser.add_argument("--hidden-diffusion", action="store_true", default=False)
    parser.add_argument("--hidden-diffusion-scale", type=float, default=1.0)
    parser.add_argument("--hidden-diffusion-last-k-blocks", type=int, default=0)
    parser.add_argument("--share-hidden-diffusion", action="store_true", default=False)
    parser.add_argument("--hidden-diffusion-bottleneck", type=int, default=32)
    parser.add_argument("--hidden-diffusion-gate-bias", type=float, default=-4.0)
    parser.add_argument("--hidden-diffusion-init-scale", type=float, default=0.005)
    parser.add_argument("--hidden-diffusion-cls-context-scale", type=float, default=0.1)
    parser.add_argument("--hidden-grid-refiner", action="store_true", default=False)
    parser.add_argument("--hidden-grid-refiner-scale", type=float, default=1.0)
    parser.add_argument("--hidden-grid-refiner-last-k-blocks", type=int, default=0)
    parser.add_argument("--share-hidden-grid-refiner", action="store_true", default=False)
    parser.add_argument("--no-share-hidden-grid-refiner", action="store_false", dest="share_hidden_grid_refiner")
    parser.add_argument("--hidden-grid-refiner-bottleneck", type=int, default=16)
    parser.add_argument("--hidden-grid-refiner-gate-bias", type=float, default=-5.0)
    parser.add_argument("--hidden-grid-refiner-init-scale", type=float, default=0.002)
    parser.add_argument("--hidden-grid-refiner-cls-context-scale", type=float, default=0.05)
    parser.add_argument("--hidden-cls-bridge", action="store_true", default=False)
    parser.add_argument("--hidden-cls-bridge-scale", type=float, default=1.0)
    parser.add_argument("--hidden-cls-bridge-last-k-blocks", type=int, default=0)
    parser.add_argument("--share-hidden-cls-bridge", action="store_true", default=False)
    parser.add_argument("--no-share-hidden-cls-bridge", action="store_false", dest="share_hidden_cls_bridge")
    parser.add_argument("--hidden-cls-bridge-bottleneck", type=int, default=16)
    parser.add_argument("--hidden-cls-bridge-gate-bias", type=float, default=-4.0)
    parser.add_argument("--hidden-cls-bridge-init-scale", type=float, default=0.01)
    parser.add_argument("--hidden-cls-bridge-patch-feedback-scale", type=float, default=0.0)
    parser.add_argument("--hidden-channel-flow", action="store_true", default=False)
    parser.add_argument("--hidden-channel-flow-scale", type=float, default=1.0)
    parser.add_argument("--hidden-channel-flow-last-k-blocks", type=int, default=0)
    parser.add_argument("--share-hidden-channel-flow", action="store_true", default=False)
    parser.add_argument("--no-share-hidden-channel-flow", action="store_false", dest="share_hidden_channel_flow")
    parser.add_argument("--hidden-channel-flow-bottleneck", type=int, default=16)
    parser.add_argument("--hidden-channel-flow-rank", type=int, default=16)
    parser.add_argument("--hidden-channel-flow-gate-bias", type=float, default=-3.5)
    parser.add_argument("--hidden-channel-flow-init-scale", type=float, default=0.01)
    parser.add_argument("--hidden-channel-flow-patch-only", action="store_true", default=False)
    parser.add_argument("--hidden-channel-flow-all-tokens", action="store_false", dest="hidden_channel_flow_patch_only")
    parser.add_argument("--hidden-channel-flow-cls-mix-scale", type=float, default=1.0)
    parser.add_argument("--hidden-channel-flow-mean-mix-scale", type=float, default=0.5)
    parser.add_argument("--response-flow-norm", action="store_true", default=False)
    parser.add_argument("--response-flow-scale", type=float, default=1.0)
    parser.add_argument("--response-flow-last-k-blocks", type=int, default=0)
    parser.add_argument("--response-flow-init-scale", type=float, default=0.01)
    parser.add_argument("--response-flow-mode", type=str, default="simple", choices=["simple", "biaxial"])
    parser.add_argument("--share-response-flow", action="store_true", default=False)
    parser.add_argument("--no-share-response-flow", action="store_false", dest="share_response_flow")
    parser.add_argument("--response-flow-bottleneck", type=int, default=12)
    parser.add_argument("--response-flow-gate-bias", type=float, default=-4.0)
    parser.add_argument("--response-flow-patch-only", action="store_true", default=False)
    parser.add_argument("--response-flow-all-tokens", action="store_false", dest="response_flow_patch_only")
    parser.add_argument("--response-flow-cls-mix-scale", type=float, default=1.0)
    parser.add_argument("--response-flow-mean-mix-scale", type=float, default=0.5)
    parser.add_argument("--response-flow-token-exponent", type=float, default=0.5)
    parser.add_argument("--response-flow-channel-exponent", type=float, default=0.5)
    parser.add_argument("--response-flow-pre-act", action="store_true", default=False)
    parser.add_argument("--dual-path-mlp", action="store_true", default=False)
    parser.add_argument("--dual-path-last-k-blocks", type=int, default=0)
    parser.add_argument("--dual-path-refine-ratio", type=float, default=0.25)
    parser.add_argument("--dual-path-cross-scale", type=float, default=1.0)
    parser.add_argument("--dual-path-gate-bias", type=float, default=-2.0)
    parser.add_argument("--hidden-group-router", action="store_true", default=False)
    parser.add_argument("--hidden-group-router-last-k-blocks", type=int, default=0)
    parser.add_argument("--hidden-group-router-groups", type=int, default=0)
    parser.add_argument("--hidden-group-router-scale", type=float, default=1.0)
    parser.add_argument("--hidden-group-router-gate-bias", type=float, default=-2.0)
    parser.add_argument("--hidden-group-router-init-scale", type=float, default=0.1)
    parser.add_argument("--hidden-group-router-cls-mix-scale", type=float, default=1.0)
    parser.add_argument("--hidden-group-router-mean-mix-scale", type=float, default=0.5)
    parser.add_argument("--attention-hidden-fusion", action="store_true", default=False)
    parser.add_argument("--attention-hidden-fusion-last-k-blocks", type=int, default=0)
    parser.add_argument("--share-attention-hidden-fusion", action="store_true", default=True)
    parser.add_argument("--no-share-attention-hidden-fusion", action="store_false", dest="share_attention_hidden_fusion")
    parser.add_argument("--attention-hidden-fusion-scale", type=float, default=1.0)
    parser.add_argument("--attention-hidden-fusion-bottleneck", type=int, default=6)
    parser.add_argument("--attention-hidden-fusion-gate-bias", type=float, default=-2.5)
    parser.add_argument("--attention-hidden-fusion-init-scale", type=float, default=0.02)
    parser.add_argument("--attention-hidden-fusion-patch-only", action="store_true", default=False)
    parser.add_argument("--attention-hidden-fusion-all-tokens", action="store_false", dest="attention_hidden_fusion_patch_only")
    parser.add_argument("--attention-hidden-fusion-cls-context-scale", type=float, default=1.0)
    parser.add_argument("--hidden-token-mixer", action="store_true", default=False)
    parser.add_argument("--hidden-token-mixer-last-k-blocks", type=int, default=0)
    parser.add_argument("--share-hidden-token-mixer", action="store_true", default=True)
    parser.add_argument("--no-share-hidden-token-mixer", action="store_false", dest="share_hidden_token_mixer")
    parser.add_argument("--hidden-token-mixer-scale", type=float, default=1.0)
    parser.add_argument("--hidden-token-mixer-gate-bias", type=float, default=-3.0)
    parser.add_argument("--hidden-token-mixer-init-scale", type=float, default=0.02)
    parser.add_argument("--hidden-token-mixer-patch-only", action="store_true", default=True)
    parser.add_argument("--hidden-token-mixer-all-tokens", action="store_false", dest="hidden_token_mixer_patch_only")
    parser.add_argument("--hidden-token-mixer-mode", type=str, default="conv")
    parser.add_argument("--hidden-token-mixer-topk", type=int, default=8)
    parser.add_argument("--competitive-residual", action="store_true", default=False)
    parser.add_argument("--competitive-residual-last-k-blocks", type=int, default=0)
    parser.add_argument("--competitive-residual-scale", type=float, default=1.0)
    parser.add_argument("--competitive-residual-gate-bias", type=float, default=0.0)
    parser.add_argument("--competitive-residual-init-scale", type=float, default=1.0)
    parser.add_argument("--competitive-residual-cls-mix-scale", type=float, default=1.0)
    parser.add_argument("--competitive-residual-mean-mix-scale", type=float, default=0.5)
    parser.add_argument("--competitive-residual-patch-only", action="store_true", default=False)
    parser.add_argument("--competitive-residual-all-tokens", action="store_false", dest="competitive_residual_patch_only")
    parser.add_argument("--parallel-block-update", action="store_true", default=False)
    parser.add_argument("--parallel-block-last-k-blocks", type=int, default=0)
    parser.add_argument("--mlp-first-update", action="store_true", default=False)
    parser.add_argument("--mlp-first-last-k-blocks", type=int, default=0)
    parser.add_argument("--tail-token-mixer", action="store_true", default=False)
    parser.add_argument("--tail-token-mixer-last-k-blocks", type=int, default=0)
    parser.add_argument("--tail-token-mixer-scale", type=float, default=1.0)
    parser.add_argument("--tail-token-mixer-gate-bias", type=float, default=-3.0)
    parser.add_argument("--tail-token-mixer-init-scale", type=float, default=0.02)
    parser.add_argument("--tail-token-mixer-patch-only", action="store_true", default=True)
    parser.add_argument("--tail-token-mixer-all-tokens", action="store_false", dest="tail_token_mixer_patch_only")
    parser.add_argument("--activation-flow", action="store_true", default=False)
    parser.add_argument("--activation-flow-scale", type=float, default=1.0)
    parser.add_argument("--activation-flow-last-k-blocks", type=int, default=0)
    parser.add_argument("--share-activation-flow", action="store_true", default=False)
    parser.add_argument("--no-share-activation-flow", action="store_false", dest="share_activation_flow")
    parser.add_argument("--activation-flow-bottleneck", type=int, default=16)
    parser.add_argument("--activation-flow-gate-bias", type=float, default=-4.0)
    parser.add_argument("--activation-flow-init-scale", type=float, default=0.01)
    parser.add_argument("--activation-flow-patch-only", action="store_true", default=False)
    parser.add_argument("--activation-flow-all-tokens", action="store_false", dest="activation_flow_patch_only")
    parser.add_argument("--activation-flow-cls-mix-scale", type=float, default=1.0)
    parser.add_argument("--activation-flow-mean-mix-scale", type=float, default=0.5)
    parser.add_argument("--activation-flow-std-mix-scale", type=float, default=0.25)
    parser.add_argument("--activation-flow-cls-token-scale", type=float, default=1.0)
    parser.add_argument("--attn-flow-modulator", action="store_true", default=False)
    parser.add_argument("--attn-flow-scale", type=float, default=1.0)
    parser.add_argument("--attn-flow-last-k-blocks", type=int, default=0)
    parser.add_argument("--share-attn-flow-modulator", action="store_true", default=False)
    parser.add_argument("--no-share-attn-flow-modulator", action="store_false", dest="share_attn_flow_modulator")
    parser.add_argument("--attn-flow-bottleneck", type=int, default=24)
    parser.add_argument("--attn-flow-gate-bias", type=float, default=-2.5)
    parser.add_argument("--attn-flow-init-scale", type=float, default=0.02)
    parser.add_argument("--attn-flow-detail-topk", type=int, default=8)
    parser.add_argument("--attn-flow-patch-only", action="store_true", default=False)
    parser.add_argument("--patch-grid-refiner", action="store_true", default=False)
    parser.add_argument("--patch-grid-refiner-scale", type=float, default=1.0)
    parser.add_argument("--patch-grid-refiner-last-k-blocks", type=int, default=0)
    parser.add_argument("--share-patch-grid-refiner", action="store_true", default=False)
    parser.add_argument("--patch-grid-refiner-bottleneck", type=int, default=16)
    parser.add_argument("--patch-grid-refiner-gate-bias", type=float, default=-5.0)
    parser.add_argument("--patch-grid-refiner-init-scale", type=float, default=0.002)
    parser.add_argument("--patch-grid-refiner-cls-context-scale", type=float, default=0.05)
    parser.add_argument("--attention-metric-adapter", action="store_true", default=False)
    parser.add_argument("--attention-metric-type", type=str, default="linear")
    parser.add_argument("--attention-metric-scale", type=float, default=1.0)
    parser.add_argument("--attention-metric-last-k-blocks", type=int, default=0)
    parser.add_argument("--share-attention-metric", action="store_true", default=True)
    parser.add_argument("--no-share-attention-metric", action="store_false", dest="share_attention_metric")
    parser.add_argument("--attention-metric-bottleneck", type=int, default=8)
    parser.add_argument("--attention-metric-patch-only", action="store_true", default=True)
    parser.add_argument("--attention-metric-all-tokens", action="store_false", dest="attention_metric_patch_only")
    parser.add_argument("--attention-metric-gate-bias", type=float, default=-3.0)
    parser.add_argument("--attention-metric-init-scale", type=float, default=0.01)
    parser.add_argument("--attention-metric-cls-context-scale", type=float, default=0.25)
    parser.add_argument("--geo-layer-scale-init", type=float, default=0.0)
    parser.add_argument(
        "--geo-block-profile",
        type=str,
        default="uniform",
        choices=[
            "uniform",
            "late_ramp",
            "late_heavy",
            "mid_peak",
            "sandwich",
            "active_late_ramp",
            "active_late_heavy",
            "active_mid_peak",
            "active_sandwich",
        ],
    )
    parser.add_argument("--geo-learnable-block-scale", action="store_true", default=False)
    parser.add_argument("--token-flow-input", action="store_true", default=False)
    parser.add_argument("--token-flow-last-k-blocks", type=int, default=0)
    parser.add_argument("--share-token-flow", action="store_true", default=True)
    parser.add_argument("--no-share-token-flow", action="store_false", dest="share_token_flow")
    parser.add_argument("--token-flow-scale", type=float, default=1.0)
    parser.add_argument("--token-flow-input-scale", type=float, default=-1.0)
    parser.add_argument("--token-flow-block-scale", type=float, default=-1.0)
    parser.add_argument("--token-flow-bottleneck", type=int, default=32)
    parser.add_argument("--token-flow-patch-only", action="store_true", default=True)
    parser.add_argument("--token-flow-all-tokens", action="store_false", dest="token_flow_patch_only")
    parser.add_argument("--token-flow-gate-bias", type=float, default=-3.0)
    parser.add_argument("--token-flow-init-scale", type=float, default=0.01)
    parser.add_argument("--token-flow-cls-context-scale", type=float, default=0.25)
    parser.add_argument("--token-flow-detail-topk", type=int, default=0)
    parser.add_argument("--token-flow-detail-boost-scale", type=float, default=0.0)
    parser.add_argument("--inter-layer-flow", action="store_true", default=False)
    parser.add_argument("--inter-layer-flow-last-k-blocks", type=int, default=0)
    parser.add_argument("--share-inter-layer-flow", action="store_true", default=True)
    parser.add_argument("--no-share-inter-layer-flow", action="store_false", dest="share_inter_layer_flow")
    parser.add_argument("--inter-layer-flow-mode", type=str, default="transport")
    parser.add_argument("--inter-layer-flow-scale", type=float, default=1.0)
    parser.add_argument("--inter-layer-flow-bottleneck", type=int, default=16)
    parser.add_argument("--inter-layer-flow-patch-only", action="store_true", default=True)
    parser.add_argument("--inter-layer-flow-all-tokens", action="store_false", dest="inter_layer_flow_patch_only")
    parser.add_argument("--inter-layer-flow-gate-bias", type=float, default=-4.0)
    parser.add_argument("--inter-layer-flow-init-scale", type=float, default=0.005)
    parser.add_argument("--inter-layer-flow-cls-context-scale", type=float, default=0.15)
    parser.add_argument("--inter-layer-flow-delta-scale", type=float, default=0.5)
    parser.add_argument("--flow-state-carrier", action="store_true", default=False)
    parser.add_argument("--flow-state-last-k-blocks", type=int, default=0)
    parser.add_argument("--share-flow-state-carrier", action="store_true", default=True)
    parser.add_argument("--no-share-flow-state-carrier", action="store_false", dest="share_flow_state_carrier")
    parser.add_argument("--flow-state-dim", type=int, default=24)
    parser.add_argument("--flow-state-scale", type=float, default=1.0)
    parser.add_argument("--flow-state-gate-bias", type=float, default=-5.0)
    parser.add_argument("--flow-state-init-scale", type=float, default=0.0025)
    parser.add_argument("--flow-state-cls-scale", type=float, default=1.0)
    parser.add_argument("--flow-state-patch-scale", type=float, default=0.1)
    return parser


def evaluate(
    model,
    loader,
    device,
    *,
    hflip_tta: bool = False,
    shift_tta: int = 0,
    diagonal_shift_tta: bool = False,
):
    def build_eval_views(images: torch.Tensor) -> list[torch.Tensor]:
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
            if diagonal_shift_tta:
                offsets.extend([(-pad, -pad), (-pad, pad), (pad, -pad), (pad, pad)])
            for base in base_views:
                padded = F.pad(base, (pad, pad, pad, pad), mode="reflect")
                for dx, dy in offsets:
                    x0 = pad + dx
                    y0 = pad + dy
                    views.append(padded[:, :, y0 : y0 + height, x0 : x0 + width])
        return views

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            views = build_eval_views(images)
            logits = torch.stack([model(view) for view in views], dim=0).mean(dim=0)
            loss = F.cross_entropy(logits, labels)
            total_loss += float(loss.item()) * int(labels.shape[0])
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_count += int(labels.shape[0])
    return total_loss / max(total_count, 1), 100.0 * total_correct / max(total_count, 1)


def build_model(
    args,
    *,
    geo_on_fc1=None,
    geo_on_fc2=None,
    geo_on_attention=None,
    geo_attention_last_k_blocks=None,
    class_texts: list[str] | None = None,
    use_class_text_router=None,
    flow_rank=None,
    flow_steps=None,
    flow_step_size=None,
):
    # FlowDepthViT: entirely separate model, bypass everything else
    if getattr(args, "flow_depth_mode", False):
        return FlowDepthViT(
            img_size=args.img_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            num_iterations=getattr(args, "flow_depth_iterations", 12),
            num_heads=args.num_heads,
            num_classes=100,
            mlp_ratio=getattr(args, "mlp_ratio", 4.0),
            drop_path_rate=args.drop_path_rate,
            input_inject_strength=getattr(args, "flow_depth_inject", 0.1),
            use_momentum=getattr(args, "flow_depth_momentum", False),
            momentum_beta=getattr(args, "flow_depth_momentum_beta", 0.9),
            num_independent_blocks=getattr(args, "flow_depth_independent_blocks", 0),
            use_time_conditioning=getattr(args, "flow_depth_time_conditioning", False),
            use_attn_pool=getattr(args, "flow_depth_attn_pool", False),
        )

    resolved_magnus_semantic_mode = bool(args.magnus_semantic_mode or getattr(args, "magnus_single_operator", False))
    resolved_geo_on_fc1 = args.geo_on_fc1 if geo_on_fc1 is None else geo_on_fc1
    resolved_geo_on_fc2 = args.geo_on_fc2 if geo_on_fc2 is None else geo_on_fc2
    resolved_geo_on_attention = args.geo_on_attention if geo_on_attention is None else geo_on_attention
    resolved_geo_attention_last_k_blocks = (
        int(getattr(args, "geo_attention_last_k_blocks", 0))
        if geo_attention_last_k_blocks is None
        else int(geo_attention_last_k_blocks)
    )
    resolved_use_class_text_router = args.use_class_text_router if use_class_text_router is None else use_class_text_router
    resolved_flow_rank = args.flow_rank if flow_rank is None else flow_rank
    resolved_flow_steps = args.flow_steps if flow_steps is None else flow_steps
    resolved_flow_step_size = args.flow_step_size if flow_step_size is None else flow_step_size
    has_token_flow = bool(getattr(args, "token_flow_input", False) or int(getattr(args, "token_flow_last_k_blocks", 0)) > 0)
    has_hidden_grid_refiner = bool(
        getattr(args, "hidden_grid_refiner", False)
        and int(getattr(args, "hidden_grid_refiner_last_k_blocks", 0)) > 0
    )
    has_hidden_cls_bridge = bool(
        getattr(args, "hidden_cls_bridge", False)
        and int(getattr(args, "hidden_cls_bridge_last_k_blocks", 0)) > 0
    )
    has_hidden_channel_flow = bool(
        getattr(args, "hidden_channel_flow", False)
        and int(getattr(args, "hidden_channel_flow_last_k_blocks", 0)) > 0
    )
    has_response_flow = bool(
        getattr(args, "response_flow_norm", False)
        and int(getattr(args, "response_flow_last_k_blocks", 0)) > 0
    )
    has_activation_flow = bool(
        getattr(args, "activation_flow", False)
        and int(getattr(args, "activation_flow_last_k_blocks", 0)) > 0
    )
    has_attn_flow_modulator = bool(
        getattr(args, "attn_flow_modulator", False)
        and int(getattr(args, "attn_flow_last_k_blocks", 0)) > 0
    )
    has_attention_metric = bool(
        getattr(args, "attention_metric_adapter", False)
        and int(getattr(args, "attention_metric_last_k_blocks", 0)) > 0
    )

    if (
        not resolved_geo_on_fc1
        and not resolved_geo_on_fc2
        and not resolved_geo_on_attention
        and resolved_geo_attention_last_k_blocks <= 0
        and not resolved_use_class_text_router
        and not has_token_flow
        and not has_hidden_grid_refiner
        and not has_hidden_cls_bridge
        and not has_hidden_channel_flow
        and not has_response_flow
        and not has_activation_flow
        and not has_attn_flow_modulator
        and not has_attention_metric
        and int(getattr(args, "agff_last_k_blocks", 0)) <= 0
        and int(getattr(args, "fnfl_last_k_blocks", 0)) <= 0
        and int(getattr(args, "gfn_last_k_blocks", 0)) <= 0
    ):
        return VisionTransformer(
            img_size=args.img_size,
            patch_size=args.patch_size,
            tokenizer_type=args.tokenizer_type,
            detail_tokens=args.detail_tokens,
            detail_score_type=args.detail_score_type,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            num_classes=100,
            drop_path_rate=args.drop_path_rate,
            use_conv_stem=args.use_conv_stem,
            stem_channels=args.stem_channels,
        )

    return GeoVisionTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        tokenizer_type=args.tokenizer_type,
        detail_tokens=args.detail_tokens,
        detail_score_type=args.detail_score_type,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_classes=100,
        mlp_ratio=args.mlp_ratio,
        drop_path_rate=args.drop_path_rate,
        use_conv_stem=args.use_conv_stem,
        stem_channels=args.stem_channels,
        stem_flow_refiner=getattr(args, "stem_flow_refiner", False),
        stem_flow_scale=getattr(args, "stem_flow_scale", 1.0),
        stem_flow_bottleneck=getattr(args, "stem_flow_bottleneck", 16),
        stem_flow_gate_bias=getattr(args, "stem_flow_gate_bias", -3.0),
        stem_flow_init_scale=getattr(args, "stem_flow_init_scale", 0.02),
        stem_flow_detail_scale=getattr(args, "stem_flow_detail_scale", 0.25),
        stem_flow_context_scale=getattr(args, "stem_flow_context_scale", 0.5),
        summary_token=getattr(args, "summary_token", False),
        summary_token_scale=getattr(args, "summary_token_scale", 1.0),
        summary_head_fusion=getattr(args, "summary_head_fusion", 0.5),
        condition_dim=args.condition_dim,
        controller_hidden_dim=args.controller_hidden_dim,
        num_spectral_bases=args.num_spectral_bases,
        low_rank_rank=args.low_rank_rank,
        orthogonal_rank=args.orthogonal_rank,
        flow_rank=resolved_flow_rank,
        flow_steps=resolved_flow_steps,
        flow_step_size=resolved_flow_step_size,
        semantic_manifold_mode=args.semantic_manifold_mode,
        semantic_num_experts=args.semantic_num_experts,
        semantic_expert_temperature=args.semantic_expert_temperature,
        magnus_semantic_mode=resolved_magnus_semantic_mode,
        magnus_single_operator=getattr(args, "magnus_single_operator", False),
        magnus_detail_topk=args.magnus_detail_topk,
        magnus_rotation_mode=args.magnus_rotation_mode,
        magnus_rotation_last_k_blocks=args.magnus_rotation_last_k_blocks,
        magnus_rotation_strength=args.magnus_rotation_strength,
        coupled_spectral_low_rank=getattr(args, "coupled_spectral_low_rank", False),
        coupled_learnable_input_basis=getattr(args, "coupled_learnable_input_basis", False),
        coupled_shared_gate=getattr(args, "coupled_shared_gate", False),
        strict_semantic_operator=args.strict_semantic_operator,
        manifold_alignment_mode=args.manifold_alignment_mode,
        fc1_base_rank=args.fc1_base_rank,
        share_fc1_bank=(
            args.share_fc1_bank
            and not resolved_magnus_semantic_mode
            and not getattr(args, "magnus_single_operator", False)
            and not (args.magnus_rotation_mode and int(args.magnus_rotation_last_k_blocks) > 0)
        ),
        fc1_bank_groups=args.fc1_bank_groups,
        share_fc1_conditioner=(args.share_fc1_conditioner and not getattr(args, "magnus_single_operator", False)),
        fc1_conditioner_groups=args.fc1_conditioner_groups,
        geo_on_fc1=resolved_geo_on_fc1,
        geo_fc1_last_k_blocks=args.geo_fc1_last_k_blocks,
        agff_last_k_blocks=getattr(args, "agff_last_k_blocks", 0),
        agff_gate_mode=getattr(args, "agff_gate_mode", "attn"),
        agff_gate_ln=getattr(args, "agff_gate_ln", True),
        agff_gate_init_scale=getattr(args, "agff_gate_init_scale", -1.0),
        fnfl_last_k_blocks=getattr(args, "fnfl_last_k_blocks", 0),
        fnfl_num_steps=getattr(args, "fnfl_num_steps", 2),
        fnfl_rank=getattr(args, "fnfl_rank", 64),
        fnfl_num_spectral_bases=getattr(args, "fnfl_num_spectral_bases", 8),
        fnfl_low_rank=getattr(args, "fnfl_low_rank", 4),
        fnfl_controller_hidden_dim=getattr(args, "fnfl_controller_hidden_dim", 128),
        fnfl_strength_init=getattr(args, "fnfl_strength_init", 0.0),
        gfn_last_k_blocks=getattr(args, "gfn_last_k_blocks", 0),
        gfn_corr_bottleneck=getattr(args, "gfn_corr_bottleneck", 32),
        gfn_n_train_iters=getattr(args, "gfn_n_train_iters", 1),
        gfn_gate_init=getattr(args, "gfn_gate_init", -4.0),
        geo_low_rank_last_k_blocks=getattr(args, "geo_low_rank_last_k_blocks", 0),
        enable_local_geo=args.enable_local_geo,
        local_geo_last_k_blocks=getattr(args, "local_geo_last_k_blocks", 0),
        geo_on_fc2=resolved_geo_on_fc2,
        geo_on_attention=resolved_geo_on_attention,
        geo_attention_last_k_blocks=resolved_geo_attention_last_k_blocks,
        hidden_diffusion=args.hidden_diffusion,
        hidden_diffusion_scale=args.hidden_diffusion_scale,
        hidden_diffusion_last_k_blocks=args.hidden_diffusion_last_k_blocks,
        share_hidden_diffusion=args.share_hidden_diffusion,
        hidden_diffusion_bottleneck=args.hidden_diffusion_bottleneck,
        hidden_diffusion_gate_bias=args.hidden_diffusion_gate_bias,
        hidden_diffusion_init_scale=args.hidden_diffusion_init_scale,
        hidden_diffusion_cls_context_scale=args.hidden_diffusion_cls_context_scale,
        hidden_grid_refiner=getattr(args, "hidden_grid_refiner", False),
        hidden_grid_refiner_scale=getattr(args, "hidden_grid_refiner_scale", 1.0),
        hidden_grid_refiner_last_k_blocks=getattr(args, "hidden_grid_refiner_last_k_blocks", 0),
        share_hidden_grid_refiner=getattr(args, "share_hidden_grid_refiner", False),
        hidden_grid_refiner_bottleneck=getattr(args, "hidden_grid_refiner_bottleneck", 16),
        hidden_grid_refiner_gate_bias=getattr(args, "hidden_grid_refiner_gate_bias", -5.0),
        hidden_grid_refiner_init_scale=getattr(args, "hidden_grid_refiner_init_scale", 0.002),
        hidden_grid_refiner_cls_context_scale=getattr(args, "hidden_grid_refiner_cls_context_scale", 0.05),
        hidden_cls_bridge=getattr(args, "hidden_cls_bridge", False),
        hidden_cls_bridge_scale=getattr(args, "hidden_cls_bridge_scale", 1.0),
        hidden_cls_bridge_last_k_blocks=getattr(args, "hidden_cls_bridge_last_k_blocks", 0),
        share_hidden_cls_bridge=getattr(args, "share_hidden_cls_bridge", False),
        hidden_cls_bridge_bottleneck=getattr(args, "hidden_cls_bridge_bottleneck", 16),
        hidden_cls_bridge_gate_bias=getattr(args, "hidden_cls_bridge_gate_bias", -4.0),
        hidden_cls_bridge_init_scale=getattr(args, "hidden_cls_bridge_init_scale", 0.01),
        hidden_cls_bridge_patch_feedback_scale=getattr(args, "hidden_cls_bridge_patch_feedback_scale", 0.0),
        hidden_channel_flow=getattr(args, "hidden_channel_flow", False),
        hidden_channel_flow_scale=getattr(args, "hidden_channel_flow_scale", 1.0),
        hidden_channel_flow_last_k_blocks=getattr(args, "hidden_channel_flow_last_k_blocks", 0),
        share_hidden_channel_flow=getattr(args, "share_hidden_channel_flow", False),
        hidden_channel_flow_bottleneck=getattr(args, "hidden_channel_flow_bottleneck", 16),
        hidden_channel_flow_rank=getattr(args, "hidden_channel_flow_rank", 16),
        hidden_channel_flow_gate_bias=getattr(args, "hidden_channel_flow_gate_bias", -3.5),
        hidden_channel_flow_init_scale=getattr(args, "hidden_channel_flow_init_scale", 0.01),
        hidden_channel_flow_patch_only=getattr(args, "hidden_channel_flow_patch_only", False),
        hidden_channel_flow_cls_mix_scale=getattr(args, "hidden_channel_flow_cls_mix_scale", 1.0),
        hidden_channel_flow_mean_mix_scale=getattr(args, "hidden_channel_flow_mean_mix_scale", 0.5),
        response_flow_norm=getattr(args, "response_flow_norm", False),
        response_flow_scale=getattr(args, "response_flow_scale", 1.0),
        response_flow_last_k_blocks=getattr(args, "response_flow_last_k_blocks", 0),
        response_flow_init_scale=getattr(args, "response_flow_init_scale", 0.01),
        response_flow_mode=getattr(args, "response_flow_mode", "simple"),
        share_response_flow=getattr(args, "share_response_flow", False),
        response_flow_bottleneck=getattr(args, "response_flow_bottleneck", 12),
        response_flow_gate_bias=getattr(args, "response_flow_gate_bias", -4.0),
        response_flow_patch_only=getattr(args, "response_flow_patch_only", False),
        response_flow_cls_mix_scale=getattr(args, "response_flow_cls_mix_scale", 1.0),
        response_flow_mean_mix_scale=getattr(args, "response_flow_mean_mix_scale", 0.5),
        response_flow_token_exponent=getattr(args, "response_flow_token_exponent", 0.5),
        response_flow_channel_exponent=getattr(args, "response_flow_channel_exponent", 0.5),
        response_flow_pre_act=getattr(args, "response_flow_pre_act", False),
        dual_path_mlp=getattr(args, "dual_path_mlp", False),
        dual_path_last_k_blocks=getattr(args, "dual_path_last_k_blocks", 0),
        dual_path_refine_ratio=getattr(args, "dual_path_refine_ratio", 0.25),
        dual_path_cross_scale=getattr(args, "dual_path_cross_scale", 1.0),
        dual_path_gate_bias=getattr(args, "dual_path_gate_bias", -2.0),
        hidden_group_router=getattr(args, "hidden_group_router", False),
        hidden_group_router_last_k_blocks=getattr(args, "hidden_group_router_last_k_blocks", 0),
        hidden_group_router_groups=getattr(args, "hidden_group_router_groups", 0),
        hidden_group_router_scale=getattr(args, "hidden_group_router_scale", 1.0),
        hidden_group_router_gate_bias=getattr(args, "hidden_group_router_gate_bias", -2.0),
        hidden_group_router_init_scale=getattr(args, "hidden_group_router_init_scale", 0.1),
        hidden_group_router_cls_mix_scale=getattr(args, "hidden_group_router_cls_mix_scale", 1.0),
        hidden_group_router_mean_mix_scale=getattr(args, "hidden_group_router_mean_mix_scale", 0.5),
        attention_hidden_fusion=getattr(args, "attention_hidden_fusion", False),
        attention_hidden_fusion_last_k_blocks=getattr(args, "attention_hidden_fusion_last_k_blocks", 0),
        share_attention_hidden_fusion=getattr(args, "share_attention_hidden_fusion", True),
        attention_hidden_fusion_scale=getattr(args, "attention_hidden_fusion_scale", 1.0),
        attention_hidden_fusion_bottleneck=getattr(args, "attention_hidden_fusion_bottleneck", 6),
        attention_hidden_fusion_gate_bias=getattr(args, "attention_hidden_fusion_gate_bias", -2.5),
        attention_hidden_fusion_init_scale=getattr(args, "attention_hidden_fusion_init_scale", 0.02),
        attention_hidden_fusion_patch_only=getattr(args, "attention_hidden_fusion_patch_only", False),
        attention_hidden_fusion_cls_context_scale=getattr(args, "attention_hidden_fusion_cls_context_scale", 1.0),
        hidden_token_mixer=getattr(args, "hidden_token_mixer", False),
        hidden_token_mixer_last_k_blocks=getattr(args, "hidden_token_mixer_last_k_blocks", 0),
        share_hidden_token_mixer=getattr(args, "share_hidden_token_mixer", True),
        hidden_token_mixer_scale=getattr(args, "hidden_token_mixer_scale", 1.0),
        hidden_token_mixer_gate_bias=getattr(args, "hidden_token_mixer_gate_bias", -3.0),
        hidden_token_mixer_init_scale=getattr(args, "hidden_token_mixer_init_scale", 0.02),
        hidden_token_mixer_patch_only=getattr(args, "hidden_token_mixer_patch_only", True),
        hidden_token_mixer_mode=getattr(args, "hidden_token_mixer_mode", "conv"),
        hidden_token_mixer_topk=getattr(args, "hidden_token_mixer_topk", 8),
        competitive_residual=getattr(args, "competitive_residual", False),
        competitive_residual_last_k_blocks=getattr(args, "competitive_residual_last_k_blocks", 0),
        competitive_residual_scale=getattr(args, "competitive_residual_scale", 1.0),
        competitive_residual_gate_bias=getattr(args, "competitive_residual_gate_bias", 0.0),
        competitive_residual_init_scale=getattr(args, "competitive_residual_init_scale", 1.0),
        competitive_residual_cls_mix_scale=getattr(args, "competitive_residual_cls_mix_scale", 1.0),
        competitive_residual_mean_mix_scale=getattr(args, "competitive_residual_mean_mix_scale", 0.5),
        competitive_residual_patch_only=getattr(args, "competitive_residual_patch_only", False),
        parallel_block_update=getattr(args, "parallel_block_update", False),
        parallel_block_last_k_blocks=getattr(args, "parallel_block_last_k_blocks", 0),
        mlp_first_update=getattr(args, "mlp_first_update", False),
        mlp_first_last_k_blocks=getattr(args, "mlp_first_last_k_blocks", 0),
        tail_token_mixer=getattr(args, "tail_token_mixer", False),
        tail_token_mixer_last_k_blocks=getattr(args, "tail_token_mixer_last_k_blocks", 0),
        tail_token_mixer_scale=getattr(args, "tail_token_mixer_scale", 1.0),
        tail_token_mixer_gate_bias=getattr(args, "tail_token_mixer_gate_bias", -3.0),
        tail_token_mixer_init_scale=getattr(args, "tail_token_mixer_init_scale", 0.02),
        tail_token_mixer_patch_only=getattr(args, "tail_token_mixer_patch_only", True),
        activation_flow=getattr(args, "activation_flow", False),
        activation_flow_scale=getattr(args, "activation_flow_scale", 1.0),
        activation_flow_last_k_blocks=getattr(args, "activation_flow_last_k_blocks", 0),
        share_activation_flow=getattr(args, "share_activation_flow", False),
        activation_flow_bottleneck=getattr(args, "activation_flow_bottleneck", 16),
        activation_flow_gate_bias=getattr(args, "activation_flow_gate_bias", -4.0),
        activation_flow_init_scale=getattr(args, "activation_flow_init_scale", 0.01),
        activation_flow_patch_only=getattr(args, "activation_flow_patch_only", False),
        activation_flow_cls_mix_scale=getattr(args, "activation_flow_cls_mix_scale", 1.0),
        activation_flow_mean_mix_scale=getattr(args, "activation_flow_mean_mix_scale", 0.5),
        activation_flow_std_mix_scale=getattr(args, "activation_flow_std_mix_scale", 0.25),
        activation_flow_cls_token_scale=getattr(args, "activation_flow_cls_token_scale", 1.0),
        attn_flow_modulator=getattr(args, "attn_flow_modulator", False),
        attn_flow_scale=getattr(args, "attn_flow_scale", 1.0),
        attn_flow_last_k_blocks=getattr(args, "attn_flow_last_k_blocks", 0),
        share_attn_flow_modulator=getattr(args, "share_attn_flow_modulator", False),
        attn_flow_bottleneck=getattr(args, "attn_flow_bottleneck", 24),
        attn_flow_gate_bias=getattr(args, "attn_flow_gate_bias", -2.5),
        attn_flow_init_scale=getattr(args, "attn_flow_init_scale", 0.02),
        attn_flow_detail_topk=getattr(args, "attn_flow_detail_topk", 8),
        attn_flow_patch_only=getattr(args, "attn_flow_patch_only", False),
        patch_grid_refiner=args.patch_grid_refiner,
        patch_grid_refiner_scale=args.patch_grid_refiner_scale,
        patch_grid_refiner_last_k_blocks=args.patch_grid_refiner_last_k_blocks,
        share_patch_grid_refiner=args.share_patch_grid_refiner,
        patch_grid_refiner_bottleneck=args.patch_grid_refiner_bottleneck,
        patch_grid_refiner_gate_bias=args.patch_grid_refiner_gate_bias,
        patch_grid_refiner_init_scale=args.patch_grid_refiner_init_scale,
        patch_grid_refiner_cls_context_scale=args.patch_grid_refiner_cls_context_scale,
        attention_metric_adapter=args.attention_metric_adapter,
        attention_metric_type=args.attention_metric_type,
        attention_metric_scale=args.attention_metric_scale,
        attention_metric_last_k_blocks=args.attention_metric_last_k_blocks,
        share_attention_metric=args.share_attention_metric,
        attention_metric_bottleneck=args.attention_metric_bottleneck,
        attention_metric_patch_only=args.attention_metric_patch_only,
        attention_metric_gate_bias=args.attention_metric_gate_bias,
        attention_metric_init_scale=args.attention_metric_init_scale,
        attention_metric_cls_context_scale=args.attention_metric_cls_context_scale,
        geo_layer_scale_init=args.geo_layer_scale_init,
        geo_block_profile=args.geo_block_profile,
        geo_learnable_block_scale=args.geo_learnable_block_scale,
        token_flow_input=args.token_flow_input,
        token_flow_last_k_blocks=args.token_flow_last_k_blocks,
        share_token_flow=args.share_token_flow,
        token_flow_scale=args.token_flow_scale,
        token_flow_input_scale=args.token_flow_input_scale,
        token_flow_block_scale=args.token_flow_block_scale,
        token_flow_bottleneck=args.token_flow_bottleneck,
        token_flow_patch_only=args.token_flow_patch_only,
        token_flow_gate_bias=args.token_flow_gate_bias,
        token_flow_init_scale=args.token_flow_init_scale,
        token_flow_cls_context_scale=args.token_flow_cls_context_scale,
        token_flow_detail_topk=args.token_flow_detail_topk,
        token_flow_detail_boost_scale=args.token_flow_detail_boost_scale,
        inter_layer_flow=getattr(args, "inter_layer_flow", False),
        inter_layer_flow_last_k_blocks=getattr(args, "inter_layer_flow_last_k_blocks", 0),
        share_inter_layer_flow=getattr(args, "share_inter_layer_flow", True),
        inter_layer_flow_mode=getattr(args, "inter_layer_flow_mode", "transport"),
        inter_layer_flow_scale=getattr(args, "inter_layer_flow_scale", 1.0),
        inter_layer_flow_bottleneck=getattr(args, "inter_layer_flow_bottleneck", 16),
        inter_layer_flow_patch_only=getattr(args, "inter_layer_flow_patch_only", True),
        inter_layer_flow_gate_bias=getattr(args, "inter_layer_flow_gate_bias", -4.0),
        inter_layer_flow_init_scale=getattr(args, "inter_layer_flow_init_scale", 0.005),
        inter_layer_flow_cls_context_scale=getattr(args, "inter_layer_flow_cls_context_scale", 0.15),
        inter_layer_flow_delta_scale=getattr(args, "inter_layer_flow_delta_scale", 0.5),
        flow_state_carrier=getattr(args, "flow_state_carrier", False),
        flow_state_last_k_blocks=getattr(args, "flow_state_last_k_blocks", 0),
        share_flow_state_carrier=getattr(args, "share_flow_state_carrier", True),
        flow_state_dim=getattr(args, "flow_state_dim", 24),
        flow_state_scale=getattr(args, "flow_state_scale", 1.0),
        flow_state_gate_bias=getattr(args, "flow_state_gate_bias", -5.0),
        flow_state_init_scale=getattr(args, "flow_state_init_scale", 0.0025),
        flow_state_cls_scale=getattr(args, "flow_state_cls_scale", 1.0),
        flow_state_patch_scale=getattr(args, "flow_state_patch_scale", 0.1),
        residual_scale=args.residual_scale,
        geo_residual_budget=args.geo_residual_budget,
        spectral_scale=args.spectral_scale,
        low_rank_scale=args.low_rank_scale,
        rotation_scale=args.rotation_scale,
        use_conditioner=args.use_conditioner,
        use_internal_conditioner=getattr(args, "use_internal_conditioner", True),
        use_class_text_router=resolved_use_class_text_router,
        class_texts=class_texts,
        text_router_temperature=args.text_router_temperature,
        text_embedding_source=args.text_embedding_source,
        text_embedding_model=args.text_embedding_model,
    )


def main():
    parser = add_args(argparse.ArgumentParser(description="Train GeoViT-Proper backbone on CIFAR-100."))
    args = parser.parse_args()
    apply_locked_smoke_protocol(args)
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_transform, test_transform = build_transforms(args.img_size, strong_aug=not args.disable_strong_aug)
    train_ds = maybe_subset(build_cifar100(args.data_root, train=True, transform=train_transform), args.limit_train)
    test_ds = maybe_subset(build_cifar100(args.data_root, train=False, transform=test_transform), args.limit_test)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    eval_generator = torch.Generator()
    eval_generator.manual_seed(args.seed + 10_000)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=train_generator,
    )
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
        worker_init_fn=seed_worker if args.num_workers > 1 else None,
        generator=eval_generator,
    )

    class_texts = get_class_texts(train_ds)
    model = build_model(args, class_texts=class_texts).to(args.device)
    condition_table = build_condition_labels(100, args.condition_dim, torch.device(args.device)) if args.use_conditioner else None
    teacher = None
    teacher_source = ""
    anchor_model = None
    anchor_source = ""
    student_source = ""
    student_load_info = None
    if args.teacher_checkpoint:
        teacher = build_model(
            args,
            geo_on_fc1=False,
            geo_on_fc2=False,
            geo_on_attention=False,
            geo_attention_last_k_blocks=0,
            class_texts=None,
            use_class_text_router=False,
            flow_rank=0,
            flow_steps=1,
            flow_step_size=0.5,
        ).to(args.device)
        teacher_ckpt = load_checkpoint_state(args.teacher_checkpoint, map_location="cpu")
        teacher_state = teacher_ckpt.get("ema_state_dict") or teacher_ckpt.get("model_state_dict")
        teacher.load_state_dict(teacher_state, strict=True)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        teacher_source = args.teacher_checkpoint
        if args.teacher_init_student:
            initialize_student_from_teacher(
                model,
                teacher,
                gate_bias=args.teacher_init_gate_bias,
            )
    if args.anchor_checkpoint:
        anchor_model = build_model(
            args,
            class_texts=class_texts,
            use_class_text_router=args.use_class_text_router,
            flow_rank=0,
            flow_steps=1,
            flow_step_size=0.5,
        ).to(args.device)
        anchor_info = load_student_checkpoint_partial(anchor_model, args.anchor_checkpoint, map_location="cpu")
        print(json.dumps({"stage": "anchor_init", **anchor_info}), flush=True)
        anchor_model.eval()
        for param in anchor_model.parameters():
            param.requires_grad_(False)
        anchor_source = args.anchor_checkpoint
    if args.student_checkpoint:
        student_load_info = load_student_checkpoint_partial(model, args.student_checkpoint, map_location="cpu")
        student_source = args.student_checkpoint
        print(json.dumps({"stage": "student_init", **student_load_info}), flush=True)
        if (args.warm_start_manifold_from_anchor or args.warm_start_semantic_from_anchor) and args.manifold_refine_only:
            warm_info = warm_start_manifold_from_anchor(
                model,
                flow_scale=args.warm_start_flow_scale,
                warm_flow=args.warm_start_manifold_from_anchor,
                warm_semantic=args.warm_start_semantic_from_anchor,
            )
            print(json.dumps({"stage": "manifold_warm_start", **warm_info}), flush=True)

    current_phase = resolve_training_phase(args, 1)
    trainable_info, trainable_parameters, optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        args,
        phase=current_phase,
    )
    total_params, trainable_params = count_parameters(model)
    phase_history = [dict(current_phase)]
    print(json.dumps({"stage": "trainable_config", "phase": current_phase["name"], **trainable_info}), flush=True)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda") if (args.device.startswith("cuda") and not args.disable_amp) else None
    ema = ModelEma(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    ema_active = False
    if ema is not None:
        ema.set(model)

    history = []
    best_acc = float("-inf")
    best_epoch = 0
    start = time.perf_counter()
    peak_memory_mb = 0.0
    best_state = None

    if args.eval_only:
        eval_model = ema.module if ema is not None else model
        eval_adapter_scale = compute_geo_adapter_scale(
            args.epochs,
            start_epoch=args.geo_start_epoch,
            ramp_epochs=args.geo_ramp_epochs,
            target_scale=args.geo_target_scale,
            total_epochs=args.epochs,
            end_scale=args.geo_end_scale,
            decay_start_epoch=args.geo_decay_start_epoch,
        )
        eval_spectral_mult = compute_geo_adapter_scale(
            args.epochs,
            start_epoch=0,
            ramp_epochs=0,
            target_scale=1.0,
            total_epochs=args.epochs,
            end_scale=args.geo_spectral_end_mult,
            decay_start_epoch=args.geo_spectral_decay_start_epoch,
        )
        eval_low_rank_mult = compute_geo_adapter_scale(
            args.epochs,
            start_epoch=0,
            ramp_epochs=0,
            target_scale=1.0,
            total_epochs=args.epochs,
            end_scale=args.geo_low_rank_end_mult,
            decay_start_epoch=args.geo_low_rank_decay_start_epoch,
        )
        eval_rotation_mult = compute_geo_adapter_scale(
            args.epochs,
            start_epoch=0,
            ramp_epochs=0,
            target_scale=1.0,
            total_epochs=args.epochs,
            end_scale=args.geo_rotation_end_mult,
            decay_start_epoch=args.geo_rotation_decay_start_epoch,
        )
        set_geo_adapter_scale(eval_model, eval_adapter_scale)
        set_geo_component_scale_multipliers(
            eval_model,
            spectral=eval_spectral_mult,
            low_rank=eval_low_rank_mult,
            rotation=eval_rotation_mult,
        )
        test_loss, test_acc = evaluate(
            eval_model,
            test_loader,
            args.device,
            hflip_tta=args.eval_hflip_tta,
            shift_tta=args.eval_shift_tta,
            diagonal_shift_tta=args.eval_diagonal_shift_tta,
        )
        best_acc = test_acc
        best_epoch = 0
        best_state = copy.deepcopy(eval_model.state_dict())
        diagnostics = eval_model.get_diagnostics()
        history.append(
            {
                "epoch": 0,
                "train_loss": 0.0,
                "train_acc": 0.0,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "best_test_acc": best_acc,
                "epoch_seconds": 0.0,
                "samples_per_second": 0.0,
                "teacher_logit_weight": args.teacher_logit_weight,
                "teacher_feature_weight": args.teacher_feature_weight,
                **diagnostics,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "eval_only",
                    "test_acc": round(test_acc, 4),
                    "best_acc": round(best_acc, 4),
                    "diagnostics": diagnostics,
                }
            ),
            flush=True,
        )

    for epoch in range(1, args.epochs + 1):
        if args.eval_only:
            break
        phase = resolve_training_phase(args, epoch)
        if phase["name"] != current_phase["name"]:
            current_phase = phase
            phase_history.append(dict(current_phase))
            trainable_info, trainable_parameters, optimizer, scheduler = build_optimizer_and_scheduler(
                model,
                args,
                phase=current_phase,
            )
            total_params, trainable_params = count_parameters(model)
            if ema is not None and ema_active:
                ema.set(model)
            print(json.dumps({"stage": "phase_change", "epoch": epoch, "phase": current_phase["name"], **trainable_info}), flush=True)
        model.train()
        adapter_scale = compute_geo_adapter_scale(
            epoch,
            start_epoch=args.geo_start_epoch,
            ramp_epochs=args.geo_ramp_epochs,
            target_scale=args.geo_target_scale,
            total_epochs=args.epochs,
            end_scale=args.geo_end_scale,
            decay_start_epoch=args.geo_decay_start_epoch,
        )
        spectral_scale_mult = compute_geo_adapter_scale(
            epoch,
            start_epoch=0,
            ramp_epochs=0,
            target_scale=1.0,
            total_epochs=args.epochs,
            end_scale=args.geo_spectral_end_mult,
            decay_start_epoch=args.geo_spectral_decay_start_epoch,
        )
        low_rank_scale_mult = compute_geo_adapter_scale(
            epoch,
            start_epoch=0,
            ramp_epochs=0,
            target_scale=1.0,
            total_epochs=args.epochs,
            end_scale=args.geo_low_rank_end_mult,
            decay_start_epoch=args.geo_low_rank_decay_start_epoch,
        )
        rotation_scale_mult = compute_geo_adapter_scale(
            epoch,
            start_epoch=0,
            ramp_epochs=0,
            target_scale=1.0,
            total_epochs=args.epochs,
            end_scale=args.geo_rotation_end_mult,
            decay_start_epoch=args.geo_rotation_decay_start_epoch,
        )
        set_geo_adapter_scale(model, adapter_scale)
        set_geo_component_scale_multipliers(
            model,
            spectral=spectral_scale_mult,
            low_rank=low_rank_scale_mult,
            rotation=rotation_scale_mult,
        )
        fnfl_strength = compute_flow_ffn_strength(
            epoch,
            anneal_epochs=int(getattr(args, "fnfl_strength_anneal_epochs", 0)),
            max_strength=float(getattr(args, "fnfl_strength_max", 1.0)),
        )
        set_flow_ffn_strength(model, fnfl_strength)
        if ema is not None:
            set_geo_adapter_scale(ema.module, adapter_scale)
            set_geo_component_scale_multipliers(
                ema.module,
                spectral=spectral_scale_mult,
                low_rank=low_rank_scale_mult,
                rotation=rotation_scale_mult,
            )
            set_flow_ffn_strength(ema.module, fnfl_strength)
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        epoch_start = time.perf_counter()
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device=args.device)
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            mixed_images, targets_a, targets_b, lam = apply_batch_mix(
                images,
                labels,
                mixup_alpha=args.mixup_alpha,
                cutmix_alpha=args.cutmix_alpha,
                mix_prob=args.mix_prob,
                switch_prob=args.cutmix_switch_prob,
            )
            condition = build_condition_vectors(labels, condition_table)

            amp_enabled = args.device.startswith("cuda") and not args.disable_amp
            amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                student_block_features = None
                need_student_blocks = (
                    args.anchor_block_feature_weight > 0 or args.teacher_block_feature_weight > 0
                ) and hasattr(model, "forward_feature_pyramid")
                if need_student_blocks:
                    student_features, student_block_features = model.forward_feature_pyramid(mixed_images, condition=condition)
                else:
                    student_features = model(mixed_images, condition=condition, return_features=True)
                logits = model.head(student_features)
                teacher_logits = None
                teacher_features = None
                teacher_block_features = None
                anchor_logits = None
                anchor_features = None
                anchor_block_features = None
                if teacher is not None:
                    with torch.no_grad():
                        if args.teacher_block_feature_weight > 0 and hasattr(teacher, "forward_feature_pyramid"):
                            teacher_features, teacher_block_features = teacher.forward_feature_pyramid(mixed_images, condition=None)
                        else:
                            teacher_features = teacher(mixed_images, return_features=True)
                        teacher_logits = teacher.head(teacher_features)
                if anchor_model is not None:
                    with torch.no_grad():
                        if args.anchor_block_feature_weight > 0 and hasattr(anchor_model, "forward_feature_pyramid"):
                            anchor_features, anchor_block_features = anchor_model.forward_feature_pyramid(mixed_images, condition=condition)
                        else:
                            anchor_features = anchor_model(mixed_images, condition=condition, return_features=True)
                        anchor_logits = anchor_model.head(anchor_features)
                loss = mixup_criterion(loss_fn, logits, targets_a, targets_b, lam)
                kd_component = torch.tensor(0.0, device=logits.device)
                feat_component = torch.tensor(0.0, device=logits.device)
                teacher_block_feat_component = torch.tensor(0.0, device=logits.device)
                anchor_kd_component = torch.tensor(0.0, device=logits.device)
                anchor_feat_component = torch.tensor(0.0, device=logits.device)
                anchor_block_feat_component = torch.tensor(0.0, device=logits.device)
                if teacher_logits is not None and args.teacher_logit_weight > 0:
                    kd_component = kd_loss(logits, teacher_logits, args.teacher_temperature)
                    loss = loss + args.teacher_logit_weight * kd_component
                if teacher_features is not None and args.teacher_feature_weight > 0:
                    feat_component = feature_distill_loss(student_features, teacher_features)
                    loss = loss + args.teacher_feature_weight * feat_component
                if (
                    teacher_block_features is not None
                    and student_block_features is not None
                    and args.teacher_block_feature_weight > 0
                ):
                    teacher_block_feat_component = block_feature_distill_loss(
                        student_block_features,
                        teacher_block_features,
                        last_k=args.teacher_block_feature_layers,
                    )
                    loss = loss + args.teacher_block_feature_weight * teacher_block_feat_component
                if anchor_logits is not None and args.anchor_logit_weight > 0:
                    anchor_kd_component = kd_loss(logits, anchor_logits, args.teacher_temperature)
                    loss = loss + args.anchor_logit_weight * anchor_kd_component
                if anchor_features is not None and args.anchor_feature_weight > 0:
                    anchor_feat_component = feature_distill_loss(student_features, anchor_features)
                    loss = loss + args.anchor_feature_weight * anchor_feat_component
                if (
                    anchor_block_features is not None
                    and student_block_features is not None
                    and args.anchor_block_feature_weight > 0
                ):
                    anchor_block_feat_component = block_feature_distill_loss(
                        student_block_features,
                        anchor_block_features,
                        last_k=args.anchor_block_feature_layers,
                    )
                    loss = loss + args.anchor_block_feature_weight * anchor_block_feat_component
                aux_losses = model.get_aux_losses()
                semantic_anchor_component = torch.tensor(0.0, device=logits.device)
                semantic_expert_component = torch.tensor(0.0, device=logits.device)
                semantic_mode_component = torch.tensor(0.0, device=logits.device)
                geo_structure_component = torch.tensor(0.0, device=logits.device)
                flow_anchor_component = torch.tensor(0.0, device=logits.device)
                flow_energy_component = torch.tensor(0.0, device=logits.device)
                magnus_anchor_component = torch.tensor(0.0, device=logits.device)
                magnus_motion_component = torch.tensor(0.0, device=logits.device)
                magnus_comm_component = torch.tensor(0.0, device=logits.device)
                magnus_mode_component = torch.tensor(0.0, device=logits.device)
                geo_structure_weight = compute_scheduled_weight(
                    epoch,
                    start_epoch=args.geo_structure_start_epoch,
                    ramp_epochs=args.geo_structure_ramp_epochs,
                    target_weight=args.geo_structure_weight,
                )
                if args.semantic_anchor_weight > 0 and "fc1_semantic_anchor_loss" in aux_losses:
                    semantic_anchor_component = aux_losses["fc1_semantic_anchor_loss"]
                    loss = loss + args.semantic_anchor_weight * semantic_anchor_component
                if args.semantic_expert_weight > 0 and "fc1_semantic_entropy_loss" in aux_losses:
                    semantic_expert_component = aux_losses["fc1_semantic_entropy_loss"]
                    loss = loss + args.semantic_expert_weight * semantic_expert_component
                if args.semantic_mode_weight > 0 and "fc1_semantic_mode_reg_loss" in aux_losses:
                    semantic_mode_component = aux_losses["fc1_semantic_mode_reg_loss"]
                    loss = loss + args.semantic_mode_weight * semantic_mode_component
                if geo_structure_weight > 0 and "fc1_geo_structure_loss" in aux_losses:
                    geo_structure_component = aux_losses["fc1_geo_structure_loss"]
                    loss = loss + geo_structure_weight * geo_structure_component
                if args.flow_anchor_weight > 0 and "fc1_flow_anchor_loss" in aux_losses:
                    flow_anchor_component = aux_losses["fc1_flow_anchor_loss"]
                    loss = loss + args.flow_anchor_weight * flow_anchor_component
                if args.flow_energy_weight > 0 and "fc1_flow_energy_loss" in aux_losses:
                    flow_energy_component = aux_losses["fc1_flow_energy_loss"]
                    loss = loss + args.flow_energy_weight * flow_energy_component
                if args.magnus_anchor_weight > 0 and "fc1_magnus_anchor_loss" in aux_losses:
                    magnus_anchor_component = aux_losses["fc1_magnus_anchor_loss"]
                    loss = loss + args.magnus_anchor_weight * magnus_anchor_component
                if args.magnus_motion_weight > 0 and "fc1_magnus_motion_loss" in aux_losses:
                    magnus_motion_component = aux_losses["fc1_magnus_motion_loss"]
                    loss = loss + args.magnus_motion_weight * magnus_motion_component
                if args.magnus_comm_weight > 0 and "fc1_magnus_comm_loss" in aux_losses:
                    magnus_comm_component = aux_losses["fc1_magnus_comm_loss"]
                    loss = loss + args.magnus_comm_weight * magnus_comm_component
                if args.magnus_mode_weight > 0 and "fc1_magnus_mode_reg_loss" in aux_losses:
                    magnus_mode_component = aux_losses["fc1_magnus_mode_reg_loss"]
                    loss = loss + args.magnus_mode_weight * magnus_mode_component
                loss = loss / max(args.grad_accum_steps, 1)

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if batch_idx % max(args.grad_accum_steps, 1) == 0 or batch_idx == len(train_loader):
                if scaler is not None:
                    if args.grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if args.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None and ema_active:
                    ema.update(model)

            total_loss += float(loss.item()) * int(labels.shape[0]) * max(args.grad_accum_steps, 1)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_count += int(labels.shape[0])

        train_loss = total_loss / max(total_count, 1)
        train_acc = 100.0 * total_correct / max(total_count, 1)
        eval_model = ema.module if (ema is not None and ema_active) else model
        set_geo_adapter_scale(eval_model, adapter_scale)
        test_loss, test_acc = evaluate(
            eval_model,
            test_loader,
            args.device,
            hflip_tta=args.eval_hflip_tta,
            shift_tta=args.eval_shift_tta,
            diagonal_shift_tta=args.eval_diagonal_shift_tta,
        )
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            best_state = copy.deepcopy(eval_model.state_dict())
        diagnostics = eval_model.get_diagnostics()
        if scheduler is not None:
            scheduler.step()
        if ema is not None and not ema_active and epoch >= args.ema_start_epoch:
            ema.set(model)
            ema_active = True
        epoch_seconds = time.perf_counter() - epoch_start
        samples_per_second = len(train_ds) / max(epoch_seconds, 1e-6)
        if args.device.startswith("cuda"):
            peak_memory_mb = max(peak_memory_mb, torch.cuda.max_memory_allocated(device=args.device) / (1024 ** 2))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "best_test_acc": best_acc,
                "epoch_seconds": epoch_seconds,
                "samples_per_second": samples_per_second,
                "phase": current_phase["name"],
                "adapter_scale": adapter_scale,
                "teacher_logit_weight": args.teacher_logit_weight,
                "teacher_feature_weight": args.teacher_feature_weight,
                "geo_structure_weight": float(geo_structure_weight),
                **diagnostics,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "epoch",
                    "epoch": epoch,
                    "phase": current_phase["name"],
                    "adapter_scale": round(adapter_scale, 4),
                    "train_acc": round(train_acc, 4),
                    "test_acc": round(test_acc, 4),
                    "best_acc": round(best_acc, 4),
                    "samples_per_second": round(samples_per_second, 2),
                    "diagnostics": diagnostics,
                    "semantic_losses": {
                        "anchor": float(semantic_anchor_component.item()),
                        "expert": float(semantic_expert_component.item()),
                        "mode": float(semantic_mode_component.item()),
                        "geo_structure": float(geo_structure_component.item()),
                        "geo_structure_weight": float(geo_structure_weight),
                        "flow_anchor": float(flow_anchor_component.item()),
                        "flow_energy": float(flow_energy_component.item()),
                        "magnus_anchor": float(magnus_anchor_component.item()),
                        "magnus_motion": float(magnus_motion_component.item()),
                        "magnus_comm": float(magnus_comm_component.item()),
                        "magnus_mode": float(magnus_mode_component.item()),
                        "teacher_block_feature": float(teacher_block_feat_component.item()),
                        "anchor_logit": float(anchor_kd_component.item()),
                        "anchor_feature": float(anchor_feat_component.item()),
                        "anchor_block_feature": float(anchor_block_feat_component.item()),
                    },
                }
            ),
            flush=True,
        )

    run_dir = Path(args.save_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    comparison_protocol = build_comparison_protocol(args)
    architecture_signature = build_architecture_signature(args)
    payload = {
        "run_name": args.run_name,
        "seed": args.seed,
        "variant": build_variant_name(args),
        "locked_smoke_protocol": str(args.locked_smoke_protocol).strip(),
        "comparison_protocol": comparison_protocol,
        "comparison_protocol_hash": stable_hash(comparison_protocol),
        "architecture_signature": architecture_signature,
        "architecture_signature_hash": stable_hash(architecture_signature),
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "best_acc": round(best_acc, 4),
        "best_epoch": best_epoch,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "peak_memory_mb": round(peak_memory_mb, 3),
        "last_epoch_seconds": round(history[-1]["epoch_seconds"], 3) if history else 0.0,
        "last_samples_per_second": round(history[-1]["samples_per_second"], 3) if history else 0.0,
        "model_config": {
            "img_size": args.img_size,
            "patch_size": args.patch_size,
            "tokenizer_type": args.tokenizer_type,
            "detail_tokens": args.detail_tokens,
            "detail_score_type": args.detail_score_type,
            "embed_dim": args.embed_dim,
            "depth": args.depth,
            "num_heads": args.num_heads,
            "num_spectral_bases": args.num_spectral_bases,
            "low_rank_rank": args.low_rank_rank,
            "orthogonal_rank": args.orthogonal_rank,
            "fc1_base_rank": args.fc1_base_rank,
            "stem_flow_refiner": getattr(args, "stem_flow_refiner", False),
            "stem_flow_scale": getattr(args, "stem_flow_scale", 1.0),
            "stem_flow_bottleneck": getattr(args, "stem_flow_bottleneck", 16),
            "stem_flow_gate_bias": getattr(args, "stem_flow_gate_bias", -3.0),
            "stem_flow_init_scale": getattr(args, "stem_flow_init_scale", 0.02),
            "stem_flow_detail_scale": getattr(args, "stem_flow_detail_scale", 0.25),
            "stem_flow_context_scale": getattr(args, "stem_flow_context_scale", 0.5),
            "summary_token": getattr(args, "summary_token", False),
            "summary_token_scale": getattr(args, "summary_token_scale", 1.0),
            "summary_head_fusion": getattr(args, "summary_head_fusion", 0.5),
            "share_fc1_bank": (
                args.share_fc1_bank
                and not (args.magnus_semantic_mode or getattr(args, "magnus_single_operator", False))
                and not (args.magnus_rotation_mode and int(args.magnus_rotation_last_k_blocks) > 0)
            ),
            "fc1_bank_groups": args.fc1_bank_groups,
            "share_fc1_conditioner": args.share_fc1_conditioner,
            "fc1_conditioner_groups": args.fc1_conditioner_groups,
            "coupled_spectral_low_rank": getattr(args, "coupled_spectral_low_rank", False),
            "coupled_learnable_input_basis": getattr(args, "coupled_learnable_input_basis", False),
            "coupled_shared_gate": getattr(args, "coupled_shared_gate", False),
            "geo_on_fc1": args.geo_on_fc1,
            "geo_fc1_last_k_blocks": args.geo_fc1_last_k_blocks,
            "enable_local_geo": args.enable_local_geo,
            "local_geo_last_k_blocks": getattr(args, "local_geo_last_k_blocks", 0),
            "geo_on_fc2": args.geo_on_fc2,
            "geo_on_attention": args.geo_on_attention,
            "geo_attention_last_k_blocks": args.geo_attention_last_k_blocks,
            "hidden_diffusion": args.hidden_diffusion,
            "hidden_diffusion_scale": args.hidden_diffusion_scale,
            "hidden_diffusion_last_k_blocks": args.hidden_diffusion_last_k_blocks,
            "share_hidden_diffusion": args.share_hidden_diffusion,
            "hidden_diffusion_bottleneck": args.hidden_diffusion_bottleneck,
            "hidden_diffusion_gate_bias": args.hidden_diffusion_gate_bias,
            "hidden_diffusion_init_scale": args.hidden_diffusion_init_scale,
            "hidden_diffusion_cls_context_scale": args.hidden_diffusion_cls_context_scale,
            "hidden_grid_refiner": getattr(args, "hidden_grid_refiner", False),
            "hidden_grid_refiner_scale": getattr(args, "hidden_grid_refiner_scale", 1.0),
            "hidden_grid_refiner_last_k_blocks": getattr(args, "hidden_grid_refiner_last_k_blocks", 0),
            "share_hidden_grid_refiner": getattr(args, "share_hidden_grid_refiner", False),
            "hidden_grid_refiner_bottleneck": getattr(args, "hidden_grid_refiner_bottleneck", 16),
            "hidden_grid_refiner_gate_bias": getattr(args, "hidden_grid_refiner_gate_bias", -5.0),
            "hidden_grid_refiner_init_scale": getattr(args, "hidden_grid_refiner_init_scale", 0.002),
            "hidden_grid_refiner_cls_context_scale": getattr(args, "hidden_grid_refiner_cls_context_scale", 0.05),
            "hidden_cls_bridge": getattr(args, "hidden_cls_bridge", False),
            "hidden_cls_bridge_scale": getattr(args, "hidden_cls_bridge_scale", 1.0),
            "hidden_cls_bridge_last_k_blocks": getattr(args, "hidden_cls_bridge_last_k_blocks", 0),
            "share_hidden_cls_bridge": getattr(args, "share_hidden_cls_bridge", False),
            "hidden_cls_bridge_bottleneck": getattr(args, "hidden_cls_bridge_bottleneck", 16),
            "hidden_cls_bridge_gate_bias": getattr(args, "hidden_cls_bridge_gate_bias", -4.0),
            "hidden_cls_bridge_init_scale": getattr(args, "hidden_cls_bridge_init_scale", 0.01),
            "hidden_cls_bridge_patch_feedback_scale": getattr(args, "hidden_cls_bridge_patch_feedback_scale", 0.0),
            "hidden_channel_flow": getattr(args, "hidden_channel_flow", False),
            "hidden_channel_flow_scale": getattr(args, "hidden_channel_flow_scale", 1.0),
            "hidden_channel_flow_last_k_blocks": getattr(args, "hidden_channel_flow_last_k_blocks", 0),
            "share_hidden_channel_flow": getattr(args, "share_hidden_channel_flow", False),
            "hidden_channel_flow_bottleneck": getattr(args, "hidden_channel_flow_bottleneck", 16),
            "hidden_channel_flow_rank": getattr(args, "hidden_channel_flow_rank", 16),
            "hidden_channel_flow_gate_bias": getattr(args, "hidden_channel_flow_gate_bias", -3.5),
            "hidden_channel_flow_init_scale": getattr(args, "hidden_channel_flow_init_scale", 0.01),
            "hidden_channel_flow_patch_only": getattr(args, "hidden_channel_flow_patch_only", False),
            "hidden_channel_flow_cls_mix_scale": getattr(args, "hidden_channel_flow_cls_mix_scale", 1.0),
            "hidden_channel_flow_mean_mix_scale": getattr(args, "hidden_channel_flow_mean_mix_scale", 0.5),
            "response_flow_norm": args.response_flow_norm,
            "response_flow_scale": args.response_flow_scale,
            "response_flow_last_k_blocks": args.response_flow_last_k_blocks,
            "response_flow_init_scale": args.response_flow_init_scale,
            "response_flow_mode": getattr(args, "response_flow_mode", "simple"),
            "share_response_flow": getattr(args, "share_response_flow", False),
            "response_flow_bottleneck": getattr(args, "response_flow_bottleneck", 12),
            "response_flow_gate_bias": getattr(args, "response_flow_gate_bias", -4.0),
            "response_flow_patch_only": getattr(args, "response_flow_patch_only", False),
            "response_flow_cls_mix_scale": getattr(args, "response_flow_cls_mix_scale", 1.0),
            "response_flow_mean_mix_scale": getattr(args, "response_flow_mean_mix_scale", 0.5),
            "response_flow_token_exponent": getattr(args, "response_flow_token_exponent", 0.5),
            "response_flow_channel_exponent": getattr(args, "response_flow_channel_exponent", 0.5),
            "response_flow_pre_act": getattr(args, "response_flow_pre_act", False),
            "dual_path_mlp": getattr(args, "dual_path_mlp", False),
            "dual_path_last_k_blocks": getattr(args, "dual_path_last_k_blocks", 0),
            "dual_path_refine_ratio": getattr(args, "dual_path_refine_ratio", 0.25),
            "dual_path_cross_scale": getattr(args, "dual_path_cross_scale", 1.0),
            "dual_path_gate_bias": getattr(args, "dual_path_gate_bias", -2.0),
            "hidden_group_router": getattr(args, "hidden_group_router", False),
            "hidden_group_router_last_k_blocks": getattr(args, "hidden_group_router_last_k_blocks", 0),
            "hidden_group_router_groups": getattr(args, "hidden_group_router_groups", 0),
            "hidden_group_router_scale": getattr(args, "hidden_group_router_scale", 1.0),
            "hidden_group_router_gate_bias": getattr(args, "hidden_group_router_gate_bias", -2.0),
            "hidden_group_router_init_scale": getattr(args, "hidden_group_router_init_scale", 0.1),
            "hidden_group_router_cls_mix_scale": getattr(args, "hidden_group_router_cls_mix_scale", 1.0),
            "hidden_group_router_mean_mix_scale": getattr(args, "hidden_group_router_mean_mix_scale", 0.5),
            "attention_hidden_fusion": getattr(args, "attention_hidden_fusion", False),
            "attention_hidden_fusion_last_k_blocks": getattr(args, "attention_hidden_fusion_last_k_blocks", 0),
            "share_attention_hidden_fusion": getattr(args, "share_attention_hidden_fusion", True),
            "attention_hidden_fusion_scale": getattr(args, "attention_hidden_fusion_scale", 1.0),
            "attention_hidden_fusion_bottleneck": getattr(args, "attention_hidden_fusion_bottleneck", 6),
            "attention_hidden_fusion_gate_bias": getattr(args, "attention_hidden_fusion_gate_bias", -2.5),
            "attention_hidden_fusion_init_scale": getattr(args, "attention_hidden_fusion_init_scale", 0.02),
            "attention_hidden_fusion_patch_only": getattr(args, "attention_hidden_fusion_patch_only", False),
            "attention_hidden_fusion_cls_context_scale": getattr(args, "attention_hidden_fusion_cls_context_scale", 1.0),
            "hidden_token_mixer": getattr(args, "hidden_token_mixer", False),
            "hidden_token_mixer_last_k_blocks": getattr(args, "hidden_token_mixer_last_k_blocks", 0),
            "share_hidden_token_mixer": getattr(args, "share_hidden_token_mixer", True),
            "hidden_token_mixer_scale": getattr(args, "hidden_token_mixer_scale", 1.0),
            "hidden_token_mixer_gate_bias": getattr(args, "hidden_token_mixer_gate_bias", -3.0),
            "hidden_token_mixer_init_scale": getattr(args, "hidden_token_mixer_init_scale", 0.02),
            "hidden_token_mixer_patch_only": getattr(args, "hidden_token_mixer_patch_only", True),
            "hidden_token_mixer_mode": getattr(args, "hidden_token_mixer_mode", "conv"),
            "hidden_token_mixer_topk": getattr(args, "hidden_token_mixer_topk", 8),
            "competitive_residual": getattr(args, "competitive_residual", False),
            "competitive_residual_last_k_blocks": getattr(args, "competitive_residual_last_k_blocks", 0),
            "competitive_residual_scale": getattr(args, "competitive_residual_scale", 1.0),
            "competitive_residual_gate_bias": getattr(args, "competitive_residual_gate_bias", 0.0),
            "competitive_residual_init_scale": getattr(args, "competitive_residual_init_scale", 1.0),
            "competitive_residual_cls_mix_scale": getattr(args, "competitive_residual_cls_mix_scale", 1.0),
            "competitive_residual_mean_mix_scale": getattr(args, "competitive_residual_mean_mix_scale", 0.5),
            "competitive_residual_patch_only": getattr(args, "competitive_residual_patch_only", False),
            "parallel_block_update": getattr(args, "parallel_block_update", False),
            "parallel_block_last_k_blocks": getattr(args, "parallel_block_last_k_blocks", 0),
            "mlp_first_update": getattr(args, "mlp_first_update", False),
            "mlp_first_last_k_blocks": getattr(args, "mlp_first_last_k_blocks", 0),
            "tail_token_mixer": getattr(args, "tail_token_mixer", False),
            "tail_token_mixer_last_k_blocks": getattr(args, "tail_token_mixer_last_k_blocks", 0),
            "tail_token_mixer_scale": getattr(args, "tail_token_mixer_scale", 1.0),
            "tail_token_mixer_gate_bias": getattr(args, "tail_token_mixer_gate_bias", -3.0),
            "tail_token_mixer_init_scale": getattr(args, "tail_token_mixer_init_scale", 0.02),
            "tail_token_mixer_patch_only": getattr(args, "tail_token_mixer_patch_only", True),
            "activation_flow": args.activation_flow,
            "activation_flow_scale": args.activation_flow_scale,
            "activation_flow_last_k_blocks": args.activation_flow_last_k_blocks,
            "share_activation_flow": args.share_activation_flow,
            "activation_flow_bottleneck": args.activation_flow_bottleneck,
            "activation_flow_gate_bias": args.activation_flow_gate_bias,
            "activation_flow_init_scale": args.activation_flow_init_scale,
            "activation_flow_patch_only": args.activation_flow_patch_only,
            "activation_flow_cls_mix_scale": args.activation_flow_cls_mix_scale,
            "activation_flow_mean_mix_scale": args.activation_flow_mean_mix_scale,
            "activation_flow_std_mix_scale": args.activation_flow_std_mix_scale,
            "activation_flow_cls_token_scale": args.activation_flow_cls_token_scale,
            "attn_flow_modulator": getattr(args, "attn_flow_modulator", False),
            "attn_flow_scale": getattr(args, "attn_flow_scale", 1.0),
            "attn_flow_last_k_blocks": getattr(args, "attn_flow_last_k_blocks", 0),
            "share_attn_flow_modulator": getattr(args, "share_attn_flow_modulator", False),
            "attn_flow_bottleneck": getattr(args, "attn_flow_bottleneck", 24),
            "attn_flow_gate_bias": getattr(args, "attn_flow_gate_bias", -2.5),
            "attn_flow_init_scale": getattr(args, "attn_flow_init_scale", 0.02),
            "attn_flow_detail_topk": getattr(args, "attn_flow_detail_topk", 8),
            "attn_flow_patch_only": getattr(args, "attn_flow_patch_only", False),
            "patch_grid_refiner": args.patch_grid_refiner,
            "patch_grid_refiner_scale": args.patch_grid_refiner_scale,
            "patch_grid_refiner_last_k_blocks": args.patch_grid_refiner_last_k_blocks,
            "share_patch_grid_refiner": args.share_patch_grid_refiner,
            "patch_grid_refiner_bottleneck": args.patch_grid_refiner_bottleneck,
            "patch_grid_refiner_gate_bias": args.patch_grid_refiner_gate_bias,
            "patch_grid_refiner_init_scale": args.patch_grid_refiner_init_scale,
            "patch_grid_refiner_cls_context_scale": args.patch_grid_refiner_cls_context_scale,
            "attention_metric_adapter": args.attention_metric_adapter,
            "attention_metric_type": args.attention_metric_type,
            "attention_metric_scale": args.attention_metric_scale,
            "attention_metric_last_k_blocks": args.attention_metric_last_k_blocks,
            "share_attention_metric": args.share_attention_metric,
            "attention_metric_bottleneck": args.attention_metric_bottleneck,
            "attention_metric_patch_only": args.attention_metric_patch_only,
            "attention_metric_gate_bias": args.attention_metric_gate_bias,
            "attention_metric_init_scale": args.attention_metric_init_scale,
            "attention_metric_cls_context_scale": args.attention_metric_cls_context_scale,
            "geo_layer_scale_init": args.geo_layer_scale_init,
            "geo_block_profile": args.geo_block_profile,
            "geo_learnable_block_scale": args.geo_learnable_block_scale,
            "token_flow_input": args.token_flow_input,
            "token_flow_last_k_blocks": args.token_flow_last_k_blocks,
            "share_token_flow": args.share_token_flow,
            "token_flow_scale": args.token_flow_scale,
            "token_flow_input_scale": args.token_flow_input_scale,
            "token_flow_block_scale": args.token_flow_block_scale,
            "token_flow_bottleneck": args.token_flow_bottleneck,
            "token_flow_patch_only": args.token_flow_patch_only,
            "token_flow_gate_bias": args.token_flow_gate_bias,
            "token_flow_init_scale": args.token_flow_init_scale,
            "token_flow_cls_context_scale": args.token_flow_cls_context_scale,
            "token_flow_detail_topk": args.token_flow_detail_topk,
            "token_flow_detail_boost_scale": args.token_flow_detail_boost_scale,
            "inter_layer_flow": getattr(args, "inter_layer_flow", False),
            "inter_layer_flow_last_k_blocks": getattr(args, "inter_layer_flow_last_k_blocks", 0),
            "share_inter_layer_flow": getattr(args, "share_inter_layer_flow", True),
            "inter_layer_flow_mode": getattr(args, "inter_layer_flow_mode", "transport"),
            "inter_layer_flow_scale": getattr(args, "inter_layer_flow_scale", 1.0),
            "inter_layer_flow_bottleneck": getattr(args, "inter_layer_flow_bottleneck", 16),
            "inter_layer_flow_patch_only": getattr(args, "inter_layer_flow_patch_only", True),
            "inter_layer_flow_gate_bias": getattr(args, "inter_layer_flow_gate_bias", -4.0),
            "inter_layer_flow_init_scale": getattr(args, "inter_layer_flow_init_scale", 0.005),
            "inter_layer_flow_cls_context_scale": getattr(args, "inter_layer_flow_cls_context_scale", 0.15),
            "inter_layer_flow_delta_scale": getattr(args, "inter_layer_flow_delta_scale", 0.5),
            "flow_state_carrier": getattr(args, "flow_state_carrier", False),
            "flow_state_last_k_blocks": getattr(args, "flow_state_last_k_blocks", 0),
            "share_flow_state_carrier": getattr(args, "share_flow_state_carrier", True),
            "flow_state_dim": getattr(args, "flow_state_dim", 24),
            "flow_state_scale": getattr(args, "flow_state_scale", 1.0),
            "flow_state_gate_bias": getattr(args, "flow_state_gate_bias", -5.0),
            "flow_state_init_scale": getattr(args, "flow_state_init_scale", 0.0025),
            "flow_state_cls_scale": getattr(args, "flow_state_cls_scale", 1.0),
            "flow_state_patch_scale": getattr(args, "flow_state_patch_scale", 0.1),
            "residual_scale": args.residual_scale,
            "geo_residual_budget": args.geo_residual_budget,
            "spectral_scale": args.spectral_scale,
            "low_rank_scale": args.low_rank_scale,
            "rotation_scale": args.rotation_scale,
            "use_conditioner": args.use_conditioner,
            "use_internal_conditioner": getattr(args, "use_internal_conditioner", True),
            "use_class_text_router": args.use_class_text_router,
            "text_router_temperature": args.text_router_temperature,
            "text_embedding_source": args.text_embedding_source,
            "text_embedding_model": args.text_embedding_model,
            "flow_rank": args.flow_rank,
            "flow_steps": args.flow_steps,
            "flow_step_size": args.flow_step_size,
            "semantic_manifold_mode": args.semantic_manifold_mode,
            "semantic_num_experts": args.semantic_num_experts,
            "semantic_expert_temperature": args.semantic_expert_temperature,
            "magnus_semantic_mode": args.magnus_semantic_mode or getattr(args, "magnus_single_operator", False),
            "magnus_single_operator": getattr(args, "magnus_single_operator", False),
            "magnus_detail_topk": args.magnus_detail_topk,
            "magnus_rotation_mode": args.magnus_rotation_mode,
            "magnus_rotation_last_k_blocks": args.magnus_rotation_last_k_blocks,
            "magnus_rotation_strength": args.magnus_rotation_strength,
            "coupled_spectral_low_rank": getattr(args, "coupled_spectral_low_rank", False),
            "coupled_learnable_input_basis": getattr(args, "coupled_learnable_input_basis", False),
            "coupled_shared_gate": getattr(args, "coupled_shared_gate", False),
            "strict_semantic_operator": args.strict_semantic_operator,
            "manifold_alignment_mode": args.manifold_alignment_mode,
            "manifold_refine_only": args.manifold_refine_only,
            "manifold_train_head": args.manifold_train_head,
            "manifold_train_output_basis": args.manifold_train_output_basis,
            "manifold_train_text_router": args.manifold_train_text_router,
            "geo_refine_only": args.geo_refine_only,
            "geo_train_head": args.geo_train_head,
            "geo_train_output_basis": args.geo_train_output_basis,
            "geo_head_warmup_epochs": args.geo_head_warmup_epochs,
            "geo_basis_last_epochs": args.geo_basis_last_epochs,
            "head_probe_lr": args.head_probe_lr,
            "geo_start_epoch": args.geo_start_epoch,
            "geo_ramp_epochs": args.geo_ramp_epochs,
            "geo_target_scale": args.geo_target_scale,
            "geo_end_scale": args.geo_end_scale,
            "geo_decay_start_epoch": args.geo_decay_start_epoch,
            "geo_spectral_end_mult": args.geo_spectral_end_mult,
            "geo_spectral_decay_start_epoch": args.geo_spectral_decay_start_epoch,
            "geo_low_rank_end_mult": args.geo_low_rank_end_mult,
            "geo_low_rank_decay_start_epoch": args.geo_low_rank_decay_start_epoch,
            "geo_rotation_end_mult": args.geo_rotation_end_mult,
            "geo_rotation_decay_start_epoch": args.geo_rotation_decay_start_epoch,
            "disable_strong_aug": args.disable_strong_aug,
            "eval_only": args.eval_only,
            "eval_hflip_tta": args.eval_hflip_tta,
            "eval_shift_tta": args.eval_shift_tta,
            "eval_diagonal_shift_tta": args.eval_diagonal_shift_tta,
            "warm_start_manifold_from_anchor": args.warm_start_manifold_from_anchor,
            "warm_start_semantic_from_anchor": args.warm_start_semantic_from_anchor,
            "warm_start_flow_scale": args.warm_start_flow_scale,
        },
        "runtime_config": {
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "eval_prefetch_factor": args.eval_prefetch_factor,
            "lr": args.lr,
            "warmup_epochs": args.warmup_epochs,
            "warmup_start_factor": args.warmup_start_factor,
            "min_lr_ratio": args.min_lr_ratio,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "ema_decay": args.ema_decay,
            "ema_start_epoch": args.ema_start_epoch,
            "grad_clip_norm": args.grad_clip_norm,
            "disable_amp": args.disable_amp,
            "amp_dtype": args.amp_dtype,
            "eval_hflip_tta": args.eval_hflip_tta,
            "eval_shift_tta": args.eval_shift_tta,
            "eval_diagonal_shift_tta": args.eval_diagonal_shift_tta,
            "matmul_precision": "high",
            "tf32_matmul": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)) if torch.cuda.is_available() else False,
            "tf32_cudnn": bool(getattr(torch.backends.cudnn, "allow_tf32", False)) if torch.cuda.is_available() else False,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else args.device,
        },
        "teacher_source": teacher_source,
        "anchor_source": anchor_source,
        "student_source": student_source,
        "student_load_info": student_load_info,
        "teacher_init_student": args.teacher_init_student,
        "manifold_refine_only": args.manifold_refine_only,
        "manifold_train_head": args.manifold_train_head,
        "manifold_train_output_basis": args.manifold_train_output_basis,
        "manifold_train_text_router": args.manifold_train_text_router,
        "geo_refine_only": args.geo_refine_only,
        "geo_train_head": args.geo_train_head,
        "geo_train_output_basis": args.geo_train_output_basis,
        "geo_head_warmup_epochs": args.geo_head_warmup_epochs,
        "geo_basis_last_epochs": args.geo_basis_last_epochs,
        "head_probe_lr": args.head_probe_lr,
        "geo_start_epoch": args.geo_start_epoch,
        "geo_ramp_epochs": args.geo_ramp_epochs,
        "geo_target_scale": args.geo_target_scale,
        "geo_end_scale": args.geo_end_scale,
        "geo_decay_start_epoch": args.geo_decay_start_epoch,
        "geo_spectral_end_mult": args.geo_spectral_end_mult,
        "geo_spectral_decay_start_epoch": args.geo_spectral_decay_start_epoch,
        "geo_low_rank_end_mult": args.geo_low_rank_end_mult,
        "geo_low_rank_decay_start_epoch": args.geo_low_rank_decay_start_epoch,
        "geo_rotation_end_mult": args.geo_rotation_end_mult,
        "geo_rotation_decay_start_epoch": args.geo_rotation_decay_start_epoch,
        "fc1_bank_groups": args.fc1_bank_groups,
        "fc1_conditioner_groups": args.fc1_conditioner_groups,
        "geo_fc1_last_k_blocks": args.geo_fc1_last_k_blocks,
        "local_geo_last_k_blocks": getattr(args, "local_geo_last_k_blocks", 0),
        "geo_attention_last_k_blocks": args.geo_attention_last_k_blocks,
        "warm_start_manifold_from_anchor": args.warm_start_manifold_from_anchor,
        "warm_start_semantic_from_anchor": args.warm_start_semantic_from_anchor,
        "warm_start_flow_scale": args.warm_start_flow_scale,
        "magnus_semantic_mode": args.magnus_semantic_mode or getattr(args, "magnus_single_operator", False),
        "magnus_single_operator": getattr(args, "magnus_single_operator", False),
        "magnus_detail_topk": args.magnus_detail_topk,
        "magnus_rotation_mode": args.magnus_rotation_mode,
        "magnus_rotation_last_k_blocks": args.magnus_rotation_last_k_blocks,
        "magnus_rotation_strength": args.magnus_rotation_strength,
        "coupled_spectral_low_rank": bool(getattr(args, "coupled_spectral_low_rank", False)),
        "coupled_learnable_input_basis": bool(getattr(args, "coupled_learnable_input_basis", False)),
        "coupled_shared_gate": bool(getattr(args, "coupled_shared_gate", False)),
        "trainable_info": trainable_info,
        "phase_history": phase_history,
        "teacher_logit_weight": args.teacher_logit_weight,
        "teacher_feature_weight": args.teacher_feature_weight,
        "teacher_block_feature_weight": args.teacher_block_feature_weight,
        "teacher_block_feature_layers": args.teacher_block_feature_layers,
        "anchor_logit_weight": args.anchor_logit_weight,
        "anchor_feature_weight": args.anchor_feature_weight,
        "anchor_block_feature_weight": args.anchor_block_feature_weight,
        "anchor_block_feature_layers": args.anchor_block_feature_layers,
        "disable_strong_aug": args.disable_strong_aug,
        "eval_only": args.eval_only,
        "eval_hflip_tta": args.eval_hflip_tta,
        "eval_shift_tta": args.eval_shift_tta,
        "eval_diagonal_shift_tta": args.eval_diagonal_shift_tta,
        "semantic_anchor_weight": args.semantic_anchor_weight,
        "semantic_expert_weight": args.semantic_expert_weight,
        "semantic_mode_weight": args.semantic_mode_weight,
        "geo_structure_weight": args.geo_structure_weight,
        "geo_structure_start_epoch": args.geo_structure_start_epoch,
        "geo_structure_ramp_epochs": args.geo_structure_ramp_epochs,
        "flow_anchor_weight": args.flow_anchor_weight,
        "flow_energy_weight": args.flow_energy_weight,
        "magnus_anchor_weight": args.magnus_anchor_weight,
        "magnus_motion_weight": args.magnus_motion_weight,
        "magnus_comm_weight": args.magnus_comm_weight,
        "magnus_mode_weight": args.magnus_mode_weight,
        "last_diagnostics": model.get_diagnostics(),
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(run_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    if args.save_best_checkpoint and best_state is not None:
        torch.save(
            {
                "model_state_dict": best_state,
                "model_config": payload["model_config"],
                "best_acc": best_acc,
                "best_epoch": best_epoch,
                "summary": payload,
            },
            run_dir / "best.pth",
        )
    if args.save_last_checkpoint:
        last_model = ema.module.state_dict() if (ema is not None and ema_active) else model.state_dict()
        torch.save(
            {
                "model_state_dict": last_model,
                "model_config": payload["model_config"],
                "best_acc": best_acc,
                "best_epoch": best_epoch,
                "summary": payload,
            },
            run_dir / "last.pth",
        )
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
