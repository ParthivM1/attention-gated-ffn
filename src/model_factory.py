from functools import partial

import torch.nn as nn

from layers.geodynamic_layer import (
    AdaptiveLocalMixerLayer,
    EqualWeightLocalBankLayer,
    FlowGeoDynamicLayer,
    GeoDynamicLayer,
    GeoLocalDynamicLayer,
    LocalOnlyGeoDynamicLayer,
    StaticLocalBankLayer,
    RotationGeoDynamicLayer,
    SandwichRotationGeoDynamicLayer,
    RotationLocalGeoDynamicLayer,
)
from models.vit import VisionTransformer


DEFAULT_MODEL_CONFIG = {
    "img_size": 32,
    "patch_size": 4,
    "embed_dim": 192,
    "depth": 6,
    "num_heads": 6,
    "num_classes": 100,
    "drop_path_rate": 0.1,
    "model_variant": "last2",
    "geo_layer_type": "local_only",
    "geo_rank": 8,
    "geo_num_bases": 8,
    "geo_num_scale_bases": 8,
    "geo_num_local_ops": 4,
    "geo_hidden_dim": 128,
    "geo_num_steps": 4,
    "geo_max_velocity": 1.5,
    "geo_residual_scale": 1.0,
    "tokenizer_type": "standard",
    "detail_tokens": 0,
    "detail_score_type": "variance",
    "controller_pool": "cls_mean_var",
    "use_conv_stem": True,
    "stem_channels": 64,
}

LEGACY_MODEL_OVERRIDES = {
    "geo_layer_type": "flow",
    "controller_pool": "mean",
    "use_conv_stem": False,
    "stem_channels": 64,
}

MODEL_VARIANTS = ("plain", "last1", "last2", "first2", "first2last2", "allgeo", "stage_local")
GEO_LAYER_TYPES = (
    "flow",
    "residual",
    "rotation",
    "sandwich_rotation",
    "rotation_local",
    "geo_local",
    "local_only",
    "adaptive_local",
    "static_local_bank",
    "equal_local_bank",
)
CONTROLLER_POOL_CHOICES = ("cls", "mean", "cls_mean", "cls_mean_var")
TOKENIZER_TYPES = ("standard", "budgeted_detail")
DETAIL_SCORE_TYPES = ("variance", "learned")


def add_model_args(parser):
    parser.add_argument("--model_variant", type=str, choices=MODEL_VARIANTS, default=DEFAULT_MODEL_CONFIG["model_variant"])
    parser.add_argument("--geo_layer_type", type=str, choices=GEO_LAYER_TYPES, default=DEFAULT_MODEL_CONFIG["geo_layer_type"])
    parser.add_argument("--geo_rank", type=int, default=DEFAULT_MODEL_CONFIG["geo_rank"])
    parser.add_argument("--geo_num_bases", type=int, default=DEFAULT_MODEL_CONFIG["geo_num_bases"])
    parser.add_argument("--geo_num_scale_bases", type=int, default=DEFAULT_MODEL_CONFIG["geo_num_scale_bases"])
    parser.add_argument("--geo_num_local_ops", type=int, default=DEFAULT_MODEL_CONFIG["geo_num_local_ops"])
    parser.add_argument("--geo_hidden_dim", type=int, default=DEFAULT_MODEL_CONFIG["geo_hidden_dim"])
    parser.add_argument("--geo_num_steps", type=int, default=DEFAULT_MODEL_CONFIG["geo_num_steps"])
    parser.add_argument("--geo_max_velocity", type=float, default=DEFAULT_MODEL_CONFIG["geo_max_velocity"])
    parser.add_argument("--geo_residual_scale", type=float, default=DEFAULT_MODEL_CONFIG["geo_residual_scale"])
    parser.add_argument("--tokenizer_type", type=str, choices=TOKENIZER_TYPES, default=DEFAULT_MODEL_CONFIG["tokenizer_type"])
    parser.add_argument("--detail_tokens", type=int, default=DEFAULT_MODEL_CONFIG["detail_tokens"])
    parser.add_argument("--detail_score_type", type=str, choices=DETAIL_SCORE_TYPES, default=DEFAULT_MODEL_CONFIG["detail_score_type"])
    parser.add_argument("--controller_pool", type=str, choices=CONTROLLER_POOL_CHOICES, default=DEFAULT_MODEL_CONFIG["controller_pool"])
    parser.add_argument("--use_conv_stem", action="store_true", default=DEFAULT_MODEL_CONFIG["use_conv_stem"])
    parser.add_argument("--no_conv_stem", action="store_false", dest="use_conv_stem")
    parser.add_argument("--stem_channels", type=int, default=DEFAULT_MODEL_CONFIG["stem_channels"])
    return parser


def config_from_args(args, num_classes=None):
    config = dict(DEFAULT_MODEL_CONFIG)
    config.update(
        {
            "model_variant": args.model_variant,
            "geo_layer_type": args.geo_layer_type,
            "geo_rank": args.geo_rank,
            "geo_num_bases": args.geo_num_bases,
            "geo_num_scale_bases": args.geo_num_scale_bases,
            "geo_num_local_ops": args.geo_num_local_ops,
            "geo_hidden_dim": args.geo_hidden_dim,
            "geo_num_steps": args.geo_num_steps,
            "geo_max_velocity": args.geo_max_velocity,
            "geo_residual_scale": args.geo_residual_scale,
            "tokenizer_type": args.tokenizer_type,
            "detail_tokens": args.detail_tokens,
            "detail_score_type": args.detail_score_type,
            "controller_pool": args.controller_pool,
            "use_conv_stem": args.use_conv_stem,
            "stem_channels": args.stem_channels,
        }
    )
    if num_classes is not None:
        config["num_classes"] = num_classes
    return config


def config_from_checkpoint(checkpoint_config, num_classes=None):
    config = dict(DEFAULT_MODEL_CONFIG)
    config.update(checkpoint_config)
    for key, value in LEGACY_MODEL_OVERRIDES.items():
        if key not in checkpoint_config:
            config[key] = value
    if num_classes is not None:
        config["num_classes"] = num_classes
    return config


def geo_block_indices(model_variant, depth):
    if model_variant == "plain":
        return []
    if model_variant == "first2":
        return list(range(min(2, depth)))
    if model_variant == "last1":
        return [depth - 1]
    if model_variant == "last2":
        return list(range(max(depth - 2, 0), depth))
    if model_variant == "first2last2":
        first = list(range(min(2, depth)))
        last = list(range(max(depth - 2, 0), depth))
        return sorted(set(first + last))
    if model_variant == "allgeo":
        return list(range(depth))
    if model_variant == "stage_local":
        return list(range(depth))
    raise ValueError(f"Unsupported model_variant: {model_variant}")


def stage_local_specs(depth):
    specs = []
    for idx in range(depth):
        if idx < 2:
            specs.append({"enabled": True, "local_strength": 1.0, "local_operator": "dw3"})
        elif idx < 4:
            specs.append({"enabled": True, "local_strength": 0.5, "local_operator": "dilated3"})
        else:
            specs.append({"enabled": False, "local_strength": 0.0, "local_operator": "dw3"})
    return specs


def build_model(model_config):
    depth = model_config["depth"]
    geo_blocks = set(geo_block_indices(model_config["model_variant"], depth))
    if model_config["geo_layer_type"] == "flow":
        geo_layer = partial(
            FlowGeoDynamicLayer,
            rank=model_config["geo_rank"],
            num_steps=model_config["geo_num_steps"],
            max_velocity=model_config["geo_max_velocity"],
            controller_hidden_dim=model_config["geo_hidden_dim"],
            controller_pool=model_config["controller_pool"],
        )
    elif model_config["geo_layer_type"] == "residual":
        geo_layer = partial(
            GeoDynamicLayer,
            rank=model_config["geo_rank"],
            num_bases=model_config["geo_num_bases"],
            controller_hidden_dim=model_config["geo_hidden_dim"],
            controller_pool=model_config["controller_pool"],
            residual_scale=model_config["geo_residual_scale"],
        )
    elif model_config["geo_layer_type"] == "rotation":
        geo_layer = partial(
            RotationGeoDynamicLayer,
            rank=model_config["geo_rank"],
            num_bases=model_config["geo_num_bases"],
            controller_hidden_dim=model_config["geo_hidden_dim"],
            controller_pool=model_config["controller_pool"],
            residual_scale=model_config["geo_residual_scale"],
        )
    elif model_config["geo_layer_type"] == "sandwich_rotation":
        geo_layer = partial(
            SandwichRotationGeoDynamicLayer,
            rank=model_config["geo_rank"],
            num_bases=model_config["geo_num_bases"],
            num_scale_bases=model_config["geo_num_scale_bases"],
            controller_hidden_dim=model_config["geo_hidden_dim"],
            controller_pool=model_config["controller_pool"],
            residual_scale=model_config["geo_residual_scale"],
        )
    elif model_config["geo_layer_type"] in {"rotation_local", "geo_local"}:
        geo_layer = partial(
            GeoLocalDynamicLayer,
            rank=model_config["geo_rank"],
            num_bases=model_config["geo_num_bases"],
            num_scale_bases=model_config["geo_num_scale_bases"],
            controller_hidden_dim=model_config["geo_hidden_dim"],
            controller_pool=model_config["controller_pool"],
            residual_scale=model_config["geo_residual_scale"],
        )
    elif model_config["geo_layer_type"] == "local_only":
        geo_layer = partial(
            LocalOnlyGeoDynamicLayer,
            controller_hidden_dim=model_config["geo_hidden_dim"],
            controller_pool=model_config["controller_pool"],
        )
    elif model_config["geo_layer_type"] == "adaptive_local":
        geo_layer = partial(
            AdaptiveLocalMixerLayer,
            num_local_ops=model_config["geo_num_local_ops"],
            controller_hidden_dim=model_config["geo_hidden_dim"],
            controller_pool=model_config["controller_pool"],
        )
    elif model_config["geo_layer_type"] == "static_local_bank":
        geo_layer = partial(
            StaticLocalBankLayer,
            num_local_ops=model_config["geo_num_local_ops"],
            controller_pool=model_config["controller_pool"],
        )
    elif model_config["geo_layer_type"] == "equal_local_bank":
        geo_layer = partial(
            EqualWeightLocalBankLayer,
            num_local_ops=model_config["geo_num_local_ops"],
            controller_pool=model_config["controller_pool"],
        )
    else:
        raise ValueError(f"Unsupported geo_layer_type: {model_config['geo_layer_type']}")
    if model_config["model_variant"] == "stage_local":
        if model_config["geo_layer_type"] != "local_only":
            raise ValueError("stage_local currently supports geo_layer_type='local_only' only")
        block_linear_layers = []
        for spec in stage_local_specs(depth):
            if spec["enabled"]:
                block_linear_layers.append(
                    partial(
                        LocalOnlyGeoDynamicLayer,
                        controller_hidden_dim=model_config["geo_hidden_dim"],
                        controller_pool=model_config["controller_pool"],
                        local_strength=spec["local_strength"],
                        local_operator=spec["local_operator"],
                    )
                )
            else:
                block_linear_layers.append(nn.Linear)
    else:
        block_linear_layers = [geo_layer if idx in geo_blocks else nn.Linear for idx in range(depth)]

    return VisionTransformer(
        img_size=model_config["img_size"],
        patch_size=model_config["patch_size"],
        embed_dim=model_config["embed_dim"],
        depth=depth,
        num_heads=model_config["num_heads"],
        num_classes=model_config["num_classes"],
        block_linear_layers=block_linear_layers,
        drop_path_rate=model_config["drop_path_rate"],
        use_conv_stem=model_config["use_conv_stem"],
        stem_channels=model_config["stem_channels"],
        tokenizer_type=model_config["tokenizer_type"],
        detail_tokens=model_config["detail_tokens"],
        detail_score_type=model_config["detail_score_type"],
    )


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable
