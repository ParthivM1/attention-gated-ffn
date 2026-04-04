from functools import partial

import torch.nn as nn

from layers.geodynamic_layer import GeoDynamicLayer
from models.vit import VisionTransformer


DEFAULT_MODEL_CONFIG = {
    "img_size": 32,
    "patch_size": 4,
    "embed_dim": 192,
    "depth": 6,
    "num_heads": 6,
    "num_classes": 100,
    "drop_path_rate": 0.1,
    "model_variant": "allgeo",
    "geo_rank": 8,
    "geo_hidden_dim": 64,
    "geo_num_steps": 4,
    "geo_max_velocity": 1.5,
    "controller_pool": "cls",
    "use_conv_stem": True,
    "stem_channels": 64,
}

LEGACY_MODEL_OVERRIDES = {
    "controller_pool": "mean",
    "use_conv_stem": False,
    "stem_channels": 64,
}

MODEL_VARIANTS = ("plain", "last1", "last2", "allgeo")


def add_model_args(parser):
    parser.add_argument("--model_variant", type=str, choices=MODEL_VARIANTS, default=DEFAULT_MODEL_CONFIG["model_variant"])
    parser.add_argument("--geo_rank", type=int, default=DEFAULT_MODEL_CONFIG["geo_rank"])
    parser.add_argument("--geo_hidden_dim", type=int, default=DEFAULT_MODEL_CONFIG["geo_hidden_dim"])
    parser.add_argument("--geo_num_steps", type=int, default=DEFAULT_MODEL_CONFIG["geo_num_steps"])
    parser.add_argument("--geo_max_velocity", type=float, default=DEFAULT_MODEL_CONFIG["geo_max_velocity"])
    parser.add_argument("--controller_pool", type=str, choices=("cls", "mean"), default=DEFAULT_MODEL_CONFIG["controller_pool"])
    parser.add_argument("--use_conv_stem", action="store_true", default=DEFAULT_MODEL_CONFIG["use_conv_stem"])
    parser.add_argument("--no_conv_stem", action="store_false", dest="use_conv_stem")
    parser.add_argument("--stem_channels", type=int, default=DEFAULT_MODEL_CONFIG["stem_channels"])
    return parser


def config_from_args(args, num_classes=None):
    config = dict(DEFAULT_MODEL_CONFIG)
    config.update(
        {
            "model_variant": args.model_variant,
            "geo_rank": args.geo_rank,
            "geo_hidden_dim": args.geo_hidden_dim,
            "geo_num_steps": args.geo_num_steps,
            "geo_max_velocity": args.geo_max_velocity,
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
    if model_variant == "last1":
        return [depth - 1]
    if model_variant == "last2":
        return list(range(max(depth - 2, 0), depth))
    if model_variant == "allgeo":
        return list(range(depth))
    raise ValueError(f"Unsupported model_variant: {model_variant}")


def build_model(model_config):
    depth = model_config["depth"]
    geo_blocks = set(geo_block_indices(model_config["model_variant"], depth))
    geo_layer = partial(
        GeoDynamicLayer,
        rank=model_config["geo_rank"],
        num_steps=model_config["geo_num_steps"],
        max_velocity=model_config["geo_max_velocity"],
        controller_hidden_dim=model_config["geo_hidden_dim"],
        controller_pool=model_config["controller_pool"],
    )
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
    )


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable
