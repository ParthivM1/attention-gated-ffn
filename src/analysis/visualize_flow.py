import os
import sys

import torch

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from layers.geodynamic_layer import FlowGeoDynamicLayer, GeoDynamicLayer
from model_factory import DEFAULT_MODEL_CONFIG, build_model


def main():
    print("Initializing model structure...")
    model_config = dict(DEFAULT_MODEL_CONFIG)
    model = build_model(model_config)

    ckpt_path = os.path.join(os.path.dirname(__file__), "../../checkpoints/epoch_5.pt")
    if os.path.exists(ckpt_path):
        print(f"   Loading checkpoint from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print("   Checkpoint loaded.")
    else:
        print("   No checkpoint found at default path, using random initialization.")

    print("\nInspecting controller outputs...")

    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, (GeoDynamicLayer, FlowGeoDynamicLayer)):
            target_layer = module
            print(f"   Targeting layer: {name}")
            break

    if target_layer is None:
        print("   No GeoDynamicLayer found in the model.")
        return

    def hook_fn(module, input, output):
        first, second = output
        print("   [Hook] Captured controller output:")
        print(f"   -> First output shape: {first.shape}")
        if isinstance(target_layer, FlowGeoDynamicLayer):
            print("   -> First output is the flow A control")
            if second is not None:
                print(f"   -> Second output shape: {second.shape} (flow B control)")
            else:
                print("   -> Second output is None")
        else:
            print("   -> First output is the tangent-basis coefficient vector")
            print(f"   -> Second output shape: {second.shape} (residual gate)")

    handle = target_layer.controller.register_forward_hook(hook_fn)

    val_input = torch.randn(2, 3, model_config["img_size"], model_config["img_size"])
    print(f"   Running forward pass with input: {val_input.shape}")

    model.eval()
    with torch.no_grad():
        try:
            model(val_input)
        except Exception as exc:
            print(f"   Forward pass encountered an error: {exc}")

    handle.remove()
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
