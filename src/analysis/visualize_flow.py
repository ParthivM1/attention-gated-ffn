import torch
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from layers.geodynamic_layer import GeoDynamicLayer
from model_factory import DEFAULT_MODEL_CONFIG, build_model

def main():
    print("🔹 Initializing Model Structure...")
    model_config = dict(DEFAULT_MODEL_CONFIG)
    model = build_model(model_config)
    
    # Check for a local checkpoint (Optional example logic)
    # In a real analysis flow, you would pass this as an arg
    ckpt_path = os.path.join(os.path.dirname(__file__), "../../checkpoints/epoch_5.pt")
    
    if os.path.exists(ckpt_path):
        print(f"   Loading checkpoint from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Checkpoint loaded.")
    else:
        print("⚠️ No checkpoint found at default path, using random initialization for demonstration.")

    print("\n🔍 Inspecting Controller Outputs (A and B)...")
    
    # 1. Find a GeoDynamicLayer to hook
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, GeoDynamicLayer):
            # We found one. Let's hook its controller.
            target_layer = module
            print(f"   Targeting Layer: {name}")
            break
            
    if target_layer is None:
        print("❌ No GeoDynamicLayer found in the model.")
        return

    # 2. Register Hook to capture (A, B) from the controller
    # The controller's forward method returns: A, B
    def hook_fn(module, input, output):
        A, B = output
        print(f"   [Hook] Captured Controller Output:")
        print(f"   -> Shape of A: {A.shape} (Skew-Symmetric)")
        if B is not None:
            print(f"   -> Shape of B: {B.shape} (Unconstrained)")
        else:
            print(f"   -> B is None")

    # Hook the 'controller' submodule of the GeoDynamicLayer
    handle = target_layer.controller.register_forward_hook(hook_fn)

    # 3. Run Dummy Pass
    val_input = torch.randn(2, 3, model_config["img_size"], model_config["img_size"])
    print(f"   Running forward pass with input: {val_input.shape}")
    
    model.eval()
    with torch.no_grad():
        try:
            model(val_input)
        except Exception as e:
            print(f"   Forward pass encountered an error (ignore if merely testing hook): {e}")
        
    handle.remove()
    print("\n✅ Analysis Complete.")

if __name__ == "__main__":
    main()
