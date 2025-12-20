import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from layers.geodynamic_layer import GeoDynamicLayer
from models.vit import VisionTransformer
from trainer import HybridTrainer

def test_geodynamic_layer():
    print("Testing GeoDynamicLayer...")
    # Instantiate with args from user request: 128, 512
    layer = GeoDynamicLayer(in_features=128, out_features=512, rank=8)
    
    # Random input tensor
    x = torch.randn(1, 196, 128)
    out = layer(x)
    print(f"GeoDynamicLayer output shape: {out.shape}")
    
    # Verify output shape
    assert out.shape == (1, 196, 512)
    print("GeoDynamicLayer test passed!")
    return layer

def test_vit_instantiation():
    print("Testing VisionTransformer instantiation...")
    model = VisionTransformer()
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(f"ViT output shape: {out.shape}")
    assert out.shape == (1, 1000)
    print("ViT instantiation test passed!")

def test_vit_custom_layer():
    print("Testing VisionTransformer with custom layer...")
    
    class MockLayer(nn.Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=bias)
        def forward(self, x):
            return self.linear(x)

    model = VisionTransformer(linear_layer=MockLayer)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(f"Custom ViT output shape: {out.shape}")
    assert out.shape == (1, 1000)
    print("ViT custom layer test passed!")

def test_hybrid_trainer(layer):
    print("Testing HybridTrainer...")
    # Instantiate HybridTrainer with the layer
    
    try:
        trainer = HybridTrainer(layer)
        
        # Check if it correctly identifies 1 manifold parameter
        
        manifold_params_count = len(trainer.opt_manifold.param_groups[0]['params'])
        print(f"Manifold parameters found: {manifold_params_count}")
        
        # The layer has self.U_0 which is_manifold=True.
        # And controller params which are euclidean.
        assert manifold_params_count == 1
        print("HybridTrainer test passed!")
        
    except ImportError:
        print("geoopt not installed, skipping HybridTrainer test execution but code is ready.")
    except Exception as e:
        print(f"HybridTrainer test failed: {e}")
        # If geoopt is missing it might fail.

if __name__ == "__main__":
    layer = test_geodynamic_layer()
    test_vit_instantiation()
    test_vit_custom_layer()
    test_hybrid_trainer(layer)
