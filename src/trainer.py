import torch
import geoopt

class HybridTrainer:
    def __init__(self, model, lr_base=1e-3, lr_controller=1e-4):
        self.model = model

        manifold_params = []
        euclidean_params = []
        
        for name, param in self.model.named_parameters():
            if hasattr(param, "is_manifold") and param.is_manifold:
                manifold_params.append(param)
            else:
                euclidean_params.append(param)
        
        print(f"Optimizer Setup: {len(manifold_params)} Manifold Matrices, {len(euclidean_params)} Euclidean Tensors.")

        # Optimizer for U_0 (Riemannian)
        self.opt_manifold = geoopt.optim.RiemannianAdam(
            manifold_params, 
            lr=lr_base, 
            stabilize=1 
        )
        
        # Optimizer for Controller (Standard)
        self.opt_euclidean = torch.optim.AdamW(
            euclidean_params, 
            lr=lr_controller, 
            weight_decay=0.05
        )