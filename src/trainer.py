import torch
import geoopt

class HybridTrainer:
    def __init__(self, model, lr_base=1e-3, lr_controller=1e-4):
        self.model = model

        # Sorts through parameters and puts them into the correct list: Controller weights and bias go in euclidian, while the tagged go into manifold params
        manifold_params = []
        euclidean_params = []
        
        for name, param in self.model.named_parameters():
            if hasattr(param, "is_manifold") and param.is_manifold:
                manifold_params.append(param)
            else:
                euclidean_params.append(param)
        
        print(f"Optimizer Setup: {len(manifold_params)} Manifold Params, {len(euclidean_params)} Euclidean Params.")

        # Optimizer UNO: Riemannian Adam (For U_0) --> Manifold Optimizer
        self.opt_manifold = geoopt.optim.RiemannianAdam(
            manifold_params, 
            lr=lr_base, # Updates Slower although lr is higher as there are fewer layers to amplify update
            stabilize=10 
        )
        
        # Optimizer DOS: AdamW --> Everything else meaning euclidean
        self.opt_euclidean = torch.optim.AdamW(
            euclidean_params, 
            lr=lr_controller, # Updates Faster as layers amplify update
            weight_decay=0.05
        )
        
    def step(self, loss):
        # Zero gradients
        self.opt_manifold.zero_grad()
        self.opt_euclidean.zero_grad()
        
        # Backward
        loss.backward()
        
        # Steps
        self.opt_euclidean.step()
        self.opt_manifold.step()





# Overall, splits the parameters into manifold and regular (euclidian); 
# then, assigns corresponding learning rates, and defines the steps for learning
