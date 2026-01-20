import torch
import torch.nn as nn
from .controller import LowRankController
from .adjoint_solver import GeodynamicSolver

class GeoDynamicLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # The Neural Controller (The "Brain")
        self.controller = LowRankController(
            embed_dim=in_features,
            manifold_dim=out_features,
            rank=rank
        )
        
        # The Manifold Basis (The "Memory")
        # Instead of a weight matrix, we store a point on the Stiefel Manifold
        self.U_0 = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.U_0)
        self.U_0.is_manifold = True

    def forward(self, x):
        # 1. Extract Context (Mean pooling over sequence tokens if needed)
        # x is (Batch, Seq, Dim) or (Batch, Dim)
        z = x.mean(dim=1) if x.dim() == 3 else x
        
        # 2. Controller predicts the dynamic geometry (A and B)
        # This tells us how to curve the manifold for this specific batch
        A, B = self.controller(z) 
        
        # 3. SPEED OPTIMIZATION: Batch-Context Averaging
        # Instead of evolving 128 separate matrices, we evolve the basis U_0
        # based on the *aggregate* semantic context of the batch.
        A_avg = A.mean(dim=0, keepdim=True) # (1, d, d)
        
        if B is not None:
            B_avg = B.mean(dim=0, keepdim=True)
            pad = torch.zeros(1, self.in_features, self.in_features, device=x.device, dtype=B.dtype)
            G_avg = torch.cat([pad, B_avg], dim=1) # (1, m, d)
        else:
            G_avg = torch.zeros(1, self.out_features, self.in_features, device=x.device, dtype=A.dtype)

        # 4. The Geodynamic Solve (Stiefel Flow)
        # We explicitly ensure contiguous memory to prevent CUDA errors
        U0_in = self.U_0.unsqueeze(0).contiguous()
        
        W_dynamic = GeodynamicSolver.apply(
            U0_in.to(torch.float32),
            A_avg.to(torch.float32).contiguous(),
            G_avg.to(torch.float32).contiguous()
        ) 
        # Result is (1, m, d)

        # 5. Apply the generated weights
        # W_dynamic is the "hallucinated" weight matrix for this batch
        W = W_dynamic.squeeze(0)
        return torch.matmul(x, W.t())