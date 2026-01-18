import torch
import torch.nn as nn
from .controller import LowRankController
from .adjoint_solver import GeodynamicSolver

# This is where Parthiv comes in Clutch
class GeoDynamicLayer(nn.Module):
   def __init__(self, in_features, out_features, rank=8, **kwargs):
       """
       replacement for nn.Linear
       """
       super().__init__()
       self.in_features = in_features # d
       self.out_features = out_features # m (usually 4d)
      
       # Initialize the Controller
       self.controller = LowRankController(
           embed_dim=in_features,
           manifold_dim=out_features,
           rank=rank
       )
      
       # 2. The Base Weight U_0
       # assume U_0 is on Stiefel(m, d)
       self.U_0 = nn.Parameter(torch.empty(out_features, in_features))
       nn.init.orthogonal_(self.U_0)
      
       # mark U_0 as manifold for the Trainer to identify
       self.U_0.is_manifold = True


   def forward(self, x):
       """
       x: (Batch, Seq_Len, in_features) or (Batch, in_features)
       """
       # 1. Get Global Context for Controller
       # Mean pool over sequence length to get (Batch, d)
       if x.dim() == 3:
           z = x.mean(dim=1)
       else:
           z = x # If input is already 2D
          
       # 2. Get A & B from controller
       A, B = self.controller(z)

       # 3. Map B to G (Relaxation Term)
       if B is not None:
            # B is (Batch, m-d, d). Concatenate with zeros to match (Batch, m, d)
            pad = torch.zeros(z.shape[0], self.in_features, self.in_features, device=x.device)
            G = torch.cat([pad, B], dim=1) 
       else:
            # If no B, G is all zeros
            G = torch.zeros(z.shape[0], self.out_features, self.in_features, device=x.device)

       # 4. Strict ODE Flow using GeodynamicSolver
       U_0_expanded = self.U_0.unsqueeze(0).expand(z.shape[0], -1, -1)
       
       # Parthiv lowkey the GOAT
       A_mean = A.mean(dim=0)
       G_mean = G.mean(dim=0)

       W_dynamic = GeodynamicSolver.apply(
            self.U_0.to(torch.float32),
            A_mean.to(torch.float32),
            G_mean.to(torch.float32),
            1,
            'euler',
            {'drift_correction': False}
        )
      
       # 5. Apply Weights
       # Linear layer: x @ W.T
       if x.dim() == 2:
           # (B, d) -> (B, 1, d) @ (B, d, m) -> (B, 1, m) -> (B, m)
           out = torch.matmul(x.unsqueeze(1), W_dynamic.transpose(-1, -2)).squeeze(1)
       else:
           # (B, S, d) @ (B, d, m) -> (B, S, m)
           out = torch.matmul(x, W_dynamic.transpose(-1, -2))
           
       return out