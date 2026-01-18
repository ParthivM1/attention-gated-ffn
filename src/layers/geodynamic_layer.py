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
        Computes ONE W_dynamic per layer (not per-sample) to avoid OOM/memory creep.
        """
        # 1) context z: (B, d)
        if x.dim() == 3:
            z = x.mean(dim=1)
        else:
            z = x

        # 2) controller outputs per-sample
        A, B = self.controller(z)  # A: (B,d,d), B: (B, m-d, d) or None

        # 3) build G per-sample, then aggregate
        if B is not None:
            pad = torch.zeros(z.shape[0], self.in_features, self.in_features, device=x.device, dtype=B.dtype)
            G = torch.cat([pad, B], dim=1)  # (B, m, d)
        else:
            G = torch.zeros(z.shape[0], self.out_features, self.in_features, device=x.device, dtype=A.dtype)

        # 4) AGGREGATE across batch -> single (d,d) and (m,d)
        A_mean = A.mean(dim=0)          # (d, d)
        G_mean = G.mean(dim=0)          # (m, d)

        # 5) solve ONCE (no batch dimension)
        W_dynamic = GeodynamicSolver.apply(
            self.U_0.to(torch.float32),       # (m, d)
            A_mean.to(torch.float32),         # (d, d)
            G_mean.to(torch.float32),         # (m, d)
            1,                                # steps (fast)
            "euler",                          # method (fast)
            {"tol": 1e-4, "drift_correction": False, "max_sv": 10.0}
        )  # -> (m, d)

        # 6) apply weights
        if x.dim() == 2:
            return x @ W_dynamic.transpose(-1, -2)   # (B,d) @ (d,m) -> (B,m)
        else:
            return torch.matmul(x, W_dynamic.transpose(-1, -2))  # (B,S,d) @ (d,m) -> (B,S,m)
