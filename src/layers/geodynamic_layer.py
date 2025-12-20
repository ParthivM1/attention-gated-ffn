import torch
import torch.nn as nn
from .controller import LowRankController


# This is where Parthiv comes in Clutch
class GeoDynamicLayer(nn.Module):
   def __init__(self, in_features, out_features, rank=8, **kwargs):
       """
       replacement for nn.Linear
       """
       super().__init__()
       self.in_features = in_features # d
       self.out_features = out_features # m (usually 4d) --> Figure out later
      
       #Initialize the Controller
       self.controller = LowRankController(
           embed_dim=in_features,
           manifold_dim=out_features,
           rank=rank
       )
      
       # 2. The Base Weight U_0
       #assume U_0 is on Stiefel(m, d)
       self.U_0 = nn.Parameter(torch.randn(out_features, in_features))
      
       # mark U_0 as manifold for the Trainer to identify
       self.U_0.is_manifold = True


   def forward(self, x):
       """
       x: (Batch, Seq_Len, in_features)
       """
       # 1. Get Global Context for Controller
       # Mean pool over sequence length to get (Batch, d)
       if x.dim() == 3:
           z = x.mean(dim=1)
       else:
           z = x # If input is already 2D
          
       # 2. Get A & B from controller
       A, B = self.controller(z)




       # This is where GPT jargon starts - Parthiv edit here
      
       # 3. MOCK FLOW (Placeholder for Person 1's ODE Solver)
       # W_approx = U_0 + (U_0 @ A + U_perp @ B) * dt
       # For the mock, we just project A onto U_0 to verify shapes match.
      
       # Delta W calculation (Simplified for debugging)
       # A is (Batch, d, d)
       # U_0 is (m, d) -> unsqueeze to (1, m, d) for broadcast
      
       U_0_expanded = self.U_0.unsqueeze(0).expand(z.shape[0], -1, -1)
      
       # Flow term: U_0 @ A
       delta_W = torch.matmul(U_0_expanded, A)
      
       # Add B term if it exists (U_perp part)
       # For mock, we just add B padded with zeros to match shape, or skip U_perp logic
       if B is not None:
            # B is (Batch, m-d, d). We pad it to (Batch, m, d) for simple testing
            pad = torch.zeros(z.shape[0], self.in_features, self.in_features, device=x.device)
            B_expanded = torch.cat([pad, B], dim=1) # This is physically wrong but dimensionally valid for testing
            # delta_W = delta_W + B_expanded
      
       # Final Dynamic Weight: W(z)
       W_dynamic = U_0_expanded + delta_W # (Batch, m, d)
      
       # 4. Apply Weights
       # Linear layer: x @ W.T
       # x: (B, S, d)
       # W_dynamic: (B, m, d) -> Transpose to (B, d, m)
      
       out = torch.matmul(x, W_dynamic.transpose(1, 2))
       return out