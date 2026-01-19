import torch
import torch.nn as nn


# Model Layer --> Note: Parthiv is a bastard
class LowRankController(nn.Module):
   def __init__(self, embed_dim, manifold_dim, rank=8, hidden_dim=64):
       """
       Args:
           embed_dim: Dimension of the input token Z (d).
           manifold_dim: The dimension 'm' of the Stiefel Manifold (output dim of layer).
           rank: The rank of the low-rank factors.
       """
       super().__init__()
       self.d = embed_dim
       self.m = manifold_dim
       self.rank = rank


       # Controller Structure: Maps pooled Z to hidden representation
       self.net = nn.Sequential(
           nn.Linear(embed_dim, hidden_dim),
           nn.GELU(),
           nn.Linear(hidden_dim, hidden_dim),
           nn.GELU()
       )
      
       # Output Heads for Matrix A (d x d, Skew Symmetric)
      
       self.head_A_u = nn.Linear(hidden_dim, self.d * rank)
       self.head_A_v = nn.Linear(hidden_dim, self.d * rank)
      
       # Output Heads for Matrix B ((m-d) x d, Arbitrary)
       # m = out_features (usually 4d).
       # We predict u, v vectors to form B = uv^T --> B Is only created when input size (d) is < output size
       self.rows_B = self.m - self.d

       nn.init.zeros_(self.head_A_u.weight)
       nn.init.zeros_(self.head_A_u.bias)
       nn.init.zeros_(self.head_A_v.weight)
       nn.init.zeros_(self.head_A_v.bias)
       
       if self.rows_B > 0:
           self.head_B_u = nn.Linear(hidden_dim, self.rows_B * rank)
           self.head_B_v = nn.Linear(hidden_dim, self.d * rank)

           # Starts at 0 --> Stable through first iteration
           nn.init.zeros_(self.head_B_u.weight)
           nn.init.zeros_(self.head_B_u.bias)
           nn.init.zeros_(self.head_B_v.weight)
           nn.init.zeros_(self.head_B_v.bias)


   def forward(self, z):
       """
       Input: z (Batch, Embed_Dim) -> Global Average Pooled Token
       Output: A (Batch, d, d), B (Batch, m-d, d)
       """
       feat = self.net(z) # (Batch, hidden)
       batch_size = z.shape[0]
      
       # Reshapes from just (batch, d*rank) to (batch, d, rank)
       u_A = self.head_A_u(feat).view(batch_size, self.d, self.rank)
       v_A = self.head_A_v(feat).view(batch_size, self.d, self.rank)
      
       # Make sure skew symetric --> Preserves meaning of matrix
       # predict u, v vectors to form A = uv^T - vu^T --> Essentially keeps dimensions lower; if not the output would be huge
       A = torch.matmul(u_A, v_A.transpose(-2, -1)) - torch.matmul(v_A, u_A.transpose(-2, -1))
      
       # Construct Arbitrary B
       if self.rows_B > 0:
           u_B = self.head_B_u(feat).view(batch_size, self.rows_B, self.rank)
           v_B = self.head_B_v(feat).view(batch_size, self.d, self.rank)
           B = torch.matmul(u_B, v_B.transpose(-2, -1))
       else:
           B = None # Case where m=d (Square matrix)

       A = 0.01 * A   # start small; you can try 0.03 later

        if 'B' in locals() and B is not None:
            B = 0.05 * B
            # Clamp B to prevent massive values
            B = torch.clamp(B, -3.0, 3.0)
            
        # Global safety clamp for A
        A = torch.clamp(A, -3.0, 3.0)

        # Replace NaNs with zeros if any generated (safety net)
        if torch.isnan(A).any():
            A = torch.where(torch.isnan(A), torch.zeros_like(A), A)
            
        if B is not None and torch.isnan(B).any():
            B = torch.where(torch.isnan(B), torch.zeros_like(B), B)

        return A, B




       # Overall, takes in context vector z and then outputs A and B, which tell the model how
       # to adjust the weights of the linear layers in a much more efficeint manner through the use
       # of Low rank factorization