import torch
import torch.nn as nn


class LowRankController(nn.Module):
    def __init__(self, embed_dim, manifold_dim, rank=8, hidden_dim=64):
        super().__init__()
        self.d = embed_dim
        self.m = manifold_dim
        self.rank = rank

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.head_A_u = nn.Linear(hidden_dim, self.d * rank)
        self.head_A_v = nn.Linear(hidden_dim, self.d * rank)

        self.rows_B = self.m - self.d
        if self.rows_B > 0:
            self.head_B_u = nn.Linear(hidden_dim, self.rows_B * rank)
            self.head_B_v = nn.Linear(hidden_dim, self.d * rank)

        # Keep the early flow conservative so the Stiefel updates stay well-behaved.
        with torch.no_grad():
            self.head_A_u.weight.mul_(0.01)
            self.head_A_v.weight.mul_(0.01)

    def forward(self, z):
        feat = self.net(z)
        batch_size = z.shape[0]

        u_A = self.head_A_u(feat).view(batch_size, self.d, self.rank)
        v_A = self.head_A_v(feat).view(batch_size, self.d, self.rank)
        A = torch.matmul(u_A, v_A.transpose(-2, -1)) - torch.matmul(v_A, u_A.transpose(-2, -1))

        B = None
        if self.rows_B > 0:
            u_B = self.head_B_u(feat).view(batch_size, self.rows_B, self.rank)
            v_B = self.head_B_v(feat).view(batch_size, self.d, self.rank)
            B = torch.matmul(u_B, v_B.transpose(-2, -1))

        scale = 0.5
        A = torch.tanh(A * 0.1) * scale
        if B is not None:
            B = torch.tanh(B * 0.1) * scale

        return A, B
