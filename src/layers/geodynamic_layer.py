import torch
import torch.nn as nn

from .adjoint_solver import GeodynamicSolver
from .controller import LowRankController


class GeoDynamicLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, num_steps=4, max_velocity=1.5, bias=True, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_steps = num_steps
        self.max_velocity = max_velocity

        self.controller = LowRankController(
            embed_dim=in_features,
            manifold_dim=out_features,
            rank=rank,
        )

        self.U_0 = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.U_0)
        self.U_0.is_manifold = True
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def _build_geodynamic_controls(self, x):
        z = x.mean(dim=1) if x.dim() == 3 else x
        A, B = self.controller(z)

        if B is not None:
            pad = torch.zeros(
                B.shape[0],
                self.in_features,
                self.in_features,
                device=x.device,
                dtype=B.dtype,
            )
            G = torch.cat([pad, B], dim=1)
        else:
            G = torch.zeros(
                A.shape[0],
                self.out_features,
                self.in_features,
                device=x.device,
                dtype=A.dtype,
            )

        control_vec = torch.cat(
            [A.reshape(A.shape[0], -1), G.reshape(G.shape[0], -1)],
            dim=1,
        )
        control_norm = torch.linalg.vector_norm(control_vec, dim=1, keepdim=True).view(-1, 1, 1)
        scale = torch.clamp(self.max_velocity / (control_norm + 1e-6), max=1.0)

        A = torch.nan_to_num(A * scale, nan=0.0, posinf=0.0, neginf=0.0)
        G = torch.nan_to_num(G * scale, nan=0.0, posinf=0.0, neginf=0.0)
        return A, G

    def _apply_dynamic_weight(self, x, weight):
        if x.dim() == 3:
            return torch.einsum("bnd,bmd->bnm", x, weight)
        if x.dim() == 2:
            return torch.einsum("bd,bmd->bm", x, weight)
        raise ValueError(f"GeoDynamicLayer expects a 2D or 3D tensor, got shape {tuple(x.shape)}")

    def forward(self, x):
        A, G = self._build_geodynamic_controls(x)
        batch_size = A.shape[0]

        U = self.U_0.unsqueeze(0).expand(batch_size, -1, -1).contiguous().to(torch.float32)
        step_scale = 1.0 / float(self.num_steps)
        A_step = (A * step_scale).to(torch.float32).contiguous()
        G_step = (G * step_scale).to(torch.float32).contiguous()

        for _ in range(self.num_steps):
            U = GeodynamicSolver.apply(U, A_step, G_step)

        out = self._apply_dynamic_weight(x, U.to(dtype=x.dtype))
        if self.bias is None:
            return out
        if out.dim() == 3:
            return out + self.bias.view(1, 1, -1)
        return out + self.bias.view(1, -1)
