import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adjoint_solver import GeodynamicSolver
from .controller import LowRankController, ResidualTangentController


def controller_feature_dim(embed_dim, controller_pool):
    if controller_pool in {"cls", "mean"}:
        return embed_dim
    if controller_pool == "cls_mean":
        return 2 * embed_dim
    if controller_pool == "cls_mean_var":
        return 3 * embed_dim
    raise ValueError(f"Unsupported controller_pool: {controller_pool}")


class FlowGeoDynamicLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        rank=8,
        num_steps=4,
        max_velocity=1.5,
        bias=True,
        controller_hidden_dim=64,
        controller_pool="cls",
        **kwargs,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_steps = num_steps
        self.max_velocity = max_velocity
        self.controller_pool = controller_pool

        self.controller = LowRankController(
            embed_dim=in_features,
            manifold_dim=out_features,
            rank=rank,
            hidden_dim=controller_hidden_dim,
            input_dim=controller_feature_dim(in_features, controller_pool),
        )

        self.U_0 = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.U_0)
        self.U_0.is_manifold = True
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def _conditioning_signal(self, x):
        if x.dim() != 3:
            return x

        cls = x[:, 0]
        mean = x.mean(dim=1)
        if self.controller_pool == "cls":
            return cls
        if self.controller_pool == "mean":
            return mean
        if self.controller_pool == "cls_mean":
            return torch.cat([cls, mean], dim=-1)
        if self.controller_pool == "cls_mean_var":
            var = x.var(dim=1, unbiased=False)
            return torch.cat([cls, mean, var], dim=-1)
        raise ValueError(f"Unsupported controller_pool: {self.controller_pool}")

    def _build_geodynamic_controls(self, x):
        z = self._conditioning_signal(x)
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


class GeoDynamicLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        rank=12,
        num_bases=16,
        bias=True,
        controller_hidden_dim=192,
        controller_pool="cls_mean_var",
        residual_scale=1.0,
        **kwargs,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.num_bases = num_bases
        self.controller_pool = controller_pool
        self.residual_scale = residual_scale

        controller_dim = controller_feature_dim(in_features, controller_pool)
        self.controller = ResidualTangentController(
            input_dim=controller_dim,
            num_bases=num_bases,
            hidden_dim=controller_hidden_dim,
        )

        self.shared_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.shared_weight, a=math.sqrt(5))

        self.U_0 = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.U_0)
        self.U_0.is_manifold = True

        self.left_basis = nn.Parameter(torch.empty(num_bases, out_features, rank))
        self.right_basis = nn.Parameter(torch.empty(num_bases, in_features, rank))
        nn.init.normal_(self.left_basis, std=0.02)
        nn.init.normal_(self.right_basis, std=0.02)

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def _conditioning_signal(self, x):
        if x.dim() != 3:
            return x

        cls = x[:, 0]
        mean = x.mean(dim=1)
        if self.controller_pool == "cls":
            return cls
        if self.controller_pool == "mean":
            return mean
        if self.controller_pool == "cls_mean":
            return torch.cat([cls, mean], dim=-1)
        if self.controller_pool == "cls_mean_var":
            var = x.var(dim=1, unbiased=False)
            return torch.cat([cls, mean, var], dim=-1)
        raise ValueError(f"Unsupported controller_pool: {self.controller_pool}")

    def _tangent_dictionary(self, dtype, device):
        raw_basis = torch.einsum("kmr,kdr->kmd", self.left_basis, self.right_basis)
        raw_basis = raw_basis.to(device=device, dtype=dtype)

        base = self.U_0.to(device=device, dtype=dtype)
        ut_raw = torch.matmul(base.transpose(0, 1).unsqueeze(0), raw_basis)
        sym = 0.5 * (ut_raw + ut_raw.transpose(-2, -1))
        tangent_basis = raw_basis - torch.matmul(base.unsqueeze(0), sym)

        flat = tangent_basis.reshape(self.num_bases, -1)
        norm = torch.linalg.vector_norm(flat, dim=1, keepdim=True).clamp_min(1e-6)
        return tangent_basis / norm.view(self.num_bases, 1, 1)

    def _retract(self, candidate):
        q, r = torch.linalg.qr(candidate, mode="reduced")
        sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        return q * sign.unsqueeze(-2)

    def _apply_dynamic_weight(self, x, weight):
        if x.dim() == 3:
            return torch.einsum("bnd,bmd->bnm", x, weight)
        if x.dim() == 2:
            return torch.einsum("bd,bmd->bm", x, weight)
        raise ValueError(f"GeoDynamicLayer expects a 2D or 3D tensor, got shape {tuple(x.shape)}")

    def forward(self, x):
        z = self._conditioning_signal(x)
        coeff, gate = self.controller(z)

        tangent_basis = self._tangent_dictionary(dtype=torch.float32, device=x.device)
        delta = torch.einsum("bk,kmd->bmd", coeff.to(torch.float32), tangent_basis)

        base = self.U_0.to(device=x.device, dtype=torch.float32).unsqueeze(0)
        candidate = base + (self.residual_scale * delta)
        dynamic_weight = self._retract(candidate)
        residual_weight = dynamic_weight - base

        out = F.linear(x, self.shared_weight, self.bias)
        residual_out = self._apply_dynamic_weight(x, residual_weight.to(dtype=x.dtype))
        if residual_out.dim() == 3:
            gate = gate.to(dtype=x.dtype).view(-1, 1, 1)
        else:
            gate = gate.to(dtype=x.dtype).view(-1, 1)
        return out + gate * residual_out
