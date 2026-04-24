import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adjoint_solver import GeodynamicSolver
from .controller import (
    AdaptiveLocalController,
    LowRankController,
    ResidualTangentController,
    SandwichRotationController,
    RotationLocalController,
    LocalTokenController,
)


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
        nn.init.trunc_normal_(self.shared_weight, std=0.02)

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


class RotationGeoDynamicLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        rank=8,
        num_bases=8,
        bias=True,
        controller_hidden_dim=128,
        controller_pool="cls_mean_var",
        residual_scale=1.0,
        gate_bias=-4.0,
        **kwargs,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.num_bases = num_bases
        self.controller_pool = controller_pool
        self.residual_scale = residual_scale
        self.adapter_scale = 1.0

        controller_dim = controller_feature_dim(in_features, controller_pool)
        self.controller = ResidualTangentController(
            input_dim=controller_dim,
            num_bases=num_bases,
            hidden_dim=controller_hidden_dim,
            gate_bias=gate_bias,
        )

        self.shared_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.trunc_normal_(self.shared_weight, std=0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.left_basis = nn.Parameter(torch.empty(num_bases, in_features, rank))
        self.right_basis = nn.Parameter(torch.empty(num_bases, in_features, rank))
        nn.init.normal_(self.left_basis, std=0.02)
        nn.init.normal_(self.right_basis, std=0.02)

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

    def _skew_basis(self, dtype, device):
        u = self.left_basis.to(device=device, dtype=dtype)
        v = self.right_basis.to(device=device, dtype=dtype)
        basis = torch.matmul(u, v.transpose(-2, -1)) - torch.matmul(v, u.transpose(-2, -1))
        flat = basis.reshape(self.num_bases, -1)
        norm = torch.linalg.vector_norm(flat, dim=1, keepdim=True).clamp_min(1e-6)
        return basis / norm.view(self.num_bases, 1, 1)

    def _cayley_rotation(self, skew):
        skew = skew.to(torch.float32)
        dim = skew.shape[-1]
        eye = torch.eye(dim, device=skew.device, dtype=skew.dtype).unsqueeze(0).expand(skew.shape[0], -1, -1)
        half = 0.5 * skew
        return torch.linalg.solve(eye - half, eye + half)

    def _apply_feature_transform(self, x, transform):
        if x.dim() == 3:
            return torch.einsum("bnd,bdh->bnh", x, transform)
        if x.dim() == 2:
            return torch.einsum("bd,bdh->bh", x, transform)
        raise ValueError(f"GeoDynamicLayer expects a 2D or 3D tensor, got shape {tuple(x.shape)}")

    def forward(self, x):
        z = self._conditioning_signal(x)
        coeff, gate = self.controller(z)

        skew_basis = self._skew_basis(dtype=torch.float32, device=x.device)
        skew = torch.einsum("bk,kde->bde", coeff.to(torch.float32), skew_basis)
        transform = self._cayley_rotation(self.residual_scale * skew)

        base_out = F.linear(x, self.shared_weight, self.bias)
        rotated_x = self._apply_feature_transform(x, transform.to(dtype=x.dtype))
        rotated_out = F.linear(rotated_x, self.shared_weight, self.bias)
        delta_out = rotated_out - base_out

        if delta_out.dim() == 3:
            gate = gate.to(dtype=x.dtype).view(-1, 1, 1)
        else:
            gate = gate.to(dtype=x.dtype).view(-1, 1)
        return base_out + (self.adapter_scale * gate) * delta_out

    def set_adapter_scale(self, scale):
        self.adapter_scale = float(scale)


class SandwichRotationGeoDynamicLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        rank=8,
        num_bases=8,
        num_scale_bases=8,
        bias=True,
        controller_hidden_dim=128,
        controller_pool="cls_mean_var",
        residual_scale=1.0,
        gate_bias=-4.0,
        scale_gate_bias=-2.5,
        **kwargs,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.num_bases = num_bases
        self.num_scale_bases = num_scale_bases
        self.controller_pool = controller_pool
        self.residual_scale = residual_scale
        self.adapter_scale = 1.0

        controller_dim = controller_feature_dim(in_features, controller_pool)
        self.controller = SandwichRotationController(
            input_dim=controller_dim,
            num_rot_bases=num_bases,
            num_scale_bases=num_scale_bases,
            hidden_dim=controller_hidden_dim,
            rot_gate_bias=gate_bias,
            scale_gate_bias=scale_gate_bias,
        )

        self.shared_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.trunc_normal_(self.shared_weight, std=0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.left_basis = nn.Parameter(torch.empty(num_bases, in_features, rank))
        self.right_basis = nn.Parameter(torch.empty(num_bases, in_features, rank))
        nn.init.normal_(self.left_basis, std=0.02)
        nn.init.normal_(self.right_basis, std=0.02)

        self.scale_basis = nn.Parameter(torch.empty(num_scale_bases, out_features))
        nn.init.normal_(self.scale_basis, std=0.01)

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

    def _skew_basis(self, dtype, device):
        u = self.left_basis.to(device=device, dtype=dtype)
        v = self.right_basis.to(device=device, dtype=dtype)
        basis = torch.matmul(u, v.transpose(-2, -1)) - torch.matmul(v, u.transpose(-2, -1))
        flat = basis.reshape(self.num_bases, -1)
        norm = torch.linalg.vector_norm(flat, dim=1, keepdim=True).clamp_min(1e-6)
        return basis / norm.view(self.num_bases, 1, 1)

    def _cayley_rotation(self, skew):
        skew = skew.to(torch.float32)
        dim = skew.shape[-1]
        eye = torch.eye(dim, device=skew.device, dtype=skew.dtype).unsqueeze(0).expand(skew.shape[0], -1, -1)
        half = 0.5 * skew
        return torch.linalg.solve(eye - half, eye + half)

    def _apply_feature_transform(self, x, transform):
        if x.dim() == 3:
            return torch.einsum("bnd,bdh->bnh", x, transform)
        if x.dim() == 2:
            return torch.einsum("bd,bdh->bh", x, transform)
        raise ValueError(f"GeoDynamicLayer expects a 2D or 3D tensor, got shape {tuple(x.shape)}")

    def forward(self, x):
        z = self._conditioning_signal(x)
        rot_coeff, rot_gate, scale_coeff, scale_gate = self.controller(z)

        skew_basis = self._skew_basis(dtype=torch.float32, device=x.device)
        skew = torch.einsum("bk,kde->bde", rot_coeff.to(torch.float32), skew_basis)
        transform = self._cayley_rotation(self.residual_scale * skew).to(dtype=x.dtype)
        rotated_x = self._apply_feature_transform(x, transform)

        if x.dim() == 3:
            rot_gate_view = rot_gate.to(dtype=x.dtype).view(-1, 1, 1)
        else:
            rot_gate_view = rot_gate.to(dtype=x.dtype).view(-1, 1)
        mixed_x = x + (self.adapter_scale * rot_gate_view) * (rotated_x - x)

        preact = F.linear(mixed_x, self.shared_weight, self.bias)

        scale_delta = torch.einsum(
            "bk,kh->bh",
            scale_coeff.to(torch.float32),
            self.scale_basis.to(device=x.device, dtype=torch.float32),
        )
        scale_delta = torch.tanh(scale_delta).to(dtype=x.dtype)

        if preact.dim() == 3:
            scale_gate_view = scale_gate.to(dtype=x.dtype).view(-1, 1, 1)
            scale = 1.0 + (self.adapter_scale * scale_gate_view) * scale_delta.view(preact.shape[0], 1, -1)
        else:
            scale_gate_view = scale_gate.to(dtype=x.dtype).view(-1, 1)
            scale = 1.0 + (self.adapter_scale * scale_gate_view) * scale_delta

        return preact * scale

    def set_adapter_scale(self, scale):
        self.adapter_scale = float(scale)


class RotationLocalGeoDynamicLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        rank=8,
        num_bases=8,
        num_scale_bases=8,
        bias=True,
        controller_hidden_dim=128,
        controller_pool="cls_mean_var",
        residual_scale=1.0,
        gate_bias=-4.0,
        local_gate_bias=-3.0,
        scale_strength_init=0.25,
        **kwargs,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.num_bases = num_bases
        self.num_scale_bases = num_scale_bases
        self.controller_pool = controller_pool
        self.residual_scale = residual_scale
        self.adapter_scale = 1.0
        self.last_stats = {}

        controller_dim = controller_feature_dim(in_features, controller_pool)
        self.controller = RotationLocalController(
            input_dim=controller_dim,
            num_rot_bases=num_bases,
            num_scale_bases=num_scale_bases,
            hidden_dim=controller_hidden_dim,
            rot_gate_bias=gate_bias,
            local_gate_bias=local_gate_bias,
        )

        self.shared_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.trunc_normal_(self.shared_weight, std=0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.left_basis = nn.Parameter(torch.empty(num_bases, in_features, rank))
        self.right_basis = nn.Parameter(torch.empty(num_bases, in_features, rank))
        nn.init.normal_(self.left_basis, std=0.02)
        nn.init.normal_(self.right_basis, std=0.02)

        self.scale_basis = nn.Parameter(torch.empty(num_scale_bases, out_features))
        nn.init.normal_(self.scale_basis, std=0.01)
        self.scale_strength = nn.Parameter(torch.tensor(float(scale_strength_init)))

        self.local_dwconv = nn.Conv2d(
            in_features,
            in_features,
            kernel_size=3,
            padding=1,
            groups=in_features,
            bias=False,
        )
        nn.init.dirac_(self.local_dwconv.weight)

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

    def _skew_basis(self, dtype, device):
        u = self.left_basis.to(device=device, dtype=dtype)
        v = self.right_basis.to(device=device, dtype=dtype)
        basis = torch.matmul(u, v.transpose(-2, -1)) - torch.matmul(v, u.transpose(-2, -1))
        flat = basis.reshape(self.num_bases, -1)
        norm = torch.linalg.vector_norm(flat, dim=1, keepdim=True).clamp_min(1e-6)
        return basis / norm.view(self.num_bases, 1, 1)

    def _cayley_rotation(self, skew):
        skew = skew.to(torch.float32)
        dim = skew.shape[-1]
        eye = torch.eye(dim, device=skew.device, dtype=skew.dtype).unsqueeze(0).expand(skew.shape[0], -1, -1)
        half = 0.5 * skew
        return torch.linalg.solve(eye - half, eye + half)

    def _apply_feature_transform(self, x, transform):
        if x.dim() == 3:
            return torch.einsum("bnd,bdh->bnh", x, transform)
        if x.dim() == 2:
            return torch.einsum("bd,bdh->bh", x, transform)
        raise ValueError(f"GeoDynamicLayer expects a 2D or 3D tensor, got shape {tuple(x.shape)}")

    def _apply_local_mixer(self, x, local_gate):
        if x.dim() != 3 or x.shape[1] <= 1:
            return x, torch.zeros((), device=x.device, dtype=torch.float32)

        cls_tok = x[:, :1, :]
        patch_tok = x[:, 1:, :]
        batch_size, num_tokens, channels = patch_tok.shape
        grid_size = int(math.isqrt(num_tokens))
        if grid_size * grid_size != num_tokens:
            return x, torch.zeros((), device=x.device, dtype=torch.float32)

        patch_grid = patch_tok.transpose(1, 2).reshape(batch_size, channels, grid_size, grid_size)
        local_grid = self.local_dwconv(patch_grid)
        local_tok = local_grid.reshape(batch_size, channels, num_tokens).transpose(1, 2)

        gate = local_gate.to(dtype=x.dtype).view(batch_size, 1, 1)
        mixed_patch = patch_tok + (self.adapter_scale * gate) * (local_tok - patch_tok)
        local_delta_norm = (local_tok - patch_tok).detach().to(torch.float32).norm(dim=-1).mean()
        return torch.cat([cls_tok, mixed_patch], dim=1), local_delta_norm

    def forward(self, x):
        z = self._conditioning_signal(x)
        rot_coeff, rot_gate, local_gate, scale_coeff = self.controller(z)

        locally_mixed_x, local_delta_norm = self._apply_local_mixer(x, local_gate)

        skew_basis = self._skew_basis(dtype=torch.float32, device=x.device)
        skew = torch.einsum("bk,kde->bde", rot_coeff.to(torch.float32), skew_basis)
        transform = self._cayley_rotation(self.residual_scale * skew).to(dtype=x.dtype)
        rotated_x = self._apply_feature_transform(locally_mixed_x, transform)

        if x.dim() == 3:
            rot_gate_view = rot_gate.to(dtype=x.dtype).view(-1, 1, 1)
        else:
            rot_gate_view = rot_gate.to(dtype=x.dtype).view(-1, 1)
        mixed_x = locally_mixed_x + (self.adapter_scale * rot_gate_view) * (rotated_x - locally_mixed_x)

        preact = F.linear(mixed_x, self.shared_weight, self.bias)

        scale_delta = torch.einsum(
            "bk,kh->bh",
            scale_coeff.to(torch.float32),
            self.scale_basis.to(device=x.device, dtype=torch.float32),
        )
        scale_delta = torch.tanh(scale_delta).to(dtype=x.dtype)
        scale_strength = self.scale_strength.to(device=x.device, dtype=x.dtype)

        if preact.dim() == 3:
            scale = 1.0 + (self.adapter_scale * scale_strength) * scale_delta.view(preact.shape[0], 1, -1)
        else:
            scale = 1.0 + (self.adapter_scale * scale_strength) * scale_delta

        out = preact * scale

        rot_delta = (rotated_x - locally_mixed_x).detach().to(torch.float32)
        scale_residual = (scale - 1.0).detach().to(torch.float32).abs().mean()
        self.last_stats = {
            "local_gate": float(local_gate.detach().to(torch.float32).mean().item()),
            "rot_gate": float(rot_gate.detach().to(torch.float32).mean().item()),
            "scale_strength": float(self.scale_strength.detach().to(torch.float32).item()),
            "local_residual_norm": float(local_delta_norm.item()),
            "rot_residual_norm": float(rot_delta.norm(dim=-1).mean().item()),
            "scale_residual_mean": float(scale_residual.item()),
        }
        return out

    def set_adapter_scale(self, scale):
        self.adapter_scale = float(scale)

    def get_diagnostics(self):
        return dict(self.last_stats)


GeoLocalDynamicLayer = RotationLocalGeoDynamicLayer


class LocalOnlyGeoDynamicLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        controller_hidden_dim=128,
        controller_pool="cls_mean_var",
        local_gate_bias=-3.0,
        local_strength=1.0,
        local_operator="dw3",
        **kwargs,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.controller_pool = controller_pool
        self.adapter_scale = 1.0
        self.last_stats = {}
        self.local_strength = float(local_strength)
        self.local_operator = local_operator

        controller_dim = controller_feature_dim(in_features, controller_pool)
        self.controller = LocalTokenController(
            input_dim=controller_dim,
            hidden_dim=controller_hidden_dim,
            local_gate_bias=local_gate_bias,
        )

        self.shared_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.trunc_normal_(self.shared_weight, std=0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        if self.local_operator == "dw3":
            self.local_mixer = nn.Conv2d(
                in_features,
                in_features,
                kernel_size=3,
                padding=1,
                groups=in_features,
                bias=False,
            )
            nn.init.dirac_(self.local_mixer.weight)
        elif self.local_operator == "dilated3":
            self.local_mixer = nn.Conv2d(
                in_features,
                in_features,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=in_features,
                bias=False,
            )
            nn.init.dirac_(self.local_mixer.weight)
        elif self.local_operator == "avg3":
            self.local_mixer = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        else:
            raise ValueError(f"Unsupported local_operator: {self.local_operator}")

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

    def _apply_local_mixer(self, x, local_gate):
        if x.dim() != 3 or x.shape[1] <= 1:
            return x, torch.zeros((), device=x.device, dtype=torch.float32)

        cls_tok = x[:, :1, :]
        patch_tok = x[:, 1:, :]
        batch_size, num_tokens, channels = patch_tok.shape
        grid_size = int(math.isqrt(num_tokens))
        if grid_size * grid_size != num_tokens:
            return x, torch.zeros((), device=x.device, dtype=torch.float32)

        patch_grid = patch_tok.transpose(1, 2).reshape(batch_size, channels, grid_size, grid_size)
        local_grid = self.local_mixer(patch_grid)
        local_tok = local_grid.reshape(batch_size, channels, num_tokens).transpose(1, 2)

        gate = local_gate.to(dtype=x.dtype).view(batch_size, 1, 1)
        mixed_patch = patch_tok + (self.adapter_scale * self.local_strength * gate) * (local_tok - patch_tok)
        local_delta_norm = (local_tok - patch_tok).detach().to(torch.float32).norm(dim=-1).mean()
        return torch.cat([cls_tok, mixed_patch], dim=1), local_delta_norm

    def forward(self, x):
        z = self._conditioning_signal(x)
        local_gate = self.controller(z)
        mixed_x, local_delta_norm = self._apply_local_mixer(x, local_gate)
        out = F.linear(mixed_x, self.shared_weight, self.bias)

        self.last_stats = {
            "local_gate": float(local_gate.detach().to(torch.float32).mean().item()),
            "local_strength": float(self.local_strength),
            "local_residual_norm": float(local_delta_norm.item()),
        }
        return out

    def set_adapter_scale(self, scale):
        self.adapter_scale = float(scale)

    def get_diagnostics(self):
        return dict(self.last_stats)


class LocalOperatorBank(nn.Module):
    def __init__(self, channels, num_ops=4):
        super().__init__()
        self.num_ops = num_ops
        self.dwconv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.dwconv3_dilated = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=channels,
            bias=False,
        )
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        nn.init.dirac_(self.dwconv3.weight)
        nn.init.dirac_(self.dwconv3_dilated.weight)

    def forward(self, x):
        outputs = [
            x,
            self.dwconv3(x),
            self.dwconv3_dilated(x),
            self.avgpool(x),
        ]
        return outputs[: self.num_ops]


class _BaseLocalBankLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_local_ops=4,
        bias=True,
        controller_pool="cls_mean_var",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_local_ops = num_local_ops
        self.controller_pool = controller_pool
        self.adapter_scale = 1.0
        self.last_stats = {}

        self.shared_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.trunc_normal_(self.shared_weight, std=0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.local_bank = LocalOperatorBank(in_features, num_ops=num_local_ops)

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

    def _apply_local_bank(self, x, mix_weights, local_gate):
        if x.dim() != 3 or x.shape[1] <= 1:
            return x, {}

        cls_tok = x[:, :1, :]
        patch_tok = x[:, 1:, :]
        batch_size, num_tokens, channels = patch_tok.shape
        grid_size = int(math.isqrt(num_tokens))
        if grid_size * grid_size != num_tokens:
            return x, {}

        patch_grid = patch_tok.transpose(1, 2).reshape(batch_size, channels, grid_size, grid_size)
        op_outputs = self.local_bank(patch_grid)
        stacked = torch.stack(op_outputs, dim=1)
        weights = mix_weights.to(dtype=patch_grid.dtype).view(batch_size, self.num_local_ops, 1, 1, 1)
        mixed_grid = (stacked * weights).sum(dim=1)
        local_tok = mixed_grid.reshape(batch_size, channels, num_tokens).transpose(1, 2)

        gate = local_gate.to(dtype=x.dtype).view(batch_size, 1, 1)
        mixed_patch = patch_tok + (self.adapter_scale * gate) * (local_tok - patch_tok)
        residual = (local_tok - patch_tok).detach().to(torch.float32)

        mix_weights_f32 = mix_weights.to(torch.float32)
        entropy = -(mix_weights_f32 * mix_weights_f32.clamp_min(1e-8).log()).sum(dim=-1).mean()
        stats = {
            "local_gate": float(local_gate.detach().to(torch.float32).mean().item()),
            "local_residual_norm": float(residual.norm(dim=-1).mean().item()),
            "routing_entropy": float(entropy.detach().to(torch.float32).item()),
        }
        weight_means = mix_weights_f32.detach().mean(dim=0)
        for idx in range(self.num_local_ops):
            stats[f"op{idx}_weight"] = float(weight_means[idx].item())

        return torch.cat([cls_tok, mixed_patch], dim=1), stats

    def set_adapter_scale(self, scale):
        self.adapter_scale = float(scale)

    def get_diagnostics(self):
        return dict(self.last_stats)


class AdaptiveLocalMixerLayer(_BaseLocalBankLayer):
    def __init__(
        self,
        in_features,
        out_features,
        num_local_ops=4,
        bias=True,
        controller_hidden_dim=128,
        controller_pool="cls_mean_var",
        local_gate_bias=-3.0,
        **kwargs,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            num_local_ops=num_local_ops,
            bias=bias,
            controller_pool=controller_pool,
        )
        controller_dim = controller_feature_dim(in_features, controller_pool)
        self.controller = AdaptiveLocalController(
            input_dim=controller_dim,
            num_ops=num_local_ops,
            hidden_dim=controller_hidden_dim,
            gate_bias=local_gate_bias,
        )

    def forward(self, x):
        z = self._conditioning_signal(x)
        mix_weights, local_gate = self.controller(z)
        mixed_x, stats = self._apply_local_bank(x, mix_weights, local_gate)
        self.last_stats = stats
        return F.linear(mixed_x, self.shared_weight, self.bias)


class StaticLocalBankLayer(_BaseLocalBankLayer):
    def __init__(
        self,
        in_features,
        out_features,
        num_local_ops=4,
        bias=True,
        controller_pool="cls_mean_var",
        local_gate_bias=-3.0,
        **kwargs,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            num_local_ops=num_local_ops,
            bias=bias,
            controller_pool=controller_pool,
        )
        self.mix_logits = nn.Parameter(torch.zeros(num_local_ops))
        self.local_gate_logit = nn.Parameter(torch.tensor(float(local_gate_bias)))

    def forward(self, x):
        batch_size = x.shape[0]
        mix_weights = torch.softmax(self.mix_logits, dim=0).view(1, -1).expand(batch_size, -1)
        local_gate = torch.sigmoid(self.local_gate_logit).view(1, 1).expand(batch_size, 1)
        mixed_x, stats = self._apply_local_bank(x, mix_weights, local_gate)
        self.last_stats = stats
        return F.linear(mixed_x, self.shared_weight, self.bias)


class EqualWeightLocalBankLayer(_BaseLocalBankLayer):
    def __init__(
        self,
        in_features,
        out_features,
        num_local_ops=4,
        bias=True,
        controller_pool="cls_mean_var",
        local_gate_bias=-3.0,
        **kwargs,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            num_local_ops=num_local_ops,
            bias=bias,
            controller_pool=controller_pool,
        )
        self.local_gate_logit = nn.Parameter(torch.tensor(float(local_gate_bias)))

    def forward(self, x):
        batch_size = x.shape[0]
        mix_weights = torch.full(
            (batch_size, self.num_local_ops),
            1.0 / float(self.num_local_ops),
            device=x.device,
            dtype=x.dtype,
        )
        local_gate = torch.sigmoid(self.local_gate_logit).view(1, 1).expand(batch_size, 1)
        mixed_x, stats = self._apply_local_bank(x, mix_weights, local_gate)
        self.last_stats = stats
        return F.linear(mixed_x, self.shared_weight, self.bias)
