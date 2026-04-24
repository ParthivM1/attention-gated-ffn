import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def infer_square_grid(num_tokens: int) -> tuple[int, int]:
    side = int(round(num_tokens ** 0.5))
    if side * side != num_tokens:
        raise ValueError(f"Expected a square token grid, got {num_tokens} visual tokens.")
    return side, side


def build_dct_basis(dim: int, num_bases: int, *, device=None, dtype=None) -> torch.Tensor:
    num_bases = min(int(num_bases), int(dim))
    positions = torch.arange(dim, device=device, dtype=torch.float32).unsqueeze(0)
    modes = torch.arange(num_bases, device=device, dtype=torch.float32).unsqueeze(1)
    basis = torch.cos(math.pi / float(dim) * (positions + 0.5) * modes)
    basis[0] = basis[0] / math.sqrt(2.0)
    basis = basis * math.sqrt(2.0 / float(dim))
    return basis.to(dtype=dtype or torch.float32)


def summarize_detail_tokens(x: torch.Tensor, topk: int = 4) -> torch.Tensor:
    if x.dim() != 3:
        return x
    if x.shape[1] <= 1:
        return x[:, 0]
    patch_tokens = x[:, 1:]
    patch_mean = patch_tokens.mean(dim=1, keepdim=True)
    detail_scores = (patch_tokens - patch_mean).pow(2).mean(dim=-1)
    k = min(max(int(topk), 1), patch_tokens.shape[1])
    top_idx = detail_scores.topk(k, dim=1).indices
    gathered = patch_tokens.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1]))
    return gathered.mean(dim=1)


class DepthwiseLocalMixer(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] <= 1:
            return x

        cls_token = x[:, :1]
        patch_tokens = x[:, 1:]
        height, width = infer_square_grid(patch_tokens.shape[1])
        patch_grid = patch_tokens.transpose(1, 2).reshape(x.shape[0], x.shape[2], height, width)
        mixed = self.conv(patch_grid).flatten(2).transpose(1, 2)
        return torch.cat([cls_token, mixed], dim=1)


class FactorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.left = nn.Linear(self.in_features, self.rank, bias=False)
        self.right = nn.Linear(self.rank, self.out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.right(self.left(x))


class SpectralBasisOperator(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_bases: int):
        super().__init__()
        self.num_bases = min(int(num_bases), int(in_features))
        basis = build_dct_basis(in_features, self.num_bases)
        self.register_buffer("input_basis", basis, persistent=False)
        self.output_basis = nn.Parameter(torch.empty(self.num_bases, out_features))
        nn.init.trunc_normal_(self.output_basis, std=0.02)

    def forward(self, x: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
        proj = torch.einsum("...d,kd->...k", x, self.input_basis.to(dtype=x.dtype))
        proj = proj * coeff.unsqueeze(-2 if x.dim() == 3 else 1)
        return torch.einsum("...k,ko->...o", proj, self.output_basis)


class SpectralLowRankOperator(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_bases: int,
        rank: int,
        *,
        learnable_input_basis: bool = False,
        shared_gate: bool = False,
    ):
        super().__init__()
        self.num_modes = min(int(num_bases), int(in_features))
        self.rank = max(1, int(rank))
        self.learnable_input_basis = bool(learnable_input_basis)
        self.shared_gate = bool(shared_gate)
        basis = build_dct_basis(in_features, self.num_modes)
        if self.learnable_input_basis:
            self.register_buffer("fixed_input_basis", basis, persistent=False)
            self.input_basis_delta = nn.Parameter(torch.zeros_like(basis))
            self.input_basis_mix_logit = nn.Parameter(torch.tensor(-3.0, dtype=torch.float32))
        else:
            self.register_buffer("input_basis", basis, persistent=False)
        self.mode_to_rank = nn.Parameter(torch.empty(self.num_modes, self.rank))
        self.output_basis = nn.Parameter(torch.empty(self.rank, out_features))
        self.shared_spectral_output_basis = None
        nn.init.zeros_(self.mode_to_rank)
        nn.init.trunc_normal_(self.output_basis, std=0.02)
        with torch.no_grad():
            diag = min(self.num_modes, self.rank)
            self.mode_to_rank[:diag, :diag] = torch.eye(diag, dtype=self.mode_to_rank.dtype)
        self.last_stats = {}

    def _current_input_basis(self) -> torch.Tensor:
        if not self.learnable_input_basis:
            return self.input_basis
        with torch.amp.autocast(device_type="cuda", enabled=False):
            fixed_basis = self.fixed_input_basis.to(dtype=torch.float32)
            mix = torch.sigmoid(self.input_basis_mix_logit.to(dtype=torch.float32))
            raw_basis = fixed_basis + mix * self.input_basis_delta.to(dtype=torch.float32)
            q = torch.linalg.qr(raw_basis.transpose(0, 1), mode="reduced").Q.transpose(0, 1)
            signs = torch.sign((q * fixed_basis).sum(dim=-1, keepdim=True))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            return q * signs

    def _current_mode_to_rank(self) -> torch.Tensor:
        with torch.amp.autocast(device_type="cuda", enabled=False):
            raw = self.mode_to_rank.to(dtype=torch.float32)
            gram = raw.transpose(0, 1) @ raw
            eye = torch.eye(self.rank, device=gram.device, dtype=gram.dtype)
            chol = torch.linalg.cholesky(gram + 1e-4 * eye)
            inv_chol = torch.linalg.solve_triangular(chol, eye, upper=False)
            q = raw @ inv_chol.transpose(0, 1)
            signs = torch.sign((q * raw).sum(dim=0, keepdim=True))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            return q * signs

    def attach_shared_spectral_output_basis(self, output_basis: torch.Tensor) -> None:
        self.shared_spectral_output_basis = output_basis

    def forward(self, x: torch.Tensor, spectral_coeff: torch.Tensor, rank_coeff: torch.Tensor) -> torch.Tensor:
        input_basis = self._current_input_basis()
        mode_to_rank = self._current_mode_to_rank()
        proj = torch.einsum("...d,kd->...k", x, input_basis.to(device=x.device, dtype=x.dtype))
        proj = proj * spectral_coeff.unsqueeze(-2 if x.dim() == 3 else 1)
        mode_to_rank_device = mode_to_rank.to(device=x.device, dtype=x.dtype)
        hidden = torch.einsum("...k,kr->...r", proj, mode_to_rank_device)
        if self.shared_gate:
            shared_rank_gate = torch.einsum(
                "...k,kr->...r",
                spectral_coeff,
                mode_to_rank_device.pow(2),
            )
            hidden = hidden * shared_rank_gate.unsqueeze(-2 if x.dim() == 3 else 1)
        else:
            hidden = hidden * rank_coeff.unsqueeze(-2 if x.dim() == 3 else 1)
        if self.shared_spectral_output_basis is not None:
            shared_output_basis = self.shared_spectral_output_basis.to(device=x.device, dtype=x.dtype)
            mode_out = torch.einsum("...r,kr->...k", hidden, mode_to_rank_device)
            out = torch.einsum("...k,ko->...o", mode_out, shared_output_basis)
        else:
            out = torch.einsum("...r,ro->...o", hidden, self.output_basis.to(device=x.device, dtype=x.dtype))
        basis_float = input_basis.detach().to(torch.float32)
        gram = basis_float @ basis_float.transpose(0, 1)
        eye = torch.eye(self.num_modes, device=gram.device, dtype=gram.dtype)
        mode_to_rank_float = mode_to_rank.detach().to(torch.float32)
        mode_gram = mode_to_rank_float.transpose(0, 1) @ mode_to_rank_float
        rank_eye = torch.eye(self.rank, device=mode_gram.device, dtype=mode_gram.dtype)
        self.last_stats = {
            "coupled_mode_hidden_norm": float(hidden.detach().to(torch.float32).norm(dim=-1).mean().item()),
            "coupled_mode_to_rank_norm": float(mode_to_rank_float.norm().item()),
            "coupled_mode_to_rank_orth_error": float(torch.linalg.vector_norm(mode_gram - rank_eye).item()),
            "coupled_output_basis_norm": float(self.output_basis.detach().to(torch.float32).norm().item()),
            "coupled_input_basis_orth_error": float(torch.linalg.vector_norm(gram - eye).item()),
            "coupled_learnable_input_basis": float(self.learnable_input_basis),
            "coupled_shared_gate": float(self.shared_gate),
            "coupled_input_basis_mix": float(torch.sigmoid(self.input_basis_mix_logit).detach().item()) if self.learnable_input_basis else 0.0,
            "coupled_shared_spectral_return": float(self.shared_spectral_output_basis is not None),
        }
        return out

    def get_diagnostics(self) -> dict[str, float]:
        return dict(self.last_stats)


class SpectralFlowResidual(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_modes: int,
        *,
        flow_rank: int,
        controller_hidden_dim: int,
        controller_rank: int = 4,
        steps: int = 1,
        step_size: float = 0.5,
        max_velocity: float = 0.75,
    ):
        super().__init__()
        self.num_modes = min(int(num_modes), int(in_features))
        self.flow_rank = min(max(int(flow_rank), 1), self.num_modes)
        self.steps = max(int(steps), 1)
        self.step_size = float(step_size)
        self.max_velocity = float(max_velocity)
        self.num_reflection_pairs = max(1, min(2, self.flow_rank))
        basis = build_dct_basis(in_features, self.num_modes)
        self.register_buffer("input_basis", basis, persistent=False)
        self.output_basis = nn.Parameter(torch.empty(self.flow_rank, out_features))
        self.shared_output_basis = None
        self.chart_delta = nn.Linear(int(controller_hidden_dim), self.num_reflection_pairs * self.flow_rank)
        self.gain_head = nn.Linear(int(controller_hidden_dim), self.flow_rank)
        self.chart_base = nn.Parameter(torch.zeros(self.num_reflection_pairs, self.flow_rank))
        nn.init.trunc_normal_(self.output_basis, std=0.02)
        nn.init.zeros_(self.chart_delta.weight)
        nn.init.zeros_(self.chart_delta.bias)
        nn.init.zeros_(self.gain_head.weight)
        nn.init.zeros_(self.gain_head.bias)
        with torch.no_grad():
            for pair in range(self.num_reflection_pairs):
                self.chart_base[pair, pair % self.flow_rank] = 1.0
        self.last_stats = {}
        self.last_aux_losses = {}

    def attach_shared_output_basis(self, output_basis: torch.Tensor) -> None:
        self.shared_output_basis = output_basis

    def _output_basis(self, device, dtype) -> torch.Tensor:
        if self.shared_output_basis is not None:
            return self.shared_output_basis[: self.flow_rank].to(device=device, dtype=dtype)
        return self.output_basis.to(device=device, dtype=dtype)

    def _householder(self, v: torch.Tensor) -> torch.Tensor:
        v = F.normalize(v, dim=-1, eps=1e-6)
        eye_shape = (1,) * (v.dim() - 1) + (self.flow_rank, self.flow_rank)
        eye = torch.eye(self.flow_rank, device=v.device, dtype=v.dtype).view(*eye_shape)
        outer = torch.einsum("...i,...j->...ij", v, v)
        return eye - 2.0 * outer

    def _build_chart(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = context.to(dtype=torch.float32)
        gains = 1.0 + 0.15 * torch.tanh(self.gain_head(context))
        base = self.chart_base.unsqueeze(0).expand(context.shape[0], -1, -1)
        delta = 0.1 * torch.tanh(self.chart_delta(context)).view(context.shape[0], self.num_reflection_pairs, self.flow_rank)
        charts = torch.eye(self.flow_rank, device=context.device, dtype=torch.float32).view(
            1, self.flow_rank, self.flow_rank
        ).expand(context.shape[0], -1, -1).clone()

        for pair in range(self.num_reflection_pairs):
            vec_a = base[:, pair]
            vec_b = base[:, pair] + delta[:, pair]
            charts = self._householder(vec_b) @ (self._householder(vec_a) @ charts)

        return charts, gains

    def forward(self, x: torch.Tensor, coeff: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        charts, gains = self._build_chart(context)
        proj = torch.einsum("...d,kd->...k", x, self.input_basis[: self.flow_rank].to(device=x.device, dtype=x.dtype))
        coeff_scale = 1.0 + coeff.to(dtype=torch.float32)
        if x.dim() == 3:
            state = proj.to(dtype=torch.float32) * coeff_scale.unsqueeze(1)
            transformed = torch.einsum("bnk,bkr->bnr", state, charts)
            transformed = transformed * gains.unsqueeze(1)
            delta_state = transformed - state
        else:
            state = proj.to(dtype=torch.float32) * coeff_scale
            transformed = torch.einsum("bk,bkr->br", state, charts)
            transformed = transformed * gains
            delta_state = transformed - state
        out = torch.einsum("...r,ro->...o", delta_state.to(dtype=x.dtype), self._output_basis(device=x.device, dtype=x.dtype))

        eye = torch.eye(self.flow_rank, device=charts.device, dtype=charts.dtype).view(1, self.flow_rank, self.flow_rank)
        gram = charts.transpose(-1, -2) @ charts
        orth_error = torch.linalg.vector_norm(gram - eye, dim=(-2, -1)).mean()
        anchor_loss = (charts - eye).pow(2).mean() + (gains - 1.0).pow(2).mean()
        energy_loss = delta_state.pow(2).mean()
        self.last_aux_losses = {
            "flow_anchor_loss": anchor_loss,
            "flow_energy_loss": energy_loss,
        }
        self.last_stats = {
            "flow_chart_orth_error": float(orth_error.item()),
            "flow_chart_norm": float(torch.linalg.vector_norm(charts, dim=(-2, -1)).mean().item()),
            "flow_delta_norm": float(delta_state.detach().to(torch.float32).norm(dim=-1).mean().item()),
            "flow_expert_entropy": 0.0,
            "flow_expert_top1": 1.0,
            "flow_gain_deviation": float((gains - 1.0).abs().mean().item()),
        }
        return out

    def get_diagnostics(self) -> dict[str, float]:
        return dict(self.last_stats)

    def get_aux_losses(self) -> dict[str, torch.Tensor]:
        return dict(self.last_aux_losses)


class SemanticManifoldOperator(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_modes: int,
        *,
        flow_rank: int,
        num_experts: int = 4,
        num_reflection_pairs: int = 2,
        expert_temperature: float = 0.5,
    ):
        super().__init__()
        self.num_modes = min(int(num_modes), int(in_features))
        self.flow_rank = min(max(int(flow_rank), 1), self.num_modes)
        self.num_experts = max(2, min(int(num_experts), self.flow_rank))
        self.num_reflection_pairs = max(1, min(int(num_reflection_pairs), self.flow_rank))
        self.expert_temperature = max(float(expert_temperature), 1e-3)
        basis = build_dct_basis(in_features, self.num_modes)
        self.register_buffer("input_basis", basis, persistent=False)
        self.output_basis = nn.Parameter(torch.empty(self.flow_rank, out_features))
        self.shared_output_basis = None
        self.expert_reflectors = nn.Parameter(torch.zeros(self.num_experts, self.num_reflection_pairs, self.flow_rank))
        nn.init.trunc_normal_(self.output_basis, std=0.02)
        with torch.no_grad():
            for expert in range(self.num_experts):
                for pair in range(self.num_reflection_pairs):
                    self.expert_reflectors[expert, pair, (expert + pair) % self.flow_rank] = 1.0
        self.last_stats = {}
        self.last_aux_losses = {}

    def attach_shared_output_basis(self, output_basis: torch.Tensor) -> None:
        self.shared_output_basis = output_basis

    def _output_basis(self, device, dtype) -> torch.Tensor:
        if self.shared_output_basis is not None:
            return self.shared_output_basis[: self.flow_rank].to(device=device, dtype=dtype)
        return self.output_basis.to(device=device, dtype=dtype)

    def _householder(self, v: torch.Tensor) -> torch.Tensor:
        v = F.normalize(v, dim=-1, eps=1e-6)
        eye_shape = (1,) * (v.dim() - 1) + (self.flow_rank, self.flow_rank)
        eye = torch.eye(self.flow_rank, device=v.device, dtype=v.dtype).view(*eye_shape)
        outer = torch.einsum("...i,...j->...ij", v, v)
        return eye - 2.0 * outer

    def _build_expert_charts(self, *, device, dtype) -> torch.Tensor:
        reflectors = self.expert_reflectors.to(device=device, dtype=dtype)
        charts = torch.eye(self.flow_rank, device=device, dtype=dtype).view(1, self.flow_rank, self.flow_rank)
        charts = charts.expand(self.num_experts, -1, -1).clone()
        for pair in range(self.num_reflection_pairs):
            charts = self._householder(reflectors[:, pair]) @ charts
        return charts

    def forward(
        self,
        x: torch.Tensor,
        expert_logits: torch.Tensor,
        mode_mask: torch.Tensor,
        gains: torch.Tensor,
    ) -> torch.Tensor:
        expert_weights = F.softmax(expert_logits.to(dtype=torch.float32) / self.expert_temperature, dim=-1)
        mode_mask = mode_mask.to(dtype=torch.float32)
        gains = gains.to(dtype=torch.float32)
        charts = self._build_expert_charts(device=x.device, dtype=torch.float32)
        proj = torch.einsum("...d,kd->...k", x, self.input_basis[: self.flow_rank].to(device=x.device, dtype=x.dtype))
        if x.dim() == 3:
            state = proj.to(dtype=torch.float32) * mode_mask.unsqueeze(1)
        else:
            state = proj.to(dtype=torch.float32) * mode_mask
        if x.dim() == 3:
            transformed = torch.einsum("bnr,erk->benk", state, charts)
            transformed = transformed * gains.unsqueeze(2)
            delta_state = (expert_weights.unsqueeze(-1).unsqueeze(-1) * (transformed - state.unsqueeze(1))).sum(dim=1)
        else:
            transformed = torch.einsum("br,erk->bek", state, charts)
            transformed = transformed * gains
            delta_state = (expert_weights.unsqueeze(-1) * (transformed - state.unsqueeze(1))).sum(dim=1)
        out = torch.einsum("...r,ro->...o", delta_state.to(dtype=x.dtype), self._output_basis(device=x.device, dtype=x.dtype))

        eye = torch.eye(self.flow_rank, device=charts.device, dtype=charts.dtype).view(1, self.flow_rank, self.flow_rank)
        gram = charts.transpose(-1, -2) @ charts
        orth_error = torch.linalg.vector_norm(gram - eye, dim=(-2, -1)).mean()
        mixed_chart = torch.einsum("be,erk->brk", expert_weights, charts)
        mixed_gain = torch.einsum("be,ber->br", expert_weights, gains)
        anchor_loss = (mixed_chart - eye).pow(2).mean() + (mixed_gain - 1.0).pow(2).mean()
        entropy = -(expert_weights * torch.log(expert_weights.clamp_min(1e-8))).sum(dim=-1).mean()
        mode_reg = (mode_mask * (1.0 - mode_mask)).mean()
        top1 = expert_weights.max(dim=-1).values.mean()
        self.last_aux_losses = {
            "semantic_anchor_loss": anchor_loss,
            "semantic_entropy_loss": entropy,
            "semantic_mode_reg_loss": mode_reg,
        }
        self.last_stats = {
            "semantic_chart_orth_error": float(orth_error.item()),
            "semantic_chart_norm": float(torch.linalg.vector_norm(charts, dim=(-2, -1)).mean().item()),
            "semantic_delta_norm": float(delta_state.detach().to(torch.float32).norm(dim=-1).mean().item()),
            "semantic_expert_entropy": float(entropy.item()),
            "semantic_expert_top1": float(top1.item()),
            "semantic_gain_deviation": float((gains - 1.0).abs().mean().item()),
            "semantic_mode_mean": float(mode_mask.mean().item()),
        }
        return out

    def get_diagnostics(self) -> dict[str, float]:
        return dict(self.last_stats)

    def get_aux_losses(self) -> dict[str, torch.Tensor]:
        return dict(self.last_aux_losses)


class MagnusSemanticOperator(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_modes: int,
        *,
        flow_rank: int,
        controller_hidden_dim: int,
        detail_topk: int = 4,
        max_gain: float = 0.1,
    ):
        super().__init__()
        self.num_modes = min(int(num_modes), int(in_features))
        self.flow_rank = min(max(int(flow_rank), 1), self.num_modes)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.controller_hidden_dim = int(controller_hidden_dim)
        self.detail_topk = max(int(detail_topk), 1)
        self.max_gain = float(max_gain)
        basis = build_dct_basis(in_features, self.flow_rank)
        self.register_buffer("input_basis", basis[: self.flow_rank].clone(), persistent=False)
        self.output_basis = nn.Parameter(torch.empty(self.flow_rank, out_features))

        self.local_proj = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, self.controller_hidden_dim),
            nn.GELU(),
            nn.Linear(self.controller_hidden_dim, self.controller_hidden_dim),
        )
        self.global_head = nn.Linear(self.controller_hidden_dim, self.flow_rank * self.flow_rank)
        self.local_head = nn.Linear(self.controller_hidden_dim, self.flow_rank * self.flow_rank)
        self.mode_head = nn.Linear(2 * self.controller_hidden_dim, self.flow_rank)
        self.gain_head = nn.Linear(2 * self.controller_hidden_dim, self.flow_rank)
        self.gate_head = nn.Linear(2 * self.controller_hidden_dim, 1)
        self.alpha_head = nn.Linear(2 * self.controller_hidden_dim, 1)
        self.beta_head = nn.Linear(2 * self.controller_hidden_dim, 1)

        nn.init.trunc_normal_(self.output_basis, std=0.02)
        self._init_heads()

        self.last_stats = {}
        self.last_aux_losses = {}

    def _init_heads(self) -> None:
        nn.init.zeros_(self.global_head.weight)
        nn.init.zeros_(self.global_head.bias)
        nn.init.zeros_(self.local_head.weight)
        nn.init.zeros_(self.local_head.bias)
        nn.init.zeros_(self.mode_head.weight)
        nn.init.constant_(self.mode_head.bias, 1.5)
        nn.init.zeros_(self.gain_head.weight)
        nn.init.zeros_(self.gain_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -1.0)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.constant_(self.alpha_head.bias, -1.0)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def _output_basis(self, device, dtype) -> torch.Tensor:
        return self.output_basis.to(device=device, dtype=dtype)

    @torch.no_grad()
    def initialize_from_weight(self, weight: torch.Tensor) -> None:
        weight = weight.detach().to(dtype=torch.float32)
        if weight.dim() != 2:
            return
        u, _, vh = torch.linalg.svd(weight, full_matrices=False)
        rank = min(self.flow_rank, u.shape[1], vh.shape[0])
        input_basis = torch.zeros(self.flow_rank, self.in_features, device=weight.device, dtype=torch.float32)
        output_basis = torch.zeros(self.flow_rank, self.out_features, device=weight.device, dtype=torch.float32)
        input_basis[:rank] = vh[:rank]
        output_basis[:rank] = u[:, :rank].transpose(0, 1)
        self.input_basis.copy_(input_basis.to(device=self.input_basis.device, dtype=self.input_basis.dtype))
        self.output_basis.copy_(output_basis.to(device=self.output_basis.device, dtype=self.output_basis.dtype))

    def _detail_summary(self, x: torch.Tensor) -> torch.Tensor:
        return summarize_detail_tokens(x, topk=self.detail_topk)

    def _skew(self, raw: torch.Tensor) -> torch.Tensor:
        return 0.5 * (raw - raw.transpose(-1, -2))

    def forward(self, x: torch.Tensor, context: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
        context = context.to(dtype=torch.float32)
        detail = self._detail_summary(x).to(dtype=torch.float32)
        local_context = self.local_proj(detail)
        fused = torch.cat([context, local_context], dim=-1)

        global_skew = self._skew(self.global_head(context).view(context.shape[0], self.flow_rank, self.flow_rank))
        local_skew = self._skew(self.local_head(local_context).view(context.shape[0], self.flow_rank, self.flow_rank))

        mode_mask = torch.sigmoid(self.mode_head(fused))
        diag_mask = torch.diag_embed(mode_mask)
        masked_global = diag_mask @ global_skew @ diag_mask
        masked_local = diag_mask @ local_skew @ diag_mask
        commutator = masked_local @ masked_global - masked_global @ masked_local

        alpha = float(scale) * 0.5 * torch.sigmoid(self.alpha_head(fused))
        beta = float(scale) * 0.25 * torch.tanh(self.beta_head(fused))
        omega = alpha.view(-1, 1, 1) * 0.5 * (masked_global + masked_local)
        omega = omega + beta.view(-1, 1, 1) * (commutator / 12.0)
        q = torch.matrix_exp(omega)

        gains = 1.0 + self.max_gain * torch.tanh(self.gain_head(fused))
        operator = q @ torch.diag_embed(gains)
        proj = torch.einsum("...d,kd->...k", x, self.input_basis.to(device=x.device, dtype=x.dtype))
        gate = float(scale) * torch.sigmoid(self.gate_head(fused))
        if x.dim() == 3:
            state = proj.to(dtype=torch.float32)
            transformed = torch.einsum("bnr,brk->bnk", state, operator)
            delta_state = gate.unsqueeze(1) * (transformed - state)
        else:
            state = proj.to(dtype=torch.float32)
            transformed = torch.einsum("br,brk->bk", state, operator)
            delta_state = gate * (transformed - state)
        out = torch.einsum(
            "...r,ro->...o",
            delta_state.to(dtype=x.dtype),
            self._output_basis(device=x.device, dtype=x.dtype),
        )

        eye = torch.eye(self.flow_rank, device=x.device, dtype=torch.float32).unsqueeze(0)
        gram = q.transpose(-1, -2) @ q
        orth_error = torch.linalg.vector_norm(gram - eye, dim=(-2, -1)).mean()
        scale_tensor = torch.tensor(float(scale), device=x.device, dtype=torch.float32)
        anchor_loss = (scale_tensor * (operator - eye)).pow(2).mean()
        motion_loss = (scale_tensor * omega).pow(2).mean()
        comm_loss = (scale_tensor * commutator).pow(2).mean()
        mode_reg = scale_tensor * (mode_mask * (1.0 - mode_mask)).mean()
        self.last_aux_losses = {
            "magnus_anchor_loss": torch.nan_to_num(anchor_loss),
            "magnus_motion_loss": torch.nan_to_num(motion_loss),
            "magnus_comm_loss": torch.nan_to_num(comm_loss),
            "magnus_mode_reg_loss": torch.nan_to_num(mode_reg),
        }
        self.last_stats = {
            "magnus_q_orth_error": float(orth_error.item()),
            "magnus_omega_norm": float(torch.linalg.vector_norm(omega, dim=(-2, -1)).mean().item()),
            "magnus_comm_norm": float(torch.linalg.vector_norm(commutator, dim=(-2, -1)).mean().item()),
            "magnus_gate": float(gate.mean().item()),
            "magnus_mode_mean": float(mode_mask.mean().item()),
            "magnus_gain_deviation": float((gains - 1.0).abs().mean().item()),
            "magnus_delta_norm": float(delta_state.detach().to(torch.float32).norm(dim=-1).mean().item()),
            "magnus_alpha": float(alpha.mean().item()),
            "magnus_beta": float(beta.abs().mean().item()),
        }
        return out

    def get_diagnostics(self) -> dict[str, float]:
        return dict(self.last_stats)

    def get_aux_losses(self) -> dict[str, torch.Tensor]:
        return dict(self.last_aux_losses)


class LowRankResidualOperator(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.rank = int(rank)
        self.left = nn.Parameter(torch.empty(out_features, self.rank))
        self.right = nn.Parameter(torch.empty(in_features, self.rank))
        nn.init.trunc_normal_(self.left, std=0.02)
        nn.init.trunc_normal_(self.right, std=0.02)

    def forward(self, x: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
        hidden = torch.einsum("...d,dr->...r", x, self.right)
        hidden = hidden * coeff.unsqueeze(-2 if x.dim() == 3 else 1)
        return torch.einsum("...r,or->...o", hidden, self.left)


class LowRankOrthogonalMixer(nn.Module):
    def __init__(self, features: int, rank: int, *, magnus_rotation_mode: bool = False):
        super().__init__()
        self.features = int(features)
        self.rank = int(rank)
        self.magnus_rotation_mode = bool(magnus_rotation_mode)
        self.left = nn.Parameter(torch.empty(self.features, self.rank))
        self.right = nn.Parameter(torch.empty(self.features, self.rank))
        nn.init.orthogonal_(self.left)
        nn.init.orthogonal_(self.right)
        if self.magnus_rotation_mode:
            self.left_local = nn.Parameter(torch.empty(self.features, self.rank))
            self.right_local = nn.Parameter(torch.empty(self.features, self.rank))
            nn.init.orthogonal_(self.left_local)
            nn.init.orthogonal_(self.right_local)
        else:
            self.left_local = None
            self.right_local = None
        self.last_stats = {}

    def _build_skew(
        self,
        coeff: torch.Tensor,
        gate: torch.Tensor,
        *,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        coeff_float = coeff.to(dtype=torch.float32)
        gate_float = gate.to(dtype=torch.float32)
        scaled_left = left.to(dtype=torch.float32).unsqueeze(0) * coeff_float.unsqueeze(1)
        right = right.to(dtype=torch.float32).unsqueeze(0).expand(coeff.shape[0], -1, -1)
        base = torch.matmul(scaled_left, right.transpose(1, 2))
        skew = base - base.transpose(1, 2)
        return gate_float.view(-1, 1, 1) * skew

    def forward(
        self,
        x: torch.Tensor,
        coeff: torch.Tensor,
        gate: torch.Tensor,
        *,
        coeff_local: torch.Tensor | None = None,
        gate_local: torch.Tensor | None = None,
        comm_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.rank <= 0:
            return x

        x_float = x.to(dtype=torch.float32)
        global_skew = self._build_skew(coeff, gate, left=self.left, right=self.right)
        if self.magnus_rotation_mode and coeff_local is not None and gate_local is not None:
            local_skew = self._build_skew(coeff_local, gate_local, left=self.left_local, right=self.right_local)
            omega = 0.5 * (global_skew + local_skew)
            commutator = local_skew @ global_skew - global_skew @ local_skew
            if comm_scale is not None:
                omega = omega + comm_scale.to(dtype=torch.float32).view(-1, 1, 1) * (commutator / 12.0)
            self.last_stats = {
                "rotation_global_skew_norm": float(torch.linalg.vector_norm(global_skew, dim=(-2, -1)).mean().item()),
                "rotation_local_skew_norm": float(torch.linalg.vector_norm(local_skew, dim=(-2, -1)).mean().item()),
                "rotation_comm_norm": float(torch.linalg.vector_norm(commutator, dim=(-2, -1)).mean().item()),
            }
        else:
            omega = global_skew
            self.last_stats = {
                "rotation_global_skew_norm": float(torch.linalg.vector_norm(global_skew, dim=(-2, -1)).mean().item()),
                "rotation_local_skew_norm": 0.0,
                "rotation_comm_norm": 0.0,
            }
        eye = torch.eye(self.features, device=x.device, dtype=torch.float32).unsqueeze(0).expand(x.shape[0], -1, -1)
        transform = torch.linalg.solve(eye - 0.5 * omega, eye + 0.5 * omega)
        mixed = torch.einsum("bnd,bdh->bnh", x_float, transform)
        return mixed.to(dtype=x.dtype)

    def get_diagnostics(self) -> dict[str, float]:
        return dict(self.last_stats)


class SharedGeoOperatorBank(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_spectral_bases: int,
        low_rank_rank: int,
        orthogonal_rank: int,
        flow_rank: int = 0,
        controller_hidden_dim: int = 128,
        flow_steps: int = 1,
        flow_step_size: float = 0.5,
        semantic_manifold_mode: bool = False,
        semantic_num_experts: int = 2,
        semantic_expert_temperature: float = 0.5,
        magnus_semantic_mode: bool = False,
        magnus_detail_topk: int = 4,
        magnus_rotation_mode: bool = False,
        coupled_spectral_low_rank: bool = False,
        coupled_learnable_input_basis: bool = False,
        coupled_shared_gate: bool = False,
        *,
        use_local_operator: bool = False,
        local_kernel_size: int = 3,
    ):
        super().__init__()
        self.local_mixer = DepthwiseLocalMixer(in_features, kernel_size=local_kernel_size) if use_local_operator else None
        self.spectral_operator = SpectralBasisOperator(in_features, out_features, num_spectral_bases)
        self.low_rank_operator = LowRankResidualOperator(in_features, out_features, low_rank_rank)
        self.coupled_operator = (
            SpectralLowRankOperator(
                in_features,
                out_features,
                num_spectral_bases,
                low_rank_rank,
                learnable_input_basis=coupled_learnable_input_basis,
                shared_gate=coupled_shared_gate,
            )
            if coupled_spectral_low_rank
            else None
        )
        if self.coupled_operator is not None:
            self.coupled_operator.attach_shared_spectral_output_basis(self.spectral_operator.output_basis)
        self.orthogonal_mixer = LowRankOrthogonalMixer(
            in_features,
            orthogonal_rank,
            magnus_rotation_mode=magnus_rotation_mode,
        )
        self.magnus_operator = (
            MagnusSemanticOperator(
                in_features,
                out_features,
                num_spectral_bases,
                flow_rank=flow_rank,
                controller_hidden_dim=controller_hidden_dim,
                detail_topk=magnus_detail_topk,
            )
            if magnus_semantic_mode and int(flow_rank) > 0
            else None
        )
        self.semantic_operator = (
            SemanticManifoldOperator(
                in_features,
                out_features,
                num_spectral_bases,
                flow_rank=flow_rank,
                num_experts=semantic_num_experts,
                expert_temperature=semantic_expert_temperature,
            )
            if self.magnus_operator is None and semantic_manifold_mode and int(flow_rank) > 0
            else None
        )
        self.flow_operator = None
        if self.magnus_operator is not None:
            pass
        elif self.semantic_operator is not None:
            self.semantic_operator.attach_shared_output_basis(self.spectral_operator.output_basis)
        else:
            self.flow_operator = (
                SpectralFlowResidual(
                    in_features,
                    out_features,
                    num_spectral_bases,
                    flow_rank=flow_rank,
                    controller_hidden_dim=controller_hidden_dim,
                    steps=flow_steps,
                    step_size=flow_step_size,
                )
                if int(flow_rank) > 0
                else None
            )
            if self.flow_operator is not None:
                self.flow_operator.attach_shared_output_basis(self.spectral_operator.output_basis)
