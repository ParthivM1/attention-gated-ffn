import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from layers.geo_operator_linear import GeoOperatorLinear
    from models.vit import DropPath
except ImportError:
    from src.layers.geo_operator_linear import GeoOperatorLinear
    from src.models.vit import DropPath


class GeoAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qkv_layer=nn.Linear,
        proj_layer=nn.Linear,
        metric_adapter: nn.Module | None = None,
        metric_scale: float = 1.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = qkv_layer(dim, dim * 3, bias=qkv_bias)
        self.proj = proj_layer(dim, dim)
        self.metric_adapter = metric_adapter
        self.metric_scale = float(metric_scale)

    def _apply_linear(self, module, x: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        if isinstance(module, GeoOperatorLinear):
            return module(x, condition)
        return module(x)

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        if self.metric_adapter is not None:
            metric_x = self.metric_adapter(x, residual_scale=self.metric_scale)
            metric_qkv = self._apply_linear(self.qkv, metric_x, condition).reshape(
                batch_size,
                num_tokens,
                3,
                self.num_heads,
                channels // self.num_heads,
            ).permute(2, 0, 3, 1, 4)
            value_qkv = self._apply_linear(self.qkv, x, condition).reshape(
                batch_size,
                num_tokens,
                3,
                self.num_heads,
                channels // self.num_heads,
            ).permute(2, 0, 3, 1, 4)
            q, k, v = metric_qkv[0], metric_qkv[1], value_qkv[2]
        else:
            qkv = self._apply_linear(self.qkv, x, condition).reshape(
                batch_size,
                num_tokens,
                3,
                self.num_heads,
                channels // self.num_heads,
            ).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, channels)
        return self._apply_linear(self.proj, out, condition)


class SharedAttentionMetricAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        bottleneck_dim: int = 8,
        *,
        patch_only: bool = True,
        gate_bias: float = -3.0,
        init_scale: float = 0.01,
        cls_context_scale: float = 0.25,
    ):
        super().__init__()
        self.dim = int(dim)
        self.bottleneck_dim = max(int(bottleneck_dim), 4)
        self.patch_only = bool(patch_only)
        self.cls_context_scale = float(cls_context_scale)
        self.reduce = nn.Linear(self.dim, self.bottleneck_dim, bias=False)
        self.expand = nn.Linear(self.bottleneck_dim, self.dim, bias=False)
        self.cls_proj = nn.Linear(self.dim, self.bottleneck_dim, bias=False)
        self.gate = nn.Linear(self.dim, 1)
        self.layer_scale = nn.Parameter(torch.full((1, 1, self.dim), float(init_scale)))
        self.act = nn.GELU()
        self.last_stats = {}
        self._init_weights(gate_bias=gate_bias)

    def _init_weights(self, *, gate_bias: float) -> None:
        with torch.no_grad():
            nn.init.trunc_normal_(self.reduce.weight, std=0.02)
            nn.init.trunc_normal_(self.expand.weight, std=0.02)
            nn.init.trunc_normal_(self.cls_proj.weight, std=0.02)
            self.expand.weight.mul_(0.1)
            self.cls_proj.weight.mul_(0.1)
            self.gate.weight.zero_()
            self.gate.bias.fill_(gate_bias)

    def forward(self, x: torch.Tensor, *, residual_scale: float = 1.0) -> torch.Tensor:
        if x.dim() != 3:
            self.last_stats = {}
            return x
        cls_hidden = x[:, 0]
        if self.patch_only and x.shape[1] > 1:
            token_hidden = x[:, 1:]
        else:
            token_hidden = x
        update = self.reduce(token_hidden)
        if self.cls_context_scale != 0.0:
            cls_context = self.cls_proj(cls_hidden).unsqueeze(1)
            update = update + self.cls_context_scale * cls_context
        update = self.act(update)
        update = self.expand(update) * self.layer_scale
        gate = torch.sigmoid(self.gate(cls_hidden)).view(x.shape[0], 1, 1)
        self.last_stats = {
            "attention_metric_gate": float(gate.detach().mean().item()),
            "attention_metric_update_norm": float(update.detach().norm(dim=-1).mean().item()),
            "attention_metric_bottleneck": float(self.bottleneck_dim),
            "attention_metric_layer_scale": float(self.layer_scale.detach().abs().mean().item()),
            "attention_metric_patch_only": float(self.patch_only),
            "attention_metric_cls_context_scale": float(self.cls_context_scale),
        }
        if self.patch_only and x.shape[1] > 1:
            delta = torch.cat((torch.zeros_like(x[:, :1]), update), dim=1)
        else:
            delta = update
        return x + residual_scale * gate * delta


class SharedActivationFlowGate(nn.Module):
    def __init__(
        self,
        dim: int,
        bottleneck_dim: int = 16,
        *,
        patch_only: bool = False,
        gate_bias: float = -4.0,
        init_scale: float = 0.01,
        cls_mix_scale: float = 1.0,
        mean_mix_scale: float = 0.5,
        std_mix_scale: float = 0.25,
        cls_token_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.bottleneck_dim = max(int(bottleneck_dim), 4)
        self.patch_only = bool(patch_only)
        self.cls_mix_scale = float(cls_mix_scale)
        self.mean_mix_scale = float(mean_mix_scale)
        self.std_mix_scale = float(std_mix_scale)
        self.cls_token_scale = float(cls_token_scale)
        self.norm = nn.LayerNorm(self.dim)
        self.reduce = nn.Linear(self.dim, self.bottleneck_dim, bias=False)
        self.expand = nn.Linear(self.bottleneck_dim, self.dim, bias=False)
        self.token_gate = nn.Linear(self.dim, 1)
        self.summary_gate = nn.Linear(self.dim, 1)
        self.layer_scale = nn.Parameter(torch.full((1, 1, self.dim), float(init_scale)))
        self.act = nn.GELU()
        self.last_stats = {}
        self._init_heads(gate_bias=gate_bias)

    def _init_heads(self, *, gate_bias: float = -4.0) -> None:
        with torch.no_grad():
            nn.init.trunc_normal_(self.reduce.weight, std=0.02)
            nn.init.trunc_normal_(self.expand.weight, std=0.02)
            self.expand.weight.mul_(0.1)
            self.token_gate.weight.zero_()
            self.token_gate.bias.zero_()
            self.summary_gate.weight.zero_()
            self.summary_gate.bias.fill_(gate_bias)

    def forward(self, x: torch.Tensor, *, residual_scale: float = 1.0) -> torch.Tensor:
        if x.dim() != 3:
            self.last_stats = {}
            return x
        hidden = self.norm(x)
        cls_hidden = hidden[:, 0]
        if hidden.shape[1] > 1:
            patch_hidden = hidden[:, 1:]
            patch_mean = patch_hidden.mean(dim=1)
            patch_std = patch_hidden.std(dim=1, unbiased=False)
        else:
            patch_mean = torch.zeros_like(cls_hidden)
            patch_std = torch.zeros_like(cls_hidden)
        summary = (
            self.cls_mix_scale * cls_hidden
            + self.mean_mix_scale * patch_mean
            + self.std_mix_scale * patch_std
        )
        channel_update = self.expand(self.act(self.reduce(summary))).unsqueeze(1)
        token_gate = torch.sigmoid(self.token_gate(hidden))
        if self.patch_only and hidden.shape[1] > 1:
            token_gate = torch.cat((torch.zeros_like(token_gate[:, :1]), token_gate[:, 1:]), dim=1)
        elif self.cls_token_scale != 1.0:
            token_gate = token_gate.clone()
            token_gate[:, :1] = token_gate[:, :1] * self.cls_token_scale
        gate = torch.sigmoid(self.summary_gate(summary)).view(hidden.shape[0], 1, 1)
        update = torch.tanh(channel_update) * token_gate * self.layer_scale
        self.last_stats = {
            "activation_flow_gate": float(gate.detach().mean().item()),
            "activation_flow_token_gate": float(token_gate.detach().mean().item()),
            "activation_flow_update_norm": float(update.detach().norm(dim=-1).mean().item()),
            "activation_flow_layer_scale": float(self.layer_scale.detach().abs().mean().item()),
            "activation_flow_bottleneck": float(self.bottleneck_dim),
            "activation_flow_patch_only": float(self.patch_only),
            "activation_flow_cls_mix_scale": float(self.cls_mix_scale),
            "activation_flow_mean_mix_scale": float(self.mean_mix_scale),
            "activation_flow_std_mix_scale": float(self.std_mix_scale),
            "activation_flow_cls_token_scale": float(self.cls_token_scale),
        }
        return x + residual_scale * gate * update


class TokenResponseFlowNorm(nn.Module):
    def __init__(self, dim: int, *, init_scale: float = 0.01, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.gamma = nn.Parameter(torch.full((1, 1, dim), float(init_scale)))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))
        self.last_stats = {}

    def forward(self, x: torch.Tensor, *, residual_scale: float = 1.0) -> torch.Tensor:
        if x.dim() != 3:
            self.last_stats = {}
            return x
        token_norm = torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
        token_response = token_norm / token_norm.mean(dim=-1, keepdim=True).clamp_min(self.eps)
        update = self.gamma * (x * token_response) + self.beta
        self.last_stats = {
            "response_flow_gamma": float(self.gamma.detach().abs().mean().item()),
            "response_flow_beta": float(self.beta.detach().abs().mean().item()),
            "response_flow_update_norm": float(update.detach().norm(dim=-1).mean().item()),
        }
        return x + residual_scale * update


class SharedBiaxialResponseFlow(nn.Module):
    def __init__(
        self,
        dim: int,
        bottleneck_dim: int = 12,
        *,
        patch_only: bool = False,
        gate_bias: float = -4.0,
        init_scale: float = 0.01,
        cls_mix_scale: float = 1.0,
        mean_mix_scale: float = 0.5,
        token_exponent: float = 0.5,
        channel_exponent: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = int(dim)
        self.bottleneck_dim = max(int(bottleneck_dim), 4)
        self.patch_only = bool(patch_only)
        self.cls_mix_scale = float(cls_mix_scale)
        self.mean_mix_scale = float(mean_mix_scale)
        self.token_exponent = float(token_exponent)
        self.channel_exponent = float(channel_exponent)
        self.eps = float(eps)
        self.norm = nn.LayerNorm(self.dim)
        self.reduce = nn.Linear(self.dim, self.bottleneck_dim, bias=False)
        self.gain_head = nn.Linear(self.bottleneck_dim, self.dim, bias=False)
        self.bias_head = nn.Linear(self.bottleneck_dim, self.dim, bias=False)
        self.token_gate = nn.Linear(self.dim, 1)
        self.summary_gate = nn.Linear(self.bottleneck_dim, 1)
        self.layer_scale = nn.Parameter(torch.full((1, 1, self.dim), float(init_scale)))
        self.act = nn.GELU()
        self.last_stats = {}
        self._init_heads(gate_bias=gate_bias)

    def _init_heads(self, *, gate_bias: float = -4.0) -> None:
        with torch.no_grad():
            nn.init.trunc_normal_(self.reduce.weight, std=0.02)
            nn.init.trunc_normal_(self.gain_head.weight, std=0.02)
            nn.init.trunc_normal_(self.bias_head.weight, std=0.02)
            self.gain_head.weight.mul_(0.1)
            self.bias_head.weight.mul_(0.1)
            self.token_gate.weight.zero_()
            self.token_gate.bias.zero_()
            self.summary_gate.weight.zero_()
            self.summary_gate.bias.fill_(gate_bias)

    def forward(self, x: torch.Tensor, *, residual_scale: float = 1.0) -> torch.Tensor:
        if x.dim() != 3:
            self.last_stats = {}
            return x
        hidden = self.norm(x)
        cls_hidden = hidden[:, 0]
        if hidden.shape[1] > 1:
            patch_mean = hidden[:, 1:].mean(dim=1)
        else:
            patch_mean = torch.zeros_like(cls_hidden)
        summary = self.cls_mix_scale * cls_hidden + self.mean_mix_scale * patch_mean
        summary_hidden = self.act(self.reduce(summary))
        channel_gain = torch.tanh(self.gain_head(summary_hidden)).unsqueeze(1)
        channel_bias = torch.tanh(self.bias_head(summary_hidden)).unsqueeze(1)
        token_gate = torch.sigmoid(self.token_gate(hidden))
        if self.patch_only and hidden.shape[1] > 1:
            token_gate = token_gate.clone()
            token_gate[:, :1] = 0.0
        summary_gate = torch.sigmoid(self.summary_gate(summary_hidden)).view(hidden.shape[0], 1, 1)
        token_response = torch.linalg.vector_norm(hidden, ord=2, dim=-1, keepdim=True)
        token_response = token_response / token_response.mean(dim=1, keepdim=True).clamp_min(self.eps)
        channel_response = torch.linalg.vector_norm(hidden, ord=2, dim=1, keepdim=True)
        channel_response = channel_response / channel_response.mean(dim=-1, keepdim=True).clamp_min(self.eps)
        response = token_response.clamp_min(self.eps).pow(self.token_exponent) * channel_response.clamp_min(self.eps).pow(self.channel_exponent)
        response = response - 1.0
        update = self.layer_scale * token_gate * (hidden * response * channel_gain + channel_bias)
        self.last_stats = {
            "biaxial_response_gate": float(summary_gate.detach().mean().item()),
            "biaxial_response_token_gate": float(token_gate.detach().mean().item()),
            "biaxial_response_update_norm": float(update.detach().norm(dim=-1).mean().item()),
            "biaxial_response_layer_scale": float(self.layer_scale.detach().abs().mean().item()),
            "biaxial_response_bottleneck": float(self.bottleneck_dim),
            "biaxial_response_patch_only": float(self.patch_only),
            "biaxial_response_token_exp": float(self.token_exponent),
            "biaxial_response_channel_exp": float(self.channel_exponent),
        }
        return x + residual_scale * summary_gate * update


class CompetitiveResidualMixer(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        gate_bias: float = 0.0,
        init_scale: float = 1.0,
        cls_mix_scale: float = 1.0,
        mean_mix_scale: float = 0.5,
        patch_only: bool = False,
    ):
        super().__init__()
        self.dim = int(dim)
        self.cls_mix_scale = float(cls_mix_scale)
        self.mean_mix_scale = float(mean_mix_scale)
        self.patch_only = bool(patch_only)
        self.summary_norm = nn.LayerNorm(self.dim * 2)
        self.branch_head = nn.Linear(self.dim * 2, 2)
        self.token_head = nn.Linear(self.dim, 2)
        self.layer_scale = nn.Parameter(torch.full((1, 1, 2), float(init_scale)))
        self.last_stats = {}
        self._gate_bias = float(gate_bias)
        self._init_heads(gate_bias=gate_bias)

    def _init_heads(self, *, gate_bias: float | None = None) -> None:
        gate_bias = self._gate_bias if gate_bias is None else float(gate_bias)
        with torch.no_grad():
            self.branch_head.weight.zero_()
            self.branch_head.bias.fill_(gate_bias)
            self.token_head.weight.zero_()
            self.token_head.bias.zero_()

    def forward(
        self,
        x: torch.Tensor,
        attn_out: torch.Tensor,
        mlp_out: torch.Tensor,
        *,
        residual_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            self.last_stats = {}
            return attn_out, mlp_out
        cls_hidden = x[:, 0]
        if x.shape[1] > 1:
            patch_mean = x[:, 1:].mean(dim=1)
        else:
            patch_mean = torch.zeros_like(cls_hidden)
        summary = self.summary_norm(
            torch.cat(
                (self.cls_mix_scale * cls_hidden, self.mean_mix_scale * patch_mean),
                dim=-1,
            )
        )
        branch_mix = torch.softmax(self.branch_head(summary), dim=-1).view(x.shape[0], 1, 2)
        token_mix = torch.softmax(self.token_head(x), dim=-1)
        if self.patch_only and token_mix.shape[1] > 1:
            token_mix = token_mix.clone()
            token_mix[:, :1, :] = branch_mix
        mix = 0.5 * (branch_mix + token_mix)
        mix = mix * self.layer_scale
        attn_weight = residual_scale * mix[..., :1]
        mlp_weight = residual_scale * mix[..., 1:]
        self.last_stats = {
            "competitive_attn_weight": float(attn_weight.detach().mean().item()),
            "competitive_mlp_weight": float(mlp_weight.detach().mean().item()),
            "competitive_attn_peak": float(attn_weight.detach().amax().item()),
            "competitive_mlp_peak": float(mlp_weight.detach().amax().item()),
            "competitive_scale": float(self.layer_scale.detach().abs().mean().item()),
            "competitive_patch_only": float(self.patch_only),
        }
        return attn_out * attn_weight, mlp_out * mlp_weight


class SharedHiddenChannelFlow(nn.Module):
    def __init__(
        self,
        dim: int,
        bottleneck_dim: int = 16,
        rank: int = 16,
        *,
        patch_only: bool = False,
        gate_bias: float = -3.5,
        init_scale: float = 0.01,
        cls_mix_scale: float = 1.0,
        mean_mix_scale: float = 0.5,
    ):
        super().__init__()
        self.dim = int(dim)
        self.bottleneck_dim = max(int(bottleneck_dim), 4)
        self.rank = max(int(rank), 4)
        self.patch_only = bool(patch_only)
        self.cls_mix_scale = float(cls_mix_scale)
        self.mean_mix_scale = float(mean_mix_scale)
        self.norm = nn.LayerNorm(self.dim)
        self.summary_reduce = nn.Linear(self.dim, self.bottleneck_dim, bias=False)
        self.channel_in = nn.Linear(self.dim, self.rank, bias=False)
        self.channel_out = nn.Linear(self.rank, self.dim, bias=False)
        self.rank_head = nn.Linear(self.bottleneck_dim, self.rank)
        self.bias_head = nn.Linear(self.bottleneck_dim, self.dim)
        self.token_gate = nn.Linear(self.dim, 1)
        self.summary_gate = nn.Linear(self.bottleneck_dim, 1)
        self.layer_scale = nn.Parameter(torch.full((1, 1, self.dim), float(init_scale)))
        self.act = nn.GELU()
        self.last_stats = {}
        self._gate_bias = float(gate_bias)
        self._init_heads(gate_bias=gate_bias)

    def _init_heads(self, *, gate_bias: float | None = None) -> None:
        gate_bias = self._gate_bias if gate_bias is None else float(gate_bias)
        with torch.no_grad():
            nn.init.trunc_normal_(self.summary_reduce.weight, std=0.02)
            nn.init.trunc_normal_(self.channel_in.weight, std=0.02)
            nn.init.trunc_normal_(self.channel_out.weight, std=0.02)
            nn.init.trunc_normal_(self.rank_head.weight, std=0.02)
            nn.init.trunc_normal_(self.bias_head.weight, std=0.02)
            self.channel_out.weight.mul_(0.1)
            self.rank_head.weight.mul_(0.1)
            self.bias_head.weight.mul_(0.1)
            self.rank_head.bias.zero_()
            self.bias_head.bias.zero_()
            self.token_gate.weight.zero_()
            self.token_gate.bias.zero_()
            self.summary_gate.weight.zero_()
            self.summary_gate.bias.fill_(gate_bias)

    def forward(self, x: torch.Tensor, *, residual_scale: float = 1.0) -> torch.Tensor:
        if x.dim() != 3:
            self.last_stats = {}
            return x
        hidden = self.norm(x)
        cls_hidden = hidden[:, 0]
        if hidden.shape[1] > 1:
            patch_mean = hidden[:, 1:].mean(dim=1)
        else:
            patch_mean = torch.zeros_like(cls_hidden)
        summary = self.cls_mix_scale * cls_hidden + self.mean_mix_scale * patch_mean
        summary_hidden = self.act(self.summary_reduce(summary))
        rank_scale = torch.tanh(self.rank_head(summary_hidden)).unsqueeze(1)
        channel_bias = 0.1 * torch.tanh(self.bias_head(summary_hidden)).unsqueeze(1)
        token_gate = torch.sigmoid(self.token_gate(hidden))
        if self.patch_only and hidden.shape[1] > 1:
            token_gate = token_gate.clone()
            token_gate[:, :1] = 0.0
        summary_gate = torch.sigmoid(self.summary_gate(summary_hidden)).view(hidden.shape[0], 1, 1)
        hidden_rank = self.channel_in(hidden)
        mixed = self.channel_out(hidden_rank * rank_scale)
        update = (mixed + channel_bias) * token_gate * self.layer_scale
        self.last_stats = {
            "hidden_channel_flow_gate": float(summary_gate.detach().mean().item()),
            "hidden_channel_flow_token_gate": float(token_gate.detach().mean().item()),
            "hidden_channel_flow_update_norm": float(update.detach().norm(dim=-1).mean().item()),
            "hidden_channel_flow_layer_scale": float(self.layer_scale.detach().abs().mean().item()),
            "hidden_channel_flow_bottleneck": float(self.bottleneck_dim),
            "hidden_channel_flow_rank": float(self.rank),
            "hidden_channel_flow_patch_only": float(self.patch_only),
            "hidden_channel_flow_cls_mix_scale": float(self.cls_mix_scale),
            "hidden_channel_flow_mean_mix_scale": float(self.mean_mix_scale),
        }
        return x + residual_scale * summary_gate * update


class HiddenClsAttentionBridge(nn.Module):
    def __init__(
        self,
        dim: int,
        bottleneck_dim: int = 16,
        *,
        gate_bias: float = -4.0,
        init_scale: float = 0.01,
        patch_feedback_scale: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.bottleneck_dim = max(int(bottleneck_dim), 4)
        self.patch_feedback_scale = float(patch_feedback_scale)
        self.scale = self.bottleneck_dim ** -0.5
        self.norm = nn.LayerNorm(self.dim)
        self.q_proj = nn.Linear(self.dim, self.bottleneck_dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.bottleneck_dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.bottleneck_dim, bias=False)
        self.cls_proj = nn.Linear(self.dim, self.bottleneck_dim, bias=False)
        self.out_proj = nn.Linear(self.bottleneck_dim, self.dim, bias=False)
        self.gate = nn.Linear(self.dim, 1)
        if self.patch_feedback_scale != 0.0:
            self.patch_proj = nn.Linear(self.bottleneck_dim, self.dim, bias=False)
            self.patch_gate = nn.Linear(self.dim, 1)
        else:
            self.patch_proj = None
            self.patch_gate = None
        self.layer_scale = nn.Parameter(torch.full((1, 1, self.dim), float(init_scale)))
        self.act = nn.GELU()
        self.last_stats = {}
        self._init_heads(gate_bias=gate_bias)

    def _init_heads(self, *, gate_bias: float = -4.0) -> None:
        with torch.no_grad():
            for layer in (self.q_proj, self.k_proj, self.v_proj, self.cls_proj, self.out_proj):
                nn.init.trunc_normal_(layer.weight, std=0.02)
            self.out_proj.weight.mul_(0.1)
            if self.patch_proj is not None:
                nn.init.trunc_normal_(self.patch_proj.weight, std=0.02)
                self.patch_proj.weight.mul_(0.1)
            self.gate.weight.zero_()
            self.gate.bias.fill_(gate_bias)
            if self.patch_gate is not None:
                self.patch_gate.weight.zero_()
                self.patch_gate.bias.zero_()

    def forward(self, x: torch.Tensor, *, residual_scale: float = 1.0) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] <= 1:
            self.last_stats = {}
            return x
        hidden = self.norm(x)
        cls_hidden = hidden[:, :1]
        patch_hidden = hidden[:, 1:]
        query = self.q_proj(cls_hidden)
        key = self.k_proj(patch_hidden)
        value = self.v_proj(patch_hidden)
        attn = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        context = torch.matmul(attn, value)
        summary = self.act(context + self.cls_proj(cls_hidden))
        cls_gate = torch.sigmoid(self.gate(cls_hidden.squeeze(1))).view(hidden.shape[0], 1, 1)
        cls_delta = self.out_proj(summary) * self.layer_scale
        out = x.clone()
        out[:, :1] = out[:, :1] + residual_scale * cls_gate * cls_delta
        patch_gate_mean = 0.0
        if self.patch_proj is not None and self.patch_gate is not None:
            patch_gate = torch.sigmoid(self.patch_gate(patch_hidden))
            patch_delta = self.patch_proj(summary).expand(-1, patch_hidden.shape[1], -1)
            out[:, 1:] = out[:, 1:] + residual_scale * cls_gate * self.patch_feedback_scale * patch_gate * patch_delta
            patch_gate_mean = float(patch_gate.detach().mean().item())
        entropy = -(attn.clamp_min(1e-8) * torch.log(attn.clamp_min(1e-8))).sum(dim=-1).mean()
        self.last_stats = {
            "hidden_cls_bridge_gate": float(cls_gate.detach().mean().item()),
            "hidden_cls_bridge_update_norm": float(cls_delta.detach().norm(dim=-1).mean().item()),
            "hidden_cls_bridge_attn_entropy": float(entropy.detach().item()),
            "hidden_cls_bridge_bottleneck": float(self.bottleneck_dim),
            "hidden_cls_bridge_layer_scale": float(self.layer_scale.detach().abs().mean().item()),
            "hidden_cls_bridge_patch_feedback_scale": float(self.patch_feedback_scale),
            "hidden_cls_bridge_patch_gate": patch_gate_mean,
        }
        return out


class AttentionFlowModulator(nn.Module):
    def __init__(
        self,
        dim: int,
        bottleneck_dim: int = 24,
        *,
        gate_bias: float = -2.5,
        init_scale: float = 0.02,
        detail_topk: int = 8,
        patch_only: bool = False,
    ):
        super().__init__()
        self.dim = int(dim)
        self.bottleneck_dim = max(int(bottleneck_dim), 8)
        self.detail_topk = max(int(detail_topk), 0)
        self.patch_only = bool(patch_only)
        self.attn_norm = nn.LayerNorm(self.dim)
        self.summary_proj = nn.Linear(self.dim * 3, self.bottleneck_dim, bias=False)
        self.token_gate = nn.Linear(self.dim, 1)
        self.summary_gate = nn.Linear(self.bottleneck_dim, 1)
        self.scale_head = nn.Linear(self.bottleneck_dim, self.dim)
        self.bias_head = nn.Linear(self.bottleneck_dim, self.dim)
        self.layer_scale = nn.Parameter(torch.full((1, 1, self.dim), float(init_scale)))
        self.act = nn.GELU()
        self.last_stats = {}
        self._gate_bias = float(gate_bias)
        self._init_heads(gate_bias=gate_bias)

    def _init_heads(self, *, gate_bias: float | None = None) -> None:
        gate_bias = self._gate_bias if gate_bias is None else float(gate_bias)
        with torch.no_grad():
            nn.init.trunc_normal_(self.summary_proj.weight, std=0.02)
            nn.init.trunc_normal_(self.scale_head.weight, std=0.02)
            nn.init.trunc_normal_(self.bias_head.weight, std=0.02)
            self.scale_head.weight.mul_(0.1)
            self.bias_head.weight.mul_(0.1)
            self.scale_head.bias.zero_()
            self.bias_head.bias.zero_()
            self.token_gate.weight.zero_()
            self.token_gate.bias.zero_()
            self.summary_gate.weight.zero_()
            self.summary_gate.bias.fill_(gate_bias)

    def _detail_summary(self, patch_hidden: torch.Tensor) -> torch.Tensor:
        if patch_hidden.shape[1] == 0:
            return patch_hidden.new_zeros(patch_hidden.shape[0], self.dim)
        if self.detail_topk <= 0 or self.detail_topk >= patch_hidden.shape[1]:
            return patch_hidden.mean(dim=1)
        scores = patch_hidden.pow(2).mean(dim=-1)
        topk = min(self.detail_topk, patch_hidden.shape[1])
        idx = scores.topk(topk, dim=1).indices
        detail = patch_hidden.gather(1, idx.unsqueeze(-1).expand(-1, -1, patch_hidden.shape[-1]))
        return detail.mean(dim=1)

    def forward(self, x: torch.Tensor, attn_out: torch.Tensor, *, residual_scale: float = 1.0) -> torch.Tensor:
        if x.dim() != 3:
            self.last_stats = {}
            return x
        attn_hidden = self.attn_norm(attn_out)
        cls_hidden = attn_hidden[:, 0]
        if attn_hidden.shape[1] > 1:
            patch_hidden = attn_hidden[:, 1:]
            patch_mean = patch_hidden.mean(dim=1)
            detail_mean = self._detail_summary(patch_hidden)
        else:
            patch_mean = torch.zeros_like(cls_hidden)
            detail_mean = torch.zeros_like(cls_hidden)
        summary = self.act(self.summary_proj(torch.cat((cls_hidden, patch_mean, detail_mean), dim=-1)))
        scale = torch.tanh(self.scale_head(summary)).unsqueeze(1)
        bias = torch.tanh(self.bias_head(summary)).unsqueeze(1)
        token_gate = torch.sigmoid(self.token_gate(attn_hidden))
        if self.patch_only and token_gate.shape[1] > 1:
            token_gate = token_gate.clone()
            token_gate[:, :1] = 0.0
        gate = torch.sigmoid(self.summary_gate(summary)).view(x.shape[0], 1, 1)
        update = self.layer_scale * token_gate * (x * scale + bias)
        self.last_stats = {
            "attn_flow_gate": float(gate.detach().mean().item()),
            "attn_flow_token_gate": float(token_gate.detach().mean().item()),
            "attn_flow_update_norm": float(update.detach().norm(dim=-1).mean().item()),
            "attn_flow_layer_scale": float(self.layer_scale.detach().abs().mean().item()),
            "attn_flow_bottleneck": float(self.bottleneck_dim),
            "attn_flow_detail_topk": float(self.detail_topk),
            "attn_flow_patch_only": float(self.patch_only),
        }
        return x + residual_scale * gate * update


class AttentionHiddenFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attn_dim: int,
        bottleneck_dim: int = 6,
        *,
        gate_bias: float = -2.5,
        init_scale: float = 0.02,
        patch_only: bool = False,
        cls_context_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.attn_dim = int(attn_dim)
        self.bottleneck_dim = max(int(bottleneck_dim), 4)
        self.patch_only = bool(patch_only)
        self.cls_context_scale = float(cls_context_scale)
        self.attn_norm = nn.LayerNorm(self.attn_dim)
        self.token_reduce = nn.Linear(self.attn_dim, self.bottleneck_dim, bias=False)
        self.cls_proj = nn.Linear(self.attn_dim, self.bottleneck_dim, bias=False)
        self.expand = nn.Linear(self.bottleneck_dim, self.hidden_dim, bias=False)
        self.token_gate = nn.Linear(self.attn_dim, 1)
        self.summary_gate = nn.Linear(self.bottleneck_dim, 1)
        self.layer_scale = nn.Parameter(torch.full((1, 1, self.hidden_dim), float(init_scale)))
        self.act = nn.GELU()
        self.last_stats: dict[str, float] = {}
        self._gate_bias = float(gate_bias)
        self._init_weights(gate_bias=gate_bias)

    def _init_weights(self, *, gate_bias: float | None = None) -> None:
        gate_bias = self._gate_bias if gate_bias is None else float(gate_bias)
        with torch.no_grad():
            nn.init.trunc_normal_(self.token_reduce.weight, std=0.02)
            nn.init.trunc_normal_(self.cls_proj.weight, std=0.02)
            nn.init.trunc_normal_(self.expand.weight, std=0.02)
            self.expand.weight.mul_(0.1)
            self.token_gate.weight.zero_()
            self.token_gate.bias.zero_()
            self.summary_gate.weight.zero_()
            self.summary_gate.bias.fill_(gate_bias)

    def forward(self, hidden: torch.Tensor, attn_out: torch.Tensor, *, residual_scale: float = 1.0) -> torch.Tensor:
        if hidden.dim() != 3 or attn_out.dim() != 3 or hidden.shape[:2] != attn_out.shape[:2]:
            self.last_stats = {}
            return hidden
        attn_hidden = self.attn_norm(attn_out)
        cls_hidden = attn_hidden[:, 0]
        token_hidden = self.token_reduce(attn_hidden)
        if self.cls_context_scale != 0.0:
            token_hidden = token_hidden + self.cls_context_scale * self.cls_proj(cls_hidden).unsqueeze(1)
        token_hidden = self.act(token_hidden)
        token_mod = torch.tanh(self.expand(token_hidden))
        token_gate = torch.sigmoid(self.token_gate(attn_hidden))
        if self.patch_only and token_gate.shape[1] > 1:
            token_gate = token_gate.clone()
            token_gate[:, :1] = 0.0
        gate = torch.sigmoid(self.summary_gate(token_hidden[:, 0])).view(hidden.shape[0], 1, 1)
        update = self.layer_scale * token_gate * (hidden * token_mod)
        self.last_stats = {
            "attn_hidden_gate": float(gate.detach().mean().item()),
            "attn_hidden_token_gate": float(token_gate.detach().mean().item()),
            "attn_hidden_update_norm": float(update.detach().norm(dim=-1).mean().item()),
            "attn_hidden_layer_scale": float(self.layer_scale.detach().abs().mean().item()),
            "attn_hidden_bottleneck": float(self.bottleneck_dim),
            "attn_hidden_patch_only": float(self.patch_only),
            "attn_hidden_cls_context_scale": float(self.cls_context_scale),
        }
        return hidden + residual_scale * gate * update


class GeoMlp(nn.Module):
    class GroupCompetitionRouter(nn.Module):
        def __init__(
            self,
            dim: int,
            num_groups: int = 4,
            *,
            gate_bias: float = -2.0,
            init_scale: float = 0.1,
            cls_mix_scale: float = 1.0,
            mean_mix_scale: float = 0.5,
        ):
            super().__init__()
            self.dim = int(dim)
            self.num_groups = max(int(num_groups), 1)
            self.cls_mix_scale = float(cls_mix_scale)
            self.mean_mix_scale = float(mean_mix_scale)
            self.valid = self.num_groups > 1 and self.dim % self.num_groups == 0
            self.group_dim = self.dim // self.num_groups if self.valid else self.dim
            self.norm = nn.LayerNorm(self.dim)
            self.summary_proj = nn.Linear(self.dim * 2, self.num_groups)
            self.summary_gate = nn.Linear(self.dim * 2, 1)
            self.layer_scale = nn.Parameter(torch.full((1, 1, self.num_groups, 1), float(init_scale)))
            self.last_stats = {}
            self._gate_bias = float(gate_bias)
            self._init_heads(gate_bias=gate_bias)

        def _init_heads(self, *, gate_bias: float | None = None) -> None:
            gate_bias = self._gate_bias if gate_bias is None else float(gate_bias)
            with torch.no_grad():
                self.summary_proj.weight.zero_()
                self.summary_proj.bias.zero_()
                self.summary_gate.weight.zero_()
                self.summary_gate.bias.fill_(gate_bias)

        def forward(self, x: torch.Tensor, *, residual_scale: float = 1.0) -> torch.Tensor:
            if x.dim() != 3 or not self.valid:
                self.last_stats = {}
                return x
            hidden = self.norm(x)
            cls_hidden = hidden[:, 0]
            if hidden.shape[1] > 1:
                patch_mean = hidden[:, 1:].mean(dim=1)
            else:
                patch_mean = torch.zeros_like(cls_hidden)
            summary = torch.cat(
                (self.cls_mix_scale * cls_hidden, self.mean_mix_scale * patch_mean),
                dim=-1,
            )
            weights = torch.softmax(self.summary_proj(summary), dim=-1).view(x.shape[0], 1, self.num_groups, 1)
            gate = torch.sigmoid(self.summary_gate(summary)).view(x.shape[0], 1, 1, 1)
            centered = weights * float(self.num_groups) - 1.0
            grouped = x.view(x.shape[0], x.shape[1], self.num_groups, self.group_dim)
            scaled = grouped * (1.0 + residual_scale * gate * self.layer_scale * centered)
            self.last_stats = {
                "group_router_gate": float(gate.detach().mean().item()),
                "group_router_weight_peak": float(weights.detach().amax(dim=2).mean().item()),
                "group_router_weight_entropy": float((-(weights.detach().squeeze(-1) * (weights.detach().squeeze(-1).clamp_min(1e-8).log())).sum(dim=-1).mean()).item()),
                "group_router_scale": float(self.layer_scale.detach().abs().mean().item()),
                "group_router_groups": float(self.num_groups),
            }
            return scaled.view_as(x)

    class TokenDiffusionMixer(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.out = nn.Linear(dim, dim, bias=False)
            self.gate = nn.Linear(dim, 1)
            self.scale = dim ** -0.5
            self.last_stats = {}
            self._init_weights()

        def _init_weights(self) -> None:
            with torch.no_grad():
                nn.init.trunc_normal_(self.out.weight, std=0.02)
                self.out.weight.mul_(0.25)
                self.gate.weight.zero_()
                self.gate.bias.fill_(-1.5)

        def forward(self, x: torch.Tensor, *, residual_scale: float = 1.0) -> torch.Tensor:
            hidden = self.norm(x)
            token_repr = F.normalize(hidden, dim=-1)
            affinity = torch.matmul(token_repr, token_repr.transpose(-2, -1)) * self.scale
            affinity = affinity.softmax(dim=-1)
            diffusion = torch.matmul(affinity, hidden) - hidden
            gate = torch.sigmoid(self.gate(hidden[:, 0])).view(hidden.shape[0], 1, 1)
            update = self.out(diffusion)
            self.last_stats = {
                "hidden_diffusion_gate": float(gate.detach().mean().item()),
                "hidden_diffusion_update_norm": float(update.detach().norm(dim=-1).mean().item()),
            }
            return x + residual_scale * gate * update

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        fc1_layer=nn.Linear,
        fc2_layer=nn.Linear,
        act_layer=nn.GELU,
        dual_path_hidden: bool = False,
        dual_path_refine_ratio: float = 0.25,
        dual_path_cross_scale: float = 1.0,
        dual_path_gate_bias: float = -2.0,
        hidden_group_router: nn.Module | None = None,
        hidden_group_router_scale: float = 1.0,
        attention_hidden_fusion: nn.Module | None = None,
        attention_hidden_fusion_scale: float = 1.0,
        hidden_token_mixer: nn.Module | None = None,
        hidden_token_mixer_scale: float = 1.0,
        hidden_diffusion: bool = False,
        hidden_diffusion_scale: float = 1.0,
        hidden_diffusion_mixer: nn.Module | None = None,
        hidden_grid_refiner: nn.Module | None = None,
        hidden_grid_refiner_scale: float = 1.0,
        hidden_cls_bridge: nn.Module | None = None,
        hidden_cls_bridge_scale: float = 1.0,
        hidden_channel_flow: nn.Module | None = None,
        hidden_channel_flow_scale: float = 1.0,
        response_flow_norm: nn.Module | None = None,
        response_flow_scale: float = 1.0,
        response_flow_pre_act: bool = False,
        activation_flow_gate: nn.Module | None = None,
        activation_flow_scale: float = 1.0,
    ):
        super().__init__()
        self.dual_path_hidden = bool(dual_path_hidden)
        self.dual_path_cross_scale = float(dual_path_cross_scale)
        self.last_dual_path_stats: dict[str, float] = {}
        if self.dual_path_hidden:
            self.fc1 = fc1_layer(in_features, hidden_features)
            min_branch = max(hidden_features // 8, 16)
            refine_hidden = int(round(hidden_features * float(dual_path_refine_ratio)))
            refine_hidden = max(min_branch, min(hidden_features, refine_hidden))
            self.refine_hidden_features = int(refine_hidden)
            self.fc1_refine = nn.Linear(in_features, self.refine_hidden_features)
            self.dual_summary_norm = nn.LayerNorm(hidden_features + self.refine_hidden_features)
            self.dual_gate = nn.Linear(hidden_features + self.refine_hidden_features, 2)
            self.main_to_refine = nn.Linear(hidden_features, self.refine_hidden_features, bias=False)
            self.refine_to_main = nn.Linear(self.refine_hidden_features, hidden_features, bias=False)
            with torch.no_grad():
                nn.init.trunc_normal_(self.fc1_refine.weight, std=0.02)
                if self.fc1_refine.bias is not None:
                    self.fc1_refine.bias.zero_()
                nn.init.trunc_normal_(self.main_to_refine.weight, std=0.02)
                nn.init.trunc_normal_(self.refine_to_main.weight, std=0.02)
                self.main_to_refine.weight.mul_(0.1)
                self.refine_to_main.weight.mul_(0.1)
                self.dual_gate.weight.zero_()
                self.dual_gate.bias.fill_(float(dual_path_gate_bias))
        else:
            self.fc1 = fc1_layer(in_features, hidden_features)
            self.fc1_refine = None
            self.dual_summary_norm = None
            self.dual_gate = None
            self.main_to_refine = None
            self.refine_to_main = None
        self.hidden_group_router = hidden_group_router
        self.hidden_group_router_scale = float(hidden_group_router_scale)
        self.attention_hidden_fusion = attention_hidden_fusion
        self.attention_hidden_fusion_scale = float(attention_hidden_fusion_scale)
        self.hidden_token_mixer = hidden_token_mixer
        self.hidden_token_mixer_scale = float(hidden_token_mixer_scale)
        if hidden_diffusion_mixer is not None:
            self.hidden_diffusion = hidden_diffusion_mixer
        else:
            self.hidden_diffusion = self.TokenDiffusionMixer(hidden_features) if hidden_diffusion else None
        self.hidden_diffusion_scale = float(hidden_diffusion_scale)
        self.hidden_grid_refiner = hidden_grid_refiner
        self.hidden_grid_refiner_scale = float(hidden_grid_refiner_scale)
        self.hidden_cls_bridge = hidden_cls_bridge
        self.hidden_cls_bridge_scale = float(hidden_cls_bridge_scale)
        self.hidden_channel_flow = hidden_channel_flow
        self.hidden_channel_flow_scale = float(hidden_channel_flow_scale)
        self.response_flow_norm = response_flow_norm
        self.response_flow_scale = float(response_flow_scale)
        self.response_flow_pre_act = bool(response_flow_pre_act)
        self.activation_flow_gate = activation_flow_gate
        self.activation_flow_scale = float(activation_flow_scale)
        self.act = act_layer()
        self.fc2 = fc2_layer(hidden_features, out_features)

    def _apply_linear(self, module, x: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        if isinstance(module, GeoOperatorLinear):
            return module(x, condition)
        return module(x)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
        *,
        attn_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.last_dual_path_stats = {}
        if self.dual_path_hidden:
            main = self._apply_linear(self.fc1, x, condition)
            if self.attention_hidden_fusion is not None and attn_out is not None:
                main = self.attention_hidden_fusion(main, attn_out, residual_scale=self.attention_hidden_fusion_scale)
            refine = self.fc1_refine(x)
            if self.response_flow_norm is not None and self.response_flow_pre_act:
                main = self.response_flow_norm(main, residual_scale=self.response_flow_scale)
            main = self.act(main)
            refine = self.act(refine)
            summary = self.dual_summary_norm(torch.cat((main.mean(dim=1), refine.mean(dim=1)), dim=-1))
            gates = torch.sigmoid(self.dual_gate(summary))
            main_gate = gates[:, :1].unsqueeze(1)
            refine_gate = gates[:, 1:].unsqueeze(1)
            main_delta = torch.tanh(self.refine_to_main(refine))
            refine_delta = torch.tanh(self.main_to_refine(main))
            refine = refine + self.dual_path_cross_scale * refine_gate * refine_delta
            x = main + self.dual_path_cross_scale * main_gate * main_delta
            self.last_dual_path_stats = {
                "dual_path_main_gate": float(main_gate.detach().mean().item()),
                "dual_path_refine_gate": float(refine_gate.detach().mean().item()),
                "dual_path_main_norm": float(main.detach().norm(dim=-1).mean().item()),
                "dual_path_refine_norm": float(refine.detach().norm(dim=-1).mean().item()),
                "dual_path_refine_hidden": float(self.refine_hidden_features),
                "dual_path_cross_scale": float(self.dual_path_cross_scale),
            }
        else:
            x = self._apply_linear(self.fc1, x, condition)
            if self.hidden_group_router is not None:
                x = self.hidden_group_router(x, residual_scale=self.hidden_group_router_scale)
            if self.attention_hidden_fusion is not None and attn_out is not None:
                x = self.attention_hidden_fusion(x, attn_out, residual_scale=self.attention_hidden_fusion_scale)
            if self.hidden_token_mixer is not None:
                x = self.hidden_token_mixer(x, residual_scale=self.hidden_token_mixer_scale)
            if self.hidden_diffusion is not None:
                x = self.hidden_diffusion(x, residual_scale=self.hidden_diffusion_scale)
            if self.hidden_channel_flow is not None:
                x = self.hidden_channel_flow(x, residual_scale=self.hidden_channel_flow_scale)
            if self.response_flow_norm is not None and self.response_flow_pre_act:
                x = self.response_flow_norm(x, residual_scale=self.response_flow_scale)
            x = self.act(x)
        if self.hidden_group_router is not None and self.dual_path_hidden:
            x = self.hidden_group_router(x, residual_scale=self.hidden_group_router_scale)
        if self.hidden_diffusion is not None and self.dual_path_hidden:
            x = self.hidden_diffusion(x, residual_scale=self.hidden_diffusion_scale)
        if self.hidden_channel_flow is not None and self.dual_path_hidden:
            x = self.hidden_channel_flow(x, residual_scale=self.hidden_channel_flow_scale)
        if self.hidden_grid_refiner is not None:
            x = self.hidden_grid_refiner(x, residual_scale=self.hidden_grid_refiner_scale)
        if self.hidden_cls_bridge is not None:
            x = self.hidden_cls_bridge(x, residual_scale=self.hidden_cls_bridge_scale)
        if self.response_flow_norm is not None and not self.response_flow_pre_act:
            x = self.response_flow_norm(x, residual_scale=self.response_flow_scale)
        if self.activation_flow_gate is not None:
            x = self.activation_flow_gate(x, residual_scale=self.activation_flow_scale)
        x = self._apply_linear(self.fc2, x, condition)
        return x

    def get_diagnostics(self) -> dict[str, float]:
        diagnostics = {}
        if self.fc1 is not None and hasattr(self.fc1, "get_diagnostics"):
            diagnostics.update({f"fc1_{k}": v for k, v in self.fc1.get_diagnostics().items()})
        if self.last_dual_path_stats:
            diagnostics.update(self.last_dual_path_stats)
        if hasattr(self.fc2, "get_diagnostics"):
            diagnostics.update({f"fc2_{k}": v for k, v in self.fc2.get_diagnostics().items()})
        if self.hidden_diffusion is not None:
            diagnostics.update(self.hidden_diffusion.last_stats)
        if self.hidden_group_router is not None:
            diagnostics.update(self.hidden_group_router.last_stats)
        if self.attention_hidden_fusion is not None:
            diagnostics.update(self.attention_hidden_fusion.last_stats)
        if self.hidden_token_mixer is not None:
            diagnostics.update({f"hidden_{k}": v for k, v in self.hidden_token_mixer.last_stats.items()})
        if self.hidden_grid_refiner is not None:
            diagnostics.update({f"hidden_{k}": v for k, v in self.hidden_grid_refiner.last_stats.items()})
        if self.hidden_cls_bridge is not None:
            diagnostics.update(self.hidden_cls_bridge.last_stats)
        if self.hidden_channel_flow is not None:
            diagnostics.update(self.hidden_channel_flow.last_stats)
        if self.response_flow_norm is not None:
            diagnostics.update(self.response_flow_norm.last_stats)
            diagnostics["response_flow_pre_act"] = float(self.response_flow_pre_act)
        if self.activation_flow_gate is not None:
            diagnostics.update(self.activation_flow_gate.last_stats)
        return diagnostics

    def get_aux_losses(self) -> dict[str, torch.Tensor]:
        aux = {}
        if self.fc1 is not None and hasattr(self.fc1, "get_aux_losses"):
            aux.update({f"fc1_{k}": v for k, v in self.fc1.get_aux_losses().items()})
        if hasattr(self.fc2, "get_aux_losses"):
            aux.update({f"fc2_{k}": v for k, v in self.fc2.get_aux_losses().items()})
        return aux


class PatchGridRefiner(nn.Module):
    def __init__(
        self,
        dim: int,
        bottleneck_dim: int = 16,
        *,
        gate_bias: float = -5.0,
        init_scale: float = 0.002,
        cls_context_scale: float = 0.05,
    ):
        super().__init__()
        self.dim = int(dim)
        self.bottleneck_dim = max(int(bottleneck_dim), 8)
        self.cls_context_scale = float(cls_context_scale)
        self.norm = nn.LayerNorm(self.dim)
        self.depthwise = nn.Conv2d(
            self.dim,
            self.dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.dim,
            bias=False,
        )
        self.reduce = nn.Conv2d(self.dim, self.bottleneck_dim, kernel_size=1, bias=False)
        self.expand = nn.Conv2d(self.bottleneck_dim, self.dim, kernel_size=1, bias=False)
        self.cls_proj = nn.Linear(self.dim, self.bottleneck_dim, bias=False)
        self.gate = nn.Linear(self.dim, 1)
        self.layer_scale = nn.Parameter(torch.full((1, self.dim, 1, 1), float(init_scale)))
        self.act = nn.GELU()
        self.last_stats = {}
        self._init_weights(gate_bias=gate_bias)

    def _init_weights(self, *, gate_bias: float) -> None:
        with torch.no_grad():
            nn.init.trunc_normal_(self.depthwise.weight, std=0.02)
            self.depthwise.weight.mul_(0.25)
            nn.init.trunc_normal_(self.reduce.weight, std=0.02)
            self.reduce.weight.mul_(0.25)
            nn.init.trunc_normal_(self.expand.weight, std=0.02)
            self.expand.weight.mul_(0.1)
            nn.init.trunc_normal_(self.cls_proj.weight, std=0.02)
            self.cls_proj.weight.mul_(0.1)
            self.gate.weight.zero_()
            self.gate.bias.fill_(gate_bias)

    def forward(self, x: torch.Tensor, *, residual_scale: float = 1.0) -> torch.Tensor:
        if x.shape[1] <= 1:
            self.last_stats = {}
            return x
        hidden = self.norm(x)
        cls_hidden = hidden[:, 0]
        patch_hidden = hidden[:, 1:]
        grid_size = int(round(math.sqrt(patch_hidden.shape[1])))
        if grid_size * grid_size != patch_hidden.shape[1]:
            self.last_stats = {}
            return x
        patch_grid = patch_hidden.transpose(1, 2).reshape(hidden.shape[0], self.dim, grid_size, grid_size)
        update = self.depthwise(patch_grid)
        update = self.reduce(update)
        if self.cls_context_scale != 0.0:
            cls_context = self.cls_proj(cls_hidden).view(hidden.shape[0], self.bottleneck_dim, 1, 1)
            update = update + self.cls_context_scale * cls_context
        update = self.act(update)
        update = self.expand(update) * self.layer_scale
        gate = torch.sigmoid(self.gate(cls_hidden)).view(hidden.shape[0], 1, 1, 1)
        patch_delta = (gate * update).reshape(hidden.shape[0], self.dim, -1).transpose(1, 2)
        self.last_stats = {
            "patch_grid_gate": float(gate.detach().mean().item()),
            "patch_grid_update_norm": float(update.detach().norm(dim=1).mean().item()),
            "patch_grid_layer_scale": float(self.layer_scale.detach().abs().mean().item()),
            "patch_grid_bottleneck": float(self.bottleneck_dim),
        }
        delta = torch.cat((torch.zeros_like(x[:, :1]), patch_delta), dim=1)
        return x + residual_scale * delta


class TailTokenMixer(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        kernel_size: int = 3,
        gate_bias: float = -3.0,
        init_scale: float = 0.02,
        patch_only: bool = True,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.patch_only = bool(patch_only)
        kernel_size = max(int(kernel_size), 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(self.dim)
        self.depthwise = nn.Conv1d(
            self.dim,
            self.dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=self.dim,
            bias=False,
        )
        self.gate = nn.Linear(self.dim, 1)
        self.layer_scale = nn.Parameter(torch.full((1, 1, self.dim), float(init_scale)))
        self.last_stats: dict[str, float] = {}
        self._init_weights(gate_bias=gate_bias)

    def _init_weights(self, *, gate_bias: float) -> None:
        with torch.no_grad():
            nn.init.trunc_normal_(self.depthwise.weight, std=0.02)
            self.depthwise.weight.mul_(0.25)
            self.gate.weight.zero_()
            self.gate.bias.fill_(gate_bias)

    def forward(self, x: torch.Tensor, *, residual_scale: float = 1.0) -> torch.Tensor:
        if x.shape[1] <= 1:
            self.last_stats = {}
            return x
        hidden = self.norm(x)
        cls_hidden = hidden[:, 0]
        if self.patch_only:
            token_hidden = hidden[:, 1:]
            mixed = self.depthwise(token_hidden.transpose(1, 2)).transpose(1, 2)
            token_delta = mixed * self.layer_scale
            gate = torch.sigmoid(self.gate(cls_hidden)).view(hidden.shape[0], 1, 1)
            delta = torch.cat((torch.zeros_like(x[:, :1]), gate * token_delta), dim=1)
        else:
            mixed = self.depthwise(hidden.transpose(1, 2)).transpose(1, 2)
            token_delta = mixed * self.layer_scale
            gate = torch.sigmoid(self.gate(cls_hidden)).view(hidden.shape[0], 1, 1)
            delta = gate * token_delta
        self.last_stats = {
            "tail_token_gate": float(gate.detach().mean().item()),
            "tail_token_mix_norm": float(token_delta.detach().norm(dim=-1).mean().item()),
            "tail_token_layer_scale": float(self.layer_scale.detach().abs().mean().item()),
            "tail_token_patch_only": float(self.patch_only),
        }
        return x + residual_scale * delta


class SparseTopKTokenMixer(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        topk: int = 8,
        gate_bias: float = -3.0,
        init_scale: float = 0.02,
        patch_only: bool = True,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.topk = max(int(topk), 1)
        self.patch_only = bool(patch_only)
        self.norm = nn.LayerNorm(self.dim)
        self.gate = nn.Linear(self.dim, 1)
        self.layer_scale = nn.Parameter(torch.full((1, 1, self.dim), float(init_scale)))
        self.last_stats: dict[str, float] = {}
        self._init_weights(gate_bias=gate_bias)

    def _init_weights(self, *, gate_bias: float) -> None:
        with torch.no_grad():
            self.gate.weight.zero_()
            self.gate.bias.fill_(gate_bias)

    def forward(self, x: torch.Tensor, *, residual_scale: float = 1.0) -> torch.Tensor:
        if x.shape[1] <= 1:
            self.last_stats = {}
            return x
        hidden = self.norm(x)
        cls_hidden = hidden[:, 0]
        if self.patch_only:
            token_hidden = hidden[:, 1:]
        else:
            token_hidden = hidden
        token_repr = F.normalize(token_hidden, dim=-1)
        affinity = torch.matmul(token_repr, token_repr.transpose(-2, -1))
        topk = min(self.topk, affinity.shape[-1])
        topk_values, topk_indices = torch.topk(affinity, k=topk, dim=-1)
        sparse_affinity = torch.full_like(affinity, float("-inf"))
        sparse_affinity.scatter_(-1, topk_indices, topk_values)
        weights = torch.softmax(sparse_affinity, dim=-1)
        mixed = torch.matmul(weights, token_hidden)
        token_delta = (mixed - token_hidden) * self.layer_scale
        gate = torch.sigmoid(self.gate(cls_hidden)).view(hidden.shape[0], 1, 1)
        if self.patch_only:
            delta = torch.cat((torch.zeros_like(x[:, :1]), gate * token_delta), dim=1)
        else:
            delta = gate * token_delta
        self.last_stats = {
            "sparse_token_gate": float(gate.detach().mean().item()),
            "sparse_token_mix_norm": float(token_delta.detach().norm(dim=-1).mean().item()),
            "sparse_token_layer_scale": float(self.layer_scale.detach().abs().mean().item()),
            "sparse_token_topk": float(topk),
            "sparse_token_patch_only": float(self.patch_only),
        }
        return x + residual_scale * delta


class GeoViTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        qkv_layer=nn.Linear,
        proj_layer=nn.Linear,
        fc1_layer=nn.Linear,
        fc2_layer=nn.Linear,
        dual_path_hidden: bool = False,
        dual_path_refine_ratio: float = 0.25,
        dual_path_cross_scale: float = 1.0,
        dual_path_gate_bias: float = -2.0,
        hidden_group_router: nn.Module | None = None,
        hidden_group_router_scale: float = 1.0,
        attention_hidden_fusion: nn.Module | None = None,
        attention_hidden_fusion_scale: float = 1.0,
        hidden_token_mixer: nn.Module | None = None,
        hidden_token_mixer_scale: float = 1.0,
        hidden_diffusion: bool = False,
        hidden_diffusion_scale: float = 1.0,
        hidden_diffusion_mixer: nn.Module | None = None,
        hidden_grid_refiner: nn.Module | None = None,
        hidden_grid_refiner_scale: float = 1.0,
        hidden_cls_bridge: nn.Module | None = None,
        hidden_cls_bridge_scale: float = 1.0,
        hidden_channel_flow: nn.Module | None = None,
        hidden_channel_flow_scale: float = 1.0,
        response_flow_norm: nn.Module | None = None,
        response_flow_scale: float = 1.0,
        response_flow_pre_act: bool = False,
        activation_flow_gate: nn.Module | None = None,
        activation_flow_scale: float = 1.0,
        competitive_residual_mixer: nn.Module | None = None,
        competitive_residual_scale: float = 1.0,
        parallel_block_update: bool = False,
        mlp_first_update: bool = False,
        tail_token_mixer: nn.Module | None = None,
        tail_token_mixer_scale: float = 1.0,
        attn_flow_modulator: nn.Module | None = None,
        attn_flow_scale: float = 1.0,
        patch_grid_refiner: bool = False,
        patch_grid_refiner_scale: float = 1.0,
        patch_grid_refiner_module: nn.Module | None = None,
        attention_metric_adapter: bool = False,
        attention_metric_scale: float = 1.0,
        attention_metric_module: nn.Module | None = None,
        layer_scale_init: float = 0.0,
        mlp_override: nn.Module | None = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if attention_metric_module is not None:
            metric_adapter = attention_metric_module
        elif attention_metric_adapter:
            metric_adapter = SharedAttentionMetricAdapter(dim)
        else:
            metric_adapter = None
        self.attn = GeoAttention(
            dim,
            num_heads=num_heads,
            qkv_layer=qkv_layer,
            proj_layer=proj_layer,
            metric_adapter=metric_adapter,
            metric_scale=attention_metric_scale,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.layer_scale_attn = nn.Parameter(torch.full((dim,), float(layer_scale_init))) if layer_scale_init > 0 else None
        self.layer_scale_mlp = nn.Parameter(torch.full((dim,), float(layer_scale_init))) if layer_scale_init > 0 else None
        self.competitive_residual_mixer = competitive_residual_mixer
        self.competitive_residual_scale = float(competitive_residual_scale)
        self.parallel_block_update = bool(parallel_block_update)
        self.mlp_first_update = bool(mlp_first_update)
        self.tail_token_mixer = tail_token_mixer
        self.tail_token_mixer_scale = float(tail_token_mixer_scale)
        self.attn_flow_modulator = attn_flow_modulator
        self.attn_flow_scale = float(attn_flow_scale)
        if patch_grid_refiner_module is not None:
            self.patch_grid_refiner = patch_grid_refiner_module
        else:
            self.patch_grid_refiner = PatchGridRefiner(dim) if patch_grid_refiner else None
        self.patch_grid_refiner_scale = float(patch_grid_refiner_scale)
        if mlp_override is not None:
            # Caller provided a fully-constructed MLP replacement (e.g. FlowFFN).
            # It must implement forward(x, condition=None, *, attn_out=None) and
            # provide get_diagnostics()/get_aux_losses() for compatibility.
            self.mlp = mlp_override
        else:
            self.mlp = GeoMlp(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                out_features=dim,
                fc1_layer=fc1_layer,
                fc2_layer=fc2_layer,
                dual_path_hidden=dual_path_hidden,
                dual_path_refine_ratio=dual_path_refine_ratio,
                dual_path_cross_scale=dual_path_cross_scale,
                dual_path_gate_bias=dual_path_gate_bias,
                hidden_group_router=hidden_group_router,
                hidden_group_router_scale=hidden_group_router_scale,
                attention_hidden_fusion=attention_hidden_fusion,
                attention_hidden_fusion_scale=attention_hidden_fusion_scale,
                hidden_token_mixer=hidden_token_mixer,
                hidden_token_mixer_scale=hidden_token_mixer_scale,
                hidden_diffusion=hidden_diffusion,
                hidden_diffusion_scale=hidden_diffusion_scale,
                hidden_diffusion_mixer=hidden_diffusion_mixer,
                hidden_grid_refiner=hidden_grid_refiner,
                hidden_grid_refiner_scale=hidden_grid_refiner_scale,
                hidden_cls_bridge=hidden_cls_bridge,
                hidden_cls_bridge_scale=hidden_cls_bridge_scale,
                hidden_channel_flow=hidden_channel_flow,
                hidden_channel_flow_scale=hidden_channel_flow_scale,
                response_flow_norm=response_flow_norm,
                response_flow_scale=response_flow_scale,
                response_flow_pre_act=response_flow_pre_act,
                activation_flow_gate=activation_flow_gate,
                activation_flow_scale=activation_flow_scale,
            )

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        if self.mlp_first_update and self.competitive_residual_mixer is None and not self.parallel_block_update:
            mlp_in = self.norm2(x)
            mlp_out = self.mlp(mlp_in, condition=condition, attn_out=None)
            if self.layer_scale_mlp is not None:
                mlp_out = mlp_out * self.layer_scale_mlp
            if self.tail_token_mixer is not None:
                mlp_out = self.tail_token_mixer(mlp_out, residual_scale=self.tail_token_mixer_scale)
            x_mid = x + self.drop_path(mlp_out)
            attn_in = self.norm1(x_mid)
            attn_out = self.attn(attn_in, condition=condition)
            if self.layer_scale_attn is not None:
                attn_out = attn_out * self.layer_scale_attn
            x = x_mid + self.drop_path(attn_out)
            if self.patch_grid_refiner is not None:
                x = self.patch_grid_refiner(x, residual_scale=self.patch_grid_refiner_scale)
            return x
        norm1_x = self.norm1(x)
        attn_out = self.attn(norm1_x, condition=condition)
        if self.layer_scale_attn is not None:
            attn_out = attn_out * self.layer_scale_attn
        x_mid = x + self.drop_path(attn_out)
        mlp_source = x if self.parallel_block_update else x_mid
        mlp_in = self.norm2(mlp_source)
        if self.attn_flow_modulator is not None:
            mlp_in = self.attn_flow_modulator(mlp_in, attn_out, residual_scale=self.attn_flow_scale)
        mlp_out = self.mlp(mlp_in, condition=condition, attn_out=attn_out)
        if self.layer_scale_mlp is not None:
            mlp_out = mlp_out * self.layer_scale_mlp
        if self.tail_token_mixer is not None:
            mlp_out = self.tail_token_mixer(mlp_out, residual_scale=self.tail_token_mixer_scale)
        if self.competitive_residual_mixer is not None:
            attn_add, mlp_add = self.competitive_residual_mixer(
                norm1_x,
                attn_out,
                mlp_out,
                residual_scale=self.competitive_residual_scale,
            )
            x = x + self.drop_path(attn_add) + self.drop_path(mlp_add)
        else:
            if self.parallel_block_update:
                x = x + self.drop_path(attn_out) + self.drop_path(mlp_out)
            else:
                x = x_mid + self.drop_path(mlp_out)
        if self.patch_grid_refiner is not None:
            x = self.patch_grid_refiner(x, residual_scale=self.patch_grid_refiner_scale)
        return x

    def get_diagnostics(self) -> dict[str, float]:
        diagnostics = self.mlp.get_diagnostics()
        if self.layer_scale_attn is not None:
            diagnostics["layer_scale_attn"] = float(self.layer_scale_attn.detach().mean().item())
        if self.layer_scale_mlp is not None:
            diagnostics["layer_scale_mlp"] = float(self.layer_scale_mlp.detach().mean().item())
        diagnostics["parallel_block_update"] = float(self.parallel_block_update)
        diagnostics["mlp_first_update"] = float(self.mlp_first_update)
        if self.tail_token_mixer is not None:
            diagnostics.update(self.tail_token_mixer.last_stats)
        if getattr(self.attn, "metric_adapter", None) is not None:
            diagnostics.update(self.attn.metric_adapter.last_stats)
        if self.competitive_residual_mixer is not None:
            diagnostics.update(self.competitive_residual_mixer.last_stats)
        if self.attn_flow_modulator is not None:
            diagnostics.update(self.attn_flow_modulator.last_stats)
        if self.patch_grid_refiner is not None:
            diagnostics.update(self.patch_grid_refiner.last_stats)
        return diagnostics

    def get_aux_losses(self) -> dict[str, torch.Tensor]:
        return self.mlp.get_aux_losses()
