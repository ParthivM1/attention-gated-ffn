import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvFeatureStem(nn.Module):
    def __init__(self, stem_channels=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.GELU(),
            nn.Conv2d(stem_channels, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.features(x)


class ConvStem(nn.Module):
    def __init__(
        self,
        embed_dim,
        patch_size,
        stem_channels=64,
        refiner: nn.Module | None = None,
        refiner_scale: float = 1.0,
    ):
        super().__init__()
        self.stem = ConvFeatureStem(stem_channels=stem_channels)
        self.refiner = refiner
        self.refiner_scale = float(refiner_scale)
        self.proj = nn.Conv2d(stem_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.last_stats = {}

    def forward(self, x):
        x = self.stem(x)
        if self.refiner is not None:
            x = self.refiner(x, residual_scale=self.refiner_scale)
            self.last_stats = dict(getattr(self.refiner, "last_stats", {}))
        else:
            self.last_stats = {}
        return self.proj(x)


class SpatialFlowStemRefiner(nn.Module):
    def __init__(
        self,
        channels: int,
        bottleneck_dim: int = 16,
        *,
        gate_bias: float = -3.0,
        init_scale: float = 0.02,
        detail_scale: float = 0.25,
        context_scale: float = 0.5,
    ):
        super().__init__()
        self.channels = int(channels)
        self.bottleneck_dim = max(int(bottleneck_dim), 8)
        self.detail_scale = float(detail_scale)
        self.context_scale = float(context_scale)
        self.norm = nn.GroupNorm(1, self.channels)
        self.depthwise = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, groups=self.channels, bias=False)
        self.pointwise = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False)
        self.context_reduce = nn.Linear(self.channels, self.bottleneck_dim, bias=False)
        self.context_expand = nn.Linear(self.bottleneck_dim, self.channels * 2, bias=False)
        self.spatial_gate = nn.Conv2d(self.channels, 1, kernel_size=3, padding=1, bias=True)
        self.layer_scale = nn.Parameter(torch.full((1, self.channels, 1, 1), float(init_scale)))
        self.act = nn.GELU()
        self.last_stats = {}
        self._init_heads(gate_bias=gate_bias)

    def _init_heads(self, *, gate_bias: float = -3.0) -> None:
        with torch.no_grad():
            nn.init.trunc_normal_(self.depthwise.weight, std=0.02)
            nn.init.trunc_normal_(self.pointwise.weight, std=0.02)
            self.pointwise.weight.mul_(0.1)
            nn.init.trunc_normal_(self.context_reduce.weight, std=0.02)
            nn.init.trunc_normal_(self.context_expand.weight, std=0.02)
            self.context_expand.weight.mul_(0.1)
            nn.init.trunc_normal_(self.spatial_gate.weight, std=0.02)
            self.spatial_gate.weight.mul_(0.1)
            self.spatial_gate.bias.fill_(gate_bias)

    def forward(self, x: torch.Tensor, *, residual_scale: float = 1.0) -> torch.Tensor:
        if x.dim() != 4:
            self.last_stats = {}
            return x
        hidden = self.norm(x)
        local = self.pointwise(self.act(self.depthwise(hidden)))
        pooled = hidden.mean(dim=(-2, -1))
        context_hidden = self.act(self.context_reduce(pooled))
        channel_gate, channel_bias = self.context_expand(context_hidden).chunk(2, dim=-1)
        channel_gate = torch.sigmoid(channel_gate).view(hidden.shape[0], hidden.shape[1], 1, 1)
        channel_bias = channel_bias.view(hidden.shape[0], hidden.shape[1], 1, 1)
        detail = hidden - F.avg_pool2d(hidden, kernel_size=3, stride=1, padding=1)
        spatial_gate = torch.sigmoid(self.spatial_gate(hidden))
        update = local + self.detail_scale * detail + self.context_scale * channel_bias
        gate = channel_gate * spatial_gate
        self.last_stats = {
            "stem_flow_gate": float(gate.detach().mean().item()),
            "stem_flow_local_norm": float(local.detach().norm(dim=1).mean().item()),
            "stem_flow_detail_norm": float(detail.detach().norm(dim=1).mean().item()),
            "stem_flow_update_norm": float(update.detach().norm(dim=1).mean().item()),
            "stem_flow_layer_scale": float(self.layer_scale.detach().abs().mean().item()),
            "stem_flow_bottleneck": float(self.bottleneck_dim),
            "stem_flow_detail_scale": float(self.detail_scale),
            "stem_flow_context_scale": float(self.context_scale),
        }
        return x + residual_scale * gate * self.layer_scale * update


class SharedTokenFlowMixer(nn.Module):
    def __init__(
        self,
        dim: int,
        bottleneck_dim: int = 32,
        *,
        patch_only: bool = True,
        gate_bias: float = -3.0,
        init_scale: float = 0.01,
        cls_context_scale: float = 0.25,
        detail_topk: int = 0,
        detail_boost_scale: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.bottleneck_dim = max(int(bottleneck_dim), 8)
        self.patch_only = bool(patch_only)
        self.gate_bias = float(gate_bias)
        self.cls_context_scale = float(cls_context_scale)
        self.detail_topk = max(int(detail_topk), 0)
        self.detail_boost_scale = float(detail_boost_scale)
        self.norm = nn.LayerNorm(self.dim)
        self.in_proj = nn.Linear(self.dim, self.bottleneck_dim, bias=False)
        self.out_proj = nn.Linear(self.bottleneck_dim, self.dim, bias=False)
        self.gate = nn.Linear(self.dim, 1)
        self.layer_scale = nn.Parameter(torch.full((1, 1, self.dim), float(init_scale)))
        self.scale = self.bottleneck_dim ** -0.5
        self.last_stats = {}
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            nn.init.trunc_normal_(self.in_proj.weight, std=0.02)
            nn.init.trunc_normal_(self.out_proj.weight, std=0.02)
            self.out_proj.weight.mul_(0.1)
            self.gate.weight.zero_()
            self.gate.bias.fill_(self.gate_bias)

    def forward(self, x: torch.Tensor, *, residual_scale: float = 1.0) -> torch.Tensor:
        hidden = self.norm(x)
        if self.patch_only and hidden.shape[1] > 1:
            cls_hidden = hidden[:, :1]
            token_hidden = hidden[:, 1:]
        else:
            cls_hidden = hidden[:, :1]
            token_hidden = hidden
        bottleneck = self.in_proj(token_hidden)
        token_repr = F.normalize(bottleneck, dim=-1)
        affinity = torch.matmul(token_repr, token_repr.transpose(-2, -1)) * self.scale
        affinity = affinity.softmax(dim=-1)
        diffusion = torch.matmul(affinity, bottleneck) - bottleneck
        detail_scores = None
        detail_weights = None
        detail_residual = None
        if cls_hidden.shape[1] > 0 and self.cls_context_scale != 0.0:
            cls_context = self.in_proj(cls_hidden).expand(-1, token_hidden.shape[1], -1)
            diffusion = diffusion + self.cls_context_scale * cls_context
        else:
            cls_context = bottleneck.mean(dim=1, keepdim=True).expand(-1, token_hidden.shape[1], -1)
        if self.detail_topk > 0 or self.detail_boost_scale != 0.0:
            detail_residual = bottleneck - cls_context
            detail_scores = detail_residual.norm(dim=-1)
            if self.detail_topk > 0:
                topk = min(self.detail_topk, detail_scores.shape[1])
                topk_idx = torch.topk(detail_scores, k=topk, dim=1).indices
                detail_mask = torch.zeros_like(detail_scores)
                detail_mask.scatter_(1, topk_idx, 1.0)
                detail_weights = detail_mask.unsqueeze(-1)
                diffusion = diffusion * detail_weights
            if self.detail_boost_scale != 0.0:
                if detail_weights is None:
                    detail_weights = (detail_scores / max(detail_scores.shape[1], 1)).unsqueeze(-1)
                diffusion = diffusion + self.detail_boost_scale * detail_weights * detail_residual
        gate = torch.sigmoid(self.gate(hidden[:, 0])).view(hidden.shape[0], 1, 1)
        update = self.out_proj(diffusion) * self.layer_scale
        self.last_stats = {
            "token_flow_gate": float(gate.detach().mean().item()),
            "token_flow_update_norm": float(update.detach().norm(dim=-1).mean().item()),
            "token_flow_bottleneck": float(self.bottleneck_dim),
            "token_flow_layer_scale": float(self.layer_scale.detach().abs().mean().item()),
            "token_flow_patch_only": float(self.patch_only),
            "token_flow_cls_context_scale": float(self.cls_context_scale),
            "token_flow_detail_topk": float(self.detail_topk),
            "token_flow_detail_boost_scale": float(self.detail_boost_scale),
            "token_flow_detail_score_mean": float(detail_scores.detach().mean().item()) if detail_scores is not None else 0.0,
            "token_flow_detail_score_topk_mean": float(
                torch.topk(detail_scores.detach(), k=min(self.detail_topk, detail_scores.shape[1]), dim=1).values.mean().item()
            )
            if detail_scores is not None and self.detail_topk > 0
            else 0.0,
        }
        if self.patch_only and hidden.shape[1] > 1:
            delta = torch.cat((torch.zeros_like(x[:, :1]), update), dim=1)
            return x + residual_scale * gate * delta
        return x + residual_scale * gate * update


class SharedInterLayerFlowMixer(nn.Module):
    def __init__(
        self,
        dim: int,
        bottleneck_dim: int = 16,
        *,
        mode: str = "transport",
        patch_only: bool = True,
        gate_bias: float = -4.0,
        init_scale: float = 0.005,
        cls_context_scale: float = 0.15,
        delta_scale: float = 0.5,
    ):
        super().__init__()
        self.dim = int(dim)
        self.bottleneck_dim = max(int(bottleneck_dim), 8)
        self.mode = str(mode).strip().lower() or "transport"
        self.patch_only = bool(patch_only)
        self.cls_context_scale = float(cls_context_scale)
        self.delta_scale = float(delta_scale)
        self.norm = nn.LayerNorm(self.dim)
        self.in_proj = nn.Linear(self.dim, self.bottleneck_dim, bias=False)
        self.out_proj = nn.Linear(self.bottleneck_dim, self.dim, bias=False)
        self.cls_proj = nn.Linear(self.dim, self.bottleneck_dim, bias=False)
        self.summary_proj = nn.Linear(self.dim * 3, self.bottleneck_dim, bias=False)
        self.gate = nn.Linear(self.dim * 2, 1)
        self.layer_scale = nn.Parameter(torch.full((1, 1, self.dim), float(init_scale)))
        self.scale = self.bottleneck_dim ** -0.5
        self.act = nn.GELU()
        self.last_stats = {}
        self._init_weights(gate_bias=gate_bias)

    def _init_weights(self, *, gate_bias: float) -> None:
        with torch.no_grad():
            nn.init.trunc_normal_(self.in_proj.weight, std=0.02)
            nn.init.trunc_normal_(self.out_proj.weight, std=0.02)
            nn.init.trunc_normal_(self.cls_proj.weight, std=0.02)
            nn.init.trunc_normal_(self.summary_proj.weight, std=0.02)
            self.out_proj.weight.mul_(0.1)
            self.cls_proj.weight.mul_(0.1)
            self.summary_proj.weight.mul_(0.1)
            self.gate.weight.zero_()
            self.gate.bias.fill_(gate_bias)

    def forward(self, prev_x: torch.Tensor | None, x: torch.Tensor, *, residual_scale: float = 1.0) -> torch.Tensor:
        if prev_x is None or prev_x.shape != x.shape or x.dim() != 3:
            self.last_stats = {}
            return x
        current_hidden = self.norm(x)
        prev_hidden = self.norm(prev_x)
        current_cls = current_hidden[:, 0]
        prev_cls = prev_hidden[:, 0]
        if self.patch_only and x.shape[1] > 1:
            current_tokens = current_hidden[:, 1:]
            prev_tokens = prev_hidden[:, 1:]
        else:
            current_tokens = current_hidden
            prev_tokens = prev_hidden
        current_bottleneck = self.in_proj(current_tokens)
        prev_bottleneck = self.in_proj(prev_tokens)
        delta_bottleneck = current_bottleneck - prev_bottleneck
        if self.mode == "summary":
            current_patch = current_tokens.mean(dim=1)
            prev_patch = prev_tokens.mean(dim=1)
            summary_input = torch.cat(
                (
                    current_cls - prev_cls,
                    current_patch - prev_patch,
                    current_cls,
                ),
                dim=-1,
            )
            summary = torch.tanh(self.summary_proj(summary_input)).unsqueeze(1)
            update = current_bottleneck * summary + self.delta_scale * 0.5 * delta_bottleneck
        else:
            current_repr = F.normalize(current_bottleneck, dim=-1)
            prev_repr = F.normalize(prev_bottleneck, dim=-1)
            affinity = torch.matmul(current_repr, prev_repr.transpose(-2, -1)) * self.scale
            affinity = affinity.softmax(dim=-1)
            transport = torch.matmul(affinity, delta_bottleneck)
            update = transport + self.delta_scale * delta_bottleneck
        if self.cls_context_scale != 0.0:
            cls_context = self.cls_proj(current_cls - prev_cls).unsqueeze(1)
            update = update + self.cls_context_scale * cls_context
        update = self.act(update)
        update = self.out_proj(update) * self.layer_scale
        gate_input = torch.cat((current_cls, prev_cls), dim=-1)
        gate = torch.sigmoid(self.gate(gate_input)).view(x.shape[0], 1, 1)
        self.last_stats = {
            "inter_layer_flow_gate": float(gate.detach().mean().item()),
            "inter_layer_flow_update_norm": float(update.detach().norm(dim=-1).mean().item()),
            "inter_layer_flow_delta_norm": float(delta_bottleneck.detach().norm(dim=-1).mean().item()),
            "inter_layer_flow_bottleneck": float(self.bottleneck_dim),
            "inter_layer_flow_layer_scale": float(self.layer_scale.detach().abs().mean().item()),
            "inter_layer_flow_patch_only": float(self.patch_only),
            "inter_layer_flow_cls_context_scale": float(self.cls_context_scale),
            "inter_layer_flow_delta_scale": float(self.delta_scale),
            "inter_layer_flow_cls_delta_norm": float((current_cls - prev_cls).detach().norm(dim=-1).mean().item()),
            "inter_layer_flow_summary_mode": float(self.mode == "summary"),
        }
        if self.patch_only and x.shape[1] > 1:
            delta = torch.cat((torch.zeros_like(x[:, :1]), update), dim=1)
        else:
            delta = update
        return x + residual_scale * gate * delta


class FlowStateCarrier(nn.Module):
    def __init__(
        self,
        dim: int,
        state_dim: int = 24,
        *,
        gate_bias: float = -5.0,
        init_scale: float = 0.0025,
        cls_scale: float = 1.0,
        patch_scale: float = 0.1,
    ):
        super().__init__()
        self.dim = int(dim)
        self.state_dim = max(int(state_dim), 8)
        self.cls_scale = float(cls_scale)
        self.patch_scale = float(patch_scale)
        self.token_norm = nn.LayerNorm(self.dim)
        self.state_norm = nn.LayerNorm(self.state_dim)
        self.summary_proj = nn.Linear(self.dim * 3, self.state_dim, bias=False)
        self.state_proj = nn.Linear(self.state_dim, self.state_dim, bias=False)
        self.update_proj = nn.Linear(self.state_dim, self.state_dim, bias=False)
        self.out_proj = nn.Linear(self.state_dim, self.dim, bias=False)
        self.gate = nn.Linear(self.dim + self.state_dim, 1)
        self.layer_scale = nn.Parameter(torch.full((1, 1, self.dim), float(init_scale)))
        self.act = nn.GELU()
        self.last_stats = {}
        self._init_weights(gate_bias=gate_bias)

    def _init_weights(self, *, gate_bias: float) -> None:
        with torch.no_grad():
            nn.init.trunc_normal_(self.summary_proj.weight, std=0.02)
            nn.init.trunc_normal_(self.state_proj.weight, std=0.02)
            nn.init.trunc_normal_(self.update_proj.weight, std=0.02)
            nn.init.trunc_normal_(self.out_proj.weight, std=0.02)
            self.state_proj.weight.mul_(0.1)
            self.update_proj.weight.mul_(0.1)
            self.out_proj.weight.mul_(0.1)
            self.gate.weight.zero_()
            self.gate.bias.fill_(gate_bias)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
        *,
        residual_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            self.last_stats = {}
            if state is None:
                state = torch.zeros(x.shape[0], self.state_dim, device=x.device, dtype=x.dtype)
            return x, state
        hidden = self.token_norm(x)
        cls = hidden[:, 0]
        if hidden.shape[1] > 1:
            patch_mean = hidden[:, 1:].mean(dim=1)
        else:
            patch_mean = cls
        summary_input = torch.cat((cls, patch_mean, cls - patch_mean), dim=-1)
        summary = torch.tanh(self.summary_proj(summary_input))
        if state is None:
            prev_state = torch.zeros(summary.shape[0], self.state_dim, device=x.device, dtype=x.dtype)
        else:
            prev_state = self.state_norm(state)
        candidate = torch.tanh(self.state_proj(prev_state) + summary)
        delta_state = self.act(self.update_proj(candidate))
        gate = torch.sigmoid(self.gate(torch.cat((cls, prev_state), dim=-1))).view(x.shape[0], 1, 1)
        next_state = prev_state + gate.squeeze(-1) * delta_state
        token_bias = self.out_proj(next_state).unsqueeze(1) * self.layer_scale
        if x.shape[1] > 1:
            cls_delta = self.cls_scale * token_bias
            patch_delta = self.patch_scale * token_bias.expand(-1, x.shape[1] - 1, -1)
            delta = torch.cat((cls_delta, patch_delta), dim=1)
        else:
            delta = self.cls_scale * token_bias
        self.last_stats = {
            "flow_state_gate": float(gate.detach().mean().item()),
            "flow_state_update_norm": float(delta_state.detach().norm(dim=-1).mean().item()),
            "flow_state_token_bias_norm": float(token_bias.detach().norm(dim=-1).mean().item()),
            "flow_state_state_norm": float(next_state.detach().norm(dim=-1).mean().item()),
            "flow_state_dim": float(self.state_dim),
            "flow_state_layer_scale": float(self.layer_scale.detach().abs().mean().item()),
            "flow_state_cls_scale": float(self.cls_scale),
            "flow_state_patch_scale": float(self.patch_scale),
        }
        return x + residual_scale * gate * delta, next_state


class BudgetedDetailTokenizer(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        embed_dim,
        stem_channels=64,
        detail_tokens=8,
        score_type="variance",
    ):
        super().__init__()
        if patch_size % 2 != 0:
            raise ValueError("BudgetedDetailTokenizer expects an even patch size")
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.detail_tokens = detail_tokens
        self.score_type = score_type
        self.coarse_grid = img_size // patch_size
        self.detail_stride = patch_size // 2

        self.stem = ConvFeatureStem(stem_channels=stem_channels)
        self.coarse_proj = nn.Conv2d(stem_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.detail_proj = nn.Conv2d(stem_channels, embed_dim, kernel_size=self.detail_stride, stride=self.detail_stride)
        self.detail_merge = nn.Linear(embed_dim * 4, embed_dim)
        if score_type == "learned":
            self.score_head = nn.Conv2d(stem_channels, 1, kernel_size=1)
        elif score_type != "variance":
            raise ValueError(f"Unsupported detail score_type: {score_type}")
        self.last_stats = {}

    def forward(self, x):
        batch_size = x.shape[0]
        features = self.stem(x)
        coarse_map = self.coarse_proj(features)
        coarse_tokens = coarse_map.flatten(2).transpose(1, 2)

        if self.detail_tokens <= 0:
            self.last_stats = {}
            return coarse_tokens, None, None

        if self.score_type == "variance":
            patches = features.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
            patches = patches.contiguous().view(batch_size, features.shape[1], self.coarse_grid, self.coarse_grid, -1)
            score_grid = patches.var(dim=-1, unbiased=False).mean(dim=1)
        else:
            score_map = self.score_head(features)
            pooled = nn.functional.avg_pool2d(score_map, kernel_size=self.patch_size, stride=self.patch_size)
            score_grid = pooled.squeeze(1)

        score_flat = score_grid.view(batch_size, -1)
        k = min(self.detail_tokens, score_flat.shape[1])
        topk_scores, topk_idx = torch.topk(score_flat, k=k, dim=1)

        detail_map = self.detail_proj(features)
        detail_patches = detail_map.unfold(2, 2, 2).unfold(3, 2, 2)
        detail_patches = detail_patches.permute(0, 2, 3, 4, 5, 1).contiguous().view(batch_size, -1, self.embed_dim * 4)
        detail_all = self.detail_merge(detail_patches)
        detail_tokens = detail_all.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        self.last_stats = {
            "detail_score_mean": float(score_flat.detach().to(torch.float32).mean().item()),
            "detail_score_topk_mean": float(topk_scores.detach().to(torch.float32).mean().item()),
            "detail_token_norm": float(detail_tokens.detach().to(torch.float32).norm(dim=-1).mean().item()),
            "detail_budget_used": float(k),
        }
        return coarse_tokens, detail_tokens, topk_idx

    def get_diagnostics(self):
        return dict(self.last_stats)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, linear_layer=nn.Linear):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.last_attn = None

        # Keep the dynamic geometry focused on the MLP expansion map only.
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            channels // self.num_heads,
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.last_attn = attn

        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, channels)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, linear_layer=nn.Linear):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = linear_layer(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, drop_path=0.0, linear_layer=nn.Linear):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, linear_layer=linear_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * 4), linear_layer=linear_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        embed_dim=192,
        depth=6,
        num_heads=6,
        num_classes=100,
        linear_layer=nn.Linear,
        block_linear_layers=None,
        drop_path_rate=0.1,
        qkv_bias=False,
        use_conv_stem=False,
        stem_channels=64,
        tokenizer_type="standard",
        detail_tokens=0,
        detail_score_type="variance",
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.tokenizer_type = tokenizer_type
        self.detail_tokens = detail_tokens

        if tokenizer_type == "budgeted_detail":
            self.tokenizer = BudgetedDetailTokenizer(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                stem_channels=stem_channels,
                detail_tokens=detail_tokens,
                score_type=detail_score_type,
            )
            self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.coarse_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
            self.detail_pos_embed = nn.Parameter(torch.zeros(1, detail_tokens, embed_dim))
            self.detail_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            if use_conv_stem:
                self.patch_embed = ConvStem(embed_dim=embed_dim, patch_size=patch_size, stem_channels=stem_channels)
            else:
                self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        if block_linear_layers is None:
            block_linear_layers = [linear_layer for _ in range(depth)]
        if len(block_linear_layers) != depth:
            raise ValueError(f"Expected {depth} block linear layers, got {len(block_linear_layers)}")
        self.blocks = nn.ModuleList(
            [
                Block(dim=embed_dim, num_heads=num_heads, drop_path=dpr[i], linear_layer=block_linear_layers[i])
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if tokenizer_type == "budgeted_detail":
            nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)
            nn.init.trunc_normal_(self.coarse_pos_embed, std=0.02)
            nn.init.trunc_normal_(self.detail_pos_embed, std=0.02)
            nn.init.trunc_normal_(self.detail_type_embed, std=0.02)
        else:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def get_attention_maps(self, block_indices=None):
        maps = []
        for idx, block in enumerate(self.blocks):
            if block_indices is not None and idx not in block_indices:
                continue
            attn = getattr(block.attn, "last_attn", None)
            if attn is not None:
                maps.append((idx, attn))
        return maps

    def forward_features(self, x, *, return_block_features=False):
        batch_size = x.shape[0]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        if self.tokenizer_type == "budgeted_detail":
            cls_tokens = cls_tokens + self.cls_pos_embed
            coarse_tokens, detail_tokens, detail_idx = self.tokenizer(x)
            coarse_pos = self.coarse_pos_embed.expand(batch_size, -1, -1)
            coarse_tokens = coarse_tokens + coarse_pos
            if detail_tokens is not None and detail_idx is not None:
                parent_pos = coarse_pos.gather(
                    1,
                    detail_idx.unsqueeze(-1).expand(-1, -1, coarse_pos.shape[-1]),
                )
                detail_rank_pos = self.detail_pos_embed[:, : detail_tokens.shape[1], :].expand(batch_size, -1, -1)
                detail_tokens = detail_tokens + parent_pos + detail_rank_pos + self.detail_type_embed
                x = torch.cat((cls_tokens, coarse_tokens, detail_tokens), dim=1)
            else:
                x = torch.cat((cls_tokens, coarse_tokens), dim=1)
        else:
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
        x = self.pos_drop(x)

        block_features = []
        for block in self.blocks:
            x = block(x)
            if return_block_features:
                block_features.append(x[:, 0])

        x = self.norm(x)
        if return_block_features:
            return x, block_features
        return x

    def forward(self, x, condition=None, return_features=False):
        x = self.forward_features(x)
        features = x[:, 0]
        if return_features:
            return features
        return self.head(features)

    def forward_feature_pyramid(self, x, condition=None):
        tokens, block_features = self.forward_features(x, return_block_features=True)
        return tokens[:, 0], block_features

    def get_diagnostics(self):
        return {}

    def get_aux_losses(self):
        return {}
