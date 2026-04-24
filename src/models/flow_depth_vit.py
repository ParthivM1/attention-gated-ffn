"""Flow-Depth ViT (FD-ViT): Weight-tied iterative transformer as continuous-depth flow.

Instead of N independent transformer blocks, FD-ViT uses ONE shared block
iterated T times. This is a discretized Neural ODE:  dh/dt = f(h, t),
where f is a single transformer block (attention + FFN) conditioned on time.

Key design elements:
- Weight-tied: ONE block, iterated T times -> effective depth T at depth-1 params
- Per-step learnable gates: control contribution of each iteration
- Per-step time embeddings: give the shared block iteration awareness (time-dependent ODE)
- Input injection: fraction of original embedding re-injected each step (stabilizes deep iteration)
- Optional momentum: h_t = h_{t-1} + gate * (beta * v_{t-1} + block(h_{t-1}))
  giving Hamiltonian-like dynamics
- Optional attention pooling head: learned multi-head pooling over all tokens

The model also supports a "hybrid" mode where the first K blocks are independent
(not weight-tied) and the remaining iterations use a shared block. This gives
the early layers freedom to learn diverse features while later iterations refine.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowDepthBlock(nn.Module):
    """Single transformer block used for weight-tied iteration."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        # Attention
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_proj = nn.Linear(dim, dim)
        self.attn_scale = head_dim ** -0.5

        # FFN
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

        # Drop path (stochastic depth)
        self.drop_path_rate = drop_path
        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            nn.init.trunc_normal_(self.qkv.weight, std=0.02)
            if self.qkv.bias is not None:
                self.qkv.bias.zero_()
            nn.init.trunc_normal_(self.attn_proj.weight, std=0.02)
            self.attn_proj.bias.zero_()
            nn.init.trunc_normal_(self.fc1.weight, std=0.02)
            self.fc1.bias.zero_()
            nn.init.trunc_normal_(self.fc2.weight, std=0.02)
            self.fc2.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Attention
        normed = self.norm1(x)
        qkv = self.qkv(normed).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.attn_scale
        attn = attn.softmax(dim=-1)
        attn_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        attn_out = self.attn_proj(attn_out)
        x = x + self.drop_path(attn_out)

        # FFN
        normed = self.norm2(x)
        mlp_out = self.fc2(self.act(self.fc1(normed)))
        x = x + self.drop_path(mlp_out)

        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor_(random_tensor + keep_prob)
        return x.div(keep_prob) * random_tensor


class FlowDepthViT(nn.Module):
    """Flow-Depth Vision Transformer.

    Architecture:
        patch_embed → cls_token + pos_embed → T iterations of shared_block → norm → head

    The shared block is applied T times with per-step gates and input injection.
    This gives effective depth T at the parameter cost of depth 1.

    Args:
        img_size: Input image size (assumes square).
        patch_size: Patch size for tokenization.
        embed_dim: Embedding dimension.
        num_iterations: Number of times to iterate the shared block (effective depth).
        num_heads: Number of attention heads.
        num_classes: Number of output classes.
        mlp_ratio: FFN hidden dim ratio.
        drop_path_rate: Stochastic depth rate.
        input_inject_strength: How much of the original embedding to re-inject each step.
        use_momentum: Whether to use Hamiltonian-style momentum dynamics.
        momentum_beta: Momentum coefficient (only used if use_momentum=True).
        num_independent_blocks: Number of initial independent (non-shared) blocks before
            the weight-tied iterations begin. 0 = fully weight-tied.
        use_time_conditioning: Add per-step time embeddings for time-dependent ODE.
        use_attn_pool: Use multi-head attention pooling instead of CLS token.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 192,
        num_iterations: int = 12,
        num_heads: int = 6,
        num_classes: int = 100,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        input_inject_strength: float = 0.1,
        use_momentum: bool = False,
        momentum_beta: float = 0.9,
        num_independent_blocks: int = 0,
        use_time_conditioning: bool = False,
        use_attn_pool: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_iterations = num_iterations
        self.num_independent_blocks = num_independent_blocks
        self.input_inject_strength = input_inject_strength
        self.use_momentum = use_momentum
        self.momentum_beta = momentum_beta
        self.use_time_conditioning = use_time_conditioning
        self.use_attn_pool = use_attn_pool
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding (linear projection, no convolution)
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Positional embedding + CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=0.0)

        # Independent blocks (if any) — these run first with their own weights
        if num_independent_blocks > 0:
            self.independent_blocks = nn.ModuleList([
                FlowDepthBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path_rate * i / max(num_iterations - 1, 1),
                )
                for i in range(num_independent_blocks)
            ])
        else:
            self.independent_blocks = nn.ModuleList()

        # The SHARED block — this is the flow operator, applied repeatedly
        shared_dp = drop_path_rate * 0.5  # moderate drop path for shared block
        self.shared_block = FlowDepthBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=shared_dp,
        )

        # Per-iteration learnable step gates
        num_shared_iters = num_iterations - num_independent_blocks
        self.step_gates = nn.Parameter(
            torch.ones(num_shared_iters) * 0.5
        )

        # Per-iteration learnable injection strength (initialized to input_inject_strength)
        self.inject_gates = nn.Parameter(
            torch.ones(num_shared_iters) * input_inject_strength
        )

        # Per-iteration LayerNorm (lightweight, helps iteration stability)
        self.iter_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_shared_iters)
        ])

        # Per-step time embeddings: make the ODE time-dependent f(h, t)
        # Each iteration adds a unique learned signal so the shared block
        # can specialize its behavior per step
        if use_time_conditioning:
            self.time_embeds = nn.Parameter(
                torch.zeros(num_shared_iters, 1, 1, embed_dim)
            )
            nn.init.trunc_normal_(self.time_embeds, std=0.02)
        else:
            self.time_embeds = None

        # Final norm + classification head
        self.norm = nn.LayerNorm(embed_dim)

        if use_attn_pool:
            # Attention pooling: learned query attends over all tokens
            self.pool_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pool_kv = nn.Linear(embed_dim, 2 * embed_dim)
            self.pool_proj = nn.Linear(embed_dim, embed_dim)
            self.pool_norm = nn.LayerNorm(embed_dim)
            nn.init.trunc_normal_(self.pool_query, std=0.02)
            nn.init.trunc_normal_(self.pool_kv.weight, std=0.02)
            nn.init.trunc_normal_(self.pool_proj.weight, std=0.02)
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            self.pool_query = None
            self.pool_kv = None
            self.pool_proj = None
            self.pool_norm = None
            self.head = nn.Linear(embed_dim, num_classes)

        self.last_diagnostics: dict[str, float] = {}
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
            self.patch_embed.bias.zero_()
            nn.init.trunc_normal_(self.head.weight, std=0.02)
            self.head.bias.zero_()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, N, D]

        # Add CLS + positional embedding
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Save initial embedding for input injection
        x0 = x

        # Phase 1: Independent blocks (if any)
        for block in self.independent_blocks:
            x = block(x)

        # Phase 2: Weight-tied flow iterations
        num_shared = self.num_iterations - self.num_independent_blocks
        velocity = torch.zeros_like(x) if self.use_momentum else None

        for t in range(num_shared):
            gate = torch.sigmoid(self.step_gates[t])
            inject = torch.sigmoid(self.inject_gates[t])

            # Apply shared block with optional time conditioning
            h = self.iter_norms[t](x)
            if self.time_embeds is not None:
                h = h + self.time_embeds[t]
            block_out = self.shared_block(h)

            if self.use_momentum and velocity is not None:
                # Hamiltonian dynamics: update velocity, then position
                velocity = self.momentum_beta * velocity + (1 - self.momentum_beta) * (block_out - x)
                delta = velocity
            else:
                delta = block_out - x

            # Flow step: h(t+1) = h(t) + gate * delta + inject * (x0 - h(t))
            x = x + gate * delta + inject * (x0 - x)

        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor, condition=None, return_features: bool = False) -> torch.Tensor:
        x = self.forward_features(x)

        if self.use_attn_pool and self.pool_kv is not None:
            # Attention pooling over all tokens
            B = x.shape[0]
            q = self.pool_query.expand(B, -1, -1)  # [B, 1, D]
            kv = self.pool_kv(x)  # [B, N, 2D]
            k, v = kv.chunk(2, dim=-1)
            attn = (q @ k.transpose(-2, -1)) * (self.embed_dim ** -0.5)
            attn = attn.softmax(dim=-1)
            pooled = (attn @ v).squeeze(1)  # [B, D]
            features = self.pool_norm(self.pool_proj(pooled))
        else:
            features = x[:, 0]  # CLS token

        if return_features:
            return features
        return self.head(features)

    def get_diagnostics(self) -> dict[str, float]:
        with torch.no_grad():
            gates = torch.sigmoid(self.step_gates)
            injects = torch.sigmoid(self.inject_gates)
            self.last_diagnostics = {
                "fd_gate_mean": float(gates.mean().item()),
                "fd_gate_min": float(gates.min().item()),
                "fd_gate_max": float(gates.max().item()),
                "fd_inject_mean": float(injects.mean().item()),
                "fd_num_iterations": float(self.num_iterations),
            }
        return dict(self.last_diagnostics)

    def get_aux_losses(self) -> dict[str, torch.Tensor]:
        return {}
