"""Generalized Flow-Depth Transformer (FD-Transformer).

Core principle: Replace N independent transformer blocks with 1 shared block
iterated N times. This decouples effective depth from parameter count.

Key innovation: Iterative refinement of representations instead of hierarchical
independent processing. Each iteration refines ALL tokens simultaneously.

Works for: Vision Transformers, Language Models (GPT, BERT, T5), Multimodal (CLIP),
any encoder/decoder transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class FDTransformerBlock(nn.Module):
    """Single transformer block for weight-tied iteration.

    This is a standard transformer block (attention + FFN) that will be
    applied multiple times. Design it to be parameter-efficient since
    it will be reused.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Attention
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # FFN
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.mlp_drop = nn.Dropout(proj_drop)

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.qkv.weight, std=0.02)
        if self.qkv.bias is not None:
            nn.init.constant_(self.qkv.bias, 0)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention
        x_norm = self.norm1(x)
        B, N, C = x_norm.shape

        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(~mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)

        x = x + self.drop_path(x_attn)

        # FFN
        x_norm = self.norm2(x)
        x_mlp = self.fc2(self.act(self.fc1(x_norm)))
        x_mlp = self.mlp_drop(x_mlp)

        x = x + self.drop_path(x_mlp)

        return x


class DropPath(nn.Module):
    """Stochastic depth."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = (random_tensor < keep_prob).float() / keep_prob
        return x * random_tensor


class FDTransformer(nn.Module):
    """Generalized Flow-Depth Transformer.

    Replaces N independent transformer blocks with 1 shared block iterated N times.
    This enables:
    - Parameter efficiency: N×fewer params, same effective depth
    - Iterative refinement: all tokens refined together at each step
    - Learnable step schedule: gates control contribution per iteration
    - Optional: Dynamic depth per token via controller network

    Args:
        dim: Hidden dimension
        num_heads: Number of attention heads
        num_iterations: Number of times to apply the shared block (effective depth)
        mlp_ratio: FFN expansion ratio
        drop_path: Stochastic depth rate
        input_inject_strength: How much to re-inject original input at each step
        use_controller: Whether to use per-token dynamic depth
        controller_hidden_dim: Controller network hidden dimension
        num_independent_blocks: How many independent blocks before shared iteration (hybrid mode)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_iterations: int = 12,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        input_inject_strength: float = 0.1,
        use_controller: bool = False,
        controller_hidden_dim: int = 64,
        num_independent_blocks: int = 0,
        time_conditioning: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_iterations = num_iterations
        self.num_independent_blocks = num_independent_blocks
        self.input_inject_strength = input_inject_strength
        self.use_controller = use_controller
        self.time_conditioning = time_conditioning

        # Independent blocks (optional hybrid mode)
        if num_independent_blocks > 0:
            self.independent_blocks = nn.ModuleList([
                FDTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path * i / max(num_iterations - 1, 1),
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                )
                for i in range(num_independent_blocks)
            ])
        else:
            self.independent_blocks = nn.ModuleList()

        # Shared block for iteration
        shared_dp = drop_path * 0.5
        self.shared_block = FDTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=shared_dp,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        # Per-iteration control parameters
        num_shared_iters = num_iterations - num_independent_blocks
        self.step_gates = nn.Parameter(torch.ones(num_shared_iters) * 0.5)
        self.inject_gates = nn.Parameter(torch.ones(num_shared_iters) * input_inject_strength)
        self.iter_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_shared_iters)])

        # Optional time conditioning (iteration awareness)
        if time_conditioning:
            self.time_embeds = nn.Parameter(torch.randn(num_shared_iters, 1, 1, dim) * 0.02)
        else:
            self.time_embeds = None

        # Optional dynamic depth controller
        if use_controller:
            self.controller = nn.Sequential(
                nn.Linear(dim, controller_hidden_dim),
                nn.GELU(),
                nn.Linear(controller_hidden_dim, 1),
                nn.Sigmoid(),  # outputs [0, 1] representing fraction of iterations to use
            )
        else:
            self.controller = None

        self.last_diagnostics: Dict[str, float] = {}

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
        """
        Args:
            x: Input tensor [B, N, D]
            mask: Optional attention mask [B, N, N]
            return_intermediate: If True, return (output, list of intermediate states)

        Returns:
            output tensor [B, N, D] or (output, intermediates) if return_intermediate=True
        """
        intermediates = [] if return_intermediate else None

        # Phase 1: Independent blocks
        for block in self.independent_blocks:
            x = block(x, mask)
            if return_intermediate:
                intermediates.append(x.detach())

        # Phase 2: Shared block iteration
        x0 = x  # Save for input injection
        num_shared = self.num_iterations - self.num_independent_blocks

        # Compute dynamic depth if controller is enabled
        max_iter_to_use = num_shared
        if self.use_controller:
            depth_scores = self.controller(x.mean(dim=1))  # [B, 1]
            max_iter_to_use_per_sample = (depth_scores * num_shared).long().squeeze(-1)  # [B]
        else:
            max_iter_to_use_per_sample = None

        for t in range(num_shared):
            # Check if any samples should exit early
            if max_iter_to_use_per_sample is not None:
                # Mask out samples that have reached their iteration limit
                should_continue = max_iter_to_use_per_sample > t  # [B]

            gate = torch.sigmoid(self.step_gates[t])
            inject = torch.sigmoid(self.inject_gates[t])

            # Apply shared block
            h = self.iter_norms[t](x)
            if self.time_embeds is not None:
                h = h + self.time_embeds[t]

            block_out = self.shared_block(h, mask)

            # Flow step: refine current state
            delta = block_out - x
            x = x + gate * delta + inject * (x0 - x)

            # Apply early exit mask if using controller
            if max_iter_to_use_per_sample is not None:
                x = torch.where(
                    should_continue.view(-1, 1, 1),
                    x,
                    x.detach()  # Stop gradient for exited samples
                )

            if return_intermediate:
                intermediates.append(x.detach())

        self.last_diagnostics = {
            "fd_gate_mean": float(torch.sigmoid(self.step_gates).mean().item()),
            "fd_gate_min": float(torch.sigmoid(self.step_gates).min().item()),
            "fd_gate_max": float(torch.sigmoid(self.step_gates).max().item()),
            "fd_inject_mean": float(torch.sigmoid(self.inject_gates).mean().item()),
            "fd_num_iterations": float(self.num_iterations),
        }

        if return_intermediate:
            return x, intermediates
        return x

    def get_diagnostics(self) -> Dict[str, float]:
        """Return training diagnostics about gate values."""
        return dict(self.last_diagnostics)
