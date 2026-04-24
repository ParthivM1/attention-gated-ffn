"""Attention-Gated Feedforward Network (AGFF).

In a standard transformer block, attention aggregates cross-token context then
discards that signal before the FFN. AGFF feeds the attention output directly to
a per-feature sigmoid gate inside the FFN, so each token's FFN transformation is
conditioned on cross-token context.

  content = act(W_content @ x)          -- token's own representation
  gate    = sigmoid(W_gate   @ attn_out) -- what attention discovered
  out     = W_out @ (content * gate)

Distinctions from prior work:
- SwiGLU: gate = sigmoid(W @ x) — gate and content both from same token, no
  cross-token information in gate.
- AGFF:   gate = sigmoid(W @ attn_out) — gate conditioned on attended context
  from ALL tokens, making the FFN input-adaptive at the cross-token level.
- MoE: routing is discrete and learned separately; AGFF is a continuous
  soft-gate that reuses the already-computed attention signal.

Parameter count matches standard GELU MLP exactly: using hidden = round(8D/3)
for two paths + output gives 3 * D * (8D/3) = 8D^2, same as fc1 (D->4D) +
fc2 (4D->D). No extra parameters relative to the baseline.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class AttentionGatedFFN(nn.Module):
    """Drop-in MLP replacement for GeoViTBlock (via mlp_override).

    Forward signature matches GeoMlp.forward so the block needs no changes.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features)  # = round(8 * D / 3) for param parity
        self.out_features = int(out_features)

        self.fc_content = nn.Linear(self.in_features, self.hidden_features)
        self.fc_gate = nn.Linear(self.in_features, self.hidden_features)
        self.fc_out = nn.Linear(self.hidden_features, self.out_features)
        self.act = act_layer()

        self.last_diagnostics: dict[str, float] = {}
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            for fc in [self.fc_content, self.fc_gate, self.fc_out]:
                nn.init.trunc_normal_(fc.weight, std=0.02)
                nn.init.zeros_(fc.bias)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
        *,
        attn_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # If no attn_out (e.g. mlp_first_update path), degrade gracefully to SwiGLU.
        gate_src = attn_out if attn_out is not None else x

        content = self.act(self.fc_content(x))
        gate = torch.sigmoid(self.fc_gate(gate_src))
        out = self.fc_out(content * gate)

        with torch.no_grad():
            self.last_diagnostics = {
                "agff_gate_mean": float(gate.mean().item()),
                "agff_gate_std": float(gate.std().item()),
                "agff_using_attn": float(attn_out is not None),
            }
        return out

    def get_diagnostics(self) -> dict[str, float]:
        return dict(self.last_diagnostics)

    def get_aux_losses(self) -> dict[str, torch.Tensor]:
        return {}
