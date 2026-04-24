"""Attention-Gated Feedforward Network (AGFF).

In a standard transformer block, attention aggregates cross-token context then
discards that signal before the FFN. AGFF feeds the attention output directly to
a per-feature sigmoid gate inside the FFN, so each token's FFN transformation is
conditioned on cross-token context.

  content = act(W_content @ x)            -- token's own representation
  gate    = sigmoid(W_gate @ LN(attn_out)) -- what attention discovered
  out     = W_out @ (content * gate)

gate_mode options
-----------------
'attn'   : gate = sigmoid(W_gate @ LN(attn_out))               -- pure cross-token per token
'dual'   : gate = sigmoid(W_a @ LN(attn_out) + W_x @ x)       -- anchored to token
'scale'  : gate = 1 + tanh(W_gate @ LN(attn_out))              -- multiplicative residual [0,2]
'cls'    : gate = sigmoid(W_gate @ LN(attn_out[:,0:1]))         -- CLS-only global gate
           (broadcast from 1 token to all; cleaner global context, less noise per patch)

Parameter count vs standard GELU MLP
--------------------------------------
hidden = round(8D/3) → 3 × D × hidden ≈ 8D²  (exact parity with fc1+fc2)
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
        gate_mode: str = "attn",   # 'attn' | 'dual' | 'scale' | 'cls'
        gate_ln: bool = True,      # LayerNorm attn_out before gate (prevents scale collapse)
        gate_init_scale: float = -1.0,  # -1 = auto 1/sqrt(hidden); >0 = explicit std
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features)
        self.out_features = int(out_features)
        self.gate_mode = gate_mode.lower().strip()

        self.fc_content = nn.Linear(self.in_features, self.hidden_features)
        self.fc_out = nn.Linear(self.hidden_features, self.out_features)
        self.act = act_layer()

        # Gate path
        self.gate_ln = nn.LayerNorm(self.in_features) if gate_ln else nn.Identity()
        self.fc_gate = nn.Linear(self.in_features, self.hidden_features)
        if self.gate_mode == "dual":
            # Second linear: x → gate logit (SwiGLU-style anchor)
            self.fc_gate_x = nn.Linear(self.in_features, self.hidden_features)
        else:
            self.fc_gate_x = None
        # 'cls' mode: fc_gate takes in_features (CLS token) → hidden_features, broadcast to all tokens

        self.gate_init_scale = float(gate_init_scale)
        self.last_diagnostics: dict[str, float] = {}
        self._init_weights()

    def _init_weights(self) -> None:
        # content and out: standard small init
        with torch.no_grad():
            for fc in [self.fc_content, self.fc_out]:
                nn.init.trunc_normal_(fc.weight, std=0.02)
                nn.init.zeros_(fc.bias)
            # Gate init: larger std so gate has variance from epoch 0
            # Auto = 1/sqrt(hidden) so W_gate @ LN(attn_out) has std~1 → gate_std~0.14
            gate_std = (
                1.0 / math.sqrt(self.hidden_features)
                if self.gate_init_scale <= 0
                else self.gate_init_scale
            )
            nn.init.trunc_normal_(self.fc_gate.weight, std=gate_std)
            nn.init.zeros_(self.fc_gate.bias)
            if self.fc_gate_x is not None:
                nn.init.trunc_normal_(self.fc_gate_x.weight, std=0.02)
                nn.init.zeros_(self.fc_gate_x.bias)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
        *,
        attn_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gate_src = self.gate_ln(attn_out) if attn_out is not None else x
        content = self.act(self.fc_content(x))

        if self.gate_mode == "cls":
            # CLS token broadcasts a single global gate to all patch tokens
            cls_src = self.gate_ln(attn_out[:, 0:1]) if attn_out is not None else x[:, 0:1]
            gate = torch.sigmoid(self.fc_gate(cls_src))  # [B, 1, hidden] → broadcasts
        elif self.gate_mode == "dual":
            # Anchor gate to token + cross-token bias
            logit = self.fc_gate(gate_src) + self.fc_gate_x(x)
            gate = torch.sigmoid(logit)
        elif self.gate_mode == "scale":
            # Multiplicative residual: default 1 (pass-through), range [0, 2]
            gate = 1.0 + torch.tanh(self.fc_gate(gate_src))
        else:  # 'attn'
            gate = torch.sigmoid(self.fc_gate(gate_src))

        out = self.fc_out(content * gate)

        with torch.no_grad():
            self.last_diagnostics = {
                "agff_gate_mean": float(gate.mean().item()),
                "agff_gate_std": float(gate.std().item()),
                "agff_using_attn": float(attn_out is not None),
                "agff_gate_mode_attn": float(self.gate_mode == "attn"),
                "agff_gate_mode_dual": float(self.gate_mode == "dual"),
                "agff_gate_mode_scale": float(self.gate_mode == "scale"),
            }
        return out

    def get_diagnostics(self) -> dict[str, float]:
        return dict(self.last_diagnostics)

    def get_aux_losses(self) -> dict[str, torch.Tensor]:
        return {}
