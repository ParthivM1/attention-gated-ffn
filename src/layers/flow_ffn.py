"""Flow-Native FFN block (FNFL).

Replaces a standard transformer MLP with an iterative flow operator in a
low-dimensional chart. The flow operator is shared across T timesteps;
expressivity comes from iteration rather than parameter count.

Design intent
-------------
Past structured-MLP attempts in this repo lived as residuals on top of a dense
fc1 baseline ("sidecars"). The empirical lesson was: when a structured branch
runs in parallel with a dense one, the optimizer leans on the dense path and
the structured branch never has to carry the computation.

FlowFFN removes that escape hatch. The forward path is:

    h_0   = proj_in(x)                       # D -> r  (low-rank compression)
    for t in 1..T:
        h_t = h_{t-1} + flow_strength * step_gates[t] * step(h_{t-1}, ctx)
    out   = proj_out(h_T)                    # r -> 4D
    out   = fc2( GELU(out) )                 # 4D -> D

The single shared `step` operator is structured (spectral basis + low-rank
residual) and conditioned on a pooled-token context. There is no parallel
dense fc1 the optimizer can hide behind.

`flow_strength` is an externally-controlled buffer (no gradient) that ramps
from 0.0 to 1.0 across training. At init the block reduces to a low-rank
linear MLP (proj_in -> GELU -> fc2) which is easy to optimize. Once
`flow_strength` reaches 1.0 the structured flow operator is the rule.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _FlowStepOperator(nn.Module):
    """One shared iteration step in the low-dim flow chart.

    The step is a structured (spectral + low-rank) update conditioned on a
    pooled-token context vector. Output magnitude is bounded by a learnable
    damping factor so iteration stays stable across T steps.
    """

    def __init__(
        self,
        chart_dim: int,
        *,
        num_spectral_bases: int = 8,
        low_rank: int = 4,
        controller_hidden_dim: int = 128,
    ):
        super().__init__()
        self.chart_dim = int(chart_dim)
        self.num_spectral_bases = max(int(num_spectral_bases), 1)
        self.low_rank = max(int(low_rank), 1)

        # Spectral component: K learned basis matrices, gated by per-sample coefficients.
        self.spectral_basis = nn.Parameter(
            torch.empty(self.num_spectral_bases, self.chart_dim, self.chart_dim)
        )
        self.spectral_head = nn.Linear(controller_hidden_dim, self.num_spectral_bases)

        # Low-rank residual: U @ diag(coeff) @ V on the chart.
        self.U = nn.Parameter(torch.empty(self.chart_dim, self.low_rank))
        self.V = nn.Parameter(torch.empty(self.low_rank, self.chart_dim))
        self.lowrank_head = nn.Linear(controller_hidden_dim, self.low_rank)

        # Stability damping in [0.25, 0.75] via tanh.
        self.damping_logit = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self) -> None:
        # Xavier-scale init so flow steps contribute meaningfully from epoch 1.
        # std=0.02 made delta≈0.002×h_norm, causing step_gate to collapse toward 0.
        spectral_std = 1.0 / math.sqrt(self.chart_dim)
        lr_std = 1.0 / math.sqrt(self.low_rank)
        with torch.no_grad():
            for k in range(self.num_spectral_bases):
                A = torch.randn(self.chart_dim, self.chart_dim) * spectral_std
                self.spectral_basis[k] = (A - A.t()) / 2.0
            nn.init.trunc_normal_(self.U, std=lr_std)
            nn.init.trunc_normal_(self.V, std=spectral_std)
            nn.init.trunc_normal_(self.spectral_head.weight, std=0.02)
            self.spectral_head.bias.zero_()
            nn.init.trunc_normal_(self.lowrank_head.weight, std=0.02)
            self.lowrank_head.bias.zero_()

    def forward(self, h: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # h:   [B, N, chart_dim]
        # ctx: [B, controller_hidden]
        spec_coeff = torch.tanh(self.spectral_head(ctx))                # [B, K]
        spec_basis = self.spectral_basis.to(dtype=h.dtype)              # [K, d, d]
        # spec_combos[b, n, k, d_out] = sum_e h[b,n,e] * basis[k, d_out, e]
        spec_combos = torch.einsum("bne,kde->bnkd", h, spec_basis)      # [B, N, K, d]
        spectral = (spec_coeff.unsqueeze(1).unsqueeze(-1) * spec_combos).sum(dim=2)  # [B, N, d]

        lr_coeff = torch.tanh(self.lowrank_head(ctx))                   # [B, low_rank]
        V_t = self.V.to(dtype=h.dtype)
        U_t = self.U.to(dtype=h.dtype)
        lr_proj = torch.einsum("bnd,rd->bnr", h, V_t)                   # [B, N, low_rank]
        lr_scaled = lr_proj * lr_coeff.unsqueeze(1)                     # [B, N, low_rank]
        lowrank = torch.einsum("bnr,dr->bnd", lr_scaled, U_t)           # [B, N, d]

        delta = spectral + lowrank
        damping = torch.tanh(self.damping_logit).abs() * 0.5 + 0.25     # in [0.25, 0.75]
        return damping * delta


class FlowFFN(nn.Module):
    """Flow-Native FFN block (a true MLP replacement, not a residual sidecar).

    Forward signature mirrors `GeoMlp.forward` so it can be dropped into
    `GeoViTBlock` via the `mlp_override` path:

        forward(x, condition=None, *, attn_out=None) -> Tensor
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        flow_rank: int = 64,
        num_steps: int = 2,
        num_spectral_bases: int = 8,
        low_rank: int = 4,
        controller_hidden_dim: int = 128,
        act_layer=nn.GELU,
        flow_strength_init: float = 0.0,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features)
        self.out_features = int(out_features)
        self.flow_rank = int(flow_rank)
        self.num_steps = max(int(num_steps), 1)

        self.proj_in = nn.Linear(self.in_features, self.flow_rank, bias=False)
        self.flow_step = _FlowStepOperator(
            self.flow_rank,
            num_spectral_bases=num_spectral_bases,
            low_rank=low_rank,
            controller_hidden_dim=controller_hidden_dim,
        )
        # Per-step gate: small learnable scalars, init at 0.5 so all steps contribute.
        self.step_gates = nn.Parameter(torch.full((self.num_steps,), 0.5))
        self.proj_out = nn.Linear(self.flow_rank, self.hidden_features, bias=True)
        # Prevents proj_out from absorbing the scale of h (which causes h→0 during training).
        self.chart_norm = nn.LayerNorm(self.flow_rank)

        self.act = act_layer()
        self.fc2 = nn.Linear(self.hidden_features, self.out_features, bias=True)

        # Conditioner: concat[CLS, mean(patches)] -> hidden context.
        self.context_proj = nn.Sequential(
            nn.LayerNorm(self.in_features * 2),
            nn.Linear(self.in_features * 2, controller_hidden_dim),
            nn.GELU(),
            nn.Linear(controller_hidden_dim, controller_hidden_dim),
        )

        # Externally controlled flow strength; not a parameter.
        # 0.0 => flow contributes nothing (block is low-rank linear MLP).
        # 1.0 => flow is full strength.
        self.register_buffer(
            "flow_strength",
            torch.tensor(float(flow_strength_init), dtype=torch.float32),
            persistent=False,
        )

        self.last_diagnostics: dict[str, float] = {}
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            nn.init.trunc_normal_(self.proj_in.weight, std=0.02)
            nn.init.trunc_normal_(self.proj_out.weight, std=0.02)
            self.proj_out.bias.zero_()
            nn.init.trunc_normal_(self.fc2.weight, std=0.02)
            self.fc2.bias.zero_()

    def set_flow_strength(self, value: float) -> None:
        self.flow_strength.fill_(float(value))

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        cls = x[:, 0]
        if x.shape[1] > 1:
            mean = x[:, 1:].mean(dim=1)
        else:
            mean = cls
        return torch.cat([cls, mean], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
        *,
        attn_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # condition and attn_out are accepted for interface compatibility with
        # GeoMlp.forward but unused in this block design.
        h = self.proj_in(x)
        ctx = self.context_proj(self._pool(x))

        strength = self.flow_strength.to(dtype=h.dtype)
        for t in range(self.num_steps):
            delta = self.flow_step(h, ctx)
            h = h + strength * self.step_gates[t] * delta

        h_raw_norm = float(h.detach().norm(dim=-1).mean().item())
        out = self.proj_out(self.chart_norm(h))
        out = self.act(out)
        out = self.fc2(out)

        self.last_diagnostics = {
            "fnfl_flow_strength": float(self.flow_strength.detach().item()),
            "fnfl_step_gate_mean": float(self.step_gates.detach().mean().item()),
            "fnfl_h_norm": h_raw_norm,
            "fnfl_damping": float(
                torch.tanh(self.flow_step.damping_logit).detach().abs().item() * 0.5 + 0.25
            ),
        }
        return out

    def get_diagnostics(self) -> dict[str, float]:
        return dict(self.last_diagnostics)

    def get_aux_losses(self) -> dict[str, torch.Tensor]:
        return {}


def set_flow_ffn_strength(model: nn.Module, value: float) -> None:
    """Walk the model and set flow_strength on every FlowFFN module.

    Uses a class-name check (not isinstance) because this repo supports two
    import paths (``layers.flow_ffn`` and ``src.layers.flow_ffn``) which can
    yield two distinct class objects for the same source file.
    """
    for module in model.modules():
        if type(module).__name__ == "FlowFFN" and hasattr(module, "set_flow_strength"):
            module.set_flow_strength(value)


def compute_flow_ffn_strength(epoch: int, *, anneal_epochs: int, max_strength: float = 1.0) -> float:
    """Linear ramp from 0.0 at epoch 1 up to `max_strength` at `anneal_epochs+1`."""
    if anneal_epochs <= 0:
        return float(max_strength)
    progress = max(0.0, min(1.0, (int(epoch) - 1) / float(anneal_epochs)))
    return float(max_strength) * progress
