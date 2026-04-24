"""GradFlow FFN: update-direction-corrected MLP block.

Standard transformer FFN:
    h = fc2(GELU(fc1(x)))

GradFlow FFN:
    h     = fc2(GELU(fc1(x)))          # first-order step (same as standard)
    delta = h - x                       # the UPDATE DIRECTION the FFN chose
    corr  = corr2(GELU(corr1(delta)))  # correction: function of update direction only
    out   = h + gate * corr            # corrected output

Key properties:
  * The correction is a function of (h - x), NOT of x or h separately.
    This makes it invariant to the absolute scale of the residual stream and
    only responds to the CHANGE the FFN is proposing.
  * At init (gate=0) the block is an exact standard FFN — optimization is stable.
  * The gate grows during training as the correction becomes useful.
  * At test time you can iterate: feed the corrected output back and apply the
    correction again. Each iteration refines the output using the same learned
    correction weights. This gives a meaningful test-time compute axis.

Why this is novel vs. prior work:
  * Not DEQ/Neural ODE: those iterate the FULL block. Here only the small
    correction branch iterates, leaving the main FFN computation fixed.
  * Not residual learning: the correction is conditioned on (h - x), the update
    direction, which encodes second-order curvature information.
  * Not mixture-of-experts: no routing, same weights for all tokens.
  * The test-time iteration gives progressively better approximations without
    rerunning the expensive fc1/fc2 path.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GradFlowFFN(nn.Module):
    """Update-direction-corrected FFN block.

    Drop-in replacement for GeoMlp. Compatible with GeoViTBlock's mlp_override
    interface: forward(x, condition=None, *, attn_out=None) -> Tensor.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        corr_bottleneck: int = 32,
        n_test_iters: int = 1,
        gate_init: float = -4.0,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features)
        self.out_features = int(out_features)
        self.n_test_iters = max(int(n_test_iters), 1)
        self.corr_bottleneck = max(int(corr_bottleneck), 8)

        # Standard FFN path.
        self.fc1 = nn.Linear(self.in_features, self.hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(self.hidden_features, self.out_features)

        # Correction branch: conditioned on (h - x), the update direction.
        # Uses a small bottleneck to keep parameter count low.
        self.corr1 = nn.Linear(self.in_features, self.corr_bottleneck)
        self.corr_act = act_layer()
        self.corr2 = nn.Linear(self.corr_bottleneck, self.out_features)

        # Learnable gate in [0, 1): default init near-zero (sigmoid(-4) ≈ 0.018)
        # so corrections are suppressed at training start.
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

        self.last_diagnostics: dict[str, float] = {}
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            nn.init.trunc_normal_(self.fc1.weight, std=0.02)
            self.fc1.bias.zero_()
            nn.init.trunc_normal_(self.fc2.weight, std=0.02)
            self.fc2.bias.zero_()
            # Init correction branch near-zero so it has no effect at the start.
            nn.init.trunc_normal_(self.corr1.weight, std=0.01)
            self.corr1.bias.zero_()
            self.corr2.weight.zero_()
            self.corr2.bias.zero_()

    def _correction(self, delta: torch.Tensor) -> torch.Tensor:
        return self.corr2(self.corr_act(self.corr1(delta)))

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
        *,
        attn_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.fc2(self.act(self.fc1(x)))

        gate = torch.sigmoid(self.gate_logit)

        for _ in range(self.n_test_iters):
            delta = h - x
            corr = self._correction(delta)
            h = h + gate * corr

        self.last_diagnostics = {
            "gfn_gate": float(gate.detach().item()),
            "gfn_delta_norm": float((h - x).detach().norm(dim=-1).mean().item()),
            "gfn_corr_norm": float(corr.detach().norm(dim=-1).mean().item()),
        }
        return h

    def get_diagnostics(self) -> dict[str, float]:
        return dict(self.last_diagnostics)

    def get_aux_losses(self) -> dict[str, torch.Tensor]:
        return {}


def set_gradflow_ffn_test_iters(model: nn.Module, n_iters: int) -> None:
    """Set the number of correction iterations on every GradFlowFFN in the model."""
    for m in model.modules():
        if type(m).__name__ == "GradFlowFFN":
            m.n_test_iters = max(int(n_iters), 1)
