import torch
import torch.nn as nn

try:
    from layers.operator_bank import (
        DepthwiseLocalMixer,
        FactorizedLinear,
        LowRankOrthogonalMixer,
        LowRankResidualOperator,
        MagnusSemanticOperator,
        SemanticManifoldOperator,
        SpectralLowRankOperator,
        SpectralFlowResidual,
        SharedGeoOperatorBank,
        SpectralBasisOperator,
        summarize_detail_tokens,
    )
    from layers.text_conditioner import PooledTokenConditioner, TextConditioner
except ImportError:
    from src.layers.operator_bank import (
        DepthwiseLocalMixer,
        FactorizedLinear,
        LowRankOrthogonalMixer,
        LowRankResidualOperator,
        MagnusSemanticOperator,
        SemanticManifoldOperator,
        SpectralLowRankOperator,
        SpectralFlowResidual,
        SharedGeoOperatorBank,
        SpectralBasisOperator,
        summarize_detail_tokens,
    )
    from src.layers.text_conditioner import PooledTokenConditioner, TextConditioner


class GeoOperatorLinear(nn.Module):
    """Structured linear map with a teacher-aligned base and optional semantic manifold operator."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        condition_dim: int = 512,
        controller_hidden_dim: int = 128,
        num_spectral_bases: int = 8,
        low_rank_rank: int = 8,
        orthogonal_rank: int = 8,
        flow_rank: int = 0,
        flow_steps: int = 1,
        flow_step_size: float = 0.5,
        base_rank: int | None = None,
        use_local_operator: bool = False,
        local_kernel_size: int = 3,
        residual_scale: float = 1.0,
        residual_budget: float = 0.0,
        spectral_scale: float = 1.0,
        low_rank_scale: float = 1.0,
        rotation_scale: float = 1.0,
        block_scale_init: float = 1.0,
        learnable_block_scale: bool = False,
        semantic_manifold_mode: bool = False,
        semantic_num_experts: int = 2,
        semantic_expert_temperature: float = 0.5,
        magnus_semantic_mode: bool = False,
        magnus_single_operator: bool = False,
        magnus_detail_topk: int = 4,
        magnus_rotation_mode: bool = False,
        magnus_rotation_strength: float = 1.0,
        coupled_spectral_low_rank: bool = False,
        coupled_learnable_input_basis: bool = False,
        coupled_shared_gate: bool = False,
        strict_semantic_operator: bool = False,
        manifold_alignment_mode: bool = False,
        use_conditioner: bool = True,
        use_internal_conditioner: bool = True,
        shared_bank: SharedGeoOperatorBank | None = None,
        shared_conditioner: nn.Module | None = None,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_spectral_bases = int(num_spectral_bases)
        self.low_rank_rank = int(low_rank_rank)
        self.orthogonal_rank = int(orthogonal_rank)
        self.flow_rank = int(flow_rank)
        self.flow_steps = max(int(flow_steps), 1)
        self.flow_step_size = float(flow_step_size)
        self.base_rank = int(base_rank) if base_rank is not None and int(base_rank) > 0 else None
        self.use_local_operator = bool(use_local_operator)
        self.residual_scale = float(residual_scale)
        self.residual_budget = float(residual_budget)
        self.spectral_scale = float(spectral_scale)
        self.low_rank_scale = float(low_rank_scale)
        self.rotation_scale = float(rotation_scale)
        self.block_scale_init = max(float(block_scale_init), 1e-4)
        self.learnable_block_scale = bool(learnable_block_scale)
        self.semantic_manifold_mode = bool(semantic_manifold_mode and flow_rank > 0)
        self.semantic_num_experts = int(semantic_num_experts)
        self.semantic_expert_temperature = float(semantic_expert_temperature)
        self.magnus_semantic_mode = bool(magnus_semantic_mode and flow_rank > 0)
        self.magnus_single_operator = bool(magnus_single_operator and self.magnus_semantic_mode)
        self.magnus_detail_topk = int(magnus_detail_topk)
        self.magnus_rotation_mode = bool(magnus_rotation_mode and orthogonal_rank > 0)
        self.magnus_rotation_strength = float(magnus_rotation_strength)
        self.coupled_spectral_low_rank = bool(coupled_spectral_low_rank)
        self.coupled_learnable_input_basis = bool(coupled_learnable_input_basis)
        self.coupled_shared_gate = bool(coupled_shared_gate and coupled_spectral_low_rank)
        self.strict_semantic_operator = bool(strict_semantic_operator and semantic_manifold_mode and flow_rank > 0)
        self.manifold_alignment_mode = bool(manifold_alignment_mode and flow_rank > 0)
        self.use_conditioner = bool(use_conditioner)
        self.use_internal_conditioner = bool(use_internal_conditioner)

        if self.base_rank is not None and self.base_rank < min(self.in_features, self.out_features):
            self.base = FactorizedLinear(self.in_features, self.out_features, self.base_rank, bias=bias)
        else:
            self.base = nn.Linear(self.in_features, self.out_features, bias=bias)
            self.base_rank = None

        self.shared_bank = shared_bank
        self.conditioner = None
        self.token_only_conditioner = None
        if self.magnus_single_operator:
            self.token_only_conditioner = shared_conditioner or PooledTokenConditioner(
                token_dim=self.in_features,
                hidden_dim=controller_hidden_dim,
            )
        elif self.use_conditioner and self.use_internal_conditioner:
            self.conditioner = shared_conditioner or TextConditioner(
                token_dim=self.in_features,
                condition_dim=condition_dim,
                hidden_dim=controller_hidden_dim,
            )
        else:
            self.token_only_conditioner = shared_conditioner or PooledTokenConditioner(
                token_dim=self.in_features,
                hidden_dim=controller_hidden_dim,
            )
        if self.shared_bank is not None:
            self.local_mixer = self.shared_bank.local_mixer if self.use_local_operator else None
            self.spectral_operator = self.shared_bank.spectral_operator
            self.low_rank_operator = self.shared_bank.low_rank_operator
            self.coupled_operator = getattr(self.shared_bank, "coupled_operator", None)
            self.orthogonal_mixer = self.shared_bank.orthogonal_mixer
            self.magnus_operator = getattr(self.shared_bank, "magnus_operator", None)
            self.flow_operator = self.shared_bank.flow_operator
            self.semantic_operator = getattr(self.shared_bank, "semantic_operator", None)
        elif self.magnus_single_operator:
            self.local_mixer = None
            self.spectral_operator = None
            self.low_rank_operator = None
            self.coupled_operator = None
            self.orthogonal_mixer = None
            self.magnus_operator = (
                MagnusSemanticOperator(
                    self.in_features,
                    self.out_features,
                    self.num_spectral_bases,
                    flow_rank=self.flow_rank,
                    controller_hidden_dim=controller_hidden_dim,
                    detail_topk=self.magnus_detail_topk,
                )
                if self.flow_rank > 0
                else None
            )
            self.semantic_operator = None
            self.flow_operator = None
        else:
            self.local_mixer = DepthwiseLocalMixer(self.in_features, kernel_size=local_kernel_size) if self.use_local_operator else None
            self.spectral_operator = SpectralBasisOperator(self.in_features, self.out_features, self.num_spectral_bases)
            self.low_rank_operator = LowRankResidualOperator(self.in_features, self.out_features, self.low_rank_rank)
            self.coupled_operator = SpectralLowRankOperator(
                self.in_features,
                self.out_features,
                self.num_spectral_bases,
                self.low_rank_rank,
                learnable_input_basis=self.coupled_learnable_input_basis,
                shared_gate=self.coupled_shared_gate,
            ) if self.coupled_spectral_low_rank else None
            if self.coupled_operator is not None:
                self.coupled_operator.attach_shared_spectral_output_basis(self.spectral_operator.output_basis)
            self.orthogonal_mixer = LowRankOrthogonalMixer(
                self.in_features,
                self.orthogonal_rank,
                magnus_rotation_mode=self.magnus_rotation_mode,
            )
            self.magnus_operator = (
                MagnusSemanticOperator(
                    self.in_features,
                    self.out_features,
                    self.num_spectral_bases,
                    flow_rank=self.flow_rank,
                    controller_hidden_dim=controller_hidden_dim,
                    detail_topk=self.magnus_detail_topk,
                )
                if self.magnus_semantic_mode and self.flow_rank > 0
                else None
            )
            self.semantic_operator = (
                SemanticManifoldOperator(
                    self.in_features,
                    self.out_features,
                    self.num_spectral_bases,
                    flow_rank=self.flow_rank,
                    num_experts=self.semantic_num_experts,
                    expert_temperature=self.semantic_expert_temperature,
                )
                if self.magnus_operator is None and self.semantic_manifold_mode and self.flow_rank > 0
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
                        self.in_features,
                        self.out_features,
                        self.num_spectral_bases,
                        flow_rank=self.flow_rank,
                        controller_hidden_dim=controller_hidden_dim,
                        steps=self.flow_steps,
                        step_size=self.flow_step_size,
                    )
                    if self.flow_rank > 0
                    else None
                )

        if self.magnus_single_operator:
            self.local_gate_head = None
            self.rotation_gate_head = None
            self.rotation_coeff_head = None
            self.rotation_local_proj = None
            self.rotation_local_gate_head = None
            self.rotation_local_coeff_head = None
            self.rotation_comm_head = None
            self.spectral_gate_head = None
            self.low_rank_head = None
            self.flow_gate_head = None
            self.flow_head = None
            self.semantic_expert_head = None
            self.semantic_mode_head = None
            self.semantic_gain_head = None
            self.semantic_gate_head = None
        else:
            self.local_gate_head = nn.Linear(controller_hidden_dim, 1)
            self.rotation_gate_head = nn.Linear(controller_hidden_dim, 1)
            self.rotation_coeff_head = nn.Linear(controller_hidden_dim, self.orthogonal_rank)
            self.rotation_local_proj = (
                nn.Sequential(
                    nn.LayerNorm(self.in_features),
                    nn.Linear(self.in_features, controller_hidden_dim),
                    nn.GELU(),
                    nn.Linear(controller_hidden_dim, controller_hidden_dim),
                )
                if self.magnus_rotation_mode
                else None
            )
            self.rotation_local_gate_head = (
                nn.Linear(controller_hidden_dim, 1) if self.magnus_rotation_mode else None
            )
            self.rotation_local_coeff_head = (
                nn.Linear(controller_hidden_dim, self.orthogonal_rank) if self.magnus_rotation_mode else None
            )
            self.rotation_comm_head = (
                nn.Linear(controller_hidden_dim, 1) if self.magnus_rotation_mode else None
            )
            self.spectral_gate_head = nn.Linear(controller_hidden_dim, self.num_spectral_bases)
            self.low_rank_head = (
                None
                if self.coupled_shared_gate and self.coupled_operator is not None
                else nn.Linear(controller_hidden_dim, self.low_rank_rank)
            )
            self.flow_gate_head = nn.Linear(controller_hidden_dim, 1) if self.flow_operator is not None else None
            self.flow_head = nn.Linear(controller_hidden_dim, self.flow_rank) if self.flow_operator is not None else None
            self.semantic_expert_head = (
                nn.Linear(controller_hidden_dim, self.semantic_operator.num_experts)
                if self.semantic_operator is not None
                else None
            )
            self.semantic_mode_head = (
                nn.Linear(controller_hidden_dim, self.flow_rank) if self.semantic_operator is not None else None
            )
            self.semantic_gain_head = (
                nn.Linear(controller_hidden_dim, self.semantic_operator.num_experts * self.flow_rank)
                if self.semantic_operator is not None
                else None
            )
            self.semantic_gate_head = (
                nn.Linear(controller_hidden_dim, 1) if self.semantic_operator is not None else None
            )
        self.last_diagnostics = {}
        self.last_aux_losses = {}
        self.adapter_scale = 1.0
        self.spectral_scale_mult = 1.0
        self.low_rank_scale_mult = 1.0
        self.rotation_scale_mult = 1.0
        if self.learnable_block_scale:
            self.block_scale_log = nn.Parameter(torch.log(torch.tensor(self.block_scale_init, dtype=torch.float32)))
        else:
            self.register_buffer("block_scale_buffer", torch.tensor(self.block_scale_init, dtype=torch.float32), persistent=False)

        self._init_controller()

    def _init_controller(self) -> None:
        with torch.no_grad():
            if self.local_gate_head is not None:
                self.local_gate_head.weight.zero_()
                self.local_gate_head.bias.fill_(-2.0)
            if self.rotation_gate_head is not None:
                self.rotation_gate_head.weight.zero_()
                self.rotation_gate_head.bias.fill_(-2.0)
            if self.rotation_coeff_head is not None:
                nn.init.trunc_normal_(self.rotation_coeff_head.weight, std=0.02)
                self.rotation_coeff_head.bias.zero_()
            if self.rotation_local_proj is not None:
                self.rotation_local_gate_head.weight.zero_()
                self.rotation_local_gate_head.bias.fill_(-2.5)
                nn.init.trunc_normal_(self.rotation_local_coeff_head.weight, std=0.02)
                self.rotation_local_coeff_head.bias.zero_()
                self.rotation_comm_head.weight.zero_()
                self.rotation_comm_head.bias.fill_(-2.5)
            if self.spectral_gate_head is not None:
                nn.init.trunc_normal_(self.spectral_gate_head.weight, std=0.02)
                self.spectral_gate_head.bias.zero_()
            if self.low_rank_head is not None:
                nn.init.trunc_normal_(self.low_rank_head.weight, std=0.02)
                self.low_rank_head.bias.zero_()
            if self.flow_gate_head is not None:
                self.flow_gate_head.weight.zero_()
                self.flow_gate_head.bias.fill_(-1.5)
            if self.flow_head is not None:
                nn.init.trunc_normal_(self.flow_head.weight, std=0.02)
                self.flow_head.bias.zero_()
            if self.semantic_expert_head is not None:
                nn.init.trunc_normal_(self.semantic_expert_head.weight, std=0.02)
                self.semantic_expert_head.bias.copy_(
                    torch.linspace(
                        -0.15,
                        0.15,
                        steps=self.semantic_expert_head.bias.numel(),
                        device=self.semantic_expert_head.bias.device,
                        dtype=self.semantic_expert_head.bias.dtype,
                    )
                )
            if self.semantic_mode_head is not None:
                self.semantic_mode_head.weight.zero_()
                self.semantic_mode_head.bias.fill_(0.25)
            if self.semantic_gain_head is not None:
                nn.init.trunc_normal_(self.semantic_gain_head.weight, std=0.02)
                self.semantic_gain_head.bias.zero_()
            if self.semantic_gate_head is not None:
                self.semantic_gate_head.weight.zero_()
                self.semantic_gate_head.bias.fill_(0.0)

    def _reshape_gate(self, gate: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return gate.view(gate.shape[0], 1, 1)
        return gate

    def _pool_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return torch.cat([x, x], dim=-1)
        cls_token = x[:, 0]
        mean_token = x[:, 1:].mean(dim=1) if x.shape[1] > 1 else cls_token
        return torch.cat([cls_token, mean_token], dim=-1)

    def _build_context(self, x: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        if self.token_only_conditioner is not None:
            return self.token_only_conditioner(self._pool_tokens(x))
        return self.conditioner(x, condition)

    def set_adapter_scale(self, scale: float) -> None:
        self.adapter_scale = float(scale)

    def _apply_residual_budget(
        self,
        base_out: torch.Tensor,
        residual_out: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        if self.residual_budget <= 0.0:
            return residual_out, 1.0
        if residual_out.dim() < 2:
            return residual_out, 1.0

        base_norm = base_out.detach().reshape(base_out.shape[0], -1).norm(dim=-1, keepdim=True)
        residual_norm = residual_out.reshape(residual_out.shape[0], -1).norm(dim=-1, keepdim=True)
        target_norm = self.residual_budget * base_norm.clamp_min(1e-6)
        scale = torch.clamp(target_norm / residual_norm.clamp_min(1e-6), max=1.0)
        view_shape = [residual_out.shape[0]] + [1] * (residual_out.dim() - 1)
        scaled = residual_out * scale.view(*view_shape)
        return scaled, float(scale.mean().item())

    def set_component_scale_multipliers(
        self,
        *,
        spectral: float = 1.0,
        low_rank: float = 1.0,
        rotation: float = 1.0,
    ) -> None:
        self.spectral_scale_mult = float(spectral)
        self.low_rank_scale_mult = float(low_rank)
        self.rotation_scale_mult = float(rotation)

    def _block_scale(self) -> torch.Tensor:
        if self.learnable_block_scale:
            return self.block_scale_log.exp()
        return self.block_scale_buffer

    def _apply_residual_budget_with_value(
        self,
        base_out: torch.Tensor,
        residual_out: torch.Tensor,
        residual_budget: float,
    ) -> tuple[torch.Tensor, float]:
        original_budget = self.residual_budget
        self.residual_budget = float(residual_budget)
        scaled, scale = self._apply_residual_budget(base_out, residual_out)
        self.residual_budget = original_budget
        return scaled, scale

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        context = self._build_context(x, condition)
        block_scale = self._block_scale()
        effective_residual_scale = self.residual_scale * block_scale
        effective_residual_budget = self.residual_budget * block_scale
        effective_rotation_scale = self.rotation_scale * self.rotation_scale_mult * block_scale
        effective_spectral_scale = self.spectral_scale * self.spectral_scale_mult * block_scale
        effective_low_rank_scale = self.low_rank_scale * self.low_rank_scale_mult * block_scale
        if self.magnus_operator is not None:
            base_out = self.base(x)
            magnus_out = self.magnus_operator(x, context, scale=self.adapter_scale)
            magnus_out, residual_budget_scale = self._apply_residual_budget_with_value(
                base_out,
                magnus_out,
                float(effective_residual_budget.detach().item()),
            )
            base_out_norm = float(base_out.norm(dim=-1).mean().item())
            out = base_out + effective_residual_scale * magnus_out
            self.last_aux_losses = self.magnus_operator.get_aux_losses()
            extra_diag = self.magnus_operator.get_diagnostics()
            self.last_diagnostics = {
                "base_rank": float(self.base_rank or 0),
                "shared_bank": float(self.shared_bank is not None),
                "semantic_manifold_mode": float(self.semantic_manifold_mode),
                "magnus_semantic_mode": float(self.magnus_semantic_mode),
                "magnus_single_operator": float(self.magnus_single_operator),
                "coupled_spectral_low_rank": float(self.coupled_spectral_low_rank),
                "coupled_learnable_input_basis": float(self.coupled_learnable_input_basis),
                "coupled_shared_gate": float(self.coupled_shared_gate),
                "strict_semantic_operator": float(self.strict_semantic_operator),
                "manifold_alignment_mode": float(self.manifold_alignment_mode),
                "local_gate": 0.0,
                "rotation_gate": 0.0,
                "local_residual_norm": 0.0,
                "spectral_residual_norm": 0.0,
                "low_rank_residual_norm": 0.0,
                "coupled_residual_norm": 0.0,
                "semantic_operator_norm": 0.0,
                "rotation_coeff_norm": 0.0,
                "spectral_coeff_norm": 0.0,
                "low_rank_coeff_norm": 0.0,
                "semantic_expert_logit_norm": 0.0,
                "semantic_mode_mean": 0.0,
                "semantic_gate": 0.0,
                "base_out_norm": base_out_norm,
                "residual_budget": float(effective_residual_budget.detach().item()),
                "residual_budget_scale": residual_budget_scale,
                "block_scale": float(block_scale.detach().item()),
                "flow_gate": 0.0,
                "flow_residual_norm": 0.0,
                "flow_coeff_norm": 0.0,
                "adapter_scale": float(self.adapter_scale),
                **extra_diag,
            }
            return out

        local_gate = self.adapter_scale * torch.sigmoid(self.local_gate_head(context))
        rotation_gate = effective_rotation_scale * self.adapter_scale * torch.sigmoid(self.rotation_gate_head(context))
        rotation_coeff = effective_rotation_scale * torch.tanh(self.rotation_coeff_head(context))
        rotation_local_gate = None
        rotation_local_coeff = None
        rotation_comm_scale = None
        strict_semantic = self.semantic_operator is not None and self.strict_semantic_operator

        mixed = x
        local_residual_norm = 0.0
        if not strict_semantic and self.local_mixer is not None and x.dim() == 3:
            local_tokens = self.local_mixer(x)
            local_residual = local_tokens - x
            mixed = x + self._reshape_gate(local_gate, x) * local_residual
            local_residual_norm = float(local_residual.norm(dim=-1).mean().item())
        elif strict_semantic:
            local_gate = torch.zeros_like(local_gate)
            rotation_gate = torch.zeros_like(rotation_gate)
            rotation_coeff = torch.zeros_like(rotation_coeff)

        if self.magnus_rotation_mode and not strict_semantic:
            detail_tokens = summarize_detail_tokens(x, topk=self.magnus_detail_topk).to(dtype=x.dtype)
            detail_hidden = self.rotation_local_proj(detail_tokens)
            local_strength = max(self.magnus_rotation_strength, 0.0)
            rotation_local_gate = (
                effective_rotation_scale
                * local_strength
                * self.adapter_scale
                * torch.sigmoid(self.rotation_local_gate_head(detail_hidden))
            )
            rotation_local_coeff = effective_rotation_scale * local_strength * torch.tanh(self.rotation_local_coeff_head(detail_hidden))
            rotation_comm_scale = (
                effective_rotation_scale
                * local_strength
                * self.adapter_scale
                * torch.sigmoid(self.rotation_comm_head(detail_hidden))
            )

        if strict_semantic:
            rotated = x
        else:
            rotated = mixed if self.manifold_alignment_mode else self.orthogonal_mixer(
                mixed,
                rotation_coeff,
                gate=rotation_gate.squeeze(-1),
                coeff_local=rotation_local_coeff,
                gate_local=None if rotation_local_gate is None else rotation_local_gate.squeeze(-1),
                comm_scale=None if rotation_comm_scale is None else rotation_comm_scale.squeeze(-1),
            )
        base_out = self.base(rotated)

        spectral_coeff = effective_spectral_scale * self.adapter_scale * torch.tanh(self.spectral_gate_head(context))
        if self.low_rank_head is not None:
            low_rank_coeff = effective_low_rank_scale * self.adapter_scale * torch.tanh(self.low_rank_head(context))
        else:
            low_rank_coeff = torch.zeros(
                context.shape[0],
                self.low_rank_rank,
                device=context.device,
                dtype=context.dtype,
            )
        flow_gate = self.adapter_scale * torch.sigmoid(self.flow_gate_head(context)) if self.flow_gate_head is not None else None
        flow_coeff = torch.tanh(self.flow_head(context)) if self.flow_head is not None else None
        if self.manifold_alignment_mode and flow_coeff is not None and spectral_coeff.shape[-1] >= flow_coeff.shape[-1]:
            flow_coeff = spectral_coeff[..., : flow_coeff.shape[-1]]
        elif flow_coeff is not None and spectral_coeff.shape[-1] >= flow_coeff.shape[-1]:
            flow_coeff = 0.5 * (flow_coeff + spectral_coeff[..., : flow_coeff.shape[-1]])

        semantic_operator_norm = 0.0
        flow_residual_norm = 0.0
        flow_coeff_norm = 0.0
        flow_gate_value = 0.0

        if self.semantic_operator is not None:
            semantic_expert_logits = self.semantic_expert_head(context)
            semantic_mode = torch.sigmoid(self.semantic_mode_head(context))
            semantic_gains = 1.0 + 0.1 * torch.tanh(
                self.semantic_gain_head(context).view(context.shape[0], self.semantic_operator.num_experts, self.flow_rank)
            )
            semantic_gate = self.adapter_scale * torch.sigmoid(self.semantic_gate_head(context))
            semantic_out = self.semantic_operator(
                rotated,
                semantic_expert_logits,
                semantic_mode,
                semantic_gains,
            )
            gated_semantic_out = self._reshape_gate(semantic_gate, semantic_out) * semantic_out
            gated_semantic_out, residual_budget_scale = self._apply_residual_budget_with_value(
                base_out,
                gated_semantic_out,
                float(effective_residual_budget.detach().item()),
            )
            semantic_operator_norm = float(gated_semantic_out.norm(dim=-1).mean().item())
            out = base_out + effective_residual_scale * gated_semantic_out
            extra_diag = self.semantic_operator.get_diagnostics()
            self.last_aux_losses = self.semantic_operator.get_aux_losses()
            spectral_residual_norm = 0.0
            low_rank_residual_norm = 0.0
            semantic_expert_logit_norm = float(semantic_expert_logits.norm(dim=-1).mean().item())
            semantic_mode_mean = float(semantic_mode.mean().item())
            semantic_gate_value = float(semantic_gate.mean().item())
        else:
            coupled_out = 0.0
            if self.manifold_alignment_mode:
                spectral_out = 0.0
                low_rank_out = 0.0
            elif self.coupled_operator is not None:
                coupled_out = self.coupled_operator(rotated, spectral_coeff, low_rank_coeff)
                spectral_out = 0.0
                low_rank_out = 0.0
                extra_diag = self.coupled_operator.get_diagnostics()
            else:
                spectral_out = self.spectral_operator(rotated, spectral_coeff)
                low_rank_out = self.low_rank_operator(rotated, low_rank_coeff)
            flow_out = 0.0
            extra_diag = extra_diag if "extra_diag" in locals() else {}
            self.last_aux_losses = {
                "geo_structure_loss": (
                    spectral_coeff.pow(2).mean()
                    + low_rank_coeff.pow(2).mean()
                    + rotation_coeff.pow(2).mean()
                ),
            }
            if self.flow_operator is not None and flow_coeff is not None and flow_gate is not None:
                raw_flow = self.flow_operator(rotated, flow_coeff, context=context)
                flow_out = self._reshape_gate(flow_gate, raw_flow) * raw_flow
                flow_residual_norm = float(raw_flow.norm(dim=-1).mean().item())
                flow_coeff_norm = float(flow_coeff.norm(dim=-1).mean().item())
                flow_gate_value = float(flow_gate.mean().item())
                extra_diag = {**extra_diag, **self.flow_operator.get_diagnostics()}
                self.last_aux_losses.update(self.flow_operator.get_aux_losses())
            combined_residual = coupled_out + spectral_out + low_rank_out + flow_out
            combined_residual, residual_budget_scale = self._apply_residual_budget_with_value(
                base_out,
                combined_residual,
                float(effective_residual_budget.detach().item()),
            )
            out = base_out + effective_residual_scale * combined_residual
            spectral_residual_norm = 0.0 if self.manifold_alignment_mode or self.coupled_operator is not None else float(spectral_out.norm(dim=-1).mean().item())
            low_rank_residual_norm = 0.0 if self.manifold_alignment_mode or self.coupled_operator is not None else float(low_rank_out.norm(dim=-1).mean().item())
            coupled_residual_norm = 0.0 if self.coupled_operator is None else float(coupled_out.norm(dim=-1).mean().item())
            semantic_expert_logit_norm = 0.0
            semantic_mode_mean = 0.0
            semantic_gate_value = 0.0

        base_out_norm = float(base_out.norm(dim=-1).mean().item())
        self.last_diagnostics = {
            "base_rank": float(self.base_rank or 0),
            "shared_bank": float(self.shared_bank is not None),
            "semantic_manifold_mode": float(self.semantic_manifold_mode),
            "magnus_semantic_mode": float(self.magnus_semantic_mode),
            "magnus_single_operator": float(self.magnus_single_operator),
            "coupled_spectral_low_rank": float(self.coupled_spectral_low_rank),
            "coupled_learnable_input_basis": float(self.coupled_learnable_input_basis),
            "coupled_shared_gate": float(self.coupled_shared_gate),
            "magnus_rotation_mode": float(self.magnus_rotation_mode),
            "magnus_rotation_strength": float(self.magnus_rotation_strength),
            "rotation_scale": float(effective_rotation_scale),
            "spectral_scale": float(effective_spectral_scale),
            "low_rank_scale": float(effective_low_rank_scale),
            "rotation_scale_mult": float(self.rotation_scale_mult),
            "spectral_scale_mult": float(self.spectral_scale_mult),
            "low_rank_scale_mult": float(self.low_rank_scale_mult),
            "strict_semantic_operator": float(self.strict_semantic_operator),
            "manifold_alignment_mode": float(self.manifold_alignment_mode),
            "local_gate": float(local_gate.mean().item()),
            "rotation_gate": float(rotation_gate.mean().item()),
            "local_residual_norm": local_residual_norm,
            "spectral_residual_norm": spectral_residual_norm,
            "low_rank_residual_norm": low_rank_residual_norm,
            "coupled_residual_norm": coupled_residual_norm if "coupled_residual_norm" in locals() else 0.0,
            "semantic_operator_norm": semantic_operator_norm,
            "rotation_coeff_norm": float(rotation_coeff.norm(dim=-1).mean().item()),
            "spectral_coeff_norm": float(spectral_coeff.norm(dim=-1).mean().item()),
            "low_rank_coeff_norm": float(low_rank_coeff.norm(dim=-1).mean().item()),
            "semantic_expert_logit_norm": semantic_expert_logit_norm,
            "semantic_mode_mean": semantic_mode_mean,
            "semantic_gate": semantic_gate_value,
            "base_out_norm": base_out_norm,
            "residual_budget": float(effective_residual_budget.detach().item()),
            "residual_budget_scale": residual_budget_scale,
            "block_scale": float(block_scale.detach().item()),
            "flow_gate": flow_gate_value,
            "flow_residual_norm": flow_residual_norm,
            "flow_coeff_norm": flow_coeff_norm,
            "adapter_scale": float(self.adapter_scale),
            **(self.orthogonal_mixer.get_diagnostics() if hasattr(self.orthogonal_mixer, "get_diagnostics") else {}),
            **extra_diag,
        }
        return out

    def get_diagnostics(self) -> dict[str, float]:
        return dict(self.last_diagnostics)

    def get_aux_losses(self) -> dict[str, torch.Tensor]:
        return dict(self.last_aux_losses)
