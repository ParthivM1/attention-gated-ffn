from functools import partial

import torch
import torch.nn as nn

try:
    from layers.operator_bank import SharedGeoOperatorBank
    from layers.text_conditioner import ClassTextRouter, PooledTokenConditioner, TextConditioner, build_text_embeddings
    from layers.geo_operator_linear import GeoOperatorLinear
    from layers.flow_ffn import FlowFFN
    from layers.gradflow_ffn import GradFlowFFN
    from layers.attn_gated_ffn import AttentionGatedFFN
    from models.geovit_block import AttentionFlowModulator, AttentionHiddenFusion, CompetitiveResidualMixer, GeoMlp, GeoViTBlock, HiddenClsAttentionBridge, PatchGridRefiner, SharedActivationFlowGate, SharedAttentionMetricAdapter, SharedBiaxialResponseFlow, SharedHiddenChannelFlow, SparseTopKTokenMixer, TailTokenMixer, TokenResponseFlowNorm
    from models.vit import BudgetedDetailTokenizer, ConvStem, FlowStateCarrier, SharedInterLayerFlowMixer, SharedTokenFlowMixer, SpatialFlowStemRefiner
except ImportError:
    from src.layers.operator_bank import SharedGeoOperatorBank
    from src.layers.text_conditioner import ClassTextRouter, PooledTokenConditioner, TextConditioner, build_text_embeddings
    from src.layers.geo_operator_linear import GeoOperatorLinear
    from src.layers.flow_ffn import FlowFFN
    from src.layers.gradflow_ffn import GradFlowFFN
    from src.layers.attn_gated_ffn import AttentionGatedFFN
    from src.models.geovit_block import AttentionFlowModulator, AttentionHiddenFusion, CompetitiveResidualMixer, GeoMlp, GeoViTBlock, HiddenClsAttentionBridge, PatchGridRefiner, SharedActivationFlowGate, SharedAttentionMetricAdapter, SharedBiaxialResponseFlow, SharedHiddenChannelFlow, SparseTopKTokenMixer, TailTokenMixer, TokenResponseFlowNorm
    from src.models.vit import BudgetedDetailTokenizer, ConvStem, FlowStateCarrier, SharedInterLayerFlowMixer, SharedTokenFlowMixer, SpatialFlowStemRefiner


class GeoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        num_classes: int = 100,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        qkv_bias: bool = False,
        use_conv_stem: bool = True,
        stem_channels: int = 64,
        stem_flow_refiner: bool = False,
        stem_flow_scale: float = 1.0,
        stem_flow_bottleneck: int = 16,
        stem_flow_gate_bias: float = -3.0,
        stem_flow_init_scale: float = 0.02,
        stem_flow_detail_scale: float = 0.25,
        stem_flow_context_scale: float = 0.5,
        summary_token: bool = False,
        summary_token_scale: float = 1.0,
        summary_head_fusion: float = 0.5,
        tokenizer_type: str = "standard",
        detail_tokens: int = 0,
        detail_score_type: str = "variance",
        condition_dim: int = 512,
        controller_hidden_dim: int = 128,
        num_spectral_bases: int = 8,
        low_rank_rank: int = 8,
        orthogonal_rank: int = 8,
        fc1_base_rank: int = 64,
        share_fc1_bank: bool = True,
        fc1_bank_groups: int = 1,
        share_fc1_conditioner: bool = True,
        fc1_conditioner_groups: int = 1,
        geo_on_fc1: bool = True,
        geo_fc1_last_k_blocks: int = 0,
        geo_low_rank_last_k_blocks: int = 0,
        enable_local_geo: bool = True,
        local_geo_last_k_blocks: int = 0,
        geo_on_fc2: bool = False,
        geo_on_attention: bool = False,
        geo_attention_last_k_blocks: int = 0,
        residual_scale: float = 1.0,
        geo_residual_budget: float = 0.0,
        spectral_scale: float = 1.0,
        low_rank_scale: float = 1.0,
        rotation_scale: float = 1.0,
        use_class_text_router: bool = False,
        use_conditioner: bool = True,
        use_internal_conditioner: bool = True,
        class_texts: list[str] | None = None,
        text_router_temperature: float = 1.0,
        text_embedding_source: str = "hashed",
        text_embedding_model: str = "openai/clip-vit-base-patch32",
        flow_rank: int = 0,
        flow_steps: int = 1,
        flow_step_size: float = 0.5,
        semantic_manifold_mode: bool = False,
        semantic_num_experts: int = 2,
        semantic_expert_temperature: float = 0.5,
        magnus_semantic_mode: bool = False,
        magnus_single_operator: bool = False,
        magnus_detail_topk: int = 4,
        magnus_rotation_mode: bool = False,
        magnus_rotation_last_k_blocks: int = 0,
        magnus_rotation_strength: float = 1.0,
        coupled_spectral_low_rank: bool = False,
        coupled_learnable_input_basis: bool = False,
        coupled_shared_gate: bool = False,
        strict_semantic_operator: bool = False,
        manifold_alignment_mode: bool = False,
        hidden_diffusion: bool = False,
        hidden_diffusion_scale: float = 1.0,
        hidden_diffusion_last_k_blocks: int = 0,
        share_hidden_diffusion: bool = False,
        hidden_diffusion_bottleneck: int = 32,
        hidden_diffusion_gate_bias: float = -4.0,
        hidden_diffusion_init_scale: float = 0.005,
        hidden_diffusion_cls_context_scale: float = 0.1,
        hidden_grid_refiner: bool = False,
        hidden_grid_refiner_scale: float = 1.0,
        hidden_grid_refiner_last_k_blocks: int = 0,
        share_hidden_grid_refiner: bool = False,
        hidden_grid_refiner_bottleneck: int = 16,
        hidden_grid_refiner_gate_bias: float = -5.0,
        hidden_grid_refiner_init_scale: float = 0.002,
        hidden_grid_refiner_cls_context_scale: float = 0.05,
        hidden_cls_bridge: bool = False,
        hidden_cls_bridge_scale: float = 1.0,
        hidden_cls_bridge_last_k_blocks: int = 0,
        share_hidden_cls_bridge: bool = False,
        hidden_cls_bridge_bottleneck: int = 16,
        hidden_cls_bridge_gate_bias: float = -4.0,
        hidden_cls_bridge_init_scale: float = 0.01,
        hidden_cls_bridge_patch_feedback_scale: float = 0.0,
        hidden_channel_flow: bool = False,
        hidden_channel_flow_scale: float = 1.0,
        hidden_channel_flow_last_k_blocks: int = 0,
        share_hidden_channel_flow: bool = False,
        hidden_channel_flow_bottleneck: int = 16,
        hidden_channel_flow_rank: int = 16,
        hidden_channel_flow_gate_bias: float = -3.5,
        hidden_channel_flow_init_scale: float = 0.01,
        hidden_channel_flow_patch_only: bool = False,
        hidden_channel_flow_cls_mix_scale: float = 1.0,
        hidden_channel_flow_mean_mix_scale: float = 0.5,
        response_flow_norm: bool = False,
        response_flow_scale: float = 1.0,
        response_flow_last_k_blocks: int = 0,
        response_flow_init_scale: float = 0.01,
        response_flow_mode: str = "simple",
        share_response_flow: bool = False,
        response_flow_bottleneck: int = 12,
        response_flow_gate_bias: float = -4.0,
        response_flow_patch_only: bool = False,
        response_flow_cls_mix_scale: float = 1.0,
        response_flow_mean_mix_scale: float = 0.5,
        response_flow_token_exponent: float = 0.5,
        response_flow_channel_exponent: float = 0.5,
        response_flow_pre_act: bool = False,
        dual_path_mlp: bool = False,
        dual_path_last_k_blocks: int = 0,
        dual_path_refine_ratio: float = 0.25,
        dual_path_cross_scale: float = 1.0,
        dual_path_gate_bias: float = -2.0,
        hidden_group_router: bool = False,
        hidden_group_router_last_k_blocks: int = 0,
        hidden_group_router_groups: int = 0,
        hidden_group_router_scale: float = 1.0,
        hidden_group_router_gate_bias: float = -2.0,
        hidden_group_router_init_scale: float = 0.1,
        hidden_group_router_cls_mix_scale: float = 1.0,
        hidden_group_router_mean_mix_scale: float = 0.5,
        attention_hidden_fusion: bool = False,
        attention_hidden_fusion_last_k_blocks: int = 0,
        share_attention_hidden_fusion: bool = True,
        attention_hidden_fusion_scale: float = 1.0,
        attention_hidden_fusion_bottleneck: int = 6,
        attention_hidden_fusion_gate_bias: float = -2.5,
        attention_hidden_fusion_init_scale: float = 0.02,
        attention_hidden_fusion_patch_only: bool = False,
        attention_hidden_fusion_cls_context_scale: float = 1.0,
        hidden_token_mixer: bool = False,
        hidden_token_mixer_last_k_blocks: int = 0,
        share_hidden_token_mixer: bool = True,
        hidden_token_mixer_scale: float = 1.0,
        hidden_token_mixer_gate_bias: float = -3.0,
        hidden_token_mixer_init_scale: float = 0.02,
        hidden_token_mixer_patch_only: bool = True,
        hidden_token_mixer_mode: str = "conv",
        hidden_token_mixer_topk: int = 8,
        competitive_residual: bool = False,
        competitive_residual_last_k_blocks: int = 0,
        competitive_residual_scale: float = 1.0,
        competitive_residual_gate_bias: float = 0.0,
        competitive_residual_init_scale: float = 1.0,
        competitive_residual_cls_mix_scale: float = 1.0,
        competitive_residual_mean_mix_scale: float = 0.5,
        competitive_residual_patch_only: bool = False,
        parallel_block_update: bool = False,
        parallel_block_last_k_blocks: int = 0,
        mlp_first_update: bool = False,
        mlp_first_last_k_blocks: int = 0,
        tail_token_mixer: bool = False,
        tail_token_mixer_last_k_blocks: int = 0,
        tail_token_mixer_scale: float = 1.0,
        tail_token_mixer_gate_bias: float = -3.0,
        tail_token_mixer_init_scale: float = 0.02,
        tail_token_mixer_patch_only: bool = True,
        activation_flow: bool = False,
        activation_flow_scale: float = 1.0,
        activation_flow_last_k_blocks: int = 0,
        share_activation_flow: bool = False,
        activation_flow_bottleneck: int = 16,
        activation_flow_gate_bias: float = -4.0,
        activation_flow_init_scale: float = 0.01,
        activation_flow_patch_only: bool = False,
        activation_flow_cls_mix_scale: float = 1.0,
        activation_flow_mean_mix_scale: float = 0.5,
        activation_flow_std_mix_scale: float = 0.25,
        activation_flow_cls_token_scale: float = 1.0,
        attn_flow_modulator: bool = False,
        attn_flow_scale: float = 1.0,
        attn_flow_last_k_blocks: int = 0,
        share_attn_flow_modulator: bool = False,
        attn_flow_bottleneck: int = 24,
        attn_flow_gate_bias: float = -2.5,
        attn_flow_init_scale: float = 0.02,
        attn_flow_detail_topk: int = 8,
        attn_flow_patch_only: bool = False,
        patch_grid_refiner: bool = False,
        patch_grid_refiner_scale: float = 1.0,
        patch_grid_refiner_last_k_blocks: int = 0,
        share_patch_grid_refiner: bool = False,
        patch_grid_refiner_bottleneck: int = 16,
        patch_grid_refiner_gate_bias: float = -5.0,
        patch_grid_refiner_init_scale: float = 0.002,
        patch_grid_refiner_cls_context_scale: float = 0.05,
        attention_metric_adapter: bool = False,
        attention_metric_type: str = "linear",
        attention_metric_scale: float = 1.0,
        attention_metric_last_k_blocks: int = 0,
        share_attention_metric: bool = True,
        attention_metric_bottleneck: int = 8,
        attention_metric_patch_only: bool = True,
        attention_metric_gate_bias: float = -3.0,
        attention_metric_init_scale: float = 0.01,
        attention_metric_cls_context_scale: float = 0.25,
        geo_layer_scale_init: float = 0.0,
        geo_block_profile: str = "uniform",
        geo_learnable_block_scale: bool = False,
        token_flow_input: bool = False,
        token_flow_last_k_blocks: int = 0,
        share_token_flow: bool = True,
        token_flow_scale: float = 1.0,
        token_flow_input_scale: float = -1.0,
        token_flow_block_scale: float = -1.0,
        token_flow_bottleneck: int = 32,
        token_flow_patch_only: bool = True,
        token_flow_gate_bias: float = -3.0,
        token_flow_init_scale: float = 0.01,
        token_flow_cls_context_scale: float = 0.25,
        token_flow_detail_topk: int = 0,
        token_flow_detail_boost_scale: float = 0.0,
        inter_layer_flow: bool = False,
        inter_layer_flow_last_k_blocks: int = 0,
        share_inter_layer_flow: bool = True,
        inter_layer_flow_mode: str = "transport",
        inter_layer_flow_scale: float = 1.0,
        inter_layer_flow_bottleneck: int = 16,
        inter_layer_flow_patch_only: bool = True,
        inter_layer_flow_gate_bias: float = -4.0,
        inter_layer_flow_init_scale: float = 0.005,
        inter_layer_flow_cls_context_scale: float = 0.15,
        inter_layer_flow_delta_scale: float = 0.5,
        flow_state_carrier: bool = False,
        flow_state_last_k_blocks: int = 0,
        share_flow_state_carrier: bool = True,
        flow_state_dim: int = 24,
        flow_state_scale: float = 1.0,
        flow_state_gate_bias: float = -5.0,
        flow_state_init_scale: float = 0.0025,
        flow_state_cls_scale: float = 1.0,
        flow_state_patch_scale: float = 0.1,
        fnfl_last_k_blocks: int = 0,
        fnfl_num_steps: int = 2,
        fnfl_rank: int = 64,
        fnfl_num_spectral_bases: int = 8,
        fnfl_low_rank: int = 4,
        fnfl_controller_hidden_dim: int = 128,
        fnfl_strength_init: float = 0.0,
        agff_last_k_blocks: int = 0,
        agff_gate_mode: str = "attn",
        agff_gate_ln: bool = True,
        agff_gate_init_scale: float = -1.0,
        agff_hidden_scale: float = 8.0 / 3.0,  # hidden = round(D * scale); default = param parity
        gfn_last_k_blocks: int = 0,
        gfn_corr_bottleneck: int = 32,
        gfn_n_train_iters: int = 1,
        gfn_gate_init: float = -4.0,
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.tokenizer_type = tokenizer_type
        self.detail_tokens = int(detail_tokens)
        self.stem_flow_refiner = bool(stem_flow_refiner and use_conv_stem and tokenizer_type == "standard")
        self.stem_flow_scale = float(stem_flow_scale)
        self.summary_token = bool(summary_token and tokenizer_type == "standard")
        self.summary_token_scale = float(summary_token_scale)
        self.summary_head_fusion = float(summary_head_fusion)
        self.fc1_bank_groups = max(int(fc1_bank_groups), 1)
        self.fc1_conditioner_groups = max(int(fc1_conditioner_groups), 1)
        self.geo_fc1_last_k_blocks = int(geo_fc1_last_k_blocks)
        self.geo_low_rank_last_k_blocks = int(geo_low_rank_last_k_blocks)
        self.local_geo_last_k_blocks = int(local_geo_last_k_blocks)
        self.geo_block_profile = str(geo_block_profile).strip().lower() or "uniform"
        self.geo_learnable_block_scale = bool(geo_learnable_block_scale)
        self.token_flow_input = bool(token_flow_input)
        self.token_flow_last_k_blocks = int(token_flow_last_k_blocks)
        self.share_token_flow = bool(share_token_flow)
        self.token_flow_scale = float(token_flow_scale)
        self.token_flow_input_scale = float(token_flow_scale if float(token_flow_input_scale) < 0.0 else token_flow_input_scale)
        self.token_flow_block_scale = float(token_flow_scale if float(token_flow_block_scale) < 0.0 else token_flow_block_scale)
        self.inter_layer_flow = bool(inter_layer_flow)
        self.inter_layer_flow_last_k_blocks = int(inter_layer_flow_last_k_blocks)
        self.share_inter_layer_flow = bool(share_inter_layer_flow)
        self.inter_layer_flow_mode = str(inter_layer_flow_mode).strip().lower() or "transport"
        self.inter_layer_flow_scale = float(inter_layer_flow_scale)
        self.flow_state_carrier = bool(flow_state_carrier)
        self.flow_state_last_k_blocks = int(flow_state_last_k_blocks)
        self.share_flow_state_carrier = bool(share_flow_state_carrier)
        self.flow_state_scale = float(flow_state_scale)
        self.hidden_grid_refiner = bool(hidden_grid_refiner)
        self.hidden_grid_refiner_last_k_blocks = int(hidden_grid_refiner_last_k_blocks)
        self.share_hidden_grid_refiner = bool(share_hidden_grid_refiner)
        self.hidden_cls_bridge = bool(hidden_cls_bridge)
        self.hidden_cls_bridge_last_k_blocks = int(hidden_cls_bridge_last_k_blocks)
        self.share_hidden_cls_bridge = bool(share_hidden_cls_bridge)
        self.hidden_channel_flow = bool(hidden_channel_flow)
        self.hidden_channel_flow_last_k_blocks = int(hidden_channel_flow_last_k_blocks)
        self.share_hidden_channel_flow = bool(share_hidden_channel_flow)
        self.response_flow_norm = bool(response_flow_norm)
        self.response_flow_last_k_blocks = int(response_flow_last_k_blocks)
        self.response_flow_mode = str(response_flow_mode).strip().lower() or "simple"
        self.share_response_flow = bool(share_response_flow)
        self.response_flow_pre_act = bool(response_flow_pre_act)
        self.dual_path_mlp = bool(dual_path_mlp)
        self.dual_path_last_k_blocks = int(dual_path_last_k_blocks)
        self.dual_path_refine_ratio = float(dual_path_refine_ratio)
        self.dual_path_cross_scale = float(dual_path_cross_scale)
        self.dual_path_gate_bias = float(dual_path_gate_bias)
        self.hidden_group_router = bool(hidden_group_router)
        self.hidden_group_router_last_k_blocks = int(hidden_group_router_last_k_blocks)
        self.hidden_group_router_groups = int(hidden_group_router_groups)
        self.hidden_group_router_scale = float(hidden_group_router_scale)
        self.hidden_group_router_gate_bias = float(hidden_group_router_gate_bias)
        self.hidden_group_router_init_scale = float(hidden_group_router_init_scale)
        self.hidden_group_router_cls_mix_scale = float(hidden_group_router_cls_mix_scale)
        self.hidden_group_router_mean_mix_scale = float(hidden_group_router_mean_mix_scale)
        self.attention_hidden_fusion = bool(attention_hidden_fusion)
        self.attention_hidden_fusion_last_k_blocks = int(attention_hidden_fusion_last_k_blocks)
        self.share_attention_hidden_fusion = bool(share_attention_hidden_fusion)
        self.attention_hidden_fusion_scale = float(attention_hidden_fusion_scale)
        self.attention_hidden_fusion_bottleneck = int(attention_hidden_fusion_bottleneck)
        self.attention_hidden_fusion_gate_bias = float(attention_hidden_fusion_gate_bias)
        self.attention_hidden_fusion_init_scale = float(attention_hidden_fusion_init_scale)
        self.attention_hidden_fusion_patch_only = bool(attention_hidden_fusion_patch_only)
        self.attention_hidden_fusion_cls_context_scale = float(attention_hidden_fusion_cls_context_scale)
        self.hidden_token_mixer = bool(hidden_token_mixer)
        self.hidden_token_mixer_last_k_blocks = int(hidden_token_mixer_last_k_blocks)
        self.share_hidden_token_mixer = bool(share_hidden_token_mixer)
        self.hidden_token_mixer_scale = float(hidden_token_mixer_scale)
        self.hidden_token_mixer_gate_bias = float(hidden_token_mixer_gate_bias)
        self.hidden_token_mixer_init_scale = float(hidden_token_mixer_init_scale)
        self.hidden_token_mixer_patch_only = bool(hidden_token_mixer_patch_only)
        self.hidden_token_mixer_mode = str(hidden_token_mixer_mode).strip().lower() or "conv"
        self.hidden_token_mixer_topk = int(hidden_token_mixer_topk)
        self.competitive_residual = bool(competitive_residual)
        self.competitive_residual_last_k_blocks = int(competitive_residual_last_k_blocks)
        self.competitive_residual_scale = float(competitive_residual_scale)
        self.competitive_residual_gate_bias = float(competitive_residual_gate_bias)
        self.competitive_residual_init_scale = float(competitive_residual_init_scale)
        self.competitive_residual_cls_mix_scale = float(competitive_residual_cls_mix_scale)
        self.competitive_residual_mean_mix_scale = float(competitive_residual_mean_mix_scale)
        self.competitive_residual_patch_only = bool(competitive_residual_patch_only)
        self.parallel_block_update = bool(parallel_block_update)
        self.parallel_block_last_k_blocks = int(parallel_block_last_k_blocks)
        self.mlp_first_update = bool(mlp_first_update)
        self.mlp_first_last_k_blocks = int(mlp_first_last_k_blocks)
        self.tail_token_mixer = bool(tail_token_mixer)
        self.tail_token_mixer_last_k_blocks = int(tail_token_mixer_last_k_blocks)
        self.tail_token_mixer_scale = float(tail_token_mixer_scale)
        self.tail_token_mixer_gate_bias = float(tail_token_mixer_gate_bias)
        self.tail_token_mixer_init_scale = float(tail_token_mixer_init_scale)
        self.tail_token_mixer_patch_only = bool(tail_token_mixer_patch_only)
        self.activation_flow = bool(activation_flow)
        self.activation_flow_last_k_blocks = int(activation_flow_last_k_blocks)
        self.share_activation_flow = bool(share_activation_flow)
        self.attn_flow_modulator = bool(attn_flow_modulator)
        self.attn_flow_last_k_blocks = int(attn_flow_last_k_blocks)
        self.share_attn_flow_modulator = bool(share_attn_flow_modulator)
        self.attention_metric_last_k_blocks = int(attention_metric_last_k_blocks)
        self.share_attention_metric = bool(share_attention_metric)
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
            self.patch_embed = None
            self.pos_embed = None
        else:
            stem_refiner = (
                SpatialFlowStemRefiner(
                    stem_channels,
                    bottleneck_dim=stem_flow_bottleneck,
                    gate_bias=stem_flow_gate_bias,
                    init_scale=stem_flow_init_scale,
                    detail_scale=stem_flow_detail_scale,
                    context_scale=stem_flow_context_scale,
                )
                if self.stem_flow_refiner
                else None
            )
            self.patch_embed = (
                ConvStem(
                    embed_dim=embed_dim,
                    patch_size=patch_size,
                    stem_channels=stem_channels,
                    refiner=stem_refiner,
                    refiner_scale=self.stem_flow_scale,
                )
                if use_conv_stem
                else nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
            )
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
            if self.summary_token:
                self.summary_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
                self.summary_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
                self.summary_norm = nn.LayerNorm(embed_dim)
            else:
                self.summary_pos_embed = None
                self.summary_type_embed = None
                self.summary_norm = None
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)
        has_token_flow = bool(self.token_flow_input or self.token_flow_last_k_blocks > 0)
        self.token_flow = None
        self.input_token_flow = None
        self.block_token_flows = None
        if has_token_flow and self.share_token_flow:
            self.token_flow = SharedTokenFlowMixer(
                embed_dim,
                bottleneck_dim=token_flow_bottleneck,
                patch_only=token_flow_patch_only,
                gate_bias=token_flow_gate_bias,
                init_scale=token_flow_init_scale,
                cls_context_scale=token_flow_cls_context_scale,
                detail_topk=token_flow_detail_topk,
                detail_boost_scale=token_flow_detail_boost_scale,
            )
        else:
            if self.token_flow_input:
                self.input_token_flow = SharedTokenFlowMixer(
                    embed_dim,
                    bottleneck_dim=token_flow_bottleneck,
                    patch_only=token_flow_patch_only,
                    gate_bias=token_flow_gate_bias,
                    init_scale=token_flow_init_scale,
                    cls_context_scale=token_flow_cls_context_scale,
                    detail_topk=token_flow_detail_topk,
                    detail_boost_scale=token_flow_detail_boost_scale,
                )
            if self.token_flow_last_k_blocks > 0:
                self.block_token_flows = nn.ModuleList(
                    [
                        SharedTokenFlowMixer(
                            embed_dim,
                            bottleneck_dim=token_flow_bottleneck,
                            patch_only=token_flow_patch_only,
                            gate_bias=token_flow_gate_bias,
                            init_scale=token_flow_init_scale,
                            cls_context_scale=token_flow_cls_context_scale,
                            detail_topk=token_flow_detail_topk,
                            detail_boost_scale=token_flow_detail_boost_scale,
                        )
                        for _ in range(self.token_flow_last_k_blocks)
                    ]
                )
        has_inter_layer_flow = bool(self.inter_layer_flow and self.inter_layer_flow_last_k_blocks > 0)
        self.shared_inter_layer_flow = None
        self.block_inter_layer_flows = None
        if has_inter_layer_flow and self.share_inter_layer_flow:
            self.shared_inter_layer_flow = SharedInterLayerFlowMixer(
                embed_dim,
                bottleneck_dim=inter_layer_flow_bottleneck,
                mode=self.inter_layer_flow_mode,
                patch_only=inter_layer_flow_patch_only,
                gate_bias=inter_layer_flow_gate_bias,
                init_scale=inter_layer_flow_init_scale,
                cls_context_scale=inter_layer_flow_cls_context_scale,
                delta_scale=inter_layer_flow_delta_scale,
            )
        elif has_inter_layer_flow:
            self.block_inter_layer_flows = nn.ModuleList(
                [
                    SharedInterLayerFlowMixer(
                        embed_dim,
                        bottleneck_dim=inter_layer_flow_bottleneck,
                        mode=self.inter_layer_flow_mode,
                        patch_only=inter_layer_flow_patch_only,
                        gate_bias=inter_layer_flow_gate_bias,
                        init_scale=inter_layer_flow_init_scale,
                        cls_context_scale=inter_layer_flow_cls_context_scale,
                        delta_scale=inter_layer_flow_delta_scale,
                    )
                    for _ in range(self.inter_layer_flow_last_k_blocks)
                ]
            )
        has_flow_state_carrier = bool(self.flow_state_carrier and self.flow_state_last_k_blocks > 0)
        self.shared_flow_state_carrier = None
        self.block_flow_state_carriers = None
        if has_flow_state_carrier and self.share_flow_state_carrier:
            self.shared_flow_state_carrier = FlowStateCarrier(
                embed_dim,
                state_dim=flow_state_dim,
                gate_bias=flow_state_gate_bias,
                init_scale=flow_state_init_scale,
                cls_scale=flow_state_cls_scale,
                patch_scale=flow_state_patch_scale,
            )
        elif has_flow_state_carrier:
            self.block_flow_state_carriers = nn.ModuleList(
                [
                    FlowStateCarrier(
                        embed_dim,
                        state_dim=flow_state_dim,
                        gate_bias=flow_state_gate_bias,
                        init_scale=flow_state_init_scale,
                        cls_scale=flow_state_cls_scale,
                        patch_scale=flow_state_patch_scale,
                    )
                    for _ in range(self.flow_state_last_k_blocks)
                ]
            )
        hidden_dim = int(embed_dim * mlp_ratio)
        shared_hidden_diffusion = None
        if hidden_diffusion and share_hidden_diffusion:
            shared_hidden_diffusion = SharedTokenFlowMixer(
                hidden_dim,
                bottleneck_dim=hidden_diffusion_bottleneck,
                patch_only=True,
                gate_bias=hidden_diffusion_gate_bias,
                init_scale=hidden_diffusion_init_scale,
                cls_context_scale=hidden_diffusion_cls_context_scale,
            )
        shared_hidden_grid_refiner = None
        if hidden_grid_refiner and share_hidden_grid_refiner:
            shared_hidden_grid_refiner = PatchGridRefiner(
                hidden_dim,
                bottleneck_dim=hidden_grid_refiner_bottleneck,
                gate_bias=hidden_grid_refiner_gate_bias,
                init_scale=hidden_grid_refiner_init_scale,
                cls_context_scale=hidden_grid_refiner_cls_context_scale,
            )
        shared_hidden_cls_bridge = None
        if hidden_cls_bridge and share_hidden_cls_bridge:
            shared_hidden_cls_bridge = HiddenClsAttentionBridge(
                hidden_dim,
                bottleneck_dim=hidden_cls_bridge_bottleneck,
                gate_bias=hidden_cls_bridge_gate_bias,
                init_scale=hidden_cls_bridge_init_scale,
                patch_feedback_scale=hidden_cls_bridge_patch_feedback_scale,
            )
        shared_hidden_channel_flow = None
        if hidden_channel_flow and share_hidden_channel_flow:
            shared_hidden_channel_flow = SharedHiddenChannelFlow(
                hidden_dim,
                bottleneck_dim=hidden_channel_flow_bottleneck,
                rank=hidden_channel_flow_rank,
                patch_only=hidden_channel_flow_patch_only,
                gate_bias=hidden_channel_flow_gate_bias,
                init_scale=hidden_channel_flow_init_scale,
                cls_mix_scale=hidden_channel_flow_cls_mix_scale,
                mean_mix_scale=hidden_channel_flow_mean_mix_scale,
            )
        def make_hidden_token_mixer_module() -> nn.Module:
            if self.hidden_token_mixer_mode == "sparse":
                return SparseTopKTokenMixer(
                    hidden_dim,
                    topk=self.hidden_token_mixer_topk,
                    gate_bias=self.hidden_token_mixer_gate_bias,
                    init_scale=self.hidden_token_mixer_init_scale,
                    patch_only=self.hidden_token_mixer_patch_only,
                )
            return TailTokenMixer(
                hidden_dim,
                gate_bias=self.hidden_token_mixer_gate_bias,
                init_scale=self.hidden_token_mixer_init_scale,
                patch_only=self.hidden_token_mixer_patch_only,
            )

        shared_hidden_token_mixer = None
        if self.hidden_token_mixer and self.share_hidden_token_mixer:
            shared_hidden_token_mixer = make_hidden_token_mixer_module()
        shared_attention_hidden_fusion = None
        if self.attention_hidden_fusion and self.share_attention_hidden_fusion:
            shared_attention_hidden_fusion = AttentionHiddenFusion(
                hidden_dim,
                embed_dim,
                bottleneck_dim=self.attention_hidden_fusion_bottleneck,
                gate_bias=self.attention_hidden_fusion_gate_bias,
                init_scale=self.attention_hidden_fusion_init_scale,
                patch_only=self.attention_hidden_fusion_patch_only,
                cls_context_scale=self.attention_hidden_fusion_cls_context_scale,
            )
        shared_response_flow = None
        if response_flow_norm and share_response_flow:
            if self.response_flow_mode == "biaxial":
                shared_response_flow = SharedBiaxialResponseFlow(
                    hidden_dim,
                    bottleneck_dim=response_flow_bottleneck,
                    patch_only=response_flow_patch_only,
                    gate_bias=response_flow_gate_bias,
                    init_scale=response_flow_init_scale,
                    cls_mix_scale=response_flow_cls_mix_scale,
                    mean_mix_scale=response_flow_mean_mix_scale,
                    token_exponent=response_flow_token_exponent,
                    channel_exponent=response_flow_channel_exponent,
                )
            else:
                shared_response_flow = TokenResponseFlowNorm(hidden_dim, init_scale=response_flow_init_scale)
        shared_activation_flow = None
        if activation_flow and share_activation_flow:
            shared_activation_flow = SharedActivationFlowGate(
                hidden_dim,
                bottleneck_dim=activation_flow_bottleneck,
                patch_only=activation_flow_patch_only,
                gate_bias=activation_flow_gate_bias,
                init_scale=activation_flow_init_scale,
                cls_mix_scale=activation_flow_cls_mix_scale,
                mean_mix_scale=activation_flow_mean_mix_scale,
                std_mix_scale=activation_flow_std_mix_scale,
                cls_token_scale=activation_flow_cls_token_scale,
            )
        shared_attn_flow_modulator = None
        if attn_flow_modulator and share_attn_flow_modulator:
            shared_attn_flow_modulator = AttentionFlowModulator(
                embed_dim,
                bottleneck_dim=attn_flow_bottleneck,
                gate_bias=attn_flow_gate_bias,
                init_scale=attn_flow_init_scale,
                detail_topk=attn_flow_detail_topk,
                patch_only=attn_flow_patch_only,
            )
        shared_patch_grid_refiner = None
        if patch_grid_refiner and share_patch_grid_refiner:
            shared_patch_grid_refiner = PatchGridRefiner(
                embed_dim,
                bottleneck_dim=patch_grid_refiner_bottleneck,
                gate_bias=patch_grid_refiner_gate_bias,
                init_scale=patch_grid_refiner_init_scale,
                cls_context_scale=patch_grid_refiner_cls_context_scale,
            )
        shared_attention_metric = None
        attention_metric_type = str(attention_metric_type).strip().lower()

        def make_attention_metric_module() -> nn.Module:
            if attention_metric_type == "grid":
                return PatchGridRefiner(
                    embed_dim,
                    bottleneck_dim=attention_metric_bottleneck,
                    gate_bias=attention_metric_gate_bias,
                    init_scale=attention_metric_init_scale,
                    cls_context_scale=attention_metric_cls_context_scale,
                )
            return SharedAttentionMetricAdapter(
                embed_dim,
                bottleneck_dim=attention_metric_bottleneck,
                patch_only=attention_metric_patch_only,
                gate_bias=attention_metric_gate_bias,
                init_scale=attention_metric_init_scale,
                cls_context_scale=attention_metric_cls_context_scale,
            )

        if attention_metric_adapter and share_attention_metric:
            shared_attention_metric = make_attention_metric_module()
        enable_local_geo = bool(enable_local_geo and tokenizer_type == "standard")
        self.enable_local_geo = enable_local_geo
        self.use_conditioner = bool(use_conditioner)
        self.use_internal_conditioner = bool(use_internal_conditioner)
        use_geo_fc1_any = bool(geo_on_fc1 and (self.geo_fc1_last_k_blocks <= 0 or depth > 0))

        partial_magnus_rotation = bool(magnus_rotation_mode and int(magnus_rotation_last_k_blocks) > 0)
        shared_fc1_banks = None
        if use_geo_fc1_any and share_fc1_bank and not partial_magnus_rotation and not magnus_single_operator:
            shared_fc1_banks = nn.ModuleList(
                [
                    SharedGeoOperatorBank(
                        in_features=embed_dim,
                        out_features=hidden_dim,
                        num_spectral_bases=num_spectral_bases,
                        low_rank_rank=low_rank_rank,
                        orthogonal_rank=orthogonal_rank,
                        flow_rank=flow_rank,
                        controller_hidden_dim=controller_hidden_dim,
                        flow_steps=flow_steps,
                        flow_step_size=flow_step_size,
                        semantic_manifold_mode=semantic_manifold_mode,
                        semantic_num_experts=semantic_num_experts,
                        semantic_expert_temperature=semantic_expert_temperature,
                        magnus_semantic_mode=magnus_semantic_mode,
                        magnus_detail_topk=magnus_detail_topk,
                        magnus_rotation_mode=magnus_rotation_mode,
                        coupled_spectral_low_rank=coupled_spectral_low_rank,
                        coupled_learnable_input_basis=coupled_learnable_input_basis,
                        coupled_shared_gate=coupled_shared_gate,
                        use_local_operator=enable_local_geo,
                    )
                    for _ in range(self.fc1_bank_groups)
                ]
            )
        shared_fc1_conditioners = None
        if use_geo_fc1_any and share_fc1_conditioner and not magnus_single_operator:
            shared_fc1_conditioners = nn.ModuleList(
                [
                    (
                        TextConditioner(
                            token_dim=embed_dim,
                            condition_dim=condition_dim,
                            hidden_dim=controller_hidden_dim,
                        )
                        if (self.use_conditioner and self.use_internal_conditioner)
                        else PooledTokenConditioner(
                            token_dim=embed_dim,
                            hidden_dim=controller_hidden_dim,
                        )
                    )
                    for _ in range(self.fc1_conditioner_groups)
                ]
            )

        geo_mlp_layer = partial(
            GeoOperatorLinear,
            condition_dim=condition_dim,
            controller_hidden_dim=controller_hidden_dim,
            num_spectral_bases=num_spectral_bases,
            low_rank_rank=low_rank_rank,
            orthogonal_rank=orthogonal_rank,
            flow_rank=flow_rank,
            flow_steps=flow_steps,
            flow_step_size=flow_step_size,
            residual_scale=residual_scale,
            residual_budget=geo_residual_budget,
            spectral_scale=spectral_scale,
            low_rank_scale=low_rank_scale,
            rotation_scale=rotation_scale,
            semantic_manifold_mode=semantic_manifold_mode,
            semantic_num_experts=semantic_num_experts,
            semantic_expert_temperature=semantic_expert_temperature,
            magnus_semantic_mode=magnus_semantic_mode,
            magnus_single_operator=magnus_single_operator,
            magnus_detail_topk=magnus_detail_topk,
            magnus_rotation_mode=False,
            magnus_rotation_strength=magnus_rotation_strength,
            coupled_spectral_low_rank=coupled_spectral_low_rank,
            coupled_learnable_input_basis=coupled_learnable_input_basis,
            coupled_shared_gate=coupled_shared_gate,
            strict_semantic_operator=strict_semantic_operator,
            manifold_alignment_mode=manifold_alignment_mode,
        )
        geo_attn_layer = partial(
            GeoOperatorLinear,
            condition_dim=condition_dim,
            controller_hidden_dim=controller_hidden_dim,
            num_spectral_bases=max(4, num_spectral_bases // 2),
            low_rank_rank=low_rank_rank,
            orthogonal_rank=orthogonal_rank,
            residual_scale=residual_scale,
            use_local_operator=False,
        )
        fc2_factory = partial(geo_mlp_layer, use_local_operator=False) if geo_on_fc2 else nn.Linear
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        geo_fc1_active_indices = [
            idx
            for idx in range(depth)
            if bool(
                geo_on_fc1
                and (
                    self.geo_fc1_last_k_blocks <= 0
                    or idx >= max(depth - self.geo_fc1_last_k_blocks, 0)
                )
            )
        ]
        geo_fc1_active_pos = {block_idx: pos for pos, block_idx in enumerate(geo_fc1_active_indices)}

        def shared_module_for_block(module_list: nn.ModuleList | None, idx: int, groups: int) -> nn.Module | None:
            if module_list is None:
                return None
            group_idx = min(groups - 1, (idx * groups) // max(depth, 1))
            return module_list[group_idx]

        def shared_geo_fc1_module_for_block(module_list: nn.ModuleList | None, idx: int, groups: int) -> nn.Module | None:
            if module_list is None:
                return None
            active_pos = geo_fc1_active_pos.get(idx, None)
            if active_pos is None:
                return None
            active_total = max(len(geo_fc1_active_indices), 1)
            group_idx = min(groups - 1, (active_pos * groups) // active_total)
            return module_list[group_idx]

        def profile_multiplier(profile: str, pos: float) -> float:
            if profile == "late_ramp":
                return 0.35 + 0.65 * pos
            if profile == "late_heavy":
                return 0.15 + 0.85 * (pos * pos)
            if profile == "mid_peak":
                return 0.35 + 0.65 * (1.0 - abs(2.0 * pos - 1.0))
            if profile == "sandwich":
                edge = abs(2.0 * pos - 1.0)
                return 0.35 + 0.65 * edge
            return 1.0

        def geo_block_multiplier(idx: int) -> float:
            if depth <= 1:
                return 1.0
            profile = self.geo_block_profile
            use_active_tail = profile.startswith("active_")
            base_profile = profile[len("active_"):] if use_active_tail else profile
            if use_active_tail:
                active_pos = geo_fc1_active_pos.get(idx, None)
                active_total = len(geo_fc1_active_indices)
                if active_pos is None or active_total <= 1:
                    pos = float(idx) / float(max(depth - 1, 1))
                else:
                    pos = float(active_pos) / float(max(active_total - 1, 1))
            else:
                pos = float(idx) / float(max(depth - 1, 1))
            return profile_multiplier(base_profile, pos)

        def fc1_factory_for_block(idx: int):
            use_geo_fc1_block = bool(
                geo_on_fc1
                and (
                    self.geo_fc1_last_k_blocks <= 0
                    or idx >= max(depth - self.geo_fc1_last_k_blocks, 0)
                )
            )
            if not use_geo_fc1_block:
                return nn.Linear
            if self.local_geo_last_k_blocks > 0:
                use_local_geo_block = bool(idx >= max(depth - self.local_geo_last_k_blocks, 0))
            else:
                use_local_geo_block = bool(enable_local_geo)
            low_rank_scale_block = float(low_rank_scale)
            if self.geo_low_rank_last_k_blocks > 0:
                active_pos = geo_fc1_active_pos.get(idx, None)
                active_total = len(geo_fc1_active_indices)
                low_rank_start = max(active_total - self.geo_low_rank_last_k_blocks, 0)
                if active_pos is None or active_pos < low_rank_start:
                    low_rank_scale_block = 0.0
            block_magnus_rotation = bool(
                magnus_rotation_mode
                and (
                    int(magnus_rotation_last_k_blocks) <= 0
                    or idx >= max(depth - int(magnus_rotation_last_k_blocks), 0)
                )
            )
            block_scale = geo_block_multiplier(idx)
            return partial(
                geo_mlp_layer,
                use_local_operator=use_local_geo_block,
                base_rank=fc1_base_rank,
                use_conditioner=self.use_conditioner,
                use_internal_conditioner=self.use_internal_conditioner,
                shared_bank=shared_geo_fc1_module_for_block(shared_fc1_banks, idx, self.fc1_bank_groups),
                shared_conditioner=shared_geo_fc1_module_for_block(shared_fc1_conditioners, idx, self.fc1_conditioner_groups),
                low_rank_scale=low_rank_scale_block,
                block_scale_init=block_scale,
                learnable_block_scale=self.geo_learnable_block_scale,
                magnus_rotation_mode=block_magnus_rotation,
                magnus_rotation_strength=magnus_rotation_strength,
            )

        self.fnfl_last_k_blocks = int(fnfl_last_k_blocks)
        self.fnfl_num_steps = max(int(fnfl_num_steps), 1)
        self.fnfl_rank = max(int(fnfl_rank), 1)
        self.fnfl_num_spectral_bases = max(int(fnfl_num_spectral_bases), 1)
        self.fnfl_low_rank = max(int(fnfl_low_rank), 1)
        self.fnfl_controller_hidden_dim = max(int(fnfl_controller_hidden_dim), 1)
        self.fnfl_strength_init = float(fnfl_strength_init)
        self.agff_last_k_blocks = int(agff_last_k_blocks)
        self.agff_gate_mode = str(agff_gate_mode)
        self.agff_gate_ln = bool(agff_gate_ln)
        self.agff_gate_init_scale = float(agff_gate_init_scale)
        self.agff_hidden_scale = float(agff_hidden_scale)
        self.gfn_last_k_blocks = int(gfn_last_k_blocks)
        self.gfn_corr_bottleneck = max(int(gfn_corr_bottleneck), 8)
        self.gfn_n_train_iters = max(int(gfn_n_train_iters), 1)
        self.gfn_gate_init = float(gfn_gate_init)

        def fnfl_factory_for_block(idx: int):
            if self.fnfl_last_k_blocks <= 0:
                return None
            if idx < max(depth - self.fnfl_last_k_blocks, 0):
                return None
            return FlowFFN(
                in_features=embed_dim,
                hidden_features=hidden_dim,
                out_features=embed_dim,
                flow_rank=self.fnfl_rank,
                num_steps=self.fnfl_num_steps,
                num_spectral_bases=self.fnfl_num_spectral_bases,
                low_rank=self.fnfl_low_rank,
                controller_hidden_dim=self.fnfl_controller_hidden_dim,
                flow_strength_init=self.fnfl_strength_init,
            )

        def gfn_factory_for_block(idx: int):
            if self.gfn_last_k_blocks <= 0:
                return None
            if idx < max(depth - self.gfn_last_k_blocks, 0):
                return None
            return GradFlowFFN(
                in_features=embed_dim,
                hidden_features=hidden_dim,
                out_features=embed_dim,
                corr_bottleneck=self.gfn_corr_bottleneck,
                n_test_iters=self.gfn_n_train_iters,
                gate_init=self.gfn_gate_init,
            )

        def agff_factory_for_block(idx: int):
            if self.agff_last_k_blocks <= 0:
                return None
            if idx < max(depth - self.agff_last_k_blocks, 0):
                return None
            agff_hidden = int(round(embed_dim * self.agff_hidden_scale))
            return AttentionGatedFFN(
                in_features=embed_dim,
                hidden_features=agff_hidden,
                out_features=embed_dim,
                gate_mode=self.agff_gate_mode,
                gate_ln=self.agff_gate_ln,
                gate_init_scale=self.agff_gate_init_scale,
            )

        def mlp_override_for_block(idx: int):
            return agff_factory_for_block(idx) or gfn_factory_for_block(idx) or fnfl_factory_for_block(idx)

        def attn_factory_for_block(idx: int):
            use_geo_attention = bool(
                geo_on_attention
                or (
                    int(geo_attention_last_k_blocks) > 0
                    and idx >= max(depth - int(geo_attention_last_k_blocks), 0)
                )
            )
            return geo_attn_layer if use_geo_attention else nn.Linear

        self.blocks = nn.ModuleList(
            [
                GeoViTBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[idx],
                    qkv_layer=attn_factory_for_block(idx),
                    proj_layer=attn_factory_for_block(idx),
                    fc1_layer=fc1_factory_for_block(idx),
                    fc2_layer=fc2_factory,
                    dual_path_hidden=bool(
                        self.dual_path_mlp
                        and (
                            self.dual_path_last_k_blocks <= 0
                            or idx >= max(depth - self.dual_path_last_k_blocks, 0)
                        )
                    ),
                    dual_path_refine_ratio=self.dual_path_refine_ratio,
                    dual_path_cross_scale=self.dual_path_cross_scale,
                    dual_path_gate_bias=self.dual_path_gate_bias,
                    hidden_group_router=(
                        GeoMlp.GroupCompetitionRouter(
                            hidden_dim,
                            num_groups=(
                                self.hidden_group_router_groups
                                if self.hidden_group_router_groups > 0
                                else max(self.fc1_bank_groups, 1)
                            ),
                            gate_bias=self.hidden_group_router_gate_bias,
                            init_scale=self.hidden_group_router_init_scale,
                            cls_mix_scale=self.hidden_group_router_cls_mix_scale,
                            mean_mix_scale=self.hidden_group_router_mean_mix_scale,
                        )
                        if (
                            self.hidden_group_router
                            and (
                                self.hidden_group_router_last_k_blocks <= 0
                                or idx >= max(depth - self.hidden_group_router_last_k_blocks, 0)
                            )
                        )
                        else None
                    ),
                    hidden_group_router_scale=self.hidden_group_router_scale,
                    attention_hidden_fusion=(
                        shared_attention_hidden_fusion
                        if (
                            self.attention_hidden_fusion
                            and shared_attention_hidden_fusion is not None
                            and (
                                self.attention_hidden_fusion_last_k_blocks <= 0
                                or idx >= max(depth - self.attention_hidden_fusion_last_k_blocks, 0)
                            )
                        )
                        else (
                            AttentionHiddenFusion(
                                hidden_dim,
                                embed_dim,
                                bottleneck_dim=self.attention_hidden_fusion_bottleneck,
                                gate_bias=self.attention_hidden_fusion_gate_bias,
                                init_scale=self.attention_hidden_fusion_init_scale,
                                patch_only=self.attention_hidden_fusion_patch_only,
                                cls_context_scale=self.attention_hidden_fusion_cls_context_scale,
                            )
                            if (
                                self.attention_hidden_fusion
                                and (
                                    self.attention_hidden_fusion_last_k_blocks <= 0
                                    or idx >= max(depth - self.attention_hidden_fusion_last_k_blocks, 0)
                                )
                                and shared_attention_hidden_fusion is None
                            )
                            else None
                        )
                    ),
                    attention_hidden_fusion_scale=self.attention_hidden_fusion_scale,
                    hidden_token_mixer=(
                        shared_hidden_token_mixer
                        if (
                            self.hidden_token_mixer
                            and shared_hidden_token_mixer is not None
                            and (
                                self.hidden_token_mixer_last_k_blocks <= 0
                                or idx >= max(depth - self.hidden_token_mixer_last_k_blocks, 0)
                            )
                        )
                        else make_hidden_token_mixer_module()
                        if (
                            self.hidden_token_mixer
                            and shared_hidden_token_mixer is None
                            and (
                                self.hidden_token_mixer_last_k_blocks <= 0
                                or idx >= max(depth - self.hidden_token_mixer_last_k_blocks, 0)
                            )
                        )
                        else None
                    ),
                    hidden_token_mixer_scale=self.hidden_token_mixer_scale,
                    competitive_residual_mixer=(
                        CompetitiveResidualMixer(
                            embed_dim,
                            gate_bias=self.competitive_residual_gate_bias,
                            init_scale=self.competitive_residual_init_scale,
                            cls_mix_scale=self.competitive_residual_cls_mix_scale,
                            mean_mix_scale=self.competitive_residual_mean_mix_scale,
                            patch_only=self.competitive_residual_patch_only,
                        )
                        if (
                            self.competitive_residual
                            and (
                                self.competitive_residual_last_k_blocks <= 0
                                or idx >= max(depth - self.competitive_residual_last_k_blocks, 0)
                            )
                        )
                        else None
                    ),
                    competitive_residual_scale=self.competitive_residual_scale,
                    parallel_block_update=bool(
                        self.parallel_block_update
                        and (
                            self.parallel_block_last_k_blocks <= 0
                            or idx >= max(depth - self.parallel_block_last_k_blocks, 0)
                        )
                    ),
                    mlp_first_update=bool(
                        self.mlp_first_update
                        and (
                            self.mlp_first_last_k_blocks <= 0
                            or idx >= max(depth - self.mlp_first_last_k_blocks, 0)
                        )
                    ),
                    tail_token_mixer=(
                        TailTokenMixer(
                            embed_dim,
                            gate_bias=self.tail_token_mixer_gate_bias,
                            init_scale=self.tail_token_mixer_init_scale,
                            patch_only=self.tail_token_mixer_patch_only,
                        )
                        if (
                            self.tail_token_mixer
                            and (
                                self.tail_token_mixer_last_k_blocks <= 0
                                or idx >= max(depth - self.tail_token_mixer_last_k_blocks, 0)
                            )
                        )
                        else None
                    ),
                    tail_token_mixer_scale=self.tail_token_mixer_scale,
                    hidden_diffusion=bool(
                        hidden_diffusion
                        and (
                            int(hidden_diffusion_last_k_blocks) <= 0
                            or idx >= max(depth - int(hidden_diffusion_last_k_blocks), 0)
                        )
                        and shared_hidden_diffusion is None
                    ),
                    hidden_diffusion_scale=hidden_diffusion_scale,
                    hidden_diffusion_mixer=(
                        shared_hidden_diffusion
                        if (
                            hidden_diffusion
                            and shared_hidden_diffusion is not None
                            and (
                                int(hidden_diffusion_last_k_blocks) <= 0
                                or idx >= max(depth - int(hidden_diffusion_last_k_blocks), 0)
                            )
                        )
                        else None
                    ),
                    hidden_grid_refiner=(
                        shared_hidden_grid_refiner
                        if (
                            hidden_grid_refiner
                            and shared_hidden_grid_refiner is not None
                            and (
                                int(hidden_grid_refiner_last_k_blocks) <= 0
                                or idx >= max(depth - int(hidden_grid_refiner_last_k_blocks), 0)
                            )
                        )
                        else PatchGridRefiner(
                            hidden_dim,
                            bottleneck_dim=hidden_grid_refiner_bottleneck,
                            gate_bias=hidden_grid_refiner_gate_bias,
                            init_scale=hidden_grid_refiner_init_scale,
                            cls_context_scale=hidden_grid_refiner_cls_context_scale,
                        )
                        if (
                            hidden_grid_refiner
                            and shared_hidden_grid_refiner is None
                            and (
                                int(hidden_grid_refiner_last_k_blocks) <= 0
                                or idx >= max(depth - int(hidden_grid_refiner_last_k_blocks), 0)
                            )
                        )
                        else None
                    ),
                    hidden_grid_refiner_scale=hidden_grid_refiner_scale,
                    hidden_cls_bridge=(
                        shared_hidden_cls_bridge
                        if (
                            hidden_cls_bridge
                            and shared_hidden_cls_bridge is not None
                            and (
                                int(hidden_cls_bridge_last_k_blocks) <= 0
                                or idx >= max(depth - int(hidden_cls_bridge_last_k_blocks), 0)
                            )
                        )
                        else HiddenClsAttentionBridge(
                            hidden_dim,
                            bottleneck_dim=hidden_cls_bridge_bottleneck,
                            gate_bias=hidden_cls_bridge_gate_bias,
                            init_scale=hidden_cls_bridge_init_scale,
                            patch_feedback_scale=hidden_cls_bridge_patch_feedback_scale,
                        )
                        if (
                            hidden_cls_bridge
                            and shared_hidden_cls_bridge is None
                            and (
                                int(hidden_cls_bridge_last_k_blocks) <= 0
                                or idx >= max(depth - int(hidden_cls_bridge_last_k_blocks), 0)
                            )
                        )
                        else None
                    ),
                    hidden_cls_bridge_scale=hidden_cls_bridge_scale,
                    hidden_channel_flow=(
                        shared_hidden_channel_flow
                        if (
                            hidden_channel_flow
                            and shared_hidden_channel_flow is not None
                            and (
                                int(hidden_channel_flow_last_k_blocks) <= 0
                                or idx >= max(depth - int(hidden_channel_flow_last_k_blocks), 0)
                            )
                        )
                        else SharedHiddenChannelFlow(
                            hidden_dim,
                            bottleneck_dim=hidden_channel_flow_bottleneck,
                            rank=hidden_channel_flow_rank,
                            patch_only=hidden_channel_flow_patch_only,
                            gate_bias=hidden_channel_flow_gate_bias,
                            init_scale=hidden_channel_flow_init_scale,
                            cls_mix_scale=hidden_channel_flow_cls_mix_scale,
                            mean_mix_scale=hidden_channel_flow_mean_mix_scale,
                        )
                        if (
                            hidden_channel_flow
                            and shared_hidden_channel_flow is None
                            and (
                                int(hidden_channel_flow_last_k_blocks) <= 0
                                or idx >= max(depth - int(hidden_channel_flow_last_k_blocks), 0)
                            )
                        )
                        else None
                    ),
                    hidden_channel_flow_scale=hidden_channel_flow_scale,
                    response_flow_norm=(
                        shared_response_flow
                        if (
                            response_flow_norm
                            and shared_response_flow is not None
                            and (
                                int(response_flow_last_k_blocks) <= 0
                                or idx >= max(depth - int(response_flow_last_k_blocks), 0)
                            )
                        )
                        else SharedBiaxialResponseFlow(
                            hidden_dim,
                            bottleneck_dim=response_flow_bottleneck,
                            patch_only=response_flow_patch_only,
                            gate_bias=response_flow_gate_bias,
                            init_scale=response_flow_init_scale,
                            cls_mix_scale=response_flow_cls_mix_scale,
                            mean_mix_scale=response_flow_mean_mix_scale,
                            token_exponent=response_flow_token_exponent,
                            channel_exponent=response_flow_channel_exponent,
                        )
                        if (
                            response_flow_norm
                            and self.response_flow_mode == "biaxial"
                            and shared_response_flow is None
                            and (
                                int(response_flow_last_k_blocks) <= 0
                                or idx >= max(depth - int(response_flow_last_k_blocks), 0)
                            )
                        )
                        else TokenResponseFlowNorm(hidden_dim, init_scale=response_flow_init_scale)
                        if (
                            response_flow_norm
                            and self.response_flow_mode != "biaxial"
                            and (
                                int(response_flow_last_k_blocks) <= 0
                                or idx >= max(depth - int(response_flow_last_k_blocks), 0)
                            )
                        )
                        else None
                    ),
                    response_flow_scale=response_flow_scale,
                    response_flow_pre_act=response_flow_pre_act,
                    activation_flow_gate=(
                        shared_activation_flow
                        if (
                            activation_flow
                            and shared_activation_flow is not None
                            and (
                                int(activation_flow_last_k_blocks) <= 0
                                or idx >= max(depth - int(activation_flow_last_k_blocks), 0)
                            )
                        )
                        else SharedActivationFlowGate(
                            hidden_dim,
                            bottleneck_dim=activation_flow_bottleneck,
                            patch_only=activation_flow_patch_only,
                            gate_bias=activation_flow_gate_bias,
                            init_scale=activation_flow_init_scale,
                            cls_mix_scale=activation_flow_cls_mix_scale,
                            mean_mix_scale=activation_flow_mean_mix_scale,
                            std_mix_scale=activation_flow_std_mix_scale,
                            cls_token_scale=activation_flow_cls_token_scale,
                        )
                        if (
                            activation_flow
                            and shared_activation_flow is None
                            and (
                                int(activation_flow_last_k_blocks) <= 0
                                or idx >= max(depth - int(activation_flow_last_k_blocks), 0)
                            )
                        )
                        else None
                    ),
                    activation_flow_scale=activation_flow_scale,
                    attn_flow_modulator=(
                        shared_attn_flow_modulator
                        if (
                            attn_flow_modulator
                            and shared_attn_flow_modulator is not None
                            and (
                                int(attn_flow_last_k_blocks) <= 0
                                or idx >= max(depth - int(attn_flow_last_k_blocks), 0)
                            )
                        )
                        else AttentionFlowModulator(
                            embed_dim,
                            bottleneck_dim=attn_flow_bottleneck,
                            gate_bias=attn_flow_gate_bias,
                            init_scale=attn_flow_init_scale,
                            detail_topk=attn_flow_detail_topk,
                            patch_only=attn_flow_patch_only,
                        )
                        if (
                            attn_flow_modulator
                            and shared_attn_flow_modulator is None
                            and (
                                int(attn_flow_last_k_blocks) <= 0
                                or idx >= max(depth - int(attn_flow_last_k_blocks), 0)
                            )
                        )
                        else None
                    ),
                    attn_flow_scale=attn_flow_scale,
                    patch_grid_refiner=bool(
                        patch_grid_refiner
                        and (
                            int(patch_grid_refiner_last_k_blocks) <= 0
                            or idx >= max(depth - int(patch_grid_refiner_last_k_blocks), 0)
                        )
                        and shared_patch_grid_refiner is None
                    ),
                    patch_grid_refiner_scale=patch_grid_refiner_scale,
                    patch_grid_refiner_module=(
                        shared_patch_grid_refiner
                        if (
                            patch_grid_refiner
                            and shared_patch_grid_refiner is not None
                            and (
                                int(patch_grid_refiner_last_k_blocks) <= 0
                                or idx >= max(depth - int(patch_grid_refiner_last_k_blocks), 0)
                            )
                        )
                        else None
                    ),
                    attention_metric_adapter=False,
                    attention_metric_scale=attention_metric_scale,
                    attention_metric_module=(
                        (
                            shared_attention_metric
                            if (
                                attention_metric_adapter
                                and shared_attention_metric is not None
                                and (
                                    int(attention_metric_last_k_blocks) <= 0
                                    or idx >= max(depth - int(attention_metric_last_k_blocks), 0)
                                )
                            )
                            else make_attention_metric_module()
                            if (
                                attention_metric_adapter
                                and shared_attention_metric is None
                                and (
                                    int(attention_metric_last_k_blocks) <= 0
                                    or idx >= max(depth - int(attention_metric_last_k_blocks), 0)
                                )
                            )
                            else None
                        )
                    ),
                    layer_scale_init=geo_layer_scale_init,
                    mlp_override=mlp_override_for_block(idx),
                )
                for idx in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.class_text_router = None
        if use_class_text_router and class_texts:
            class_embeddings = build_text_embeddings(
                class_texts,
                condition_dim,
                source=text_embedding_source,
                clip_model_name=text_embedding_model,
            )
            self.class_text_router = ClassTextRouter(
                token_dim=embed_dim,
                class_embeddings=class_embeddings,
                hidden_dim=controller_hidden_dim,
                temperature=text_router_temperature,
            )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if tokenizer_type == "budgeted_detail":
            nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)
            nn.init.trunc_normal_(self.coarse_pos_embed, std=0.02)
            nn.init.trunc_normal_(self.detail_pos_embed, std=0.02)
            nn.init.trunc_normal_(self.detail_type_embed, std=0.02)
        else:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            if self.summary_pos_embed is not None:
                nn.init.trunc_normal_(self.summary_pos_embed, std=0.02)
                nn.init.trunc_normal_(self.summary_type_embed, std=0.02)
        self.apply(self._init_weights)
        # Re-run custom inits AFTER apply() so they take precedence
        for module in self.modules():
            if type(module).__name__ == "AttentionGatedFFN":
                module._init_weights()
        self._reset_geo_operator_controllers()
        self.last_summary_stats: dict[str, float] = {}

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _reset_geo_operator_controllers(self) -> None:
        for module in self.modules():
            if isinstance(module, GeoOperatorLinear):
                module._init_controller()
            reset_heads = getattr(module, "_init_heads", None)
            if callable(reset_heads):
                reset_heads()

    def forward_features(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
        *,
        return_block_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
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
            patch_map = self.patch_embed(x)
            patch_tokens = patch_map.flatten(2).transpose(1, 2)
            cls_tokens = cls_tokens + self.pos_embed[:, :1, :]
            patch_tokens = patch_tokens + self.pos_embed[:, 1:, :]
            if self.summary_token:
                summary_token = self.summary_norm(patch_map.mean(dim=(-2, -1))).unsqueeze(1)
                summary_token = self.summary_token_scale * summary_token
                summary_token = summary_token + self.summary_pos_embed + self.summary_type_embed
                x = torch.cat((cls_tokens, summary_token, patch_tokens), dim=1)
                self.last_summary_stats = {
                    "summary_token_norm": float(summary_token.detach().norm(dim=-1).mean().item()),
                    "summary_token_scale": float(self.summary_token_scale),
                    "summary_head_fusion": float(self.summary_head_fusion),
                }
            else:
                x = torch.cat((cls_tokens, patch_tokens), dim=1)
                self.last_summary_stats = {}
        x = self.pos_drop(x)
        if self.token_flow is not None and self.token_flow_input:
            x = self.token_flow(x, residual_scale=self.token_flow_input_scale)
        elif self.input_token_flow is not None:
            x = self.input_token_flow(x, residual_scale=self.token_flow_input_scale)
        if self.class_text_router is not None:
            routed_condition = self.class_text_router(x)
            if condition is None:
                condition = routed_condition
            else:
                condition = torch.nn.functional.normalize(condition + routed_condition, dim=-1)

        block_features: list[torch.Tensor] = []
        flow_state: torch.Tensor | None = None
        for idx, block in enumerate(self.blocks):
            block_input = x
            x = block(x, condition=condition)
            if self.token_flow_last_k_blocks > 0:
                token_flow_start = max(len(self.blocks) - self.token_flow_last_k_blocks, 0)
                if idx >= token_flow_start:
                    if self.token_flow is not None:
                        x = self.token_flow(x, residual_scale=self.token_flow_block_scale)
                    elif self.block_token_flows is not None:
                        x = self.block_token_flows[idx - token_flow_start](x, residual_scale=self.token_flow_block_scale)
            if self.inter_layer_flow and self.inter_layer_flow_last_k_blocks > 0:
                inter_layer_start = max(len(self.blocks) - self.inter_layer_flow_last_k_blocks, 0)
                if idx >= inter_layer_start:
                    if self.shared_inter_layer_flow is not None:
                        x = self.shared_inter_layer_flow(block_input, x, residual_scale=self.inter_layer_flow_scale)
                    elif self.block_inter_layer_flows is not None:
                        x = self.block_inter_layer_flows[idx - inter_layer_start](
                            block_input,
                            x,
                            residual_scale=self.inter_layer_flow_scale,
                        )
            if self.flow_state_carrier and self.flow_state_last_k_blocks > 0:
                flow_state_start = max(len(self.blocks) - self.flow_state_last_k_blocks, 0)
                if idx >= flow_state_start:
                    if self.shared_flow_state_carrier is not None:
                        x, flow_state = self.shared_flow_state_carrier(
                            x,
                            flow_state,
                            residual_scale=self.flow_state_scale,
                        )
                    elif self.block_flow_state_carriers is not None:
                        x, flow_state = self.block_flow_state_carriers[idx - flow_state_start](
                            x,
                            flow_state,
                            residual_scale=self.flow_state_scale,
                        )
            if return_block_features:
                block_features.append(x[:, 0])
        x = self.norm(x)
        if return_block_features:
            return x, block_features
        return x

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None, return_features: bool = False):
        x = self.forward_features(x, condition=condition)
        features = x[:, 0]
        if self.summary_token:
            features = features + self.summary_head_fusion * x[:, 1]
        if return_features:
            return features
        return self.head(features)

    def forward_feature_pyramid(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        tokens, block_features = self.forward_features(x, condition=condition, return_block_features=True)
        return tokens[:, 0], block_features

    def get_diagnostics(self) -> dict[str, float]:
        block_diags = [block.get_diagnostics() for block in self.blocks]
        diagnostics = {}
        if block_diags:
            keys = {key for diag in block_diags for key in diag}
            diagnostics.update(
                {
            key: sum(diag.get(key, 0.0) for diag in block_diags) / max(len(block_diags), 1)
            for key in keys
                }
            )
        if self.class_text_router is not None:
            diagnostics.update(self.class_text_router.get_diagnostics())
        if self.tokenizer_type == "budgeted_detail":
            diagnostics.update(self.tokenizer.get_diagnostics())
        elif self.patch_embed is not None and hasattr(self.patch_embed, "last_stats"):
            diagnostics.update(self.patch_embed.last_stats)
        diagnostics.update(self.last_summary_stats)
        if self.token_flow is not None:
            diagnostics.update(self.token_flow.last_stats)
        elif self.input_token_flow is not None or self.block_token_flows is not None:
            if self.input_token_flow is not None:
                diagnostics.update({f"input_{k}": v for k, v in self.input_token_flow.last_stats.items()})
            if self.block_token_flows is not None and len(self.block_token_flows) > 0:
                block_diags = [module.last_stats for module in self.block_token_flows if module.last_stats]
                if block_diags:
                    keys = {key for diag in block_diags for key in diag}
                    diagnostics.update(
                        {
                            key: sum(diag.get(key, 0.0) for diag in block_diags) / len(block_diags)
                            for key in keys
                        }
                    )
        if self.shared_inter_layer_flow is not None:
            diagnostics.update(self.shared_inter_layer_flow.last_stats)
        elif self.block_inter_layer_flows is not None and len(self.block_inter_layer_flows) > 0:
            block_diags = [module.last_stats for module in self.block_inter_layer_flows if module.last_stats]
            if block_diags:
                keys = {key for diag in block_diags for key in diag}
                diagnostics.update(
                    {
                        key: sum(diag.get(key, 0.0) for diag in block_diags) / len(block_diags)
                        for key in keys
                    }
                )
        if self.shared_flow_state_carrier is not None:
            diagnostics.update(self.shared_flow_state_carrier.last_stats)
        elif self.block_flow_state_carriers is not None and len(self.block_flow_state_carriers) > 0:
            block_diags = [module.last_stats for module in self.block_flow_state_carriers if module.last_stats]
            if block_diags:
                keys = {key for diag in block_diags for key in diag}
                diagnostics.update(
                    {
                        key: sum(diag.get(key, 0.0) for diag in block_diags) / len(block_diags)
                        for key in keys
                    }
                )
        return diagnostics

    def get_aux_losses(self) -> dict[str, torch.Tensor]:
        block_aux = [block.get_aux_losses() for block in self.blocks]
        aux = {}
        keys = {key for losses in block_aux for key in losses}
        for key in keys:
            values = [losses[key] for losses in block_aux if key in losses]
            if values:
                aux[key] = torch.stack(values).mean()
        return aux
