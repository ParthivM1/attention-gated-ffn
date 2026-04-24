# Attention-Gated FFN (AGFF) for Vision Transformers

Novel FFN block where the gate is conditioned on attention output (cross-token context) rather than the token itself.

## Core Idea

Standard SwiGLU:
```
gate    = sigmoid(W_gate @ x)       # gate only sees current token
content = GELU(W_content @ x)
out     = W_out @ (content * gate)
```

AGFF:
```
gate    = sigmoid(W_gate @ attn_out) # gate sees ALL tokens via attention
content = GELU(W_content @ x)
out     = W_out @ (content * gate)
```

The FFN transformation is now conditioned on what attention discovered across the full sequence. Same parameter count as a standard GELU MLP.

## Results (CIFAR-100, ViT-S/4)

| Config | Scale | Accuracy | vs Plain ViT |
|---|---|---|---|
| Plain ViT | smoke (2k train, 40ep) | 19.85% ±1.33 | — |
| **AGFF k=6** | smoke | **22.2% ±2.4** | **+2.35pp** |
| Plain ViT | full (50k, 150ep) | 70.29% | — |
| **AGFF k=6** | full | **70.77%** | **+0.48pp** |

First architecture to beat plain ViT at full scale in this codebase.

## Setup

```bash
pip install torch torchvision modal numpy scipy pillow tqdm
```

For Modal (A100) runs you need a Modal account: `modal setup`

## Running Experiments

Local smoke test (CPU, 1 epoch):
```bash
python src/train_geovit_smoke_local.py
```

Modal A100 runs via presets:
```bash
# Smoke (40 epochs, 2k train samples) — ~$0.40
GEOVIT_PROPER_PRESET=smoke_agff_k6_locked_mid192d6_e40 modal run modal_geovit.py

# Full CIFAR-100 (150 epochs) — ~$2.50
GEOVIT_PROPER_PRESET=full_agff_k6_mid192d6_e150 modal run modal_geovit.py

# Plain ViT baseline
GEOVIT_PROPER_PRESET=smoke_plain_locked_mid192d6_e40 modal run modal_geovit.py
GEOVIT_PROPER_PRESET=full_plain_mid192d6_e150 modal run modal_geovit.py
```

Results land in Modal volume `geovit-proper-results`:
```bash
modal volume get geovit-proper-results geovit_proper/<run_name>/summary.json -
```

## Key Files

```
src/layers/attn_gated_ffn.py   # AGFF implementation
src/models/geovit.py           # GeoVisionTransformer (--agff-last-k-blocks)
src/train_geovit.py            # Training loop + CLI
src/geovit_presets.py          # All experiment presets
modal_geovit.py                # Modal A100 launcher
```

## Architecture Details

- `--agff-last-k-blocks N`: replace FFN in the last N blocks (use N=6 for all blocks)
- Hidden dim = `round(8D/3)` for exact param parity with standard GELU MLP
- `attn_out` is the raw attention output before the residual add — already available in `GeoViTBlock.forward`
- Degrades gracefully to SwiGLU if `attn_out` is not passed

## Model Config (mid192d6)

ViT with patch_size=4, embed_dim=192, depth=6, num_heads=6 on 32×32 CIFAR-100.
