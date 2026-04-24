import torch
import torch.nn as nn
import torch.nn.functional as F


def build_hashed_text_embeddings(texts: list[str], dim: int) -> torch.Tensor:
    dim = int(dim)
    embeddings = torch.zeros(len(texts), dim, dtype=torch.float32)
    for row, text in enumerate(texts):
        encoded = text.lower().encode("utf-8")
        if not encoded:
            embeddings[row, 0] = 1.0
            continue
        for idx, byte in enumerate(encoded):
            pos = (byte * 131 + idx * 17) % dim
            sign = 1.0 if ((byte + idx) % 2 == 0) else -1.0
            embeddings[row, pos] += sign
            pos2 = (byte * 313 + idx * 37 + 7) % dim
            embeddings[row, pos2] += 0.5 * sign
            if idx + 1 < len(encoded):
                bigram = byte * 257 + encoded[idx + 1]
                pos3 = (bigram * 53 + idx * 19) % dim
                embeddings[row, pos3] += 0.75
        embeddings[row] = F.normalize(embeddings[row], dim=0)
    return embeddings


def build_clip_text_embeddings(
    texts: list[str],
    *,
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "cpu",
) -> torch.Tensor:
    from transformers import AutoTokenizer, CLIPTextModelWithProjection

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = CLIPTextModelWithProjection.from_pretrained(model_name, use_safetensors=True).to(device)
    model.eval()
    with torch.inference_mode():
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = model(**tokens)
        embeddings = outputs.text_embeds.detach().to(dtype=torch.float32).cpu()
    return F.normalize(embeddings, dim=-1)


def build_text_embeddings(
    texts: list[str],
    dim: int,
    *,
    source: str = "hashed",
    clip_model_name: str = "openai/clip-vit-base-patch32",
    device: str = "cpu",
) -> torch.Tensor:
    source = str(source).lower()
    if source == "hashed":
        return build_hashed_text_embeddings(texts, dim)
    if source == "clip":
        embeddings = build_clip_text_embeddings(texts, model_name=clip_model_name, device=device)
        if embeddings.shape[-1] != int(dim):
            raise ValueError(
                f"CLIP text embedding dim {embeddings.shape[-1]} does not match requested condition dim {dim}."
            )
        return embeddings
    raise ValueError(f"Unsupported text embedding source: {source}")


class TextConditioner(nn.Module):
    """Generic external-condition projector with a text-first interface."""

    def __init__(
        self,
        token_dim: int,
        condition_dim: int = 512,
        hidden_dim: int = 128,
        pool_mode: str = "cls_mean",
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.pool_mode = str(pool_mode)

        pooled_dim = self.token_dim if self.pool_mode in {"cls", "mean"} else 2 * self.token_dim
        self.token_proj = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.condition_proj = nn.Sequential(
            nn.LayerNorm(self.condition_dim),
            nn.Linear(self.condition_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.null_condition = nn.Parameter(torch.zeros(1, self.condition_dim))

    def _pool_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x

        cls_token = x[:, 0]
        mean_token = x[:, 1:].mean(dim=1) if x.shape[1] > 1 else cls_token
        if self.pool_mode == "cls":
            return cls_token
        if self.pool_mode == "mean":
            return mean_token
        return torch.cat([cls_token, mean_token], dim=-1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        pooled = self._pool_tokens(x)
        token_hidden = self.token_proj(pooled)

        if condition is None:
            condition = self.null_condition.expand(pooled.shape[0], -1)
        cond_hidden = self.condition_proj(condition)
        return self.fuse(token_hidden + cond_hidden)


class PooledTokenConditioner(nn.Module):
    """Lightweight token-only conditioner for architecture-only GeoViT runs."""

    def __init__(self, token_dim: int, hidden_dim: int = 128, pool_mode: str = "cls_mean"):
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.pool_mode = str(pool_mode)
        pooled_dim = self.token_dim if self.pool_mode in {"cls", "mean"} else 2 * self.token_dim
        self.token_proj = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def _pool_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x

        cls_token = x[:, 0]
        mean_token = x[:, 1:].mean(dim=1) if x.shape[1] > 1 else cls_token
        if self.pool_mode == "cls":
            return cls_token
        if self.pool_mode == "mean":
            return mean_token
        return torch.cat([cls_token, mean_token], dim=-1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        del condition
        return self.token_proj(self._pool_tokens(x))


class ClassTextRouter(nn.Module):
    def __init__(
        self,
        token_dim: int,
        class_embeddings: torch.Tensor,
        *,
        hidden_dim: int = 128,
        temperature: float = 1.0,
    ):
        super().__init__()
        class_embeddings = F.normalize(class_embeddings.to(dtype=torch.float32), dim=-1)
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)
        self.register_buffer("class_embeddings", class_embeddings, persistent=False)
        self.query_proj = nn.Sequential(
            nn.LayerNorm(2 * self.token_dim),
            nn.Linear(2 * self.token_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, class_embeddings.shape[-1]),
        )
        self.last_stats = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token = x[:, 0]
        mean_token = x[:, 1:].mean(dim=1) if x.shape[1] > 1 else cls_token
        pooled = torch.cat([cls_token, mean_token], dim=-1)
        query = F.normalize(self.query_proj(pooled), dim=-1)
        logits = query @ self.class_embeddings.transpose(0, 1)
        logits = logits / max(self.temperature, 1e-6)
        weights = F.softmax(logits, dim=-1)
        routed = weights @ self.class_embeddings
        entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=-1).mean()
        top1 = weights.max(dim=-1).values.mean()
        self.last_stats = {
            "text_condition_entropy": float(entropy.item()),
            "text_condition_top1": float(top1.item()),
        }
        return routed

    def get_diagnostics(self) -> dict[str, float]:
        return dict(self.last_stats)
