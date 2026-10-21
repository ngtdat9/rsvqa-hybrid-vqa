"""Question encoder, fusion, and classification/count-regression heads.

Fusion has two modes (``fusion_type``):
  * ``cross_attention`` (default): question tokens attend over the image patch
    tokens; the fused tokens are pooled and passed to the heads. Spatial
    information survives until after fusion, and the attention weights
    ``[B, L, N]`` can be returned for explanation.
  * ``concat``: the original Week 1 baseline -- CLS vector + mean-pooled question
    vector, concatenated and fed to an MLP. Kept for the ablation.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from src.models.hybrid_encoder import EncoderOutput, HybridEncoder

FUSION_TYPES = ("cross_attention", "concat")
COUNT_TRANSFORMS = ("log1p", "none")


def _sinusoidal_positions(max_len: int, dim: int) -> torch.Tensor:
    position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim))
    pe = torch.zeros(max_len, dim)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div[: dim // 2])
    return pe


class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: int = 0, max_len: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        # fixed (non-learned) word-order signal; not saved in checkpoints
        self.register_buffer("positions", _sinusoidal_positions(max_len, embed_dim), persistent=False)

    def forward(self, question_tokens: torch.Tensor, question_len: torch.Tensor) -> torch.Tensor:
        """Mean-pooled question vector [B,E] (used by the ``concat`` fusion)."""
        embedded = self.embedding(question_tokens)  # [B,L,E]
        positions = torch.arange(question_tokens.size(1), device=question_tokens.device)
        mask = positions[None, :] < question_len[:, None]  # [B,L] bool
        summed = (embedded * mask.unsqueeze(-1)).sum(dim=1)
        return summed / question_len.clamp(min=1).unsqueeze(-1).float()  # [B,E]

    def encode_tokens(
        self, question_tokens: torch.Tensor, question_len: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-word tokens [B,L,E] with word-order signal, plus the valid-token mask [B,L]."""
        if question_tokens.size(1) > self.positions.size(0):
            raise ValueError(
                f"question length {question_tokens.size(1)} exceeds max_question_len "
                f"{self.positions.size(0)}; raise max_question_len in the config"
            )
        embedded = self.embedding(question_tokens)
        embedded = embedded + self.positions[: question_tokens.size(1)].to(embedded.dtype)
        positions = torch.arange(question_tokens.size(1), device=question_tokens.device)
        mask = positions[None, :] < question_len[:, None]
        return embedded, mask


class FusionModule(nn.Module):
    """Week 1 baseline fusion: concatenate two pooled vectors, then an MLP."""

    def __init__(self, image_dim: int, question_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(image_dim + question_dim)
        self.mlp = nn.Sequential(
            nn.Linear(image_dim + question_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, image_embed: torch.Tensor, question_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image_embed, question_embed], dim=-1)
        return self.mlp(self.norm(x))  # [B,hidden_dim]


class CrossAttentionFusion(nn.Module):
    """Question tokens (queries) attend over image patch tokens (keys/values).

    Pooling happens only AFTER attention, so the model can focus on specific
    image regions per question word.
    """

    def __init__(
        self, image_dim: int, question_dim: int, hidden_dim: int, num_heads: int, dropout: float
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(question_dim, image_dim) if question_dim != image_dim else nn.Identity()
        self.attn = nn.MultiheadAttention(image_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(image_dim)
        self.mlp = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,     # [B,N,d]
        question_tokens: torch.Tensor,  # [B,L,E]
        question_mask: torch.Tensor,    # [B,L] bool, True = real word
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        q = self.q_proj(question_tokens)  # [B,L,d]
        attn_out, weights = self.attn(
            q, patch_tokens, patch_tokens,
            need_weights=return_attention, average_attn_weights=True,
        )  # weights: [B,L,N] (mean over heads) or None
        fused_tokens = self.norm(q + attn_out)  # [B,L,d]
        mask = question_mask.unsqueeze(-1).to(fused_tokens.dtype)
        pooled = (fused_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B,d]
        return self.mlp(pooled), weights


class HybridVQAModel(nn.Module):
    def __init__(
        self,
        image_size: int,
        cnn_channels: list[int],
        d_model: int,
        num_transformer_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        question_vocab_size: int,
        question_embed_dim: int,
        fusion_hidden_dim: int,
        vocab_size: int,
        fusion_type: str,
        max_question_len: int,
    ) -> None:
        super().__init__()
        if fusion_type not in FUSION_TYPES:
            raise ValueError(f"fusion_type must be one of {FUSION_TYPES}, got {fusion_type!r}")
        self.fusion_type = fusion_type
        self.encoder = HybridEncoder(
            image_size=image_size,
            cnn_channels=cnn_channels,
            d_model=d_model,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.question_encoder = QuestionEncoder(
            question_vocab_size, question_embed_dim, max_len=max_question_len
        )
        if fusion_type == "concat":
            self.fusion = FusionModule(d_model, question_embed_dim, fusion_hidden_dim, dropout)
        else:
            self.fusion = CrossAttentionFusion(
                d_model, question_embed_dim, fusion_hidden_dim, num_heads, dropout
            )
        self.classifier = nn.Linear(fusion_hidden_dim, vocab_size)
        self.count_head = nn.Linear(fusion_hidden_dim, 1)

    def forward(
        self, batch: dict[str, torch.Tensor], return_attention: bool = False
    ) -> dict[str, torch.Tensor]:
        enc_out: EncoderOutput = self.encoder(batch["image"])
        attention = None
        if self.fusion_type == "concat":
            q_embed = self.question_encoder(batch["question_tokens"], batch["question_len"])
            fused = self.fusion(enc_out.pooled, q_embed)
        else:
            q_tokens, q_mask = self.question_encoder.encode_tokens(
                batch["question_tokens"], batch["question_len"]
            )
            fused, attention = self.fusion(enc_out.patch_tokens, q_tokens, q_mask, return_attention)
        class_logits = self.classifier(fused)                                     # [B, vocab_size]
        count_pred = nn.functional.softplus(self.count_head(fused)).squeeze(-1)    # [B], >=0, in raw counts
        outputs = {"class_logits": class_logits, "count_pred": count_pred}
        if attention is not None:
            outputs["cross_attention"] = attention                                 # [B, L, N]
        return outputs


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    count_loss_weight: float = 1.0,
    count_transform: str = "log1p",
) -> torch.Tensor:
    """Cross-entropy on classification rows + MSE on count rows.

    Raw counts reach the thousands, so a raw MSE would dwarf the cross-entropy
    (< 1) and starve the classifier. With ``count_transform="log1p"`` the MSE is
    taken on log(1 + count); the model still predicts raw counts, so evaluation
    MAE/RMSE stay in objects. ``"none"`` reproduces the Week 1 baseline loss.
    """
    if count_transform not in COUNT_TRANSFORMS:
        raise ValueError(f"count_transform must be one of {COUNT_TRANSFORMS}, got {count_transform!r}")
    class_mask = batch["label"] != -1
    count_mask = ~batch["count_target"].isnan()

    class_loss = torch.tensor(0.0, device=outputs["class_logits"].device)
    if class_mask.any():
        class_loss = nn.functional.cross_entropy(
            outputs["class_logits"][class_mask], batch["label"][class_mask]
        )

    count_loss = torch.tensor(0.0, device=outputs["count_pred"].device)
    if count_mask.any():
        pred = outputs["count_pred"][count_mask]
        target = batch["count_target"][count_mask]
        if count_transform == "log1p":
            pred, target = torch.log1p(pred), torch.log1p(target)
        count_loss = nn.functional.mse_loss(pred, target)

    return class_loss + count_loss_weight * count_loss
