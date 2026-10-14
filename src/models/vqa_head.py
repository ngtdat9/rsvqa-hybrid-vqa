"""Question encoder, fusion, and classification/count-regression heads (Week 1 first pass)."""
from __future__ import annotations

import torch
from torch import nn

from src.models.hybrid_encoder import EncoderOutput, HybridEncoder


class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: int = 0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

    def forward(self, question_tokens: torch.Tensor, question_len: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(question_tokens)  # [B,L,E]
        positions = torch.arange(question_tokens.size(1), device=question_tokens.device)
        mask = positions[None, :] < question_len[:, None]  # [B,L] bool
        summed = (embedded * mask.unsqueeze(-1)).sum(dim=1)
        return summed / question_len.clamp(min=1).unsqueeze(-1).float()  # [B,E]


class FusionModule(nn.Module):
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
    ) -> None:
        super().__init__()
        self.encoder = HybridEncoder(
            image_size=image_size,
            cnn_channels=cnn_channels,
            d_model=d_model,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.question_encoder = QuestionEncoder(question_vocab_size, question_embed_dim)
        self.fusion = FusionModule(d_model, question_embed_dim, fusion_hidden_dim, dropout)
        self.classifier = nn.Linear(fusion_hidden_dim, vocab_size)
        self.count_head = nn.Linear(fusion_hidden_dim, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        enc_out: EncoderOutput = self.encoder(batch["image"])
        q_embed = self.question_encoder(batch["question_tokens"], batch["question_len"])
        fused = self.fusion(enc_out.pooled, q_embed)
        class_logits = self.classifier(fused)                                     # [B, vocab_size]
        count_pred = nn.functional.softplus(self.count_head(fused)).squeeze(-1)    # [B], >=0
        return {"class_logits": class_logits, "count_pred": count_pred}


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    count_loss_weight: float = 1.0,
) -> torch.Tensor:
    class_mask = batch["label"] != -1
    count_mask = ~batch["count_target"].isnan()

    class_loss = torch.tensor(0.0, device=outputs["class_logits"].device)
    if class_mask.any():
        class_loss = nn.functional.cross_entropy(
            outputs["class_logits"][class_mask], batch["label"][class_mask]
        )

    count_loss = torch.tensor(0.0, device=outputs["count_pred"].device)
    if count_mask.any():
        count_loss = nn.functional.mse_loss(
            outputs["count_pred"][count_mask], batch["count_target"][count_mask]
        )

    return class_loss + count_loss_weight * count_loss
