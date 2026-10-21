import math

import torch

import pytest

from src.models.vqa_head import HybridVQAModel, QuestionEncoder, compute_loss

TEST_CONFIG = dict(
    image_size=32,
    cnn_channels=[8, 16],
    d_model=32,
    num_transformer_layers=1,
    num_heads=2,
    ffn_dim=64,
    dropout=0.0,
    question_vocab_size=20,
    question_embed_dim=32,
    fusion_hidden_dim=32,
    vocab_size=4,
    fusion_type="cross_attention",
    max_question_len=16,
)


def make_model(**overrides) -> HybridVQAModel:
    kwargs = {**TEST_CONFIG, **overrides}
    return HybridVQAModel(**kwargs)


def make_batch(batch_size: int, max_len: int = 6) -> dict[str, torch.Tensor]:
    image = torch.randn(batch_size, 3, TEST_CONFIG["image_size"], TEST_CONFIG["image_size"])
    question_tokens = torch.randint(2, TEST_CONFIG["question_vocab_size"], (batch_size, max_len))
    question_len = torch.full((batch_size,), max_len, dtype=torch.long)
    # alternate rows between classification and count type
    label = torch.tensor([i % TEST_CONFIG["vocab_size"] for i in range(batch_size)], dtype=torch.long)
    count_target = torch.full((batch_size,), float("nan"))
    for i in range(0, batch_size, 2):
        label[i] = -1
        count_target[i] = float(i + 1)
    return {
        "image": image,
        "question_tokens": question_tokens,
        "question_len": question_len,
        "label": label,
        "count_target": count_target,
    }


def test_question_encoder_ignores_padding():
    encoder = QuestionEncoder(vocab_size=20, embed_dim=16, padding_idx=0)
    real_tokens = torch.tensor([[5, 6, 7]])
    padded_tokens = torch.tensor([[5, 6, 7, 0, 0]])
    len_a = torch.tensor([3])
    len_b = torch.tensor([3])

    out_a = encoder(real_tokens, len_a)
    out_b = encoder(padded_tokens, len_b)
    assert torch.equal(out_a, out_b)


def test_forward_output_shapes():
    model = make_model()
    batch = make_batch(batch_size=4)
    out = model(batch)
    assert out["class_logits"].shape == (4, TEST_CONFIG["vocab_size"])
    assert out["count_pred"].shape == (4,)


def test_count_pred_nonnegative():
    model = make_model()
    batch = make_batch(batch_size=6)
    out = model(batch)
    assert torch.all(out["count_pred"] >= 0)


def test_compute_loss_masks_correctly():
    model = make_model()
    batch = make_batch(batch_size=6)
    out = model(batch)

    loss = compute_loss(out, batch)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    # perturbing class_logits on count-only rows (label == -1) must not change loss
    out_perturbed = {k: v.clone() for k, v in out.items()}
    count_rows = batch["label"] == -1
    out_perturbed["class_logits"][count_rows] += 100.0
    loss_perturbed = compute_loss(out_perturbed, batch)
    assert torch.isclose(loss, loss_perturbed)

    # perturbing count_pred on classification-only rows must not change loss
    out_perturbed2 = {k: v.clone() for k, v in out.items()}
    class_rows = batch["label"] != -1
    out_perturbed2["count_pred"][class_rows] += 100.0
    loss_perturbed2 = compute_loss(out_perturbed2, batch)
    assert torch.isclose(loss, loss_perturbed2)


def test_compute_loss_handles_single_type_batches():
    model = make_model()

    all_class_batch = make_batch(batch_size=4)
    all_class_batch["label"] = torch.tensor([0, 1, 2, 3])
    all_class_batch["count_target"] = torch.full((4,), float("nan"))
    out = model(all_class_batch)
    loss = compute_loss(out, all_class_batch)
    assert not torch.isnan(loss)

    all_count_batch = make_batch(batch_size=4)
    all_count_batch["label"] = torch.full((4,), -1, dtype=torch.long)
    all_count_batch["count_target"] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = model(all_count_batch)
    loss = compute_loss(out, all_count_batch)
    assert not torch.isnan(loss)


def test_gradient_flow_backward():
    model = make_model()
    batch = make_batch(batch_size=6)
    out = model(batch)
    loss = compute_loss(out, batch)
    loss.backward()

    classifier_grad = any(p.grad is not None and torch.any(p.grad != 0) for p in model.classifier.parameters())
    count_head_grad = any(p.grad is not None and torch.any(p.grad != 0) for p in model.count_head.parameters())
    assert classifier_grad
    assert count_head_grad


# ---------------------------------------------------------------------------
# Cross-attention fusion, concat ablation, log1p count loss
# ---------------------------------------------------------------------------

def test_default_fusion_is_cross_attention():
    assert make_model().fusion_type == TEST_CONFIG["fusion_type"]


def test_unknown_fusion_type_rejected():
    with pytest.raises(ValueError):
        make_model(fusion_type="bogus")


def test_cross_attention_weights_shape_and_normalised():
    model = make_model().eval()
    batch = make_batch(batch_size=3, max_len=6)
    with torch.no_grad():
        out = model(batch, return_attention=True)
    attn = out["cross_attention"]
    num_patches = model.encoder.pos_embed.shape[1] - 1
    assert attn.shape == (3, 6, num_patches)  # [B, question words, image patches]
    assert torch.allclose(attn.sum(dim=-1), torch.ones(3, 6), atol=1e-5)


def test_attention_not_returned_by_default():
    out = make_model()(make_batch(batch_size=2))
    assert "cross_attention" not in out


def test_cross_attention_ignores_padded_question_tokens():
    model = make_model().eval()
    image = torch.randn(1, 3, TEST_CONFIG["image_size"], TEST_CONFIG["image_size"])
    short = {"image": image, "question_tokens": torch.tensor([[5, 6, 7]]), "question_len": torch.tensor([3])}
    padded = {"image": image, "question_tokens": torch.tensor([[5, 6, 7, 0, 0]]), "question_len": torch.tensor([3])}
    with torch.no_grad():
        a, b = model(short), model(padded)
    assert torch.allclose(a["class_logits"], b["class_logits"], atol=1e-5)
    assert torch.allclose(a["count_pred"], b["count_pred"], atol=1e-5)


def test_cross_attention_output_depends_on_image_patches():
    model = make_model().eval()
    batch = make_batch(batch_size=2)
    other = {**batch, "image": torch.randn_like(batch["image"])}
    with torch.no_grad():
        a, b = model(batch), model(other)
    assert not torch.allclose(a["class_logits"], b["class_logits"])


def test_concat_fusion_still_works():
    model = make_model(fusion_type="concat")
    batch = make_batch(batch_size=4)
    out = model(batch)
    assert out["class_logits"].shape == (4, TEST_CONFIG["vocab_size"])
    assert out["count_pred"].shape == (4,)
    assert "cross_attention" not in out


def test_cross_attention_gradients_reach_encoder_and_both_heads():
    model = make_model()
    batch = make_batch(batch_size=6)
    compute_loss(model(batch), batch).backward()
    for module in (model.encoder.cnn_stem, model.fusion, model.question_encoder, model.classifier, model.count_head):
        assert any(p.grad is not None and torch.any(p.grad != 0) for p in module.parameters())


def test_log1p_count_loss_matches_manual_value():
    outputs = {"class_logits": torch.zeros(2, 4), "count_pred": torch.tensor([9.0, 99.0])}
    batch = {"label": torch.tensor([-1, -1]), "count_target": torch.tensor([19.0, 999.0])}
    expected = torch.mean((torch.log1p(outputs["count_pred"]) - torch.log1p(batch["count_target"])) ** 2)
    assert torch.isclose(compute_loss(outputs, batch, count_transform="log1p"), expected)
    raw = torch.mean((outputs["count_pred"] - batch["count_target"]) ** 2)
    assert torch.isclose(compute_loss(outputs, batch, count_transform="none"), raw)


def test_log1p_keeps_count_loss_on_classification_scale():
    outputs = {"class_logits": torch.zeros(1, 4), "count_pred": torch.tensor([10.0])}
    batch = {"label": torch.tensor([-1]), "count_target": torch.tensor([5000.0])}
    assert compute_loss(outputs, batch, count_transform="none") > 1e6
    assert compute_loss(outputs, batch, count_transform="log1p") < 100


def test_unknown_count_transform_rejected():
    outputs = {"class_logits": torch.zeros(1, 4), "count_pred": torch.tensor([1.0])}
    batch = {"label": torch.tensor([-1]), "count_target": torch.tensor([1.0])}
    with pytest.raises(ValueError):
        compute_loss(outputs, batch, count_transform="sqrt")


def test_question_longer_than_max_len_raises_clear_error():
    model = make_model(max_question_len=4)
    batch = make_batch(batch_size=2, max_len=6)
    with pytest.raises(ValueError, match="max_question_len"):
        model(batch)
