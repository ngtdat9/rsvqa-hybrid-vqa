import math

import torch

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
