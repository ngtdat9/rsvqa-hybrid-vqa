import math
from pathlib import Path

import pytest
import torch

from src.explain import (
    blur_image,
    choose_map,
    deletion_effect,
    delete_top_patches,
    _attention_with_weights,
    explain_batch,
    explain_methods,
    gradient_attention_map,
    balanced_sample,
    occlusion_map,
    paired_summary,
    question_pairs,
    question_pool,
    question_rollout_map,
    random_block_map,
    randomized_copy,
    rank_scores,
    rollout_matrix,
    run_xai,
    spearman,
    top_overlap,
    word_attention_map,
)
from src.models.vqa_head import HybridVQAModel
from src.train import build_model
from tests.test_data_layout import hr_config, write_fake_hr


def make_model() -> HybridVQAModel:
    torch.manual_seed(0)
    model = HybridVQAModel(
        image_size=32, cnn_channels=[8, 16], d_model=32, num_transformer_layers=2, num_heads=2, ffn_dim=64,
        dropout=0.1, question_vocab_size=20, question_embed_dim=32, fusion_hidden_dim=32, vocab_size=4,
        fusion_type="cross_attention", max_question_len=16,
    )
    return model.eval()


def make_batch(count_rows: tuple[int, ...] = (1,)) -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    label = torch.tensor([0, 1, 2])
    count_target = torch.full((3,), float("nan"))
    for row in count_rows:
        label[row] = -1
        count_target[row] = 3.0
    return {
        "image": torch.randn(3, 3, 32, 32), "question_tokens": torch.randint(2, 20, (3, 6)),
        "question_len": torch.tensor([6, 4, 2]), "label": label, "count_target": count_target,
    }


def take(batch: dict[str, torch.Tensor], row: int) -> dict[str, torch.Tensor]:
    return {k: v[row:row + 1] for k, v in batch.items()}


def test_explaining_does_not_change_the_prediction():
    model, batch = make_model(), make_batch()
    with torch.no_grad():
        plain = model(batch)
    explanation = explain_batch(model, batch)
    assert torch.allclose(explanation.class_logits, plain["class_logits"], atol=1e-6)
    assert torch.allclose(explanation.count_pred, plain["count_pred"], atol=1e-6)
    for method in ("cross_attention", "rollout", "grad_cam"):
        assert explanation.maps[method].shape == (3, 64)
        assert explanation.maps[method].isfinite().all()


@pytest.mark.parametrize("grad_enabled", [True, False])
def test_self_attention_is_captured_on_every_layer_in_eval_mode(grad_enabled: bool):
    # under no_grad PyTorch may take its fast path, which skips self_attn.forward
    with torch.set_grad_enabled(grad_enabled):
        explanation = explain_batch(make_model(), make_batch())
    assert len(explanation.layer_attention) == 2
    for weights in explanation.layer_attention:
        assert weights.shape == (3, 65, 65)
        assert torch.allclose(weights.sum(-1), torch.ones(3, 65), atol=1e-5)


def test_captured_attention_comes_from_each_layers_real_input():
    model, batch = make_model(), make_batch()
    explanation = explain_batch(model, batch)
    image = batch["image"]
    features = model.encoder.cnn_stem(image).flatten(2).transpose(1, 2)
    tokens = torch.cat([model.encoder.cls_token.expand(3, -1, -1), model.encoder.patch_proj(features)], dim=1)
    x = tokens + model.encoder.pos_embed
    with torch.no_grad():
        for layer, weights in zip(model.encoder.transformer.layers, explanation.layer_attention):
            expected = layer.self_attn(x, x, x, need_weights=True, average_attn_weights=True)[1]
            assert torch.allclose(weights, expected, atol=1e-5)
            x = layer(x)


def test_rollout_matches_a_hand_computed_product():
    first = torch.tensor([[[0.5, 0.5], [0.0, 1.0]]])
    second = torch.tensor([[[1.0, 0.0], [0.5, 0.5]]])
    # with the identity: first -> [[.75,.25],[0,1]], second -> [[1,0],[.25,.75]]; second @ first
    expected = torch.tensor([[[0.75, 0.25], [0.1875, 0.8125]]])
    assert torch.allclose(rollout_matrix([first, second]), expected)


def test_word_attention_leaves_out_padding():
    weights = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])  # the third word is padding
    assert torch.allclose(word_attention_map(weights, torch.tensor([2])), torch.tensor([[0.5, 0.5]]))


def test_question_rollout_with_identity_attention_returns_the_word_map():
    word_map = torch.tensor([[0.25, 0.75]])
    identity = torch.eye(3).unsqueeze(0)
    assert torch.allclose(question_rollout_map(word_map, [identity, identity]), word_map)


def test_question_rollout_moves_attention_along_self_attention():
    # patch 0 draws wholly on patch 1 (token 0 is CLS); after rollout's identity mix it keeps half of itself
    attention = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
    result = question_rollout_map(torch.tensor([[1.0, 0.0]]), [attention])
    assert torch.allclose(result, torch.tensor([[0.5, 0.5]]))


def test_maps_of_a_row_do_not_depend_on_the_rest_of_the_batch():
    model, batch = make_model(), make_batch()
    together = explain_batch(model, batch).maps
    alone = explain_batch(model, take(batch, 2)).maps
    for method, values in together.items():
        assert torch.allclose(values[2], alone[method][0], atol=1e-4, rtol=1e-3), method


def test_grad_cam_targets_the_count_on_count_rows_and_the_class_otherwise():
    model, batch = make_model(), make_batch(count_rows=())
    as_class = explain_batch(model, batch).maps["grad_cam"]
    as_count = explain_batch(model, {**batch, "count_target": torch.full((3,), 3.0)}).maps["grad_cam"]
    assert not torch.allclose(as_class, as_count)


def test_grad_cam_follows_the_definition():
    model, batch = make_model(), take(make_batch(count_rows=()), 0)
    captured = []
    hook = model.encoder.cnn_stem.blocks[-1].register_forward_hook(lambda _m, _a, output: captured.append(output))
    logits = model(batch)["class_logits"]
    hook.remove()
    gradient = torch.autograd.grad(logits.max(), captured[0])[0]
    expected = torch.relu((gradient.mean((2, 3), keepdim=True) * captured[0]).sum(1)).flatten(1)
    assert torch.allclose(explain_batch(model, batch).maps["grad_cam"], expected.detach(), atol=1e-6)


def test_attention_with_weights_matches_pytorch():
    attention = make_model().fusion.attn
    query, key = torch.randn(2, 5, 32), torch.randn(2, 9, 32)
    with torch.no_grad():
        expected, expected_weights = attention(query, key, key, need_weights=True, average_attn_weights=False)
        output, weights = _attention_with_weights(attention, query, key, key)
    assert torch.allclose(output, expected, atol=1e-5)
    assert torch.allclose(weights, expected_weights, atol=1e-5)


def test_gradient_attention_leaves_the_model_as_it_was_and_gives_a_finite_map():
    model, batch = make_model(), make_batch()
    with torch.no_grad():
        before = model(batch)
    result = gradient_attention_map(model, batch)
    assert result.shape == (3, 64) and result.isfinite().all() and (result >= 0).all()
    assert all("forward" not in layer.self_attn.__dict__ for layer in model.encoder.transformer.layers)
    with torch.no_grad():
        after = model(batch)
    assert torch.equal(before["class_logits"], after["class_logits"])


def test_gradient_attention_sees_the_real_forward_pass():
    # with the patch in place the model must still give the same answer as PyTorch's attention
    model, batch = make_model(), make_batch()
    modules = [layer.self_attn for layer in model.encoder.transformer.layers] + [model.fusion.attn]
    with torch.no_grad():
        expected = model(batch)["class_logits"]
    for module in modules:
        module.forward = lambda q, k, v, _m=module, **_: (_attention_with_weights(_m, q, k, v)[0], None)
    logits = model(batch)["class_logits"]
    for module in modules:
        del module.forward
    assert torch.allclose(logits, expected, atol=1e-5)


def test_gradient_attention_depends_on_what_is_being_explained():
    model, batch = make_model(), make_batch(count_rows=())
    as_class = gradient_attention_map(model, batch)
    as_count = gradient_attention_map(model, {**batch, "count_target": torch.full((3,), 3.0)})
    assert not torch.allclose(as_class, as_count)


def test_occlusion_matches_blanking_each_cell_by_hand():
    model, batch = make_model(), make_batch()
    base = explain_batch(model, batch)
    result = occlusion_map(model, batch, base.class_logits, base.count_pred, 64)
    assert result.shape == (3, 64)
    scores = torch.zeros(3, 64)
    scores[:, 10] = 1.0
    with torch.no_grad():
        outputs = model({**batch, "image": delete_top_patches(batch["image"], scores, 1 / 64)})
    expected = deletion_effect(
        base.class_logits, base.count_pred, outputs["class_logits"], outputs["count_pred"], ~batch["count_target"].isnan(),
    )[0]
    assert torch.allclose(result[:, 10], expected, atol=1e-6)


def test_explain_methods_keeps_only_what_was_asked_for():
    explanation = explain_methods(make_model(), make_batch(), ("gradient_attention", "occlusion"))
    assert set(explanation.maps) == {"gradient_attention", "occlusion"}


def test_explaining_refuses_training_mode_and_concat_fusion():
    model = make_model().train()
    with pytest.raises(ValueError, match="eval"):
        explain_batch(model, make_batch())
    model = make_model()
    model.fusion_type = "concat"
    with pytest.raises(ValueError, match="cross_attention"):
        explain_batch(model, make_batch())


def test_delete_top_patches_blanks_the_right_cells_only():
    image = torch.ones(1, 3, 4, 4)
    scores = torch.tensor([[0.0, 3.0, 2.0, 1.0]])  # grid: [[0, 3], [2, 1]]
    expected = torch.ones(4, 4)
    expected[:2, 2:] = 0.0
    assert torch.equal(delete_top_patches(image, scores, 0.25)[0, 0], expected)
    expected[2:, :2] = 0.0
    assert torch.equal(delete_top_patches(image, scores, 0.5)[0, 0], expected)


def test_delete_top_patches_can_fill_from_another_image():
    image, fill = torch.ones(1, 3, 4, 4), torch.full((1, 3, 4, 4), 5.0)
    result = delete_top_patches(image, torch.tensor([[0.0, 3.0, 2.0, 1.0]]), 0.25, fill)[0, 0]
    assert result[0, 3] == 5.0 and result[3, 0] == 1.0


def test_blur_keeps_the_shape_and_removes_detail():
    image = torch.randn(2, 3, 32, 32)
    blurred = blur_image(image, 4.0)
    assert blurred.shape == image.shape
    assert blurred.std() < image.std() / 3


def test_a_non_square_map_raises():
    with pytest.raises(ValueError, match="square"):
        delete_top_patches(torch.ones(1, 3, 4, 4), torch.ones(1, 5), 0.5)


def test_ties_break_at_random_but_real_differences_keep_their_order():
    generator = torch.Generator().manual_seed(0)
    flat = rank_scores(torch.zeros(200, 64), generator).topk(1, dim=1).indices
    assert flat.unique().numel() > 10
    real = rank_scores(torch.tensor([[0.0, 0.5, 1.0, 0.25]]), generator)
    assert real.argsort(descending=True).tolist() == [[2, 1, 3, 0]]


def test_a_tiny_map_keeps_its_ranking():
    # softplus can make a count row's Grad-CAM ~1e-20; its shape still counts
    scores = rank_scores(torch.tensor([[1e-20, 3e-20, 2e-20, 0.0]]), torch.Generator().manual_seed(0))
    assert scores.argsort(descending=True).tolist() == [[1, 2, 0, 3]]


def test_random_block_deletes_one_connected_patch_of_the_right_size():
    generator = torch.Generator().manual_seed(0)
    block = random_block_map(torch.Size((50, 64)), generator)
    assert block.argmax(dim=1).unique().numel() > 10  # centres vary
    scores = rank_scores(block, generator)
    for row in scores:
        deleted = delete_top_patches(torch.ones(1, 3, 8, 8), row[None], 0.2)[0, 0] == 0
        assert int(deleted.sum()) == round(0.2 * 64)
        cells = {tuple(cell) for cell in deleted.nonzero().tolist()}
        seen, stack = set(), [next(iter(cells))]
        while stack:
            r, c = stack.pop()
            if (r, c) in cells and (r, c) not in seen:
                seen.add((r, c))
                stack += [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        assert seen == cells


def test_deletion_effect_uses_the_right_measure_per_row():
    base = torch.log(torch.tensor([[0.8, 0.2], [0.5, 0.5]]))
    after = torch.log(torch.tensor([[0.3, 0.7], [0.5, 0.5]]))
    is_count = torch.tensor([False, True])
    effect, changed = deletion_effect(
        base, torch.tensor([9.0, math.e - 1]), after, torch.tensor([9.0, 0.0]), is_count,
    )
    assert effect[0].item() == pytest.approx(0.5, abs=1e-6)
    assert effect[1].item() == pytest.approx(1.0, abs=1e-6)
    assert changed.tolist() == [True, True]


def test_spearman_averages_ties_and_gives_nan_for_a_flat_map():
    a = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]])
    b = torch.tensor([[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]])
    result = spearman(a, b)
    assert result[0].item() == pytest.approx(1.0)
    assert math.isnan(result[1].item())
    # ranks [1.5, 1.5, 3, 4] and [1, 2, 3, 4]
    tied = spearman(torch.tensor([[0.0, 0.0, 1.0, 2.0]]), torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    assert tied.item() == pytest.approx(4.5 / math.sqrt(22.5), abs=1e-5)


def test_top_overlap_is_intersection_over_union():
    a = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    b = torch.tensor([[4.0, 1.0, 3.0, 2.0]])
    assert top_overlap(a, b, 0.5).item() == pytest.approx(1 / 3)


def test_randomized_copy_changes_every_weight_and_leaves_the_original():
    model = make_model()
    with torch.no_grad():  # as after training: a fresh model's norm weights already equal their reset value
        for parameter in model.parameters():
            parameter.add_(0.1)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    clone = randomized_copy(model)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    for name, value in clone.named_parameters():
        assert not torch.equal(value, before[name]), name
    assert not clone.training


def records_for_sampling() -> list[dict]:
    return [
        {"img_id": tile, "type": qtype, "question": f"{qtype} {i}", "answer": "yes"}
        for tile in range(5) for qtype in ("presence", "count") for i in range(3)
    ]


def test_pool_has_one_question_per_tile_per_type_and_is_seeded():
    records = records_for_sampling()
    pool = question_pool(records, 0)
    assert sorted((records[i]["img_id"], records[i]["type"]) for i in pool) == sorted(
        (tile, qtype) for tile in range(5) for qtype in ("presence", "count")
    )
    assert pool == question_pool(records, 0)
    assert pool != question_pool(records, 1)


def test_balanced_sample_caps_each_type_and_outcome():
    records = records_for_sampling()
    pool = question_pool(records, 0)
    correct = [i % 2 == 0 for i in range(len(pool))]
    sample = balanced_sample(pool, records, correct, 1)
    outcome = dict(zip(pool, correct))
    keys = [(records[i]["type"], outcome[i]) for i in sample]
    assert len(keys) == len(set(keys)) == 4


def test_question_pairs_share_a_tile_and_differ_in_type():
    records = records_for_sampling()
    pairs = question_pairs(records, 0, 3)
    assert len(pairs) == 3
    for a, b in pairs:
        assert records[a]["img_id"] == records[b]["img_id"]
        assert records[a]["type"] != records[b]["type"]


def test_paired_summary_matches_hand_computed_values():
    summary = paired_summary(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    std = math.sqrt(5 / 3)
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["lower"] == pytest.approx(2.5 - 1.96 * std / 2, abs=1e-5)
    assert summary["standardized"] == pytest.approx(2.5 / std, abs=1e-5)


def good_row(standardized: float = 1.0) -> dict[str, float]:
    return {"mean": 0.1, "lower": 0.05, "standardized": standardized}


def test_choose_map_picks_the_largest_effect_among_those_that_qualify():
    comparison = {
        "a": {"presence": good_row(0.5), "count": good_row(0.5)},
        "b": {"presence": good_row(1.0), "count": good_row(0.8)},
    }
    chosen, _ = choose_map(comparison, {"a": 0.1, "b": 0.1}, {"a": 0.1, "b": 0.1}, 0.5, 0.9)
    assert chosen == "b"


@pytest.mark.parametrize("sanity,question,bound,reason", [
    (0.5, 0.1, 0.05, "randomization"),
    (0.1, 0.9, 0.05, "question"),
    (0.1, 0.1, 0.0, "beat random"),
    (float("nan"), 0.1, 0.05, "randomization"),
])
def test_choose_map_rejects_a_map_that_fails_any_condition(sanity, question, bound, reason):
    comparison = {"a": {"presence": {"mean": 0.1, "lower": bound, "standardized": 9.0}}}
    chosen, reasons = choose_map(comparison, {"a": sanity}, {"a": question}, 0.5, 0.9)
    assert chosen is None
    assert reason in reasons["a"]


def xai_config(tmp_path: Path) -> dict:
    config = hr_config(tmp_path)
    config.update(
        cnn_channels=[8, 16], d_model=32, num_transformer_layers=2, num_heads=2, ffn_dim=64, dropout=0.1,
        question_embed_dim=32, fusion_hidden_dim=32, fusion_type="cross_attention", mixed_precision=False,
        xai_seed=0, xai_batch_size=2, xai_questions_per_group=3, xai_pilot_questions_per_group=1,
        xai_check_tiles=2, xai_pilot_check_tiles=1, xai_delete_fractions=[0.1, 0.5], xai_top_overlap_fraction=0.2,
        xai_deletion_fill="mean", xai_blur_sigma=4, xai_latency_runs=1, xai_figure_dir=str(tmp_path / "figures"), xai_sanity_max_correlation=0.5,
        xai_question_max_correlation=0.9,
    )
    return config


@pytest.fixture
def trained_run(tmp_path: Path) -> tuple[Path, dict]:
    write_fake_hr(tmp_path / "raw")
    config = xai_config(tmp_path)
    from src.data_pipeline import get_dataloaders

    get_dataloaders(config)
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    torch.save({"model_state_dict": build_model(config).state_dict(), "config": config},
               run_dir / "checkpoints" / "best.pt")
    return run_dir, config


def test_run_xai_writes_the_report_and_figures(trained_run, tmp_path: Path):
    run_dir, config = trained_run
    report = run_xai(run_dir, config)
    assert "**Map for the RAG layer:" in report
    assert "## Deletion test: test\n" in report and "## Deletion test: test_phili\n" in report
    for method in ("cross_attention", "rollout", "grad_cam", "random", "random_block"):
        assert f"| {method} |" in report
    assert "## Against a random block (secondary" in report
    assert "## Against the random map: test alone" in report and "## Against the random map: test_phili alone" in report
    assert "## Against the random map" in report and "s for one answer and its maps" in report
    figures = list((tmp_path / "figures" / "run").glob("*.png"))
    assert figures and all(path.stat().st_size > 0 for path in figures)
    assert {path.name.split("_")[0] for path in figures} == {"test"}  # test_* and test_phili_*
    assert any(path.name.startswith("test_phili_") for path in figures)


def test_run_xai_pilot_uses_its_own_sample_size_and_folder(trained_run, tmp_path: Path):
    run_dir, config = trained_run
    report = run_xai(run_dir, config, pilot=True)
    assert "(pilot subsets)" in report
    assert list((tmp_path / "figures" / "pilot").glob("*.png"))


def test_new_methods_run_on_val_with_blurred_deletion(trained_run, tmp_path: Path):
    run_dir, config = trained_run
    config = {**config, "xai_deletion_fill": "blur"}
    report = run_xai(run_dir, config, methods=("gradient_attention", "occlusion"), stage="val", tag="_new_val")
    assert "Splits: val. Deleted cells are replaced with: blur." in report
    assert "## Deletion test: val\n" in report and "## Deletion test: test" not in report
    assert "| gradient_attention |" in report and "| occlusion |" in report
    assert "| rollout |" not in report
    assert "alone (does not decide" not in report  # only one split
    assert list((tmp_path / "figures" / "run_new_val").glob("*.png"))
