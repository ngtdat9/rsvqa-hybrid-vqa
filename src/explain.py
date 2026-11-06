"""Attention and saliency maps for HybridVQAModel, and tests of whether they are faithful.

Three maps over the patch grid, each [B,N] with N = 16x16 tokens for HR:
  * ``cross_attention``: attention from the question's real words to the patch tokens.
  * ``rollout``: that attention traced back through the encoder's self-attention. The
    usual rollout follows the CLS token, but cross-attention fusion never reads CLS, so
    it would explain an output the answer does not come from.
  * ``grad_cam``: gradient-weighted activations of the last CNN stage, for the predicted
    class (classification rows) or the predicted count (count rows).

Two further maps, added after those three failed, are selected on val and confirmed on test:
  * ``gradient_attention``: every attention step weighted by its gradient for this answer
    (Chefer et al., 2021), so that the path depends on the question.
  * ``occlusion``: blank each grid cell in turn and measure how much the answer changes.

    python -m src.explain <run_dir> configs/hr_config.yaml [--pilot]
    python -m src.explain <run_dir> configs/hr_config.yaml --new-methods occlusion,gradient_attention --stage val

A follow-up after occlusion was confirmed, which cannot change that choice: whether black no-data
strips drive the result, and how the first three maps do under the same blur deletion.

    python -m src.explain <run_dir> configs/hr_config.yaml --followup [--pilot]
"""
from __future__ import annotations

import argparse
import copy
import math
import random
import textwrap
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from torch import nn
from torch.nn.init import trunc_normal_
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from src.data_pipeline import prepare_batch, vqa_collate_fn
from src.evaluate import load_run

METHODS = ("cross_attention", "rollout", "grad_cam")
NEW_METHODS = ("gradient_attention", "occlusion")
CONTROL = "random"
BLOCK_CONTROL = "random_block"  # secondary: reported, but has no say in choose_map
_TIE_BREAK = 1e-5  # far below any real score difference on a map scaled to max 1


@dataclass
class Explanation:
    maps: dict[str, torch.Tensor]  # method -> [B,N]
    class_logits: torch.Tensor     # [B,vocab_size]
    count_pred: torch.Tensor       # [B]
    layer_attention: list[torch.Tensor]  # per encoder layer, first to last: [B,N+1,N+1]


def rollout_matrix(layer_attention: list[torch.Tensor]) -> torch.Tensor:
    """Abnar & Zuidema's rollout: row i says how much output token i draws on each input
    token. Each layer is averaged with the identity to stand for its residual connection."""
    result: torch.Tensor | None = None
    for weights in layer_attention:
        identity = torch.eye(weights.size(-1), device=weights.device, dtype=weights.dtype)
        step = 0.5 * weights + 0.5 * identity
        step = step / step.sum(dim=-1, keepdim=True)
        result = step if result is None else step @ result
    if result is None:
        raise ValueError("layer_attention is empty")
    return result


def word_attention_map(cross_attention: torch.Tensor, question_len: torch.Tensor) -> torch.Tensor:
    """Mean over the question's real words: [B,L,N] -> [B,N]. Padding also gets weights, so
    it must be left out."""
    positions = torch.arange(cross_attention.size(1), device=cross_attention.device)
    mask = (positions[None, :] < question_len[:, None]).unsqueeze(-1).to(cross_attention.dtype)
    return (cross_attention * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def question_rollout_map(word_map: torch.Tensor, layer_attention: list[torch.Tensor]) -> torch.Tensor:
    """Trace [B,N] word attention over output patches back to the input patches. Token 0 is
    CLS: the fusion reads no CLS output, and what patches drew from it maps to no location."""
    patches = rollout_matrix(layer_attention)[:, 1:, 1:]
    return (word_map.unsqueeze(1) @ patches).squeeze(1)


def _target_scores(outputs: dict[str, torch.Tensor], is_count: torch.Tensor) -> torch.Tensor:
    """What a map explains: the predicted class's logit, or the predicted count."""
    predicted = outputs["class_logits"].argmax(dim=-1, keepdim=True)
    return torch.where(is_count, outputs["count_pred"], outputs["class_logits"].gather(1, predicted).squeeze(1))


def explain_batch(model: nn.Module, batch: dict[str, Any]) -> Explanation:
    """All three maps from one forward and one backward pass, in float32.

    PyTorch's encoder layers discard their attention weights and can skip
    ``self_attn.forward`` altogether on an evaluation fast path, so a hook there may capture
    nothing. Each layer's input is captured instead, and the weights recomputed from it.
    """
    if model.training:
        raise ValueError("explain_batch needs model.eval(): dropout would change the maps")
    if model.fusion_type != "cross_attention":
        raise ValueError(f"explain_batch needs cross_attention fusion, got {model.fusion_type!r}")

    layer_inputs: list[torch.Tensor] = []
    stage_output: list[torch.Tensor] = []
    layers = list(model.encoder.transformer.layers)
    hooks = [layer.register_forward_pre_hook(lambda _m, args: layer_inputs.append(args[0])) for layer in layers]
    hooks.append(model.encoder.cnn_stem.blocks[-1].register_forward_hook(
        lambda _m, _args, output: stage_output.append(output)
    ))
    try:
        with torch.enable_grad():
            outputs = model(batch, return_attention=True)
            is_count = ~batch["count_target"].isnan()
            activations = stage_output[0]
            gradients = torch.autograd.grad(_target_scores(outputs, is_count).sum(), activations)[0]
    finally:
        for hook in hooks:
            hook.remove()

    with torch.no_grad():
        layer_attention = [
            layer.self_attn(x, x, x, need_weights=True, average_attn_weights=True)[1]
            for layer, x in zip(layers, layer_inputs)
        ]
        word_map = word_attention_map(outputs["cross_attention"], batch["question_len"])
        channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
        grad_cam = F.relu((channel_weights * activations).sum(dim=1)).flatten(1)
    return Explanation(
        maps={
            "cross_attention": word_map.detach(),
            "rollout": question_rollout_map(word_map, layer_attention).detach(),
            "grad_cam": grad_cam.detach(),
        },
        class_logits=outputs["class_logits"].detach(),
        count_pred=outputs["count_pred"].detach(),
        layer_attention=[w.detach() for w in layer_attention],
    )


def _attention_with_weights(
    attention: nn.MultiheadAttention, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MultiheadAttention's computation in eval mode (batch first). Its own returned weights
    are a copy taken after the output, so no gradient reaches them; these [B,H,L,N] do feed it."""
    batch_size, length, dim = query.shape
    head_dim = attention.head_dim

    def project(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return F.linear(x, weight, bias).view(x.size(0), x.size(1), attention.num_heads, head_dim).transpose(1, 2)

    weights_q, weights_k, weights_v = attention.in_proj_weight.chunk(3)
    bias_q, bias_k, bias_v = attention.in_proj_bias.chunk(3)
    q = project(query, weights_q, bias_q)
    k = project(key, weights_k, bias_k)
    v = project(value, weights_v, bias_v)
    weights = (q @ k.transpose(-2, -1) / math.sqrt(head_dim)).softmax(dim=-1)
    return attention.out_proj((weights @ v).transpose(1, 2).reshape(batch_size, length, dim)), weights


def gradient_attention_map(model: nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    """Chefer et al.'s relevance through the encoder and the fusion: at each attention step keep
    the positive part of gradient x attention, averaged over heads. [B,N] over input patches."""
    modules = [layer.self_attn for layer in model.encoder.transformer.layers] + [model.fusion.attn]
    captured: dict[nn.Module, torch.Tensor] = {}

    def patched(module: nn.Module) -> Any:
        def forward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **_: Any) -> tuple[torch.Tensor, None]:
            output, captured[module] = _attention_with_weights(module, query, key, value)
            return output, None
        return forward

    for module in modules:
        module.forward = patched(module)
    try:
        with torch.enable_grad():
            outputs = model(batch)
            scores = _target_scores(outputs, ~batch["count_target"].isnan())
            gradients = torch.autograd.grad(scores.sum(), [captured[m] for m in modules])
    finally:
        for module in modules:
            del module.forward

    with torch.no_grad():
        steps = [
            ((gradient * captured[module]).clamp(min=0)).mean(dim=1)
            for module, gradient in zip(modules, gradients)
        ]
        size = steps[0].size(-1)
        relevance = torch.eye(size, device=steps[0].device, dtype=steps[0].dtype).expand(steps[0].size(0), -1, -1)
        for step in steps[:-1]:
            relevance = relevance + step @ relevance
        word_map = word_attention_map(steps[-1], batch["question_len"])
        return (word_map.unsqueeze(1) @ relevance[:, 1:, 1:]).squeeze(1)


@torch.no_grad()
def occlusion_map(
    model: nn.Module, batch: dict[str, Any], base_logits: torch.Tensor, base_count: torch.Tensor, num_patches: int,
) -> torch.Tensor:
    """Per cell, the change in the answer when that cell alone is blanked to the train mean. [B,N]."""
    is_count = ~batch["count_target"].isnan()
    cells = []
    for cell in range(num_patches):
        scores = torch.zeros(base_count.size(0), num_patches, device=base_count.device)
        scores[:, cell] = 1.0
        outputs = model({**batch, "image": delete_top_patches(batch["image"], scores, 1 / num_patches)})
        cells.append(deletion_effect(base_logits, base_count, outputs["class_logits"], outputs["count_pred"], is_count)[0])
    return torch.stack(cells, dim=1)


def explain_methods(model: nn.Module, batch: dict[str, Any], methods: tuple[str, ...]) -> Explanation:
    """``explain_batch``, keeping only ``methods`` and adding any of the slower maps asked for."""
    explanation = explain_batch(model, batch)
    maps = {method: explanation.maps[method] for method in methods if method in explanation.maps}
    if "gradient_attention" in methods:
        maps["gradient_attention"] = gradient_attention_map(model, batch)
    if "occlusion" in methods:
        num_patches = explanation.maps["rollout"].size(1)
        maps["occlusion"] = occlusion_map(model, batch, explanation.class_logits, explanation.count_pred, num_patches)
    explanation.maps = maps
    return explanation


def rank_scores(maps: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Scores for ranking patches, scaled to max 1. Ties break at random, or an all-zero
    Grad-CAM would always delete the top-left corner."""
    scale = maps.amax(dim=1, keepdim=True)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))  # scale, not clamp: softplus makes count maps tiny
    noise = torch.rand(maps.shape, generator=generator).to(maps.device)
    return maps / scale + _TIE_BREAK * noise


def random_block_map(shape: torch.Size, generator: torch.Generator) -> torch.Tensor:
    """A [B,N] map whose top patches form one compact patch around a random centre, at any
    deletion size. Scattered random cells can disturb more of the image than clustered ones,
    since each token sees beyond its own cell; this control measures clustered deletion."""
    batch_size, num_patches = shape
    grid = round(math.sqrt(num_patches))
    rows, cols = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    centres = torch.randint(0, grid, (batch_size, 2), generator=generator)
    squared = (rows.flatten() - centres[:, :1]) ** 2 + (cols.flatten() - centres[:, 1:]) ** 2
    return 1 / (1 + squared.float().sqrt())


def blur_image(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Heavy Gaussian blur, a second way to remove detail: occlusion builds its map by blanking
    to the mean, so testing it the same way would favour it by construction."""
    radius = math.ceil(2 * sigma)
    return TF.gaussian_blur(image, kernel_size=[2 * radius + 1] * 2, sigma=[sigma, sigma])


def delete_top_patches(
    image: torch.Tensor, scores: torch.Tensor, fraction: float, fill: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replace the top ``fraction`` of grid cells with ``fill``. By default that is zero: images
    are normalized by the train-split mean and std, so zero is the train mean colour and no new
    statistic is needed."""
    num_patches = scores.size(1)
    grid = round(math.sqrt(num_patches))
    if grid * grid != num_patches:
        raise ValueError(f"{num_patches} patches do not form a square grid")
    top = scores.topk(max(1, round(fraction * num_patches)), dim=1).indices
    mask = torch.zeros_like(scores).scatter(1, top, 1.0).view(-1, 1, grid, grid)
    mask = F.interpolate(mask, size=image.shape[-2:], mode="nearest")
    return image * (1 - mask) if fill is None else image * (1 - mask) + fill * mask


def no_data_cells(pixels: torch.Tensor, grid: int, pixel_value: float, cell_fraction: float) -> torch.Tensor:
    """[B,N], True for grid cells where at least ``cell_fraction`` of the pixels are black in
    every channel. ``pixels`` is [B,3,H,W] in the original 0-1 range, before normalization."""
    black = (pixels.amax(dim=1, keepdim=True) <= pixel_value).float()
    return F.adaptive_avg_pool2d(black, grid).flatten(1) >= cell_fraction


def cells_in_or_beside(cells: torch.Tensor) -> torch.Tensor:
    """[B,N] mask grown to the eight neighbouring cells."""
    grid = round(math.sqrt(cells.size(1)))
    grown = F.max_pool2d(cells.float().view(-1, 1, grid, grid), kernel_size=3, stride=1, padding=1)
    return grown.flatten(1).bool()


def outer_ring(num_patches: int) -> torch.Tensor:
    """[N] mask of the grid's outermost cells. Strips lie along edges, so a map that favours
    edges anyway would seem drawn to strips; this is the control for that."""
    grid = round(math.sqrt(num_patches))
    rows, cols = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    return ((rows == 0) | (cols == 0) | (rows == grid - 1) | (cols == grid - 1)).flatten()


def top_cell_share(scores: torch.Tensor, zone: torch.Tensor, fraction: float) -> torch.Tensor:
    """Per row, the share of the top ``fraction`` of cells that fall inside ``zone``."""
    top = scores.topk(max(1, round(fraction * scores.size(1))), dim=1).indices
    return zone.gather(1, top).float().mean(dim=1)


def deletion_effect(
    base_logits: torch.Tensor, base_count: torch.Tensor, logits: torch.Tensor, count: torch.Tensor,
    is_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row: (effect, answer changed). The effect is the drop in the predicted class's
    probability, or for count rows the absolute change in log1p(count), the scale of the
    count loss, since a few huge counts would otherwise dominate."""
    predicted = base_logits.argmax(dim=-1, keepdim=True)
    drop = (base_logits.softmax(-1).gather(1, predicted) - logits.softmax(-1).gather(1, predicted)).squeeze(1)
    log_change = (torch.log1p(base_count) - torch.log1p(count)).abs()
    class_changed = logits.argmax(dim=-1) != predicted.squeeze(1)
    count_changed = base_count.round() != count.round()
    return torch.where(is_count, log_change, drop), torch.where(is_count, count_changed, class_changed)


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    _, inverse, counts = torch.unique(values, return_inverse=True, return_counts=True)
    ends = counts.cumsum(0)
    return ((ends - counts + ends - 1).float() / 2)[inverse]


def spearman(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Rank correlation per row of two [B,N] maps, ties averaged. NaN where a map is constant,
    since a flat map has no ranking to compare."""
    results = []
    for row_a, row_b in zip(a.cpu(), b.cpu()):
        if row_a.unique().numel() < 2 or row_b.unique().numel() < 2:
            results.append(torch.tensor(float("nan")))
        else:
            results.append(torch.corrcoef(torch.stack([_average_ranks(row_a), _average_ranks(row_b)]))[0, 1])
    return torch.stack(results)


def top_overlap(a: torch.Tensor, b: torch.Tensor, fraction: float) -> torch.Tensor:
    """Intersection over union of each row's top ``fraction`` of patches."""
    k = max(1, round(fraction * a.size(1)))
    top_a = torch.zeros_like(a, dtype=torch.bool).scatter(1, a.topk(k, dim=1).indices, True)
    top_b = torch.zeros_like(b, dtype=torch.bool).scatter(1, b.topk(k, dim=1).indices, True)
    return (top_a & top_b).sum(1).float() / (top_a | top_b).sum(1).float()


def randomized_copy(model: nn.Module) -> nn.Module:
    """A copy with every weight re-initialized. Maps that survive this were not explaining
    what the model learned."""
    clone = copy.deepcopy(model)
    for module in clone.modules():
        for name in ("reset_parameters", "_reset_parameters"):
            if module is not clone and hasattr(module, name):
                getattr(module, name)()
    trunc_normal_(clone.encoder.cls_token, std=0.02)
    trunc_normal_(clone.encoder.pos_embed, std=0.02)
    return clone.eval()


def question_pool(records: list[dict], seed: int) -> list[int]:
    """Indices of one random question per tile per type, so no tile dominates a type."""
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    chosen: dict[tuple[int, str], int] = {}
    for idx in order:
        chosen.setdefault((records[idx]["img_id"], records[idx]["type"]), idx)
    return list(chosen.values())


def balanced_sample(
    pool: list[int], records: list[dict], correct: list[bool], per_group: int,
) -> list[int]:
    """Up to ``per_group`` indices per (type, right or wrong), keeping ``pool``'s random order."""
    taken: dict[tuple[str, bool], int] = defaultdict(int)
    sample = []
    for idx, is_correct in zip(pool, correct):
        key = (records[idx]["type"], is_correct)
        if taken[key] < per_group:
            taken[key] += 1
            sample.append(idx)
    return sample


def question_pairs(records: list[dict], seed: int, max_tiles: int) -> list[tuple[int, int]]:
    """Per tile, two questions of different types, for the question-sensitivity check."""
    by_tile: dict[int, dict[str, int]] = defaultdict(dict)
    for idx in question_pool(records, seed):
        by_tile[records[idx]["img_id"]][records[idx]["type"]] = idx
    tiles = sorted(tile for tile, by_type in by_tile.items() if len(by_type) >= 2)
    random.Random(seed).shuffle(tiles)
    return [tuple(list(by_tile[tile].values())[:2]) for tile in tiles[:max_tiles]]


def paired_summary(differences: torch.Tensor) -> dict[str, float]:
    """Mean per-question difference from the random map, its 95% lower bound, and the mean
    in standard deviations (comparable between confidence and count effects)."""
    mean = differences.mean().item()
    std = differences.std().item() if differences.numel() > 1 else 0.0
    lower = mean - 1.96 * std / math.sqrt(differences.numel())
    return {"mean": mean, "lower": lower, "standardized": mean / std if std > 0 else 0.0}


def choose_map(
    random_comparison: dict[str, dict[str, dict[str, float]]], sanity: dict[str, float],
    question: dict[str, float], sanity_limit: float, question_limit: float,
) -> tuple[str | None, dict[str, str]]:
    """The rule fixed before any result: the largest mean standardized deletion effect across
    question types, among maps that beat random on every type and pass both checks."""
    reasons: dict[str, str] = {}
    effects: dict[str, float] = {}
    for method, by_type in random_comparison.items():
        failed = [qtype for qtype, row in by_type.items() if not row["lower"] > 0]
        if failed:
            reasons[method] = f"does not beat random on: {', '.join(sorted(failed))}"
        elif not sanity[method] < sanity_limit:
            reasons[method] = f"survives weight randomization (correlation {sanity[method]:.3f})"
        elif not question[method] < question_limit:
            reasons[method] = f"barely changes with the question (correlation {question[method]:.3f})"
        else:
            effects[method] = sum(row["standardized"] for row in by_type.values()) / len(by_type)
            reasons[method] = f"qualifies, mean standardized effect {effects[method]:.3f}"
    return (max(effects, key=effects.get) if effects else None), reasons


def _answer_text(class_logits: torch.Tensor, count_pred: torch.Tensor, is_count: bool, idx_to_class: list[str]) -> str:
    return f"{count_pred.item():.1f}" if is_count else idx_to_class[class_logits.argmax().item()]


def save_example_figure(
    pixels: torch.Tensor, maps: dict[str, torch.Tensor], question: str, truth: str, predicted: str, path: Path,
) -> None:
    """The tile beside each method's map, as blocks over the grid: smoothing would suggest
    more precision than the maps have. Each map is scaled to its own range."""
    fig, axes = plt.subplots(1, len(maps) + 1, figsize=(3.2 * (len(maps) + 1), 3.9), layout="constrained")
    tile = pixels.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    axes[0].imshow(tile)
    axes[0].set_title("Tile", fontsize=10)
    size = pixels.shape[-1]
    for axis, (method, values) in zip(axes[1:], maps.items()):
        grid = round(math.sqrt(values.numel()))
        low, high = values.min(), values.max()
        span = high - low
        scaled = ((values - low) / (span if span > 0 else 1.0)).view(grid, grid).cpu().numpy()
        axis.imshow(tile, alpha=0.65)
        axis.imshow(scaled, cmap="inferno", alpha=0.6, extent=(0, size, size, 0), interpolation="nearest", vmin=0, vmax=1)
        axis.set_title(method.replace("_", " "), fontsize=10)
    for axis in axes:
        axis.set_axis_off()
    fig.suptitle(textwrap.fill(f"{question}  Truth: {truth}.  Model: {predicted}.", 90), fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


@dataclass
class SplitResult:
    types: list[str] = field(default_factory=list)
    correct: list[bool] = field(default_factory=list)
    effects: dict[str, list[torch.Tensor]] = field(default_factory=lambda: defaultdict(list))  # [B,fractions]
    changed: dict[str, list[torch.Tensor]] = field(default_factory=lambda: defaultdict(list))  # [B,fractions]
    agreement: dict[tuple[str, str], list[torch.Tensor]] = field(default_factory=lambda: defaultdict(list))
    overlap: dict[tuple[str, str], list[torch.Tensor]] = field(default_factory=lambda: defaultdict(list))
    flat_maps: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    figures: list[Path] = field(default_factory=list)
    no_data: list[torch.Tensor] = field(default_factory=list)  # [B], no-data cells per question
    zone_share: list[torch.Tensor] = field(default_factory=list)  # [B], share of the tile in or beside no-data cells
    top_in_zone: dict[str, list[torch.Tensor]] = field(default_factory=lambda: defaultdict(list))  # [B]
    top_on_edge: dict[str, list[torch.Tensor]] = field(default_factory=lambda: defaultdict(list))  # [B]

    def has_strip(self) -> torch.Tensor:
        return torch.cat(self.no_data) > 0


def _loader(dataset: Dataset, indices: list[int], config: dict[str, Any]) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices), batch_size=config["xai_batch_size"], shuffle=False, collate_fn=vqa_collate_fn,
        num_workers=config["num_workers"], pin_memory=torch.cuda.is_available(),
    )


def _is_correct(class_logits: torch.Tensor, count_pred: torch.Tensor, batch: dict[str, Any]) -> list[bool]:
    is_count = ~batch["count_target"].isnan()
    target = batch["count_target"].nan_to_num(-1.0)
    return torch.where(is_count, count_pred.round() == target, class_logits.argmax(-1) == batch["label"]).tolist()


@torch.no_grad()
def predict_correct(
    model: nn.Module, dataset: Dataset, indices: list[int], config: dict[str, Any], device: torch.device, desc: str,
    strips: list[bool] | None = None,
) -> list[bool]:
    """Whether the model answers each question correctly (count: rounded prediction == truth).
    If ``strips`` is given, whether each question's tile has a no-data strip is appended to it."""
    correct: list[bool] = []
    for batch in tqdm(_loader(dataset, indices, config), desc=desc, leave=False, unit="batch"):
        batch = prepare_batch(batch, device, dataset)
        outputs = model(batch)
        correct += _is_correct(outputs["class_logits"], outputs["count_pred"], batch)
        if strips is not None:
            pixels = batch["image"] * dataset.image_std.to(device).view(3, 1, 1) + dataset.image_mean.to(device).view(3, 1, 1)
            grid = round(math.sqrt(model.encoder.pos_embed.size(1) - 1))  # pos_embed also holds CLS
            found = no_data_cells(pixels, grid, config["xai_nodata_pixel_value"], config["xai_nodata_cell_fraction"])
            strips += found.any(dim=1).tolist()
    return correct


def score_split(
    model: nn.Module, dataset: Any, indices: list[int], config: dict[str, Any], device: torch.device,
    generator: torch.Generator, split: str, figure_dir: Path, methods: tuple[str, ...] = METHODS,
    extra_figures: int = 0,
) -> SplitResult:
    """Deletion test, agreement between maps, and one right and one wrong figure per type.
    The figures are the first of each kind in the random sample, whatever their maps show.
    ``extra_figures`` adds that many of two more kinds: presence questions answered "yes"
    correctly, where there is an object to point at, and tiles with a no-data strip."""
    result = SplitResult()
    fractions = config["xai_delete_fractions"]
    records = dataset.qa_records
    drawn: set[tuple[str, bool]] = set()
    extra_drawn: dict[str, int] = defaultdict(int)
    position = 0
    for batch in tqdm(_loader(dataset, indices, config), desc=f"explain {split}", leave=False, unit="batch"):
        batch = prepare_batch(batch, device, dataset)
        explanation = explain_methods(model, batch, methods)
        fill = blur_image(batch["image"], config["xai_blur_sigma"]) if config["xai_deletion_fill"] == "blur" else None
        mean = dataset.image_mean.to(device).view(3, 1, 1)
        std = dataset.image_std.to(device).view(3, 1, 1)
        pixels = batch["image"] * std + mean
        is_count = ~batch["count_target"].isnan()
        correct = _is_correct(explanation.class_logits, explanation.count_pred, batch)
        maps = dict(explanation.maps)
        shape = maps[methods[0]].shape
        no_data = no_data_cells(
            pixels, round(math.sqrt(shape[1])), config["xai_nodata_pixel_value"], config["xai_nodata_cell_fraction"],
        )
        zone = cells_in_or_beside(no_data)
        result.no_data.append(no_data.sum(dim=1).cpu())
        result.zone_share.append(zone.float().mean(dim=1).cpu())
        maps[CONTROL] = torch.rand(shape, generator=generator).to(device)
        maps[BLOCK_CONTROL] = random_block_map(shape, generator).to(device)

        for method in methods:
            result.flat_maps[method] += int((maps[method].amax(1) == maps[method].amin(1)).sum())
        for i, a in enumerate(methods):
            for b in methods[i + 1:]:
                result.agreement[(a, b)].append(spearman(maps[a], maps[b]))
                result.overlap[(a, b)].append(top_overlap(maps[a], maps[b], config["xai_top_overlap_fraction"]).cpu())

        for method, values in maps.items():
            scores = rank_scores(values, generator)
            result.top_in_zone[method].append(top_cell_share(scores, zone, config["xai_top_overlap_fraction"]).cpu())
            edge = outer_ring(shape[1]).to(device).expand(shape[0], -1)
            result.top_on_edge[method].append(top_cell_share(scores, edge, config["xai_top_overlap_fraction"]).cpu())
            effects, changed = [], []
            for fraction in fractions:
                deleted = {**batch, "image": delete_top_patches(batch["image"], scores, fraction, fill)}
                with torch.no_grad():
                    outputs = model(deleted)
                effect, flip = deletion_effect(
                    explanation.class_logits, explanation.count_pred, outputs["class_logits"], outputs["count_pred"],
                    is_count,
                )
                effects.append(effect.cpu())
                changed.append(flip.cpu())
            result.effects[method].append(torch.stack(effects, dim=1))
            result.changed[method].append(torch.stack(changed, dim=1))

        for row, is_right in enumerate(correct):
            record = records[indices[position + row]]
            kind = (record["type"], is_right)
            names = []
            if kind not in drawn:
                drawn.add(kind)
                names.append(f"{split}_{record['type']}_{'right' if is_right else 'wrong'}")
            extras = []
            if record["type"] == "presence" and record["answer"] == "yes" and is_right:
                extras.append("presence_yes")
            if no_data[row].any():
                extras.append("strip")
            for extra in extras:
                if extra_drawn[extra] < extra_figures:
                    extra_drawn[extra] += 1
                    names.append(f"{split}_{extra}_{extra_drawn[extra]}")
            for name in names:
                path = figure_dir / f"{name}.png"
                predicted = _answer_text(
                    explanation.class_logits[row], explanation.count_pred[row], bool(is_count[row]),
                    dataset.answer_vocab.idx_to_class,
                )
                save_example_figure(
                    pixels[row], {m: explanation.maps[m][row] for m in methods},
                    record["question"], record["answer"], predicted, path,
                )
                result.figures.append(path)
            result.types.append(record["type"])
            result.correct.append(is_right)
        position += len(correct)
    return result


def sanity_check(
    model: nn.Module, dataset: Any, indices: list[int], config: dict[str, Any], device: torch.device,
    methods: tuple[str, ...] = METHODS,
) -> dict[str, torch.Tensor]:
    """Rank correlation between each map from the trained model and from random weights."""
    random_model = randomized_copy(model)
    correlations: dict[str, list[torch.Tensor]] = defaultdict(list)
    for batch in tqdm(_loader(dataset, indices, config), desc="randomized weights", leave=False, unit="batch"):
        batch = prepare_batch(batch, device, dataset)
        trained = explain_methods(model, batch, methods)
        randomized = explain_methods(random_model, batch, methods)
        for method in methods:
            correlations[method].append(spearman(trained.maps[method], randomized.maps[method]))
    return {method: torch.cat(values) for method, values in correlations.items()}


def question_check(
    model: nn.Module, dataset: Any, pairs: list[tuple[int, int]], config: dict[str, Any], device: torch.device,
    methods: tuple[str, ...] = METHODS,
) -> dict[str, torch.Tensor]:
    """Rank correlation between the maps of two different questions on the same tile."""
    correlations: dict[str, list[torch.Tensor]] = defaultdict(list)
    first = _loader(dataset, [a for a, _ in pairs], config)
    second = _loader(dataset, [b for _, b in pairs], config)
    for batch_a, batch_b in tqdm(zip(first, second), total=len(first), desc="question pairs", leave=False, unit="batch"):
        maps_a = explain_methods(model, prepare_batch(batch_a, device, dataset), methods).maps
        maps_b = explain_methods(model, prepare_batch(batch_b, device, dataset), methods).maps
        for method in methods:
            correlations[method].append(spearman(maps_a[method], maps_b[method]))
    return {method: torch.cat(values) for method, values in correlations.items()}


def measure_latency(
    model: nn.Module, dataset: Any, index: int, config: dict[str, Any], device: torch.device, runs: int,
    methods: tuple[str, ...] = METHODS,
) -> float:
    """Seconds for one answer and its maps, for one question: what a demo would wait for."""
    batch = prepare_batch(next(iter(_loader(dataset, [index], config))), device, dataset)
    explain_methods(model, batch, methods)  # the first call also pays for warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.monotonic()
    for _ in range(runs):
        explain_methods(model, batch, methods)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.monotonic() - started) / runs


def _nan_mean(values: torch.Tensor) -> float:
    return values.nanmean().item()


def _split_table(
    split: str, result: SplitResult, fractions: list[float], types: list[str], methods: tuple[str, ...],
) -> str:
    type_of = result.types
    header = " | ".join(f"{fraction:.0%}" for fraction in fractions)
    lines = [
        f"## Deletion test: {split}\n",
        "Effect: drop in the predicted class's probability, or for count the absolute change in "
        "log1p(count). Averages over questions; the sample is balanced between right and wrong answers.\n",
        f"| Type | Questions (right) | Map | {header} | Mean | Answer changed (mean) |",
        "|---|---|---|" + "---|" * (len(fractions) + 2),
    ]
    for qtype in types:
        rows = [i for i, t in enumerate(type_of) if t == qtype]
        if not rows:
            continue
        num_right = sum(result.correct[i] for i in rows)
        for method in (*methods, CONTROL, BLOCK_CONTROL):
            effects = torch.cat(result.effects[method])[rows]
            changed = torch.cat(result.changed[method])[rows].float()
            cells = " | ".join(f"{value:.4f}" for value in effects.mean(0).tolist())
            lines.append(
                f"| {qtype} | {len(rows)} ({num_right}) | {method} | {cells} | {effects.mean().item():.4f} "
                f"| {changed.mean().item():.1%} |"
            )
    return "\n".join(lines) + "\n"


def _comparison_table(title: str, intro: str, comparison: dict[str, dict[str, dict[str, float]]]) -> str:
    return (
        f"## {title}\n\n{intro}\n\n"
        "| Map | Type | Mean difference | 95% lower bound | Standardized |\n|---|---|---|---|---|\n"
        + "".join(
            f"| {method} | {qtype} | {row['mean']:.4f} | {row['lower']:.4f} | {row['standardized']:.3f} |\n"
            for method, by_type in comparison.items() for qtype, row in by_type.items()
        )
    )


def compare_with_control(
    results: dict[str, SplitResult], types: list[str], control: str, methods: tuple[str, ...] = METHODS,
    with_strip: bool | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Per map and type, pooled over test splits: each question's effect minus the control's.
    ``with_strip`` keeps only the questions on tiles with (True) or without (False) a no-data strip."""
    comparison: dict[str, dict[str, dict[str, float]]] = {}
    for method in methods:
        comparison[method] = {}
        for qtype in types:
            differences = []
            for result in results.values():
                strips = result.has_strip().tolist() if with_strip is not None else [None] * len(result.types)
                rows = [
                    i for i, t in enumerate(result.types)
                    if t == qtype and (with_strip is None or strips[i] == with_strip)
                ]
                if rows:
                    per_question = torch.cat(result.effects[method])[rows] - torch.cat(result.effects[control])[rows]
                    differences.append(per_question.mean(dim=1))
            if differences:
                comparison[method][qtype] = paired_summary(torch.cat(differences))
    return comparison


def run_xai(
    run_dir: Path, config: dict[str, Any], pilot: bool = False, methods: tuple[str, ...] = METHODS,
    stage: str = "test", tag: str = "",
) -> str:
    """Returns the markdown report; figures go to ``xai_figure_dir``. ``stage`` is ``val`` or
    ``test``: the later maps are selected on val before test sees them."""
    config = dict(config)
    config["cache_images"] = False  # a sample of tiles: decoding them beats caching 8 GB
    if not pilot:
        config["max_images_per_split"] = None
    per_group = config["xai_pilot_questions_per_group" if pilot else "xai_questions_per_group"]
    seed = config["xai_seed"]
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    figure_dir = Path(config["xai_figure_dir"]) / (("pilot" if pilot else run_dir.name) + tag)

    model, dataloaders, device = load_run(run_dir, config)
    model.eval()
    splits = ["val"] if stage == "val" else [s for s in config["splits"] if s.startswith("test")]
    types = [*config["question_types"], "count"]
    fractions = config["xai_delete_fractions"]

    results: dict[str, SplitResult] = {}
    sanity: dict[str, list[torch.Tensor]] = defaultdict(list)
    question: dict[str, list[torch.Tensor]] = defaultdict(list)
    latency: float | None = None
    for split in splits:
        dataset = dataloaders[split].dataset
        records = dataset.qa_records
        pool = question_pool(records, seed)
        correct = predict_correct(model, dataset, pool, config, device, f"predict {split}")
        sample = balanced_sample(pool, records, correct, per_group)
        if latency is None:
            latency = measure_latency(model, dataset, sample[0], config, device, config["xai_latency_runs"], methods)
        results[split] = score_split(model, dataset, sample, config, device, generator, split, figure_dir, methods)

        check_size = config["xai_pilot_check_tiles" if pilot else "xai_check_tiles"]
        for method, values in sanity_check(model, dataset, sample[:check_size], config, device, methods).items():
            sanity[method].append(values)
        pairs = question_pairs(records, seed, check_size)
        for method, values in question_check(model, dataset, pairs, config, device, methods).items():
            question[method].append(values)

    sanity_means = {m: _nan_mean(torch.cat(sanity[m])) for m in methods}
    question_means = {m: _nan_mean(torch.cat(question[m])) for m in methods}
    comparison = compare_with_control(results, types, CONTROL, methods)
    block_comparison = compare_with_control(results, types, BLOCK_CONTROL, methods)
    sanity_limit, question_limit = config["xai_sanity_max_correlation"], config["xai_question_max_correlation"]
    chosen, reasons = choose_map(comparison, sanity_means, question_means, sanity_limit, question_limit)

    sections = [
        f"# XAI results: {run_dir.name}" + (" (pilot subsets)" if pilot else "") + "\n\n"
        f"Splits: {', '.join(splits)}. Deleted cells are replaced with: {config['xai_deletion_fill']}.\n"
    ]
    sections.append(
        f"**Map for the RAG layer: {chosen or 'none qualifies; the maps must not be presented as evidence'}**\n\n"
        + "".join(f"- `{method}`: {reasons[method]}\n" for method in methods)
    )
    sections.append(_comparison_table(
        "Against the random map (scored splits pooled; this decides the choice)",
        "Per question, the effect averaged over deletion sizes, minus the random map's.", comparison,
    ))
    sections.append(_comparison_table(
        "Against a random block (secondary, scored splits pooled)",
        "The same, against one compact random patch of the same size. Added after the pilot, so it "
        "does not affect the choice of map; it shows whether scattered deletion made the random map a harsher baseline.",
        block_comparison,
    ))
    if len(results) > 1:  # pooling can hide a map that only works on one test set
        sections += [
            _comparison_table(
                f"Against the random map: {split} alone (does not decide the choice)",
                "The same as the pooled table, for this split only.",
                compare_with_control({split: result}, types, CONTROL, methods),
            )
            for split, result in results.items()
        ]
    sections.append(
        "## Checks\n\nMean rank correlation; a map fails at or above the limit.\n\n"
        f"| Map | Trained vs random weights (limit {sanity_limit}) | Two questions, same tile (limit {question_limit}) "
        "| All-constant maps |\n|---|---|---|---|\n"
        + "".join(
            f"| {m} | {sanity_means[m]:.3f} | {question_means[m]:.3f} "
            f"| {sum(r.flat_maps[m] for r in results.values())} of {sum(len(r.types) for r in results.values())} |\n"
            for m in methods
        )
    )
    agreement_rows = []
    for a, b in results[splits[0]].agreement:
        correlation = _nan_mean(torch.cat([v for r in results.values() for v in r.agreement[(a, b)]]))
        overlap = torch.cat([v for r in results.values() for v in r.overlap[(a, b)]]).mean().item()
        agreement_rows.append(f"| {a} vs {b} | {correlation:.3f} | {overlap:.3f} |\n")
    if agreement_rows:
        sections.append(
            "## Agreement between maps\n\n"
            f"| Pair | Rank correlation | Overlap of top {config['xai_top_overlap_fraction']:.0%} (IoU) |\n|---|---|---|\n"
            + "".join(agreement_rows)
        )
    sections += [_split_table(split, results[split], fractions, types, methods) for split in splits]
    sections.append(f"## Speed\n\n{latency:.3f} s for one answer and its maps, on {device.type}.\n")
    return "\n".join(sections)


def _beats_random(comparison: dict[str, dict[str, float]], types: list[str]) -> str:
    """The types a map fails to beat random on, or says so if it beats it on all of them."""
    failed = [qtype for qtype in types if qtype not in comparison or not comparison[qtype]["lower"] > 0]
    return f"does not beat random on: {', '.join(failed)}" if failed else "beats random on every type"


def run_followup(run_dir: Path, config: dict[str, Any], pilot: bool = False) -> str:
    """Scores the questions of the test run again, with all four maps under blur deletion, and
    reports what no-data strips do to the result. Occlusion was already chosen and confirmed, so
    this cannot change the choice; it can only limit what may be claimed for it."""
    config = {**config, "cache_images": False, "xai_deletion_fill": "blur"}
    if not pilot:
        config["max_images_per_split"] = None
    per_group = config["xai_pilot_questions_per_group" if pilot else "xai_questions_per_group"]
    seed = config["xai_seed"]
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    figure_dir = Path(config["xai_figure_dir"]) / (("pilot" if pilot else run_dir.name) + "_followup")
    methods = (*METHODS, "occlusion")

    model, dataloaders, device = load_run(run_dir, config)
    model.eval()
    splits = [s for s in config["splits"] if s.startswith("test")]
    types = [*config["question_types"], "count"]
    results: dict[str, SplitResult] = {}
    accuracy_rows = []
    for split in splits:
        dataset = dataloaders[split].dataset
        pool = question_pool(dataset.qa_records, seed)
        pool_strips: list[bool] = []
        correct = predict_correct(model, dataset, pool, config, device, f"predict {split}", pool_strips)
        for qtype in types:
            for label, keep in (("strip tiles", True), ("other tiles", False)):
                hits = [c for idx, c, s in zip(pool, correct, pool_strips)
                        if dataset.qa_records[idx]["type"] == qtype and s == keep]
                if hits:
                    accuracy_rows.append(f"| {split} | {qtype} | {label} | {len(hits)} | {sum(hits) / len(hits):.1%} |\n")
        sample = balanced_sample(pool, dataset.qa_records, correct, per_group)
        results[split] = score_split(
            model, dataset, sample, config, device, generator, split, figure_dir, methods,
            extra_figures=config["xai_followup_figures"],
        )

    accuracy_section = (
        "## Accuracy on strip tiles and other tiles\n\n"
        "Over one question per tile and type (count: rounded prediction equals the truth), not the sample below, "
        "and not the full test sets. A difference here does not show that strips cause errors: strip tiles may "
        "also differ in other ways.\n\n"
        "| Split | Type | Tiles | Questions | Accuracy |\n|---|---|---|---|---|\n" + "".join(accuracy_rows)
    )
    groups = {"pooled": results, **{split: {split: results[split]} for split in splits}}
    sections = [
        f"# XAI follow-up: {run_dir.name}" + (" (pilot subsets)" if pilot else "") + "\n\n"
        f"Splits: {', '.join(splits)}. Deleted cells are replaced with: blur. Same selection of questions as the "
        "test run. This does not change the choice of map; it limits what may be claimed for it.\n"
    ]

    strip_rows = []
    for split, result in results.items():
        strips, cells = result.has_strip(), torch.cat(result.no_data).float()
        for qtype in types:
            rows = torch.tensor([t == qtype for t in result.types])
            if rows.any():
                with_strip = strips[rows]
                size = cells[rows][with_strip].mean().item() if with_strip.any() else 0.0
                strip_rows.append(
                    f"| {split} | {qtype} | {rows.sum().item()} | {with_strip.sum().item()} "
                    f"| {with_strip.float().mean().item():.1%} | {size:.1f} |\n"
                )
    sections.append(
        "## How common no-data strips are\n\n"
        f"A no-data cell has at least {config['xai_nodata_cell_fraction']:.0%} of its pixels black "
        f"(at most {config['xai_nodata_pixel_value']} in every channel). A strip tile has at least one, so a large "
        "dark shadow could also count: check the strip figures.\n\n"
        "| Split | Type | Questions | On strip tiles | Share | No-data cells per strip tile |\n|---|---|---|---|---|---|\n"
        + "".join(strip_rows)
    )

    sections.append(accuracy_section)
    verdicts, occlusion_rows = [], []
    for group, scored in groups.items():
        for label, with_strip in (("all", None), ("without strips", False), ("only strips", True)):
            comparison = compare_with_control(scored, types, CONTROL, ("occlusion",), with_strip)["occlusion"]
            if with_strip is False:
                verdicts.append(f"- {group}: {_beats_random(comparison, types)}\n")
            for qtype, row in comparison.items():
                count = 0
                for result in scored.values():
                    strips = result.has_strip().tolist()
                    count += sum(
                        t == qtype and (with_strip is None or strips[i] == with_strip)
                        for i, t in enumerate(result.types)
                    )
                occlusion_rows.append(
                    f"| {group} | {label} | {qtype} | {count} | {row['mean']:.4f} | {row['lower']:.4f} "
                    f"| {row['standardized']:.3f} |\n"
                )
    sections.append(
        "## Occlusion without the strip tiles\n\n" + "".join(verdicts) + "\n"
        "Per question, the effect averaged over deletion sizes, minus the random map's. \"only strips\" can rest on few "
        "questions.\n\n"
        "| Split | Questions kept | Type | Questions | Mean difference | 95% lower bound | Standardized |\n"
        "|---|---|---|---|---|---|---|\n" + "".join(occlusion_rows)
    )

    fraction = config["xai_top_overlap_fraction"]
    zone_rows = []
    for split, result in results.items():
        strips = result.has_strip()
        if strips.any():
            chance = torch.cat(result.zone_share)[strips].mean().item()
            for method in (*methods, CONTROL, BLOCK_CONTROL):
                share = torch.cat(result.top_in_zone[method])[strips].mean().item()
                zone_rows.append(f"| {split} | {method} | {strips.sum().item()} | {share:.1%} | {chance:.1%} |\n")
    sections.append(
        f"## Where the top {fraction:.0%} of cells fall on strip tiles\n\n"
        "Share of a map's top cells in or beside no-data cells, against that area's share of the tile.\n\n"
        "| Split | Map | Questions | Top cells in or beside no-data | Share of the tile |\n|---|---|---|---|---|\n"
        + "".join(zone_rows)
    )
    edge_rows = []
    ring_share = outer_ring(model.encoder.pos_embed.size(1) - 1).float().mean().item()
    for split, result in results.items():
        others = ~result.has_strip()
        if others.any():
            for method in (*methods, CONTROL, BLOCK_CONTROL):
                share = torch.cat(result.top_on_edge[method])[others].mean().item()
                edge_rows.append(f"| {split} | {method} | {others.sum().item()} | {share:.1%} | {ring_share:.1%} |\n")
    sections.append(
        "## Edge control: tiles without strips\n\n"
        f"Strips lie along edges. If a map's top {fraction:.0%} of cells favour the outermost ring on tiles without "
        "strips too, part of the pull towards strips is a pull towards edges.\n\n"
        "| Split | Map | Questions | Top cells on the outer ring | Share of the tile |\n|---|---|---|---|---|\n"
        + "".join(edge_rows)
    )

    sections.append("## All four maps under blur deletion\n\nThe first three were first tested with mean-colour "
                    "deletion. Their two correlation checks do not depend on deletion, so they stand as first reported.\n\n"
                    + "".join(
                        f"- `{method}`: {_beats_random(compare_with_control(results, types, CONTROL, methods)[method], types)}\n"
                        for method in methods
                    ))
    for group, scored in groups.items():
        sections.append(_comparison_table(
            f"Against the random map, blur: {group}", "Effect averaged over deletion sizes, minus the random map's.",
            compare_with_control(scored, types, CONTROL, methods),
        ))
    sections.append(_comparison_table(
        "Against a random block, blur (secondary, pooled)", "The same, against one compact random patch.",
        compare_with_control(results, types, BLOCK_CONTROL, methods),
    ))
    total = sum(len(result.types) for result in results.values())
    sections.append(
        "## All-constant maps\n\n| Map | All-constant |\n|---|---|\n" + "".join(
            f"| {method} | {sum(result.flat_maps[method] for result in results.values())} of {total} |\n"
            for method in methods
        )
    )
    sections += [_split_table(split, results[split], config["xai_delete_fractions"], types, methods) for split in splits]
    return "\n".join(sections)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("config", type=Path, nargs="?", default=Path("configs/hr_config.yaml"))
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--new-methods", help=f"comma-separated, from {','.join(NEW_METHODS)}; deletes by blurring")
    parser.add_argument("--stage", choices=("val", "test"), default="test")
    parser.add_argument("--followup", action="store_true", help="no-data strips, and all maps under blur deletion")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())

    if args.followup and args.new_methods:
        parser.error("--followup and --new-methods are separate runs")
    methods, tag = METHODS, ""
    if args.followup:
        tag = "_followup"
    elif args.new_methods:
        methods = tuple(args.new_methods.split(","))
        unknown = set(methods) - set(NEW_METHODS)
        if unknown:
            parser.error(f"unknown methods: {sorted(unknown)}")
        config["xai_deletion_fill"] = "blur"
        tag = f"_new_{args.stage}"
    started = time.monotonic()
    if args.followup:
        report = run_followup(args.run_dir, config, args.pilot)
    else:
        report = run_xai(args.run_dir, config, args.pilot, methods, args.stage, tag)
    report += f"\n{(time.monotonic() - started) / 60:.1f} min in total.\n"
    out_path = Path("reports") / f"xai_results_{args.run_dir.name}{tag}{'_pilot' if args.pilot else ''}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Wrote {out_path}")
    print(report)


if __name__ == "__main__":
    _main()
