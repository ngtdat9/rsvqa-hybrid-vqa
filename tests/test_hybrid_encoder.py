import torch

from src.models.hybrid_encoder import HybridEncoder

TEST_CONFIG = dict(
    image_size=32,
    cnn_channels=[8, 16],
    d_model=32,
    num_transformer_layers=1,
    num_heads=2,
    ffn_dim=64,
    dropout=0.0,
)


def make_encoder(**overrides) -> HybridEncoder:
    kwargs = {**TEST_CONFIG, **overrides}
    return HybridEncoder(**kwargs)


def test_forward_output_shapes():
    encoder = make_encoder()
    batch_size = 4
    image = torch.randn(batch_size, 3, TEST_CONFIG["image_size"], TEST_CONFIG["image_size"])
    out = encoder(image)

    expected_n = (TEST_CONFIG["image_size"] // (2 ** len(TEST_CONFIG["cnn_channels"]))) ** 2
    assert out.pooled.shape == (batch_size, TEST_CONFIG["d_model"])
    assert out.patch_tokens.shape == (batch_size, expected_n, TEST_CONFIG["d_model"])


def test_patch_count_matches_stem_downsampling():
    for image_size, cnn_channels in [(32, [8, 16]), (64, [8, 16, 16]), (256, [32, 64, 128, 128])]:
        encoder = make_encoder(image_size=image_size, cnn_channels=cnn_channels)
        image = torch.randn(2, 3, image_size, image_size)
        out = encoder(image)
        expected_n = (image_size // (2 ** len(cnn_channels))) ** 2
        assert out.patch_tokens.shape[1] == expected_n


def test_gradient_flows_to_cnn_and_transformer_params():
    encoder = make_encoder()
    image = torch.randn(4, 3, TEST_CONFIG["image_size"], TEST_CONFIG["image_size"])
    out = encoder(image)
    # A uniform .sum() can structurally cancel to an exact zero gradient for
    # some LayerNorm-adjacent bias terms (LayerNorm centers its output, so
    # summing all output features with equal weight has a degenerate
    # gradient for certain random inits). A random weighting avoids that.
    weights = torch.randn_like(out.pooled)
    (out.pooled * weights).sum().backward()

    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.any(param.grad != 0), f"{name} has an all-zero gradient"


def test_batch_independence():
    encoder = make_encoder()
    images = torch.randn(2, 3, TEST_CONFIG["image_size"], TEST_CONFIG["image_size"])
    out = encoder(images)
    assert not torch.allclose(out.pooled[0], out.pooled[1])
