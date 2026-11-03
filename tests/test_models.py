import numpy as np
import pytest
import torch

from ballistic_forecast.models import build_model, predict_px
from ballistic_forecast.models.mamba import CausalDepthwiseConv, MambaBlock


@pytest.mark.parametrize("name", ["gru", "mamba"])
def test_output_shape(name):
    model = build_model(name, pred_len=12)
    out = model(torch.randn(3, 30, 2) * 0.1)
    assert out.shape == (3, 12, 2)


def test_mamba_block_is_causal():
    torch.manual_seed(0)
    block = MambaBlock(d_model=16, d_state=4)
    x = torch.randn(1, 20, 16)
    x2 = x.clone()
    x2[:, 12:] += torch.randn(1, 8, 16)  # perturb the future only
    y, y2 = block(x), block(x2)
    torch.testing.assert_close(y[:, :12], y2[:, :12])
    assert not torch.allclose(y[:, 12:], y2[:, 12:])


def test_causal_conv_matches_torch_conv1d():
    torch.manual_seed(0)
    ours = CausalDepthwiseConv(channels=8, kernel_size=4)
    ref = torch.nn.Conv1d(8, 8, 4, groups=8, padding=3)
    with torch.no_grad():
        ref.weight.copy_(ours.weight.T.unsqueeze(1))
        ref.bias.copy_(ours.bias)
    x = torch.randn(2, 15, 8)
    expected = ref(x.transpose(1, 2))[..., :15].transpose(1, 2)
    torch.testing.assert_close(ours(x), expected)


@pytest.mark.parametrize("name", ["gru", "mamba"])
def test_translation_invariance(name):
    """Inputs are re-centred on the last point, so shifting a track shifts its forecast."""
    torch.manual_seed(0)
    model = build_model(name, pred_len=10)
    obs = np.cumsum(np.random.default_rng(0).normal(size=(4, 30, 2)) * 5, axis=1) + 500
    shift = np.array([123.0, -45.0])
    np.testing.assert_allclose(predict_px(model, obs + shift), predict_px(model, obs) + shift, atol=1e-3)
