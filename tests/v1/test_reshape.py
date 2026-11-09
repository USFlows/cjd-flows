import math

import pytest
import torch

from cjd_flows.bijections.linear import LULinear
from cjd_flows.bijections.reshape import Flatten, HaarWavelet, Patchify, Reshape, Squeeze
from cjd_flows.core import Chain
from cjd_flows.testing import check_bijection


def check_shape_changing_bijection(layer, x, atol=1e-10):
    """``check_bijection`` analogue for layers with ``out_shape != event_shape``."""
    B = x.shape[0]
    z = layer(x)
    assert z.shape == (B, *layer.out_shape)
    x_rec = layer.inverse(z)
    assert x_rec.shape == x.shape
    assert (x_rec - x).abs().max() < atol
    z2 = torch.randn_like(z)
    assert (layer(layer.inverse(z2)) - z2).abs().max() < atol
    ladj = layer.log_abs_det()
    assert ladj.dim() == 0 and ladj.item() == 0.0
    assert layer.volume_preserving and layer.piecewise_affine
    # Jacobian: the (flattened) map is linear with |det| = 1
    d = math.prod(x.shape[1:])
    if d <= 256:
        J = torch.autograd.functional.jacobian(lambda v: layer(v.reshape(1, *x.shape[1:])).reshape(d), x[0].reshape(d))
        assert abs(torch.linalg.slogdet(J).logabsdet.item()) < 1e-10
    # works inside a Chain and as inverse view
    torch.testing.assert_close(layer.inv(z), x)


@pytest.mark.parametrize("shape", [(2, 8), (3, 4, 4), (2, 2, 4, 6), (1, 6, 4)], ids=str)
def test_squeeze(shape):
    layer = Squeeze(shape).double()
    k = len(shape) - 1
    assert layer.out_shape == (shape[0] * 2**k, *[s // 2 for s in shape[1:]])
    x = torch.randn(5, *shape, dtype=torch.float64)
    check_shape_changing_bijection(layer, x)
    # it is a permutation of the entries
    assert torch.equal(layer(x).flatten(1).sort(1).values, x.flatten(1).sort(1).values)


def test_squeeze_layout_2d():
    x = torch.arange(16.0).reshape(1, 1, 4, 4)
    z = Squeeze((1, 4, 4))(x)
    # channel o = (dy, dx) holds x[2i+dy, 2j+dx]
    assert z[0, :, 0, 0].tolist() == [0.0, 1.0, 4.0, 5.0]
    assert z[0, 3].tolist() == [[5.0, 7.0], [13.0, 15.0]]
    z3 = Squeeze((1, 6, 6), factor=3)(torch.arange(36.0).reshape(1, 1, 6, 6))
    assert z3.shape == (1, 9, 2, 2) and z3[0, :, 0, 0].tolist() == [0, 1, 2, 6, 7, 8, 12, 13, 14]


@pytest.mark.parametrize("shape", [(2, 8), (3, 4, 4), (2, 2, 2, 2)], ids=str)
def test_haar(shape):
    layer = HaarWavelet(shape).double()
    x = torch.randn(4, *shape, dtype=torch.float64)
    check_shape_changing_bijection(layer, x)
    # orthonormal: norms preserved
    torch.testing.assert_close(layer(x).flatten(1).norm(dim=1), x.flatten(1).norm(dim=1))
    # low-pass band = scaled block mean
    k = len(shape) - 1
    pool = {1: torch.nn.functional.avg_pool1d, 2: torch.nn.functional.avg_pool2d, 3: torch.nn.functional.avg_pool3d}[k]
    torch.testing.assert_close(layer(x)[:, : shape[0]], pool(x, 2) * 2 ** (k / 2))


def test_haar_constant_image_has_no_details():
    x = torch.ones(1, 3, 4, 4)
    z = HaarWavelet((3, 4, 4))(x)
    assert torch.allclose(z[:, 3:], torch.zeros(()), atol=1e-6)


@pytest.mark.parametrize("shape,p", [((3, 4, 4), 2), ((2, 8), 4), ((2, 4, 4, 4), 2)], ids=str)
def test_patchify(shape, p):
    layer = Patchify(shape, p).double()
    x = torch.randn(3, *shape, dtype=torch.float64)
    check_shape_changing_bijection(layer, x)


def test_patchify_layout():
    # einops: b c (h p1) (w p2) -> b (h w) (p1 p2 c)
    B, C, H, W, p = 2, 3, 4, 6, 2
    x = torch.randn(B, C, H, W)
    ref = x.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 3, 5, 1).reshape(B, (H // p) * (W // p), p * p * C)
    torch.testing.assert_close(Patchify((C, H, W), p)(x), ref)


def test_flatten_reshape():
    x = torch.randn(4, 3, 4, 4, dtype=torch.float64)
    f = Flatten((3, 4, 4))
    assert f.out_shape == (48,)
    check_shape_changing_bijection(f, x)
    r = Reshape((3, 4, 4), (6, 8))
    check_shape_changing_bijection(r, x)
    with pytest.raises(ValueError):
        Reshape((3, 4), (5,))


def test_invalid_shapes():
    with pytest.raises(ValueError):
        Squeeze((3, 5, 4))
    with pytest.raises(ValueError):
        Squeeze((6,))
    with pytest.raises(ValueError):
        HaarWavelet((3, 3))


def test_chain_with_shape_change():
    """Shape-changing layers compose with channel-linear layers on the new shape."""
    torch.manual_seed(0)
    s = (3, 4, 4)
    sq = Squeeze(s)
    chain = Chain(LULinear(s, identity_init=False), sq, LULinear(sq.out_shape, identity_init=False), Flatten(sq.out_shape))
    chain = chain.double()
    with torch.no_grad():
        chain[0].log_s.add_(0.3)
        chain[2].log_s.sub_(0.1)
    x = torch.randn(3, *s, dtype=torch.float64)
    z = chain(x)
    assert z.shape == (3, 48)
    torch.testing.assert_close(chain.inverse(z), x)
    J = torch.autograd.functional.jacobian(lambda v: chain(v.reshape(1, *s)).reshape(48), x[0].reshape(48))
    torch.testing.assert_close(torch.linalg.slogdet(J).logabsdet, chain.log_abs_det())


def test_state_dict_double_cuda():
    h = HaarWavelet((3, 4, 4))
    assert set(h.state_dict()) == set()
    HaarWavelet((3, 4, 4)).load_state_dict(h.state_dict())
    assert h.double().log_abs_det().dtype == torch.float64
    if torch.cuda.is_available():
        for layer in [Squeeze((3, 4, 4)), HaarWavelet((3, 4, 4)), Patchify((3, 4, 4), 2), Flatten((3, 4, 4))]:
            layer = layer.cuda()
            x = torch.randn(2, 3, 4, 4, device="cuda")
            torch.testing.assert_close(layer.inverse(layer(x)), x)


def test_squeeze_check_bijection_when_shape_preserved():
    """Sanity: the generic contract checker works on squeeze∘unsqueeze."""
    s = (2, 4, 4)
    sq = Squeeze(s).double()
    check_bijection(Chain(sq, sq.inv), torch.randn(3, *s, dtype=torch.float64))
