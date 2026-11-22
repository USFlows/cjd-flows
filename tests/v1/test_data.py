import math

import pytest
import torch

from cjd_flows.data import (
    AffineScale,
    Compose,
    Logit,
    Standardize,
    ToSymmetric,
    ToUnitInterval,
    UniformDequantize,
    bits_per_dim,
    default_root,
    to_symmetric,
    to_unit_interval,
    toy,
)
from cjd_flows.testing import jacobian_log_abs_det


# ------------------------------------------------------------- preprocess
def test_dequantize_range():
    x = torch.randint(0, 256, (100, 3, 4, 4), dtype=torch.uint8)
    y = UniformDequantize()(x)
    assert y.dtype == torch.float32
    assert ((y - x.float()) >= 0).all() and ((y - x.float()) < 1).all()
    assert torch.equal(UniformDequantize().inverse(y), x.float())
    assert not torch.equal(UniformDequantize()(x), y)  # fresh noise


@pytest.mark.parametrize("pre", [ToUnitInterval(), ToSymmetric(), AffineScale(0.3, 2.0), AffineScale(torch.tensor([0.5, 2.0, 3.0]), 1.0)])
def test_affine_logdet_matches_autograd(pre):
    x = torch.rand(4, 3, dtype=torch.float64) * 256
    ref = jacobian_log_abs_det(pre, x)
    ld = pre.log_abs_det(x)
    assert torch.allclose(ld, ref)
    assert pre.constant_det
    assert pre.constant_log_abs_det(3) == pytest.approx(ref[0].item())
    assert torch.allclose(pre.inverse(pre(x)), x)


def test_unit_symmetric_values():
    x = torch.tensor([[0.0, 128.0, 255.999]])
    assert torch.allclose(to_unit_interval(x), x / 256)
    assert torch.allclose(to_symmetric(x), x / 128 - 1)
    assert ToUnitInterval().constant_log_abs_det((1, 28, 28)) == pytest.approx(-784 * math.log(256))
    assert ToSymmetric().constant_log_abs_det((3, 32, 32)) == pytest.approx(3072 * math.log(2 / 256))


def test_logit_logdet_and_flag():
    lg = Logit(0.05)
    assert not lg.constant_det
    u = torch.rand(5, 4, dtype=torch.float64)
    assert torch.allclose(lg.log_abs_det(u), jacobian_log_abs_det(lg, u))
    assert torch.allclose(lg.inverse(lg(u)), u)
    with pytest.raises(ValueError):
        lg.constant_log_abs_det(4)
    comp = Compose(ToUnitInterval(), lg)
    assert not comp.constant_det
    x = torch.rand(3, 4, dtype=torch.float64) * 256
    assert torch.allclose(comp.log_abs_det(x), jacobian_log_abs_det(comp, x))


def test_standardize():
    x = torch.randn(1000, 3, dtype=torch.float64) * torch.tensor([1.0, 2.0, 5.0]) + 3
    st = Standardize().fit(x)
    y = st(x)
    assert torch.allclose(y.mean(0), torch.zeros(3, dtype=torch.float64), atol=1e-5)
    assert torch.allclose(y.std(0, unbiased=False), torch.ones(3, dtype=torch.float64), atol=1e-5)
    ref = jacobian_log_abs_det(st, x[:3])
    assert st.constant_log_abs_det() == pytest.approx(ref[0].item(), rel=1e-6)
    assert torch.allclose(st.inverse(y), x, atol=1e-5)


def test_bits_per_dim_uniform_is_8():
    D = 10
    # uniform density on [0,256)^D in raw space, identity preprocessing
    assert bits_per_dim(torch.tensor(-D * math.log(256)), D, 0.0).item() == pytest.approx(8.0)
    # uniform on [0,1)^D in model space with y = x/256 (default convention)
    assert bits_per_dim(0.0, D) == pytest.approx(8.0)
    assert bits_per_dim(0.0, D, ToUnitInterval().constant_log_abs_det(D)) == pytest.approx(8.0)
    # uniform on [-1,1)^D with the symmetric scaling
    lp = -D * math.log(2)
    assert bits_per_dim(lp, D, ToSymmetric().constant_log_abs_det(D)) == pytest.approx(8.0)


def test_bits_per_dim_monte_carlo():
    # model p(y) for y = x̃/256 on [0,1): density 2y (D=1) -> exact discrete NLL via MC
    torch.manual_seed(0)
    x = torch.randint(0, 256, (200_000,)).float()
    y = (x + torch.rand_like(x)) / 256
    lp = torch.log(2 * y)
    bpd = bits_per_dim(lp, 1).mean()
    # E[-log2 p(x̃)] with p(x̃) = 2 x̃ / 256^2 under uniform x̃
    exact = -(math.log(2) - 2 * math.log(256) + (math.log(256) - 1)) / math.log(2)
    assert bpd.item() == pytest.approx(exact, abs=0.02)


def test_default_root(monkeypatch, tmp_path):
    monkeypatch.setenv("CJD_FLOWS_DATA", str(tmp_path))
    assert default_root() == tmp_path
    monkeypatch.delenv("CJD_FLOWS_DATA")
    assert default_root().name == "cjd_flows"


# ------------------------------------------------------------- toy data
@pytest.mark.parametrize("name", sorted(toy.TOY_DATASETS))
def test_toy_shapes(name):
    x = toy.make_toy(name, 1001, seed=0)
    assert x.shape == (1001, 2) and x.dtype == torch.float32
    assert torch.isfinite(x).all()
    assert torch.equal(x, toy.make_toy(name, 1001, seed=0))
    assert x.abs().max() < 20


def test_gmm_normalization_and_shapes():
    for d in (2, 5):
        gmm = toy.asymmetric_gmm(d)
        assert torch.allclose(gmm.means[0], torch.ones(d, dtype=torch.float64))
        assert gmm.variances[1, 0] == 5.0 and gmm.variances[1, -1] == 0.5
        x = gmm.sample(10_000, seed=1)
        assert x.shape == (10_000, d)
        # importance sampling estimate of ∫ p = E_q[p/q], q = N(0, 9 I)
        g = torch.Generator().manual_seed(2)
        q = torch.randn(400_000, d, generator=g, dtype=torch.float64) * 3
        log_q = -0.5 * (q / 3).pow(2).sum(-1) - d * math.log(3) - 0.5 * d * math.log(2 * math.pi)
        est = (gmm.log_prob(q) - log_q).exp().mean().item()
        assert est == pytest.approx(1.0, abs=0.02)
    assert toy.toy_log_prob("asymmetric_gmm", torch.zeros(3, 4)).shape == (3,)


def test_exact_toy_densities_match_samples():
    # eight gaussians: compare mean log density of samples with entropy-ish check via MC
    x = toy.eight_gaussians(20000, seed=0).double()
    lp = toy.toy_log_prob("eight_gaussians", x)
    # samples are from the density -> average lp approximates -H; KL to itself ~ 0:
    # compare against a mismatched density (shifted) which must be worse
    assert lp.mean() > toy.toy_log_prob("eight_gaussians", x + 0.3).mean()
    # normalization on a grid
    g = torch.linspace(-5, 5, 801, dtype=torch.float64)
    xx, yy = torch.meshgrid(g, g, indexing="ij")
    grid = torch.stack([xx.flatten(), yy.flatten()], 1)
    h = (g[1] - g[0]).item()
    mass = toy.toy_log_prob("eight_gaussians", grid).exp().sum().item() * h * h
    assert mass == pytest.approx(1.0, abs=1e-3)
    # checkerboard: all samples are on the support, mass 1
    xc = toy.checkerboard(5000, seed=0)
    assert torch.isfinite(toy.toy_log_prob("checkerboard", xc)).all()
    mass = toy.toy_log_prob("checkerboard", grid).exp().sum().item() * h * h
    assert mass == pytest.approx(1.0, abs=0.02)
    with pytest.raises(NotImplementedError):
        toy.toy_log_prob("two_moons", x)


# ------------------------------------------------------------- tabular
TAB_SHAPES = {
    "power": (6, 1_659_917, 184_435, 204_928),
    "gas": (8, 852_174, 94_685, 105_206),
    "hepmass": (21, 315_123, 35_013, 174_987),
    "miniboone": (43, 29_556, 3_284, 3_648),
    "bsds300": (63, 1_000_000, 50_000, 250_000),
}


@pytest.mark.parametrize("name", sorted(TAB_SHAPES))
def test_tabular_shapes(name):
    from cjd_flows.data.tabular import load_tabular

    proc = default_root() / "maf" / "processed" / f"{name}_train.npy"
    if not proc.exists():
        pytest.skip("processed MAF data not available (run load_tabular once to download)")
    tr, va, te = load_tabular(name, download=False)
    d, ntr, nva, nte = TAB_SHAPES[name]
    assert tr.shape == (ntr, d) and va.shape == (nva, d) and te.shape == (nte, d)
    assert tr.dtype == torch.float32
    assert torch.isfinite(tr).all() and torch.isfinite(te).all()
    if name != "bsds300":
        both = torch.cat([tr, va]) if name in ("power", "miniboone") else tr
        assert both.double().mean(0).abs().max() < 0.05


# ------------------------------------------------------------- images
@pytest.mark.parametrize("backend", ["auto", "raw"])
def test_mnist_loader(backend):
    from cjd_flows.data.images import load_images

    try:
        ds = load_images("mnist", classes=[3], split="test", backend=backend)
    except (OSError, RuntimeError) as e:  # offline
        pytest.skip(f"MNIST not available: {e}")
    assert len(ds) == 1010  # number of 3s in the MNIST test set
    assert (ds.labels == 3).all()
    y, lab = ds[0]
    assert y.shape == (1, 28, 28) and y.dtype == torch.float32 and int(lab) == 3
    assert y.min() >= -1 and y.max() < 1
    y2, _ = ds[0]
    assert not torch.equal(y, y2)  # fresh noise each access
    assert torch.equal(ds.to_raw(y.unsqueeze(0))[0], ds.images[0])
    xb, lb = ds.get_batch(torch.arange(8))
    assert xb.shape == (8, 1, 28, 28)
    assert ds.preprocessing_log_det == pytest.approx(784 * math.log(2 / 256))
    # uniform model density on [-1,1)^D -> 8 bits/dim
    assert ds.bits_per_dim(-784 * math.log(2)) == pytest.approx(8.0)
    flat = load_images("mnist", classes=3, split="test", scale="unit", flatten=True, backend=backend)
    v, _ = flat[0]
    assert v.shape == (784,) and v.min() >= 0 and v.max() < 1
    rest = load_images("mnist", classes=[3], split="test", exclude_classes=True, backend=backend)
    assert len(rest) == 10000 - 1010


def test_tabular_missing_dependency_has_install_hint(monkeypatch):
    import importlib

    from cjd_flows.data import tabular

    real = importlib.import_module

    def fake(name, *args, **kwargs):
        if name == "h5py":
            raise ImportError("No module named 'h5py'")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake)
    with pytest.raises(ImportError, match=r"cjd-flows\[tabular\]"):
        tabular._require("h5py", "BSDS300")
