"""Regression tests for bugs fixed in the legacy (pre-v1) modules.

Each test references the bug number of the audit it covers. Tests that need
pyro (all of ``flows``/``transforms``/``distributions``) are skipped if pyro is
not installed (``pip install cjd-flows[legacy]``).
"""

import math
import os
import subprocess
import sys
import textwrap

import pytest
import torch

pyro = pytest.importorskip("pyro")

from cjd_flows.distributions import (  # noqa: E402
    Categorical,
    Gamma,
    LMM,
    RadialDistribution,
    RadialMM,
    UniformUnitLpBall,
)
from cjd_flows.flows import USFlow  # noqa: E402
from cjd_flows.networks import CondConvNet, ConvNet, ConvNet2D, GatedConv  # noqa: E402
from cjd_flows.transforms import (  # noqa: E402
    BlockLUTransform,
    HouseholderTransform,
    LeakyReLUTransform,
    LUTransform,
    MaskedCoupling,
    Rotation,
    ScaleTransform,
    SequentialAffineTransform,
)

DIM = 4


def _gamma_norm():
    return Gamma(torch.tensor([2.0]), torch.tensor([1.0]))


def _make_flow(soft_training: bool = False, prior_scale=None, dim: int = DIM) -> USFlow:
    torch.manual_seed(0)
    base = RadialDistribution(torch.zeros(dim), _gamma_norm(), p=2.0)
    if soft_training:
        cond_cls, cond_args = CondConvNet, {"in_dims": [dim], "c_hidden": [16]}
    else:
        cond_cls, cond_args = ConvNet, {"in_dims": [dim], "c_hidden": [16]}
    return USFlow(
        base,
        in_dims=[dim],
        coupling_blocks=2,
        conditioner_cls=cond_cls,
        conditioner_args=cond_args,
        soft_training=soft_training,
        prior_scale=prior_scale,
        training_noise_prior=pyro.distributions.Uniform(0.0, 0.1) if soft_training else None,
    )


# --------------------------------------------------------------------------- 1
@pytest.mark.parametrize("shuffle", [False, True])
def test_fit_trains_with_and_without_shuffle(shuffle):
    """Bug 1: ``shuffle=False`` raised a NameError that was swallowed by a bare
    except, so the model silently trained on nothing."""
    flow = _make_flow()
    data = torch.utils.data.TensorDataset(torch.randn(64, DIM))
    before = [p.detach().clone() for p in flow.parameters()]
    losses = flow.fit(
        data,
        optim=torch.optim.Adam,
        optim_params={"lr": 1e-2},
        batch_size=16,
        shuffle=shuffle,
        device="cpu",
        epochs=2,
    )
    assert len(losses) == 2 and all(math.isfinite(v) for v in losses)
    after = list(flow.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_fit_accepts_plain_tensor():
    flow = _make_flow()
    losses = flow.fit(
        torch.randn(32, DIM), optim=torch.optim.Adam, batch_size=8, shuffle=False, device="cpu"
    )
    assert len(losses) == 1 and math.isfinite(losses[0])


def test_fit_reuses_optimizer_instance():
    """Bug 1/7: passing an optimizer instance keeps its state across calls."""
    flow = _make_flow()
    data = torch.randn(32, DIM)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
    flow.fit(data, optim=opt, batch_size=8, device="cpu")
    flow.fit(data, optim=opt, batch_size=8, device="cpu")
    steps = {int(s["step"]) for s in opt.state.values()}
    assert steps == {8}  # 2 calls x 4 batches, state was not reset


def test_fit_does_not_swallow_errors():
    """Bug 1: errors while loading a batch must propagate (no bare except)."""

    class Broken(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            raise RuntimeError("broken dataset")

    flow = _make_flow()
    with pytest.raises(RuntimeError, match="broken dataset"):
        flow.fit(Broken(), optim=torch.optim.Adam, device="cpu")


# --------------------------------------------------------------------------- 12
def test_fit_prior_weight():
    """Bug 12: the log prior is weighted with 1/N by default (was weight 1)."""
    data = torch.randn(16, DIM)
    for prior_weight, expected_weight in [(None, 1 / 16), (1.0, 1.0), (0.0, 0.0)]:
        flow = _make_flow(prior_scale=1.0)
        with torch.no_grad():
            nll = -flow.log_prob(data).mean()
            log_prior = flow.log_prior()
        assert float(log_prior) != 0.0
        opt = torch.optim.SGD(flow.parameters(), lr=0.0)
        (loss,) = flow.fit(
            data, optim=opt, batch_size=16, shuffle=False, device="cpu", prior_weight=prior_weight
        )
        assert loss == pytest.approx(float(nll - expected_weight * log_prior), rel=1e-5)


# --------------------------------------------------------------------------- 2
def test_cond_conv_net_default_c_out_excludes_context_channel():
    """Bug 2: CondConvNet(c_out=-1) returned C + 1 channels."""
    net = CondConvNet(in_dims=[3, 8, 8], c_hidden=[8])
    out = net(torch.randn(2, 3, 8, 8), torch.zeros(2, 1))
    assert out.shape == (2, 3, 8, 8)
    net = CondConvNet(in_dims=[DIM], c_hidden=[8])
    assert net(torch.randn(5, DIM), torch.zeros(5, 1)).shape == (5, DIM)


@pytest.mark.parametrize("gating", [True, False])
def test_conv_net_2d_preserves_shape(gating):
    """Bug 2: ConvNet2D's default padding=0 and GatedConv's padded 1x1 conv
    produced shape mismatches."""
    net = ConvNet2D(c_in=2, c_hidden=4, num_layers=2, gating=gating)
    assert net(torch.randn(3, 2, 6, 6)).shape == (3, 2, 6, 6)


def test_gated_conv_preserves_shape():
    gc = GatedConv(4, 8, kernel_size=3, padding=1)
    assert gc(torch.randn(2, 4, 5, 5)).shape == (2, 4, 5, 5)


# --------------------------------------------------------------------------- 3
def test_soft_training_default_context_in_all_directions():
    """Bug 3: sample/_forward/backward passed no context while log_prob passed
    zeros for soft-trained models."""
    flow = _make_flow(soft_training=True)
    flow.eval()
    x = torch.randn(7, DIM)
    zeros = torch.zeros(7, 1)
    with torch.no_grad():
        z = flow.backward(x)
        assert torch.allclose(z, flow.backward(x, context=zeros))
        assert torch.allclose(flow._forward(z), x, atol=1e-4)
        assert torch.allclose(flow.log_prob(x), flow.log_prob(x, context=zeros))
        assert flow.sample([5]).shape == (5, DIM)
    # also trains
    losses = flow.fit(torch.randn(16, DIM), optim=torch.optim.Adam, batch_size=8, device="cpu")
    assert math.isfinite(losses[0])


# --------------------------------------------------------------------------- 4
def test_uniform_linf_sphere_samples_both_faces():
    """Bug 4: the p=inf sampler never produced points on the -1 faces."""
    torch.manual_seed(0)
    x = UniformUnitLpBall(3, math.inf).sample((2000,))
    assert torch.allclose(x.abs().max(dim=-1).values, torch.ones(2000))
    assert (x == -1.0).any(dim=-1).float().mean() > 0.4
    assert (x == 1.0).any(dim=-1).float().mean() > 0.4


@pytest.mark.parametrize("d", [1, 2, 3, 5])
def test_uniform_lp_sphere_normalizer_is_cone_measure(d):
    """Bug 4: the L1 normaliser used (3/2) log d instead of log(d * vol(B_1))."""
    log_vol = {
        1.0: d * math.log(2) - math.lgamma(d + 1),
        2.0: (d / 2) * math.log(math.pi) - math.lgamma(d / 2 + 1),
        math.inf: d * math.log(2),
    }
    for p, lv in log_vol.items():
        dist = UniformUnitLpBall(d, p)
        assert float(dist.log_surface_area_unit_ball) == pytest.approx(math.log(d) + lv)


@pytest.mark.parametrize("p", [1.0, 2.0, math.inf])
def test_radial_density_integrates_to_one_2d(p):
    """Radial density normalisation (log_delta_volume) in d = 2."""
    dist = RadialDistribution(torch.zeros(2), _gamma_norm(), p=p)
    n, lim = 801, 14.0
    g = torch.linspace(-lim, lim, n, dtype=torch.float64)
    xx, yy = torch.meshgrid(g, g, indexing="ij")
    pts = torch.stack([xx.flatten(), yy.flatten()], -1).float()
    with torch.no_grad():
        dens = dist.log_prob(pts).double().exp()
    integral = float(dens.sum() * (g[1] - g[0]) ** 2)
    assert integral == pytest.approx(1.0, abs=2e-2)


# --------------------------------------------------------------------------- 5
def test_categorical_not_shadowed():
    """Bug 5: a broken duplicate Categorical shadowed the working one."""
    cat = Categorical(torch.zeros(3))
    assert cat.sample((10,)).shape == (10,)
    assert torch.allclose(cat.log_prob(torch.tensor([0, 1, 2])), torch.full((3,), -math.log(3)))


def test_radial_mm_raises_clear_error():
    with pytest.raises(NotImplementedError, match="RadialMixture"):
        RadialMM(torch.zeros(2, DIM), _gamma_norm(), p=2.0)


def test_lmm_constructs_and_evaluates():
    lmm = LMM(torch.randn(3, DIM), torch.ones(3, DIM), torch.zeros(3))
    x = lmm.sample((6,))
    assert x.shape == (6, DIM)
    lp = lmm.log_prob(x)
    assert lp.shape == (6,) and torch.isfinite(lp).all()


@pytest.mark.parametrize("p", [1, 2, 1.0, math.inf])
def test_radial_distribution_sample_shape_and_int_p(p):
    """Bug 5: wrong sample shapes for multi-dim events; int p was rejected."""
    shape = (2, 3, 4)
    dist = RadialDistribution(torch.ones(shape), _gamma_norm(), p=p)
    assert isinstance(dist.p, float)
    x = dist.sample((5,))
    assert x.shape == (5, *shape)
    # every sample lies on its own sphere: ||x - loc||_p equals a positive radius
    r = (x - 1.0).flatten(1).norm(p=dist.p, dim=-1)
    assert (r > 0).all()
    assert dist.sample().shape == shape
    assert dist.log_prob(x).shape == (5,)


# --------------------------------------------------------------------------- 6
def test_explib_eval_energy_distance_and_norm():
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    pytest.importorskip("matplotlib")
    from cjd_flows.explib import eval as explib_eval

    assert hasattr(explib_eval, "norm")  # was used without import
    X = torch.rand(50, 3).double().numpy()
    d = explib_eval.RadialFlowEvaluator._energy_distance(None, X, X)
    assert math.isfinite(d) and abs(d) < 1e-6


# --------------------------------------------------------------------------- 7
def test_hyperopt_anomaly_detection_is_opt_in(monkeypatch):
    pytest.importorskip("ray")
    from cjd_flows.explib import hyperopt

    monkeypatch.delenv("CJD_FLOWS_DETECT_ANOMALY", raising=False)
    assert not hyperopt._detect_anomaly_requested()
    monkeypatch.setenv("CJD_FLOWS_DETECT_ANOMALY", "1")
    assert hyperopt._detect_anomaly_requested()


# --------------------------------------------------------------------------- 8
def test_scale_transform_init_and_jitter():
    t = ScaleTransform([DIM])
    assert torch.equal(t.scale.detach(), torch.ones(DIM))
    t.add_jitter(1e-3)  # previously referenced the missing attribute U_raw
    assert not torch.equal(t.scale.detach(), torch.ones(DIM))


# --------------------------------------------------------------------------- 9
def test_householder_backward_inverts_forward():
    torch.manual_seed(0)
    t = HouseholderTransform(DIM, nvs=3)
    x = torch.randn(10, DIM)
    with torch.no_grad():
        assert torch.allclose(t.backward(t.forward(x)), x, atol=1e-5)
        w = t.matrix()
        assert torch.allclose(t.forward(x), x @ w.T, atol=1e-6)


def test_rotation_backward_inverts_forward():
    t = Rotation(3, (0, 2), 0.7)
    x = torch.randn(10, 3)
    assert torch.allclose(t.backward(t.forward(x)), x, atol=1e-6)
    assert torch.allclose(t.forward(x), x @ t.as_matrix().T, atol=1e-6)


def test_leaky_relu_log_det_is_per_sample():
    t = LeakyReLUTransform(alpha=0.1)
    x = torch.tensor([[1.0, -1.0, -2.0], [1.0, 2.0, 0.0]])
    ladj = t.log_abs_det_jacobian(x, t.forward(x))
    assert ladj.shape == (2,)
    assert torch.allclose(ladj, torch.tensor([2 * math.log(0.1), 0.0]))


def test_sequential_affine_forward_matches_matrix_and_bias():
    torch.manual_seed(0)
    t = SequentialAffineTransform([LUTransform(DIM), HouseholderTransform(DIM, nvs=2)])
    x = torch.randn(8, DIM)
    with torch.no_grad():
        y = t.forward(x)
        assert torch.allclose(y, x @ t.matrix().T + t.bias(), atol=1e-5)
        assert torch.allclose(t.backward(y), x, atol=1e-4)
        assert torch.allclose(t.matrix() @ t.inverse_matrix(), torch.eye(DIM), atol=1e-4)


def test_block_lu_sign():
    t = BlockLUTransform([2, 3])
    s = t.sign()  # previously AttributeError (self.block_transform)
    assert float(s) in (-1.0, 1.0)


# --------------------------------------------------------------------------- 10
def test_masked_coupling_mask_is_non_persistent_buffer():
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    layer = MaskedCoupling(mask, ConvNet(in_dims=[4], c_hidden=[8]))
    assert "mask" in dict(layer.named_buffers())
    assert "mask" not in layer.state_dict()  # checkpoint compatibility
    layer = layer.double()
    assert layer.mask.dtype == torch.float64


def test_lu_transform_follows_dtype():
    t = LUTransform(DIM).double()
    assert t.L.dtype == torch.float64
    assert t.L_mask.dtype == torch.float64
    x = torch.randn(3, DIM, dtype=torch.float64)
    assert t.log_abs_det_jacobian(x, t.forward(x)).dtype == torch.float64


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA/ROCm device")
def test_flow_on_gpu():
    flow = _make_flow()
    losses = flow.fit(torch.randn(32, DIM), optim=torch.optim.Adam, device="cuda", batch_size=8)
    assert math.isfinite(losses[0])
    with torch.no_grad():
        assert flow.log_prob(torch.randn(4, DIM, device="cuda")).device.type == "cuda"


# --------------------------------------------------------------------------- 11
def test_import_without_pyro():
    """Bug 11: ``import cjd_flows`` / ``cjd_flows.networks`` must not need pyro."""
    code = textwrap.dedent(
        """
        import sys

        class _Block:
            def find_spec(self, name, path=None, target=None):
                if name == "pyro" or name.startswith("pyro."):
                    raise ImportError("pyro blocked")
                return None

        sys.meta_path.insert(0, _Block())
        import cjd_flows
        import cjd_flows.networks as n
        n.ConvNet(in_dims=[4], c_hidden=[8])
        assert "pyro" not in sys.modules
        try:
            n.AdditiveAffineNN(4, [8], 4)
        except ImportError:
            pass
        else:
            raise AssertionError("expected ImportError")
        """
    )
    env = dict(os.environ)
    src = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "src")
    env["PYTHONPATH"] = os.pathsep.join([os.path.abspath(src), env.get("PYTHONPATH", "")])
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
