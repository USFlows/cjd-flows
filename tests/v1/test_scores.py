
import pytest
import torch
from torch import nn

from cjd_flows.core import BaseDistribution, Bijection, Flow, StandardNormal
from cjd_flows.scores import (
    TypicalityScorer,
    anomaly_scores,
    aupr,
    auroc,
    detection_metrics,
    fpr_at_tpr,
    latent_norm,
    one_class_protocol,
)


class Shift(Bijection):
    volume_preserving = True

    def __init__(self, d):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, x, context=None):
        return x - self.b

    def inverse(self, z, context=None):
        return z + self.b


class GlobalScale(Bijection):
    def __init__(self, d, log_scale=0.0):
        super().__init__()
        self.d = d
        self.s = nn.Parameter(torch.tensor(float(log_scale)))

    def forward(self, x, context=None):
        return x * self.s.exp()

    def inverse(self, z, context=None):
        return z * (-self.s).exp()

    def log_abs_det(self):
        return self.d * self.s


class Rotation(Bijection):
    volume_preserving = True

    def __init__(self, d, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
        self.register_buffer("q", q)

    def forward(self, x, context=None):
        return x @ self.q.T

    def inverse(self, z, context=None):
        return z @ self.q


def make_flow(d=4):
    return Flow([Shift(d), Rotation(d), GlobalScale(d, 0.3)], StandardNormal(d))


# --------------------------------------------------------------- metrics
def _sk():
    return pytest.importorskip("sklearn.metrics")


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("ties", [False, True])
def test_metrics_match_sklearn(seed, ties):
    skm = _sk()
    import numpy as np

    g = torch.Generator().manual_seed(seed)
    s_in = torch.randn(300, generator=g)
    s_out = torch.randn(200, generator=g) + 1.0
    if ties:
        s_in, s_out = s_in.round(), s_out.round()
    y = np.r_[np.zeros(300), np.ones(200)]
    s = np.r_[s_in.numpy(), s_out.numpy()]
    assert auroc(s_in, s_out) == pytest.approx(skm.roc_auc_score(y, s), abs=1e-10)
    assert aupr(s_in, s_out) == pytest.approx(skm.average_precision_score(y, s), abs=1e-10)
    fpr, tpr, _ = skm.roc_curve(y, s, drop_intermediate=False)
    ref = fpr[np.searchsorted(tpr, 0.95, side="left")]
    assert fpr_at_tpr(s_in, s_out, 0.95) == pytest.approx(ref, abs=1e-10)


def test_metrics_edge_cases():
    assert auroc(torch.zeros(5), torch.ones(5)) == 1.0
    assert auroc(torch.ones(5), torch.zeros(5)) == 0.0
    assert auroc(torch.ones(5), torch.ones(5)) == 0.5
    assert fpr_at_tpr(torch.zeros(5), torch.ones(5)) == 0.0
    assert aupr(torch.zeros(5), torch.ones(5)) == 1.0
    m = detection_metrics(torch.zeros(5), torch.ones(5))
    assert set(m) == {"auroc", "aupr", "fpr@95"}
    with pytest.raises(ValueError):
        auroc(torch.tensor([]), torch.ones(2))


# --------------------------------------------------------------- scores
def test_scores_shapes_and_values():
    flow = make_flow()
    x = torch.randn(100, 4) * 2
    s = anomaly_scores(flow, x, batch_size=33)
    assert set(s) == {"nll", "latent_norm", "typicality", "hdr"}
    for v in s.values():
        assert v.shape == (100,)
    with torch.no_grad():
        assert torch.allclose(s["nll"], -flow.log_prob(x), atol=1e-5)
        r = flow.transform(x).norm(dim=-1)
    assert torch.allclose(s["latent_norm"], r, atol=1e-5)
    assert ((s["hdr"] >= 0) & (s["hdr"] <= 1)).all()
    # chi CDF / log density sanity vs scipy
    stats = pytest.importorskip("scipy.stats")
    assert torch.allclose(s["hdr"].double(), torch.from_numpy(stats.chi(4).cdf(r.numpy())), atol=1e-5)
    assert torch.allclose(s["typicality"].double(), -torch.from_numpy(stats.chi(4).logpdf(r.numpy())), atol=1e-4)


def test_nll_and_latent_norm_same_auroc():
    """Constant det + radially decreasing base -> nll is monotone in the latent norm."""
    torch.manual_seed(0)
    flow = make_flow(6)
    x_in = flow.sample((400,))
    x_out = flow.sample((300,)) * 1.3 + 0.2
    s_in = anomaly_scores(flow, x_in, ["nll", "latent_norm", "hdr"])
    s_out = anomaly_scores(flow, x_out, ["nll", "latent_norm", "hdr"])
    a_nll = auroc(s_in["nll"], s_out["nll"])
    assert a_nll == pytest.approx(auroc(s_in["latent_norm"], s_out["latent_norm"]), abs=1e-12)
    assert a_nll == pytest.approx(auroc(s_in["hdr"], s_out["hdr"]), abs=1e-12)
    assert a_nll > 0.6


def test_typicality_reference_and_availability():
    class OddBase(BaseDistribution):
        """Standard normal not recognised as radial."""

        def __init__(self, d):
            super().__init__()
            self.event_shape = (d,)
            self.inner = StandardNormal(d)

        def log_prob(self, z):
            return self.inner.log_prob(z)

        def sample(self, sample_shape=()):
            return self.inner.sample(sample_shape)

    flow = Flow([Shift(3)], OddBase(3))
    x = torch.randn(50, 3)
    s = anomaly_scores(flow, x)
    assert set(s) == {"nll", "latent_norm"}
    with pytest.raises(ValueError):
        anomaly_scores(flow, x, ["hdr"])
    with pytest.raises(ValueError):
        anomaly_scores(flow, x, ["typicality"])
    ref = TypicalityScorer().fit(flow, torch.randn(1000, 3))
    s = anomaly_scores(flow, x, ["nll", "typicality"], reference=ref)
    assert torch.allclose(s["typicality"], (s["nll"] - ref.mean).abs())
    assert torch.allclose(ref.score(flow, x), s["typicality"])


def test_radial_base_duck_typing():
    class Norm(nn.Module):
        def log_prob(self, r):
            return -r  # Exp(1)

        def cdf(self, r):
            return 1 - torch.exp(-r)

    class RadialLike(StandardNormal):
        def __init__(self, d):
            super().__init__(d)
            self.norm = Norm()
            self.p = 1.0

        def norm_of(self, z):
            return z.abs().sum(-1)

    flow = Flow([Shift(3)], RadialLike(3))
    x = torch.randn(20, 3)
    s = anomaly_scores(flow, x)
    assert torch.allclose(s["latent_norm"], x.abs().sum(-1))
    assert torch.allclose(s["typicality"], x.abs().sum(-1))
    assert torch.allclose(s["hdr"], 1 - torch.exp(-x.abs().sum(-1)))


def test_latent_norm_loc_p():
    class B(StandardNormal):
        def __init__(self):
            super().__init__(2)
            self.loc = torch.tensor([1.0, 1.0])
            self.p = 1

    z = torch.tensor([[2.0, 3.0]])
    assert latent_norm(B(), z).item() == pytest.approx(3.0)


def test_dataset_input_and_protocol():
    from torch.utils.data import TensorDataset

    x = torch.randn(64, 4)
    flow = make_flow()
    s1 = anomaly_scores(flow, TensorDataset(x, torch.zeros(64)), ["nll"])
    s2 = anomaly_scores(flow, x, ["nll"])
    assert torch.allclose(s1["nll"], s2["nll"])

    def get_data(c):
        g = torch.Generator().manual_seed(c)
        tr = torch.randn(512, 2, generator=g) + c
        return tr, torch.randn(200, 2, generator=g) + c, torch.randn(200, 2, generator=g) * 3 + c

    res = one_class_protocol(
        [0, 1],
        lambda c: Flow([Shift(2), GlobalScale(2)], StandardNormal(2)),
        get_data,
        fit_kwargs=dict(epochs=5, lr=5e-2),
        kinds=("nll", "latent_norm"),
    )
    assert set(res) == {0, 1}
    for c in res:
        assert res[c]["nll"]["auroc"] > 0.7
        assert res[c]["nll"]["auroc"] == pytest.approx(res[c]["latent_norm"]["auroc"])
