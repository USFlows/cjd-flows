import pytest
import torch
from torch import nn

from cjd_flows.core import Bijection, Flow, StandardNormal
from cjd_flows.export import export_module, to_onnx, verify_onnx

onnx = pytest.importorskip("onnx")


class AffineCtx(Bijection):
    """z = (x - b - W c) * exp(s): constant det, optional context."""

    def __init__(self, d, context_dim=0):
        super().__init__()
        self.b = nn.Parameter(torch.randn(d) * 0.1)
        self.s = nn.Parameter(torch.tensor(0.2))
        self.d = d
        self.ctx = nn.Linear(context_dim, d, bias=False) if context_dim else None

    def _shift(self, context):
        s = self.b
        if self.ctx is not None and context is not None:
            s = s + self.ctx(context)
        return s

    def forward(self, x, context=None):
        return (x - self._shift(context)) * self.s.exp()

    def inverse(self, z, context=None):
        return z * (-self.s).exp() + self._shift(context)

    def log_abs_det(self):
        return self.d * self.s


class ReluMix(Bijection):
    """Additive coupling with a ReLU MLP conditioner (exercises nonlinear graphs)."""

    volume_preserving = True

    def __init__(self, d):
        super().__init__()
        self.h = d // 2
        self.net = nn.Sequential(nn.Linear(self.h, 8), nn.ReLU(), nn.Linear(8, d - self.h))

    def forward(self, x, context=None):
        a, b = x[:, : self.h], x[:, self.h :]
        return torch.cat([a, b + self.net(a)], 1)

    def inverse(self, z, context=None):
        a, b = z[:, : self.h], z[:, self.h :]
        return torch.cat([a, b - self.net(a)], 1)


def make_flow(d=4, context_dim=0):
    return Flow([AffineCtx(d, context_dim), ReluMix(d)], StandardNormal(d), context_dim=context_dim)


def test_wrapper_matches_flow():
    flow = make_flow()
    x = torch.randn(5, 4)
    assert torch.allclose(export_module(flow, "log_prob")(x), flow.log_prob(x))
    assert torch.allclose(export_module(flow, "forward")(x), flow.transform(x))
    assert torch.allclose(export_module(flow, "sample")(x), flow.generate(x))
    with pytest.raises(ValueError):
        export_module(flow, "nope")


@pytest.mark.parametrize("mode", ["log_prob", "forward", "inverse", "sample"])
def test_onnx_roundtrip(tmp_path, mode):
    torch.manual_seed(0)
    flow = make_flow()
    path = to_onnx(flow, tmp_path / f"{mode}.onnx", mode=mode, batch_size=2)
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    pytest.importorskip("onnxruntime")
    x = torch.randn(7, 4)  # different batch size -> dynamic batch works
    err = verify_onnx(flow, path, x)
    assert err < 1e-4


def test_onnx_context(tmp_path):
    pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    flow = make_flow(context_dim=2)
    with torch.no_grad():
        flow.bijection[0].ctx.weight.normal_()
    path = to_onnx(flow, tmp_path / "ctx.onnx", mode="log_prob")
    x, c = torch.randn(6, 4), torch.rand(6, 2)
    assert verify_onnx(flow, path, x, c) < 1e-4
    # baked-in default context
    path2 = to_onnx(flow, tmp_path / "noctx.onnx", mode="log_prob", with_context=False)
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path2), providers=["CPUExecutionProvider"])
    assert len(sess.get_inputs()) == 1
    got = sess.run(None, {"x": x.numpy()})[0]
    assert torch.allclose(torch.from_numpy(got), flow.log_prob(x).detach(), atol=1e-4)


def test_onnx_dynamo_path(tmp_path):
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")
    flow = make_flow()
    try:
        path = to_onnx(flow, tmp_path / "dyn.onnx", mode="log_prob", dynamo=True)
    except Exception as e:  # pragma: no cover - exporter environment specific
        pytest.skip(f"dynamo exporter unavailable: {e}")
    assert verify_onnx(flow, path, torch.randn(3, 4)) < 1e-4


@pytest.mark.parametrize("mode", ["log_prob", "forward", "inverse", "sample"])
def test_onnx_transformer_autoregressive(tmp_path, mode):
    # QK-RMSNorm needs the dynamo exporter (to_onnx falls back to it)
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")
    from cjd_flows.bijections import TransformerAutoregressive

    torch.manual_seed(0)
    tok = (4, 3)
    flow = Flow(
        [
            TransformerAutoregressive(tok, embed_dim=16, depth=2, num_heads=2),
            TransformerAutoregressive(tok, embed_dim=16, depth=1, num_heads=2, reverse=True, scale=True),
        ],
        StandardNormal(tok),
    )
    with torch.no_grad():
        for p in flow.parameters():
            p.add_(torch.randn_like(p) * 0.2)
    path = to_onnx(flow, tmp_path / f"tar_{mode}.onnx", mode=mode, batch_size=2)
    assert verify_onnx(flow, path, torch.randn(5, *tok)) < 1e-4
