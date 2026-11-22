import copy
import warnings

import pytest
import torch

import cjd_flows.io as cio
from cjd_flows import Flow, StandardNormal, USFlow, VeriFlow, load, save
from cjd_flows.bijections import LULinear, Shift
from cjd_flows.io import load_metadata, register_builder


def _perturb(module, scale=0.1):
    with torch.no_grad():
        for p in module.parameters():
            p.add_(torch.randn_like(p) * scale)
    return module


CASES = {
    "vector-additive": (lambda: USFlow((6,), blocks=3, hidden=(16,), base="ggmix"), (6,), 0),
    "vector-gin-householder": (lambda: USFlow((6,), blocks=2, hidden=(16,), coupling="gin", mixing="householder", base="spline"), (6,), 0),
    "vector-autoregressive-perm": (lambda: USFlow((5,), blocks=2, hidden=(16,), coupling="autoregressive", mixing="permutation", base="khinchine"), (5,), 0),
    "vector-masked-conjugate-ctx": (lambda: USFlow((4,), blocks=2, hidden=(8,), masked=True, conjugate=True, context_dim=3, base="radial-mixture"), (4,), 3),
    "image-conv": (lambda: USFlow((2, 4, 4), blocks=2, hidden=(8,), base="gg-monotone", p=1.0), (2, 4, 4), 0),
    "image-jet": (lambda: USFlow((2, 4, 4), blocks=1, conditioner="jet", conditioner_kwargs={"embed_dim": 32, "depth": 1, "num_heads": 2}), (2, 4, 4), 0),
    "veriflow": (lambda: VeriFlow((6,), blocks=2, hidden=(8,)), (6,), 0),
}


@pytest.mark.parametrize("case", list(CASES))
def test_roundtrip_rebuilds_identical_flow(tmp_path, case):
    build, shape, ctx = CASES[case]
    torch.manual_seed(0)
    flow = build().double()
    x = torch.randn(7, *shape, dtype=torch.float64)
    c = torch.randn(7, ctx, dtype=torch.float64) if ctx else None
    flow.train()
    flow.log_prob(x, c)  # data-dependent initialization (Shift / ActNorm)
    _perturb(flow)
    flow.eval()

    path = save(flow, tmp_path / "flow.pt")
    torch.manual_seed(123)  # a different RNG state must not matter
    loaded = load(path)

    assert not loaded.training
    assert loaded.builder_config == flow.builder_config
    assert next(loaded.parameters()).dtype == torch.float64
    assert torch.equal(loaded.log_prob(x, c), flow.log_prob(x, c))
    z = torch.randn(4, *shape, dtype=torch.float64)
    cz = c[:4] if c is not None else None
    assert torch.equal(loaded.generate(z, cz), flow.generate(z, cz))
    for a, b in zip(flow.state_dict().items(), loaded.state_dict().items()):
        assert a[0] == b[0] and torch.equal(a[1], b[1])


def test_data_init_flags_survive(tmp_path):
    torch.manual_seed(0)
    flow = USFlow((3,), blocks=1, hidden=(8,))
    flow.train()
    flow.log_prob(torch.randn(16, 3) + 5.0)
    shift = [m for m in flow.modules() if isinstance(m, Shift)][0]
    assert bool(shift.initialized)
    loaded = load(save(flow, tmp_path / "f.pt"))
    loaded_shift = [m for m in loaded.modules() if isinstance(m, Shift)][0]
    assert bool(loaded_shift.initialized) and torch.equal(loaded_shift.shift, shift.shift)
    loaded.train()
    loaded.log_prob(torch.randn(16, 3) - 5.0)  # must not re-initialize
    assert torch.equal(loaded_shift.shift, shift.shift)


def test_config_records_defaults_and_outermost_builder():
    flow = VeriFlow((4,), blocks=1, hidden=(8,))
    cfg = flow.builder_config
    assert cfg["builder"] == "VeriFlow"
    assert cfg["kwargs"] == {"event_shape": [4], "blocks": 1, "hidden": [8], "base": "ggmix-monotone", "p": 1.0}
    inner = cfg["delegate"]  # the fully resolved USFlow call, defaults included
    assert inner["builder"] == "USFlow" and inner["kwargs"]["masked"] is True
    assert inner["kwargs"]["shift"] == "coordinate" and inner["kwargs"]["global_scale"] is True
    usf = USFlow((4,), blocks=1)
    assert usf.builder_config["kwargs"]["mixing"] == "lu"  # defaults are recorded
    assert copy.deepcopy(usf).builder_config == usf.builder_config  # e.g. EMA copies


def test_unserializable_config_needs_model(tmp_path):
    from cjd_flows.conditioners import MLP

    flow = _perturb(USFlow((4,), blocks=1, conditioner=lambda i, o: MLP(i, o, hidden=(8,))))
    assert flow.builder_config is None and "JSON" in flow.builder_config_error
    with pytest.warns(UserWarning, match="without builder configuration"):
        path = save(flow, tmp_path / "f.pt")
    with pytest.raises(ValueError, match="model="):
        load(path)
    target = USFlow((4,), blocks=1, conditioner=lambda i, o: MLP(i, o, hidden=(8,)))
    loaded = load(path, model=target)
    x = torch.randn(5, 4)
    assert torch.equal(loaded.log_prob(x), flow.eval().log_prob(x))


def test_custom_flow_with_model(tmp_path):
    flow = Flow([LULinear((3,))], StandardNormal(3))
    _perturb(flow)
    with pytest.warns(UserWarning):
        path = save(flow, tmp_path / "f.pt")
    loaded = load(path, model=Flow([LULinear((3,))], StandardNormal(3)))
    x = torch.randn(4, 3)
    assert torch.equal(loaded.log_prob(x), flow.log_prob(x))


def test_registered_custom_builder(tmp_path):
    @register_builder(name="TinyTestFlow")
    def tiny(dim: int, scale: float = 1.0):
        return USFlow((dim,), blocks=1, hidden=(4,))

    flow = tiny(3, scale=2.0)
    assert flow.builder_config["builder"] == "TinyTestFlow"
    assert flow.builder_config["kwargs"] == {"dim": 3, "scale": 2.0}
    assert flow.builder_config["delegate"]["builder"] == "USFlow"
    _perturb(flow)
    loaded = load(save(flow, tmp_path / "f.pt"))
    assert loaded.builder_config == flow.builder_config
    x = torch.randn(4, 3)
    assert torch.equal(loaded.log_prob(x), flow.eval().log_prob(x))

    with pytest.raises(ValueError, match="already registered"):
        @register_builder(name="TinyTestFlow")
        def other(dim: int):
            return USFlow((dim,))


def test_metadata_and_file_checks(tmp_path):
    flow = USFlow((2,), blocks=1, hidden=(4,))
    path = save(flow, tmp_path / "f.pt", metadata={"epochs": 3, "note": "test"})
    info = load_metadata(path)
    assert info["metadata"] == {"epochs": 3, "note": "test"}
    assert info["config"]["builder"] == "USFlow" and info["dtype"] == "float32"
    # weights_only loading works on the raw file (no pickled code)
    assert torch.load(path, weights_only=True)["format"] == "cjd-flows"

    with pytest.raises(TypeError):
        save(flow, tmp_path / "g.pt", metadata={"bad": object()})
    torch.save({"something": 1}, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="not a file written"):
        load(tmp_path / "other.pt")

    payload = torch.load(path, weights_only=True)
    payload["format_version"] = cio.FORMAT_VERSION + 1
    torch.save(payload, tmp_path / "future.pt")
    with pytest.raises(ValueError, match="Upgrade"):
        load(tmp_path / "future.pt")

    payload = torch.load(path, weights_only=True)
    payload["library_version"] = "0.9.0"
    torch.save(payload, tmp_path / "old.pt")
    with pytest.warns(UserWarning, match="0.9.0"):
        load(tmp_path / "old.pt")


def test_no_warning_for_builder_flows(tmp_path):
    flow = USFlow((2,), blocks=1, hidden=(4,))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load(save(flow, tmp_path / "f.pt"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_map_location_cuda(tmp_path):
    flow = USFlow((3,), blocks=1, hidden=(8,))
    loaded = load(save(flow, tmp_path / "f.pt"), map_location="cuda")
    assert next(loaded.parameters()).is_cuda


def test_trained_flow_roundtrip(tmp_path):
    from cjd_flows import fit

    torch.manual_seed(0)
    x = torch.randn(256, 2) * torch.tensor([2.0, 0.5]) + 1.0
    flow = USFlow((2,), blocks=2, hidden=(16,), base="ggmix")
    fit(flow, x, epochs=2, batch_size=64, lr=1e-3, ema_decay=0.9, verbose=False)
    assert flow.builder_config is not None
    loaded = load(save(flow, tmp_path / "trained.pt"))
    assert torch.equal(loaded.log_prob(x), flow.eval().log_prob(x))


def test_delegated_defaults_are_frozen(tmp_path, monkeypatch):
    # a later version changes a USFlow default that VeriFlow does not set itself
    flow = _perturb(VeriFlow((4,), blocks=1, hidden=(8,)))
    path = save(flow, tmp_path / "f.pt")
    import cjd_flows.models as models

    monkeypatch.setitem(models.USFlow.__wrapped__.__kwdefaults__, "shift", None)
    assert not any(isinstance(m, Shift) for m in VeriFlow((4,), blocks=1, hidden=(8,)).modules())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = load(path)
    assert any(isinstance(m, Shift) for m in loaded.modules())
    assert loaded.builder_config == flow.builder_config
    x = torch.randn(5, 4)
    assert torch.equal(loaded.log_prob(x), flow.eval().log_prob(x))


def test_outer_builder_that_changes_the_model_falls_back(tmp_path):
    @register_builder(name="WrappedTestFlow")
    def wrapped(dim: int):
        f = USFlow((dim,), blocks=1, hidden=(4,))
        f.bijection.bijections.append(LULinear((dim,)))  # changed after the inner call
        return f

    flow = _perturb(wrapped(3))
    path = save(flow, tmp_path / "f.pt")
    with pytest.warns(UserWarning, match="WrappedTestFlow"):
        loaded = load(path)
    x = torch.randn(4, 3)
    assert torch.equal(loaded.log_prob(x), flow.eval().log_prob(x))


def test_positional_only_builder(tmp_path):
    @register_builder(name="PosOnlyTestFlow")
    def posonly(dim, /, blocks=1):
        return USFlow((dim,), blocks=blocks, hidden=(4,))

    flow = _perturb(posonly(3, blocks=2))
    loaded = load(save(flow, tmp_path / "f.pt"))
    x = torch.randn(4, 3)
    assert torch.equal(loaded.log_prob(x), flow.eval().log_prob(x))
