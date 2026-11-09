import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from cjd_flows.core import Bijection, Flow, StandardNormal
from cjd_flows.train import EMA, NoiseAugmentation, TrainResult, evaluate_nll, fit, nats_to_bits_per_dim, tweedie_denoise


# ----------------------------------------------------------------- tiny flow
class Shift(Bijection):
    """z = x - (b + W c): volume preserving, optionally context dependent."""

    volume_preserving = True

    def __init__(self, d, context_dim=0):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(d))
        self.ctx = nn.Linear(context_dim, d, bias=False) if context_dim else None
        if self.ctx is not None:
            nn.init.zeros_(self.ctx.weight)

    def _shift(self, x, context):
        s = self.b
        if self.ctx is not None and context is not None:
            s = s + self.ctx(context)
        return s

    def forward(self, x, context=None):
        return x - self._shift(x, context)

    def inverse(self, z, context=None):
        return z + self._shift(z, context)


class DiagScale(Bijection):
    """z = x * exp(s) (per-dimension), log|det| = sum(s)."""

    def __init__(self, d):
        super().__init__()
        self.s = nn.Parameter(torch.zeros(d))

    def forward(self, x, context=None):
        return x * self.s.exp()

    def inverse(self, z, context=None):
        return z * (-self.s).exp()

    def log_abs_det(self):
        return self.s.sum()


class PriorScale(DiagScale):
    """DiagScale with a Gaussian prior on the log-scales."""

    def log_prior(self):
        return -0.5 * (self.s**2).sum() * 100.0


def tiny_flow(d=2, context_dim=0, prior=False):
    scale = PriorScale(d) if prior else DiagScale(d)
    return Flow([Shift(d, context_dim), scale], StandardNormal(d), context_dim=context_dim)


def gauss_data(n=2000, d=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g) * torch.tensor([3.0, 0.5])[:d] + torch.tensor([2.0, -1.0])[:d]


# ------------------------------------------------------------------- tests
def test_fit_decreases_nll_toy():
    from cjd_flows.data.toy import two_moons

    x = two_moons(3000, seed=0)
    flow = tiny_flow()
    before = evaluate_nll(flow, x)
    res = fit(flow, x, x[:500], epochs=10, batch_size=128, lr=5e-2, seed=0)
    after = evaluate_nll(flow, x)
    assert isinstance(res, TrainResult)
    assert res.steps == 10 * math.ceil(3000 / 128)
    assert len(res.train_losses) == 10 and len(res.val_losses) == 10
    assert after < before - 0.4  # Gaussian fit of moons: optimum ~3.40
    assert res.train_losses[-1] < res.train_losses[0]
    assert res.wall_time > 0


def test_fit_recovers_gaussian_mle():
    x = gauss_data(4000)
    flow = tiny_flow()
    fit(flow, x, epochs=30, batch_size=256, lr=5e-2, seed=0)
    # MLE of a diagonal Gaussian: mean and std
    assert torch.allclose(flow.bijection[0].b.detach(), x.mean(0), atol=0.15)
    assert torch.allclose((-flow.bijection[1].s.detach()).exp(), x.std(0), rtol=0.05)


@pytest.mark.parametrize("kind", ["tensor", "tensordataset", "xy_dataset", "dataloader", "plain_dataset"])
def test_input_types(kind):
    x = gauss_data(512)
    y = torch.zeros(512, dtype=torch.long)

    class XY(Dataset):
        def __len__(self):
            return 512

        def __getitem__(self, i):
            return x[i], y[i]

    class Plain(Dataset):
        def __len__(self):
            return 512

        def __getitem__(self, i):
            return x[i]

    data = {
        "tensor": x,
        "tensordataset": TensorDataset(x, y),
        "xy_dataset": XY(),
        "dataloader": DataLoader(TensorDataset(x), batch_size=64, shuffle=True),
        "plain_dataset": Plain(),
    }[kind]
    flow = tiny_flow()
    res = fit(flow, data, data, epochs=3, batch_size=64, lr=5e-2, seed=0)
    assert res.steps == 3 * 8
    assert res.val_losses[-1] < res.val_losses[0] or res.val_losses[-1] < 4.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_moves_batches_to_device():
    x = gauss_data(512)
    flow = tiny_flow()
    res = fit(flow, x, x, epochs=2, batch_size=128, device="cuda", lr=1e-2)
    assert next(flow.parameters()).device.type == "cuda"
    assert math.isfinite(res.val_losses[-1])


def test_ema_class_semantics():
    lin = nn.Linear(2, 2)
    ema = EMA(lin, decay=0.5, warmup=False)
    w0 = lin.weight.detach().clone()
    with torch.no_grad():
        lin.weight.add_(1.0)
    ema.update()
    assert torch.allclose(ema.shadow["weight"], w0 + 0.5)
    with ema.swap():
        assert torch.allclose(lin.weight, w0 + 0.5)
    assert torch.allclose(lin.weight, w0 + 1.0)  # restored
    ema.copy_to()
    assert torch.allclose(lin.weight, w0 + 0.5)


def test_restore_semantics():
    x = gauss_data(1024)
    v = gauss_data(256, seed=1)
    # last
    flow = tiny_flow()
    res = fit(flow, x, v, epochs=3, lr=5e-2, restore="last", seed=0)
    assert res.restored == "last"
    # best: state equals best_state
    flow = tiny_flow()
    res = fit(flow, x, v, epochs=5, lr=5e-2, restore="best", seed=0)
    assert res.restored == "best"
    for k, t in flow.state_dict().items():
        assert torch.equal(t.cpu(), res.best_state[k])
    assert res.best_val_loss == min(res.val_losses)
    assert abs(evaluate_nll(flow, v) - res.best_val_loss) < 1e-5
    # ema without validation -> ema weights restored
    flow = tiny_flow()
    res = fit(flow, x, epochs=3, lr=5e-2, ema_decay=0.9, restore="best", seed=0)
    assert res.restored == "ema" and res.best_state is None
    # ema with validation: val losses computed with EMA weights, best state = EMA weights
    flow = tiny_flow()
    res = fit(flow, x, v, epochs=3, lr=5e-2, ema_decay=0.9, restore="best", seed=0)
    assert abs(evaluate_nll(flow, v) - res.best_val_loss) < 1e-5
    with pytest.raises(ValueError):
        fit(tiny_flow(), x, epochs=1, restore="ema")


def test_early_stopping():
    x = gauss_data(256)
    v = gauss_data(256, seed=1)
    calls = []

    # lr = 0 -> no improvement after the first epoch
    flow = tiny_flow()
    res = fit(flow, x, v, epochs=50, lr=0.0, early_stopping_patience=3, callback=lambda info: calls.append(info["epoch"]))
    assert res.stopped_early
    assert res.epochs == 4 and len(res.val_losses) == 4
    assert calls == [1, 2, 3, 4]
    assert res.best_epoch == 1
    with pytest.raises(ValueError):
        fit(tiny_flow(), x, epochs=1, early_stopping_patience=2)


def test_callback_stop_and_max_steps():
    x = gauss_data(1000)
    res = fit(tiny_flow(), x, epochs=10, batch_size=100, callback=lambda info: info["epoch"] >= 2)
    assert res.epochs == 2 and res.steps == 20
    res = fit(tiny_flow(), x, epochs=10, batch_size=100, max_steps=25)
    assert res.steps == 25 and res.epochs == 3


class NaNOnce(Bijection):
    """Identity that returns NaN for the first `k` calls in training mode."""

    volume_preserving = True

    def __init__(self, k):
        super().__init__()
        self.k = k
        self.calls = 0
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, x, context=None):
        if self.training:
            self.calls += 1
            if self.calls <= self.k:
                return x * float("nan") + self.dummy
        return x + 0 * self.dummy

    def inverse(self, z, context=None):
        return z


def test_nonfinite_steps_are_skipped():
    x = gauss_data(640)
    flow = Flow([NaNOnce(3), Shift(2), DiagScale(2)], StandardNormal(2))
    res = fit(flow, x, epochs=1, batch_size=64, lr=1e-2)
    assert res.skipped_steps == 3 and not res.aborted
    assert all(math.isfinite(p.sum().item()) for p in flow.parameters())
    assert math.isfinite(res.train_losses[0])


def test_nonfinite_abort():
    x = gauss_data(640)
    flow = Flow([NaNOnce(10**6), Shift(2), DiagScale(2)], StandardNormal(2))
    with pytest.warns(UserWarning, match="aborting"):
        res = fit(flow, x, epochs=5, batch_size=64, max_nonfinite=4)
    assert res.aborted and res.skipped_steps == 4


class ContextSpy(Shift):
    def __init__(self, d, context_dim):
        super().__init__(d, context_dim)
        self.seen = []

    def forward(self, x, context=None):
        if self.training:
            self.seen.append(None if context is None else context.detach().clone())
        return super().forward(x, context)


def test_soft_training_passes_context():
    x = gauss_data(512)
    spy = ContextSpy(2, 1)
    flow = Flow([spy, DiagScale(2)], StandardNormal(2), context_dim=1, default_context=0.0)
    noise = NoiseAugmentation("soft", sigma_max=0.5)
    fit(flow, x, x, epochs=2, batch_size=64, lr=1e-2, noise=noise)
    assert len(spy.seen) == 16
    c = torch.cat(spy.seen)
    assert c.shape == (1024, 1)
    assert (c >= 0).all() and (c <= 1).all() and c.std() > 0.1
    # the context actually influences the flow after training
    assert spy.ctx.weight.abs().sum() > 0
    # soft training requires a context
    with pytest.raises(ValueError):
        fit(tiny_flow(), x, epochs=1, noise=noise)


def test_noise_modes():
    x = torch.zeros(1000, 3)
    g = NoiseAugmentation("gaussian", sigma=0.3)
    xn, c = g(x)
    assert c is None and abs(xn.std().item() - 0.3) < 0.03
    cur = NoiseAugmentation("curriculum", sigma_max=1.0, decay_frac=0.5, schedule="linear")
    assert cur.current_sigma(0, 100) == 1.0
    assert abs(cur.current_sigma(25, 100) - 0.5) < 1e-9
    assert cur.current_sigma(60, 100) == 0.0
    xn, _ = cur(x, 60, 100)
    assert torch.equal(xn, x)
    s = NoiseAugmentation("soft", sigma_max=2.0)
    xn, c = s(x)
    # per-sample sigma = 2 * context
    ratio = (xn.std(1) / (2 * c[:, 0])).median()
    assert 0.6 < ratio < 1.4
    # gaussian noise training runs
    fit(tiny_flow(), gauss_data(256), epochs=1, noise=NoiseAugmentation("gaussian", sigma=0.1))


def test_tweedie_denoise_gaussian():
    # flow = N(m, diag(v)): score = -(x-m)/v
    flow = tiny_flow()
    with torch.no_grad():
        flow.bijection[0].b.copy_(torch.tensor([1.0, -1.0]))
        flow.bijection[1].s.copy_(torch.tensor([0.0, math.log(2.0)]))  # std (1, 0.5)
    x = torch.randn(10, 2)
    out = tweedie_denoise(flow, x, sigma=0.1)
    m = torch.tensor([1.0, -1.0])
    v = torch.tensor([1.0, 0.25])
    assert torch.allclose(out, x - 0.01 * (x - m) / v, atol=1e-6)
    assert not out.requires_grad


def test_prior_term_is_used():
    x = gauss_data(2000)
    f0 = tiny_flow()
    fit(f0, x, epochs=10, lr=5e-2, prior_weight=0.0, seed=0)
    f1 = tiny_flow(prior=True)
    fit(f1, x, epochs=10, lr=5e-2, seed=0)  # prior_weight=None -> auto 1.0
    f2 = tiny_flow(prior=True)
    fit(f2, x, epochs=10, lr=5e-2, prior_weight=500.0, seed=0)
    s0 = f0.bijection[1].s.detach().abs().sum()
    s1 = f1.bijection[1].s.detach().abs().sum()
    s2 = f2.bijection[1].s.detach().abs().sum()
    assert s2 < s1 <= s0 + 1e-3


@pytest.mark.parametrize("amp", [False, "bf16"])
def test_amp_and_options(amp):
    x = gauss_data(512)
    res = fit(tiny_flow(), x, x, epochs=2, amp=amp, schedule="constant", warmup_steps=2, grad_clip=None,
              optimizer=lambda p: torch.optim.SGD(p, lr=1e-2), log_every=2)
    assert math.isfinite(res.val_losses[-1])
    assert len(res.step_losses) == 4


def test_bits_per_dim_helper():
    res = TrainResult(val_losses=[8 * math.log(2) * 3])
    assert res.bits_per_dim(3) == [pytest.approx(8.0)]
    assert nats_to_bits_per_dim(0.0, 4, -4 * math.log(256)) == pytest.approx(8.0)


def test_flow_fit_method():
    flow = tiny_flow()
    res = flow.fit(gauss_data(256), epochs=1)
    assert isinstance(res, TrainResult)
