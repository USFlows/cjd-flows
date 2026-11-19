"""CIFAR-10 scaling study: uniformly scaling flows with learnable radial bases.

Example::

    PYTHONPATH=src:. python benchmarks/cifar/train.py --variant us-ggmix --size small --epochs 20

Variants:
    us-gauss   additive CJD flow, standard normal base
    us-ggmix   additive CJD flow, learnable radial base (generalized-gamma mixture norm)
    us-spline  additive CJD flow, learnable radial base (RQ-spline on the log-radius)
    us-mono    additive CJD flow, learnable monotone radial profile
    affine     Jet-style affine coupling flow (non-uniformly-scaling baseline), Gaussian base

bpd convention: pixels are uniformly dequantized to x~ in [0, 256)^D and
affinely mapped to y = 2 x~/256 - 1 in [-1, 1]; then
bpd = (-log p_y(y) - D log(2/256)) / (D ln 2).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.cifar.models import SIZES, build_flow
from cjd_flows.radial import (
    GeneralizedGammaMixture,
    MonotoneProfileNorm,
    RadialBase,
    SplineNorm,
)

ROOT = Path(os.environ.get("CJD_FLOWS_DATA", Path.home() / ".cache" / "cjd_flows"))
LOG2_128 = math.log2(128.0)


def load_cifar(split: str, classes=None) -> torch.Tensor:
    d = ROOT / "torchvision" / "cifar-10-batches-py"
    files = [f"data_batch_{i}" for i in range(1, 6)] if split == "train" else ["test_batch"]
    xs, ys = [], []
    for f in files:
        with open(d / f, "rb") as fh:
            b = pickle.load(fh, encoding="latin1")
        xs.append(np.asarray(b["data"], dtype=np.uint8).reshape(-1, 3, 32, 32))
        ys.append(np.asarray(b["labels"]))
    x, y = torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ys))
    if classes is not None:
        keep = torch.isin(y, torch.tensor(classes))
        x, y = x[keep], y[keep]
    return x, y


def dequantize(xu8: torch.Tensor) -> torch.Tensor:
    x = xu8.float() + torch.rand_like(xu8, dtype=torch.float32)
    return x * (2.0 / 256.0) - 1.0


def make_base(variant: str, latent_shape):
    d = math.prod(latent_shape)
    if variant == "us-ggmix":
        norm = GeneralizedGammaMixture(d, 2.0, n_components=4, learn_scale=False, spread=1.0)
    elif variant == "us-spline":
        norm = SplineNorm(d, 2.0, num_bins=16, bound=6.0, learn_scale=False)
    elif variant == "us-mono":
        norm = MonotoneProfileNorm(d, 2.0, learn_scale=False)
    else:
        return None
    return RadialBase(latent_shape, p=2.0, norm=norm)


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.lerp_(p.detach(), 1 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)


@torch.no_grad()
def evaluate_bpd(flow, data_u8, batch_size, device, amp_dtype, max_n=None):
    flow.eval()
    n = len(data_u8) if max_n is None else min(max_n, len(data_u8))
    tot = 0.0
    D = 3 * 32 * 32
    g = torch.Generator(device="cpu").manual_seed(1234)
    for i in range(0, n, batch_size):
        xb = data_u8[i : i + batch_size]
        x = (xb.float() + torch.rand(xb.shape, generator=g)) * (2 / 256) - 1
        x = x.to(device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            lp = flow.log_prob(x).float()
        tot += (-lp).sum().item()
    nll = tot / n
    return nll / (D * math.log(2)) + LOG2_128


def tweedie(flow, x, sigma, amp_dtype):
    x = x.detach().requires_grad_(True)
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        lp = flow.log_prob(x).float().sum()
    (g,) = torch.autograd.grad(lp, x)
    return (x + sigma**2 * g).detach()


@torch.no_grad()
def sample_grid(flow, n, temperature, device, amp_dtype, seed=0):
    torch.manual_seed(seed)
    z = flow.base.sample((n,)) * temperature
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        x = flow.generate(z.to(device)).float()
    return x


def to_img(x):
    return ((x.clamp(-1, 1) + 1) / 2).cpu()


def save_grid(x, path, nrow=8):
    import torchvision

    torchvision.utils.save_image(to_img(x), path, nrow=nrow)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="us-ggmix")
    ap.add_argument("--size", default="small", choices=list(SIZES))
    ap.add_argument("--epochs", type=float, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--base-lr-mult", type=float, default=10.0, help="lr multiplier for base-distribution params")
    ap.add_argument("--scale-lr-mult", type=float, default=20.0, help="lr multiplier for the global scale")
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--noise", type=float, default=0.0, help="Gaussian noise sigma on [-1,1] (TarFlow-style)")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "none"])
    ap.add_argument("--eval-every", type=float, default=2.0, help="epochs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditioner", default="jet", choices=["jet", "conv"])
    ap.add_argument("--no-shift", action="store_true", help="omit the data-initialized latent Shift (pre-v1 study architecture)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="benchmarks/runs/cifar")
    args = ap.parse_args()

    device = "cuda"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else None
    # Optional per-run overrides (lets a running job queue be re-planned):
    # <out>/OVERRIDES.json maps a default run name to a dict of argument overrides.
    ov_file = Path(args.out) / "OVERRIDES.json"
    default_name = f"{args.size}-{args.variant}" + (f"-noise{args.noise}" if args.noise else "") + f"-s{args.seed}"
    if ov_file.exists():
        for k, v in json.loads(ov_file.read_text()).get(default_name, {}).items():
            setattr(args, k.replace("-", "_"), v)
    name = args.name or f"{args.size}-{args.variant}" + (f"-noise{args.noise}" if args.noise else "") + f"-s{args.seed}"
    out = Path(args.out) / name
    skip_file = Path(args.out) / "SKIP"
    if skip_file.exists() and name in skip_file.read_text().split():
        print(f"[{name}] skipped (listed in {skip_file})", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter(str(out))
    log_f = open(out / "metrics.jsonl", "a")

    def log(step, **kw):
        rec = {"step": step, "time": time.time(), **kw}
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        for k, v in kw.items():
            if isinstance(v, (int, float)):
                tb.add_scalar(k, v, step)

    xtr, _ = load_cifar("train")
    xte, _ = load_cifar("test")
    xtr_dev = xtr.to(device)

    torch.manual_seed(args.seed)
    tmp = build_flow("us-gauss", "tiny")
    latent_shape = tmp.latent_shape
    del tmp
    base = make_base(args.variant, latent_shape)
    flow = build_flow(args.variant, args.size, base=base, seed=args.seed, conditioner=args.conditioner,
                      shift=not args.no_shift).to(device)
    # Data-dependent init of the global scale (MLE for fixed preceding layers;
    # at init all couplings are the identity).
    with torch.no_grad():
        xi = dequantize(xtr[:4096]).to(device)
        h = xi
        layers = list(flow.bijection)
        for b in layers[:-1]:
            h = b(h)
        c = layers[-1].set_from_latents(h, method="mle")
    print(f"[{name}] init global scale c={c:.4f}", flush=True)
    n_params = sum(p.numel() for p in flow.parameters())
    print(f"[{name}] params={n_params/1e6:.1f}M latent={latent_shape}", flush=True)
    log(0, n_params=n_params)

    base_params = list(flow.base.parameters())
    # Scalar global scale: Adam moves a parameter by ~lr per step, so a single
    # scale that has to track the shrinking latent variance needs a larger lr.
    scale_params = [m.log_scale for m in flow.modules() if m.__class__.__name__ == "GlobalScale"]
    special = {id(p) for p in base_params + scale_params}
    other = [p for p in flow.parameters() if id(p) not in special]
    groups = [{"params": other, "weight_decay": args.weight_decay},
              {"params": scale_params, "weight_decay": 0.0, "lr": args.lr * args.scale_lr_mult}]
    if base_params:
        groups.append({"params": base_params, "weight_decay": 0.0, "lr": args.lr * args.base_lr_mult})
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]

    steps_per_epoch = len(xtr) // args.batch_size
    total = int(args.epochs * steps_per_epoch)
    eval_every = max(1, int(args.eval_every * steps_per_epoch))
    ema = EMA(flow, args.ema) if args.ema else None
    D = 3 * 32 * 32

    def lr_factor(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, total - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    step, t0, bad = 0, time.time(), 0
    best = float("inf")
    run_loss = []
    while step < total:
        perm = torch.randperm(len(xtr_dev), device=device)
        for i in range(steps_per_epoch):
            if step >= total:
                break
            idx = perm[i * args.batch_size : (i + 1) * args.batch_size]
            xb = xtr_dev[idx]
            x = (xb.float() + torch.rand(xb.shape, device=device)) * (2 / 256) - 1
            if args.noise > 0:
                x = x + args.noise * torch.randn_like(x)
            f = lr_factor(step)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            flow.train()
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                lp = flow.log_prob(x).float()
            loss = -lp.mean() / D  # nats per dim: keeps grad norms O(1)
            if not torch.isfinite(loss):
                bad += 1
                opt.zero_grad(set_to_none=True)
                if bad > 50:
                    raise RuntimeError("too many non-finite losses")
                step += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(flow.parameters(), args.grad_clip)
            opt.step()
            if ema is not None:
                ema.update(flow)
            run_loss.append(loss.detach())
            step += 1
            if step % 100 == 0:
                nll = torch.stack(run_loss).mean().item() * D
                run_loss = []
                bpd = nll / (D * math.log(2)) + LOG2_128
                scale = None
                for m in flow.modules():
                    if m.__class__.__name__ == "GlobalScale":
                        scale = float(m.log_abs_det().detach()) / D
                log(step, train_bpd=bpd, lr=opt.param_groups[0]["lr"], grad_norm=float(gn),
                    log_global_scale=scale, epoch=step / steps_per_epoch,
                    steps_per_sec=step / (time.time() - t0))
            if step % eval_every == 0 or step == total:
                model = ema.shadow if ema is not None else flow
                n_eval = None if step == total else 2000
                tbpd = evaluate_bpd(model, xte, 500, device, amp_dtype, max_n=n_eval)
                rec = {"test_bpd": tbpd, "epoch": step / steps_per_epoch}
                for T in (1.0, 0.9, 0.8, 0.7):
                    xs = sample_grid(model, 64, T, device, amp_dtype)
                    if args.noise > 0:
                        xs = tweedie(model, xs, args.noise, amp_dtype)
                    tb.add_images(f"samples/T{T}", to_img(xs), step)
                    save_grid(xs, out / f"samples_T{T}_step{step}.png")
                log(step, **rec)
                print(f"[{name}] step {step}/{total} ep {step/steps_per_epoch:.1f} test_bpd {tbpd:.4f} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
                if tbpd < best:
                    best = tbpd
                torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, out / "model.pt")
    final = {"final_test_bpd": tbpd, "best_test_bpd_subset": best, "minutes": (time.time() - t0) / 60, "params": n_params}
    (out / "final.json").write_text(json.dumps(final, indent=2))
    print(f"[{name}] DONE {final}", flush=True)


if __name__ == "__main__":
    main()
