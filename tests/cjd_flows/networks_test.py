import torch

from cjd_flows.networks import JetConditioner, sincos_pos_embed
from cjd_flows.transforms import MaskedCoupling


def test_sincos_pos_embed():
    """Positional embeddings have the right shape and are unique per token."""
    emb = sincos_pos_embed(64, (4, 4))
    assert emb.shape == (16, 64)
    # all pairwise distinct
    dists = torch.cdist(emb, emb) + torch.eye(16)
    assert (dists > 1e-3).all()


def test_jet_conditioner_shape_image():
    """Output shape equals input shape for spatial inputs."""
    in_dims = [3, 8, 8]
    net = JetConditioner(in_dims, patch_size=2, embed_dim=32, depth=2, num_heads=4)
    x = torch.randn(5, *in_dims)
    assert net(x).shape == x.shape
    assert net(x, context=torch.rand(5, 1)).shape == x.shape


def test_jet_conditioner_shape_vector():
    """Output shape equals input shape for vector inputs (incl. padding case)."""
    in_dims = [10]
    net = JetConditioner(in_dims, embed_dim=32, depth=2, num_heads=4, token_size=4)
    x = torch.randn(5, *in_dims)
    assert net(x).shape == x.shape
    assert net(x, context=torch.rand(5, 1)).shape == x.shape


def test_jet_conditioner_identity_at_init():
    """Zero-initialized head makes the conditioner output zero at init, so the
    enclosing additive coupling starts as the identity."""
    net = JetConditioner([3, 8, 8], patch_size=2, embed_dim=32, depth=2, num_heads=4)
    x = torch.randn(4, 3, 8, 8)
    assert (net(x) == 0).all()
    assert (net(x, context=torch.rand(4, 1)) == 0).all()


def test_jet_masked_coupling_invertibility():
    """Coupling with a Jet conditioner is exactly invertible with zero log-det."""
    torch.manual_seed(0)
    in_dims = [4, 8, 8]
    net = JetConditioner(in_dims, patch_size=2, embed_dim=32, depth=2, num_heads=4)
    # randomize the zero-initialized head so the coupling is non-trivial
    with torch.no_grad():
        net.head.weight.normal_(0, 0.1)
        net.head.bias.normal_(0, 0.1)

    # channel mask
    mask = torch.zeros(in_dims)
    mask[: in_dims[0] // 2] = 1
    coupling = MaskedCoupling(mask, net)

    x = torch.randn(5, *in_dims)
    y = coupling.forward(x)
    assert not torch.allclose(y, x)
    x_rec = coupling.backward(y)
    assert torch.allclose(x_rec, x, atol=1e-5)
    assert coupling.log_abs_det_jacobian(x, y) == 0.0


def test_jet_conditioner_context_sensitivity():
    """Different contexts produce different outputs (after de-zeroing init)."""
    torch.manual_seed(0)
    net = JetConditioner(
        [3, 8, 8], patch_size=2, embed_dim=32, depth=2, num_heads=4
    )
    with torch.no_grad():
        net.head.weight.normal_(0, 0.1)
        for block in net.blocks:
            block.ada[-1].weight.normal_(0, 0.1)

    x = torch.randn(4, 3, 8, 8)
    out_a = net(x, context=torch.zeros(4, 1))
    out_b = net(x, context=torch.ones(4, 1))
    assert not torch.allclose(out_a, out_b)


def test_jet_conditioner_gradients():
    """Gradients flow to head, trunk input path, and context embedding."""
    net = JetConditioner([3, 8, 8], patch_size=2, embed_dim=32, depth=2, num_heads=4)
    x = torch.randn(4, 3, 8, 8)
    out = net(x, context=torch.rand(4, 1))
    out.sum().backward()
    assert net.head.weight.grad is not None
    assert net.head.weight.grad.abs().sum() > 0
    assert net.patch.weight.grad is not None
    assert net.blocks[0].attn.qkv.weight.grad is not None


def test_jet_conditioner_grad_checkpointing():
    """Checkpointed forward matches the non-checkpointed one."""
    torch.manual_seed(0)
    net = JetConditioner(
        [3, 8, 8], patch_size=2, embed_dim=32, depth=2, num_heads=4
    )
    with torch.no_grad():
        net.head.weight.normal_(0, 0.1)
    net.train()
    x = torch.randn(4, 3, 8, 8, requires_grad=True)
    out_plain = net(x)
    net.grad_checkpointing = True
    out_ckpt = net(x)
    assert torch.allclose(out_plain, out_ckpt, atol=1e-6)
    out_ckpt.sum().backward()
    assert net.head.weight.grad is not None


def _dezeroed_seq_net(causal: bool) -> JetConditioner:
    """Sequence-shaped conditioner with randomized head so outputs are non-trivial."""
    torch.manual_seed(0)
    net = JetConditioner(
        [4, 8], patch_size=1, embed_dim=32, depth=2, num_heads=4, causal=causal
    )
    with torch.no_grad():
        net.head.weight.normal_(0, 0.1)
        net.head.bias.normal_(0, 0.1)
        # open the adaLN gates so attention actually mixes tokens
        for block in net.blocks:
            block.ada[-1].weight.normal_(0, 0.1)
            block.ada[-1].bias.normal_(0, 0.1)
    return net


def test_jet_conditioner_causal():
    """With causal=True, outputs at position i are independent of inputs at j > i."""
    net = _dezeroed_seq_net(causal=True)
    x_a = torch.randn(3, 4, 8)
    x_b = x_a.clone()
    x_b[:, :, -3:] = torch.randn(3, 4, 3)  # perturb the last three tokens

    out_a, out_b = net(x_a), net(x_b)
    assert torch.allclose(out_a[:, :, :-3], out_b[:, :, :-3], atol=1e-6)
    assert not torch.allclose(out_a[:, :, -3:], out_b[:, :, -3:])

    # sanity: without causal masking the perturbation propagates everywhere
    net_bidir = _dezeroed_seq_net(causal=False)
    assert not torch.allclose(net_bidir(x_a)[:, :, 0], net_bidir(x_b)[:, :, 0])


def test_jet_conditioner_padding_mask():
    """Padded tokens are excluded as attention keys: outputs at valid positions
    do not depend on inputs at padded positions, and nothing is NaN."""
    net = _dezeroed_seq_net(causal=False)
    padding_mask = torch.ones(3, 8, dtype=torch.bool)
    padding_mask[:, -2:] = False

    x_a = torch.randn(3, 4, 8)
    x_b = x_a.clone()
    x_b[:, :, -2:] = torch.randn(3, 4, 2)  # perturb only padded tokens

    out_a = net(x_a, padding_mask=padding_mask)
    out_b = net(x_b, padding_mask=padding_mask)
    assert not out_a.isnan().any()
    assert torch.allclose(out_a[:, :, :-2], out_b[:, :, :-2], atol=1e-6)


def test_jet_conditioner_causal_with_padding():
    """Causal and padding masks combine; outputs stay finite."""
    net = _dezeroed_seq_net(causal=True)
    padding_mask = torch.ones(2, 8, dtype=torch.bool)
    padding_mask[:, -2:] = False

    x_a = torch.randn(2, 4, 8)
    x_b = x_a.clone()
    x_b[:, :, -2:] = torch.randn(2, 4, 2)

    out_a = net(x_a, padding_mask=padding_mask)
    out_b = net(x_b, padding_mask=padding_mask)
    assert not out_a.isnan().any()
    assert torch.allclose(out_a[:, :, :-2], out_b[:, :, :-2], atol=1e-6)
