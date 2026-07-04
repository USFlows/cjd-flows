import torch
import math

from src.cjd_flows.transforms import (
    ScaleTransform,
    Permute,
    LUTransform,
    LeakyReLUTransform,
    BlockAffineTransform,
    SequentialAffineTransform,
)

def test_scale_transform():
    """Test scale transform."""
    # Test input
    dim = 10
    transform = ScaleTransform(10)
    with torch.no_grad():
        transform.scale.copy_(torch.ones(dim) * 2)
    x = torch.ones(dim)
    
    # Test forward, inverse, and log det
    y = transform(x)
    assert (y == 2 * x).all()
    assert (transform.backward(y) == x).all()
    log_det = transform.log_abs_det_jacobian(x, y)
    assert log_det == dim * torch.log(torch.tensor(2))
    
def test_permute():
    """Test permute."""
    # Test input
    dim = 10
    transform = Permute(torch.arange(dim))
    x = torch.arange(dim)
    
    # Test forward, inverse, and log det
    y = transform(x)
    assert (y == x).all()
    assert (transform._inverse(y) == x).all()
    log_det = transform.log_abs_det_jacobian(x, y)
    assert log_det == 0
    
def test_lu_transform():
    """Test LU transform."""
    # Test input
    dim = 10
    transform = LUTransform(dim)
    with torch.no_grad():
        transform.L_raw.copy_(torch.tril(torch.ones(dim, dim)))
        transform.U_raw.copy_(torch.eye(dim))
        transform.bias_vector.copy_(torch.zeros(dim))
    x = torch.ones(dim)
    
    # Test forward, inverse, and log det
    y = transform(x) # LU-factorization parametrizes inverse
    assert (y == (torch.arange(dim) + 1.)).all()
    assert (transform.backward(y) == x).all()
    log_det = transform.log_abs_det_jacobian(x, y)
    assert log_det == 0
    
def test_leaky_relu_transform():
    """Test leaky ReLU transform."""
    # Test input
    base_dim = 5
    dim = 2 * base_dim
    transform = LeakyReLUTransform()
    x = torch.tensor([1., -1.] * base_dim)
    y_true = x * torch.tensor([1., .01] * base_dim)
    
    # Test forward, inverse, and log det
    y = transform(x) 
    assert (y == y_true).all()
    assert (transform.backward(y) == x).all()
    log_det = transform.log_abs_det_jacobian(x, y)
    assert log_det == base_dim * torch.log(torch.tensor(.01))


def test_lu_log_prior_include_constants_delta():
    dim = 5
    prior_scale = 1.7
    transform = LUTransform(dim, prior_scale=prior_scale)

    lp_unnormalized = transform.log_prior(include_constants=False)
    lp_exact = transform.log_prior(include_constants=True)

    expected_delta = -dim * math.log(prior_scale * math.sqrt(2 * math.pi))
    observed_delta = float(lp_exact - lp_unnormalized)
    assert abs(observed_delta - expected_delta) < 1e-6


def test_block_and_sequential_forward_log_prior_flags():
    lu1 = LUTransform(4, prior_scale=1.2)
    lu2 = LUTransform(4, prior_scale=0.8)
    seq = SequentialAffineTransform([lu1, lu2])
    block = BlockAffineTransform([4], seq)

    lp_seq = seq.log_prior(include_constants=False)
    lp_block = block.log_prior(include_constants=False)
    assert torch.allclose(lp_block, lp_seq)

    lp_seq_exact = seq.log_prior(include_constants=True)
    lp_block_exact = block.log_prior(include_constants=True)
    assert torch.allclose(lp_block_exact, lp_seq_exact)
