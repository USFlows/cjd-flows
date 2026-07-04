import torch
from cjd_flows.linalg import solve_triangular

test_size = 10
tol =  1e-5

def test_solve_triangular():
    M_base = torch.ones(10, 10)
    x = torch.stack([torch.rand(10) for _ in range(test_size)])
    
    M_lower = torch.tril(M_base)
    y_lower = torch.stack([M_lower @ x_i for x_i in x])
    x_lower = solve_triangular(M_lower, y_lower)
    
    assert  ((x_lower - x).abs() < tol).all()
    
    M_upper = torch.triu(M_base)
    y_upper = torch.stack([M_upper @ x_i for x_i in x])
    x_upper = solve_triangular(M_upper, y_upper)
    assert ((x_upper - x).abs() < tol).all()