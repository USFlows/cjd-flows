from typing import Optional

import torch
import time
import numpy as np
import math

def solve_triangular(M: torch.Tensor, y: torch.Tensor, pivot: Optional[int]=None, backend: str="torch") -> torch.Tensor:
    """ Re-implementation of torch solve_triangular. Since Onnx export of the native method is currently not supported,
    we implement it with basic torch operations.  
    
    Args:
        M: triangular matrix. May be upper or lower triangular
        y: input vector.
        pivot: If given, determines wether to treat $M$ as a lower or upper triangular matrix. Note that in this case 
            there is no check wether $M$ is actually lower or upper triangular, respectively. It therefore 
            speeds up computation but should be used with care.
        backend: backend to use. If set to torch, call torch.linalg to improve function runtime.
    Returns:
        (torch.Tensor): Solution of the system $Mx=y$
    """
    
    if backend == "torch":
        is_upper = np.allclose(M.cpu().detach().numpy(), np.triu(M.cpu().detach().numpy()))
        if not is_upper:
            if not np.allclose(M.cpu().detach().numpy(), np.tril(M.cpu().detach().numpy())):
                raise ValueError("M needs to be triangular.")
          
        # From: https://pytorch.org/docs/stable/generated/torch.linalg.solve_triangular.html 
        # M size (*, n, n)
        # y size (*, n, k)
        
        if len(y.size()) == 1:
            y = y.unsqueeze(0)
            
        if (M.size(-2) != y.size(-1)) and (M.size(-2) != y.size(-2)):
            raise ValueError(f"M and y must have the same size. Got M={M.size()}, y={y.size()}.") 
        
        if M.size(-2) == y.size(-1):
            y = y.transpose(-1, -2)
        
        x = torch.linalg.solve_triangular(M, y, upper=is_upper).transpose(-1, -2)
        
        return x
    
    else:
        if (M.size(-2) != y.size(-1)) or (M.size(-1) != y.size(-1)):
            raise ValueError(f"M and y must have the same size. Got M={M.size()}, y={y.size()}.") 
        
        dim = y.size(-1)
         
        if dim == 0:
            raise ValueError(f"M and y must be at least 1 dimensional. Got M={M}, y={y}.")
        if dim == 1:
            return y / M
        
        if len(M.size()) != len(y.size()) + 1:
            new_shape = [1] * (len(y.size()) + 1 - len(M.size())) + list(M.size())
            M = M.reshape(new_shape)
       
        # Validate inputs
     
        if pivot is None:
            # Determine orientation of Matrix
            if all([(M[..., i, j] == 0.).all() for i in range(dim) for j in range(i+1, dim)]):
                pivot = 0
            elif all([(M[..., i, j] == 0.).all() for i in range(dim) for j in range(0, i)]):
                pivot = -1
            else:
                raise ValueError("M needs to be triangular.")
        elif pivot not in [0, -1]:
            raise ValueError("pivot needs to be either None, 0, or -1.")
        

        x = torch.zeros_like(y)
        x[..., pivot] = y[..., pivot] / M[..., pivot, pivot]
        x_p = x[..., pivot]
        new_shape = list(x_p.size()) + [1]
        x_p = x_p.reshape(new_shape)
        y_next = (y - x_p * M[..., :, pivot])
        if pivot == 0:
            y_next = y_next[..., 1:] 
            M_next = M[..., 1:, 1:]
            x[..., 1:] = solve_triangular(M_next, y_next, pivot=pivot) 
        else:
            y_next = y_next[..., :-1] 
            M_next = M[..., :-1, :-1]
            x[..., :-1] = solve_triangular(M_next, y_next, pivot=pivot) 

        return x

def random_orthonormal_matrix(n: int) -> torch.Tensor:
    """ Generates a random orthonormal matrix of size n x n.
    
    Args:
        n: The size of the orthogonal matrix.
    """
    # Generate random matrix
    A = torch.randn(n, n)
    A = A / torch.norm(A, dim=0)
    # Apply Gram-Schmidt process
    Q, _ = torch.qr(A)
    return Q