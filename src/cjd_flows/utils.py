import torch

def inv_softplus(x: torch.Tensor) -> torch.Tensor:
    """Computes the inverse of the softplus function.

    :param x: The input tensor.
    :return: The inverse of the softplus function applied to x.
    """
    return torch.log(torch.exp(x) - 1)
