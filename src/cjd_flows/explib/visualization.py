from typing import Dict, Iterable, Literal
from matplotlib import pyplot as plt
import numpy as np
from cjd_flows.explib import datasets
import torch

from cjd_flows.flows import Flow

Norm = Literal[-1, 1, 2]
SampleType = Literal["conditional", "boundary", "boundary_basis"]

class FakeModel(torch.nn.Module):
    """A fake model that samples from a dataset.
    
    Args:
        dataset: The dataset to sample from.
    """
    def __init__(self, dataset: datasets):
        super().__init__()
        self.dataset = dataset
        self.n = len(dataset)
    
    def sample(self, shape):
        """Samples from the dataset.
        
        Args:
            shape: The shape of the samples.
        
        Returns:
            A tensor of shape `shape`.
        """
        return self.dataset[np.random.choice(self.n, shape)][0]

def visualize_udl(
    model: Flow, 
    p: Norm, 
    sqrt_n_sample: int = 3,
    n_udl_estimation: int = 100000,
    saveto = None
):
    """ Visualizes the UDL of a model. Assumes that the UDL is of the form $p(x | |z|_p <= r)$. 
    
    Args:
        model: The model to visualize.
        p: The norm to use for the UDL.
        sqrt_n_sample: The number of samples to draw in each row and column.
    
    """
    with torch.no_grad():
        sample = norm(model.base_distribution.sample((n_udl_estimation,)), p)
    
    sample_types = ["conditional", "boundary", "boundary_basis"]
    fig, axes = plt.subplots(len(sample_types), len(thresholds), figsize=(4*len(thresholds),4*len(sample_types)))
    
    first = True
    for sample_type, ax_row in zip(sample_types, axes):
        for prob, ax in zip(thresholds, ax_row):
            radius = sample.quantile(prob)
            sample_cond = udl_multisample(model, radius, sample_type, p, sqrt_n_sample)
            ax.imshow(sample_cond, cmap="gray")
            ax.set_title(f"{sample_type}\n{prob} UDL \n($|z|_p = {radius:.2f}$)")
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    if saveto is not None:
        plt.savefig(saveto)
    plt.show()
    
def norm(x: torch.Tensor, p: Norm):
    """ Computes the L_p norm of a tensor x.
    
    Args:
        x: A tensor of shape (..., dim)
        p: The norm to compute. If p is -1 or inf, the maximum norm is computed.
        
    Returns:
            A tensor of shape (...,)
    """
    if p == -1 or p == float("inf"):
        return x.abs().max(dim=-1)[0]
    elif p == 1:
        return x.abs().sum(dim=-1)
    elif p == 2:
        return torch.sqrt((x * x).sum(dim=-1))
    
def udl_multisample(
    model: Flow, 
    radius: float, 
    sample_type: SampleType,
    p: Norm, 
    sqrtn: int, 
    reshape: Iterable[int] = (10, 10)
):
    """Samples from models's base distribution conditioned on the the specified radius. Depending on the sample_type, different conditional distributions are drawn.
    
    Args:
        model: The model to sample from.
        radius: The radius of the conditional distribution.
        sample_type: The type of the conditional distribution. The options are "conditional": $p(x | |z|_p <= radius)$, "boundary": $p(x | |z|_p = radius)$, and "boundary_basis": "boundary": $p(x | for all q: |z|_q = radius)$. 
        p: The norm to use for the conditioning.
        sqrtn: The number of samples to draw in each row and column.
        reshape: The shape of the samples.
    """
    latent_vars = []
    n = sqrtn**2
    while len(latent_vars) < n:
        latent = model.base_distribution.sample()
        if sample_type == "conditional":
            if norm(latent, p) < radius:
                latent_vars.append(latent)
        elif sample_type == "boundary":
            latent = radius * latent / norm(latent, p)
            latent_vars.append(latent)
        elif sample_type == "boundary_basis":
            dim = latent.shape[0]
            i = torch.distributions.Categorical(torch.ones(dim) / dim).sample()
            rei = radius * torch.eye(dim)[i]
            latent_vars.append(rei)
            
    
    latent_shape =  (sqrtn, sqrtn) + tuple(latent_vars[0].shape) 
    latent_vars = torch.stack(latent_vars).reshape(latent_shape)
    model.export = "forward"
    with torch.no_grad():
        sample = torch.clip(model.forward(latent_vars), 0, 1) * 255
     
    if reshape is not None:
        reshape = (sqrtn, sqrtn) + tuple(reshape)
        sample = sample.reshape(reshape)
    
    sample = torch.cat([x for x in sample], dim=-1)
    sample = torch.cat([x for x in sample], dim=0)
    
    return sample


def plot_digits(models: Flow, sqrtn: int, save_to=None):
    """ Plot the samples from the models in a grid.
    
    Args:
        models: A dictionary of models. The keys are the digits and the values are dictionaries of models.
        sqrtn: The number of samples to plot in each row and column.
        save_to: If not None, the plot is saved to this path.
    """
    with torch.no_grad():
        experiments = ["MNIST"] + sorted([
            exp for exp in models["0"].keys()
        ])
        for digit in models.keys():
            models[digit]["MNIST"] = FakeModel(digit)   
    
        ncols = len(models)
        nrows = len(experiments) 
        figsize = (7 * ncols, 25)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        for digit, axcol in zip(sorted(models.keys()), axes.T):
            digit_models = models[digit]
            for exp, ax in zip(experiments, axcol):                 
                model = digit_models[exp]
                
                sample = nsample(model, sqrtn)
                
                ax.imshow(sample, cmap="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(exp, fontsize=55)
        plt.tight_layout()
        if save_to:
            plt.savefig(save_to)
        plt.show()
        
def latent_radial_qqplot(models: Dict[str, Flow], data: datasets, p, n_samples, save_to=None):
    """Plots a QQ-plot of the empirical and theoretical distribution of the L_p norm of the latent variables.
    
    Args:
        model: The model to visualize.
        p: The norm to use.
        n_samples: The number of samples to draw from the base distribution.
        n_bins: The number of bins to use in the histogram.
        save_to: If not None, the plot is saved to this path.
    """
    fakemodel = FakeModel(data)
    true_samples = fakemodel.sample((n_samples,))
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.set_xlabel("Latent radial quantiles of the true distribution under the model")
    ax.set_ylabel("Latent radial quantiles of the learned distribution under the model")
    curves = {"Optimal": plt.plot([0, 1], [0, 1])}
    for name, model in models.items():
        learned_samples =  model.sample((n_samples,))
        
        model.export = "backward" 
        true_latent_norms = norm(model.forward(true_samples.to(model.device)), p).sort()[0].cpu().detach()
        learned_latent_norms = norm(model.forward(learned_samples), p).sort()[0].cpu().detach()
        
        def cdf(r, samples):
            return (samples <= r).sum()/samples.shape[0]

        tqs = [cdf(r, true_latent_norms).detach() for r in true_latent_norms] 
        lqs = [cdf(r, learned_latent_norms).detach() for r in true_latent_norms] 


        curves[name] = ax.plot([0.] + list(tqs) + [1.], [0.] + list(lqs) + [1.])
    
    plt.legend(curves.keys())
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to)
    plt.show() 