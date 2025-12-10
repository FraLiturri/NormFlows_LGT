import torch


class SimpleNormal:  # this class implements a gaussian prior distribution, with diagonal covariance matrix, i.e., independent normal variables;
    """
    Simple normal distribution with diagonal covariance matrix. The output have the same shape as loc.

    Parameters
    ----------
    loc: Tensor
        mean of the distribution
    var: Tensor
        variance of the distribution
    """

    def __init__(
        self, loc: torch.Tensor, var: torch.Tensor, device: torch.device | None = None
    ):
        self.dist = torch.distributions.normal.Normal(
            torch.flatten(loc.to(device)), torch.flatten(var.to(device))
        )  # passing means and variances to the built-in Normal distribution;
        self.shape = loc.shape #the shape of the output is equivalent to the shape of the vector loc;

    def log_prob(self, x: torch.Tensor): 
        logp = self.dist.log_prob(x.reshape(x.shape[0], -1)) #log_prob here is a built-in function: returns the log_prob; 
        return torch.sum(logp, dim=1)                        #x.shape[0] return the first dimension of x;

    def sample_n(self, batch_size: int):
        x = self.dist.sample((batch_size,))
        return x.reshape(batch_size, *self.shape)


class MultivariateUniform:
    """Uniformly draw samples from ``[low, high)``.

    Parameters
    ----------
    low: Tensor
        lower range (inclusive)
    high: Tensor
        upper range (exclusive)
    """

    def __init__(self, low: torch.Tensor, high: torch.Tensor, device=None):
        self.dist = torch.distributions.uniform.Uniform(low.to(device), high.to(device))

    def log_prob(self, x: torch.Tensor):
        axes = range(1, len(x.shape))
        return torch.sum(self.dist.log_prob(x), dim=tuple(axes))

    def sample_n(self, batch_size: int):
        return self.dist.sample((batch_size,))
    
