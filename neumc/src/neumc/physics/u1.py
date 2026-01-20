# Status: done ✅
import numpy as np
import torch

scipy_installed = True
try:
    from scipy.special import iv
except ImportError as e:
    scipy_installed = False
    print(f"scipy is not installed: {e}")


M_2PI = 2 * torch.pi  # defines a reusable constant (=2pi);


def torch_mod(x):  # assures that all the elements in x are in (0, 2pi);
    """
    - torch.remainder calculates the remainder of division element-wise;
    - torch.where(condition, x, y) return a tensor of elements in x for which condition is True, and elements in y elsewhere;
      in this way the elements of the new tensor are all lesser than 2pi.
    """
    x = torch.remainder(x, M_2PI)
    x = torch.where(
        x >= M_2PI, x - M_2PI, x
    )  # safety check to ensure x < 2pi (floating point issues can erase equality);
    return x


def torch_wrap(x):  # transforms angles to the interval [-pi, pi);
    return torch_mod(x + np.pi) - np.pi


debug_info = {}


if scipy_installed:

    def logZ(L, beta, *, n=2):
        z = L * L * np.log(iv(0, beta))
        x = np.sum(
            2 * np.power(iv(np.arange(1, n + 1), beta) / iv(0, beta), L * L)
        )  # iv(k, beta) is the modified Bessel function of k order;
        return z + x - x * x / 2


def set_weights(m):  # sets weights and biases in the CNN;
    if hasattr(m, "weight") and m.weight is not None:
        torch.nn.init.normal_(
            m.weight, mean=1, std=2
        )  # weights are guassian distributed...!;
    if hasattr(m, "bias") and m.bias is not None:
        m.bias.data.fill_(-1)


def compute_u1_plaq(links, mu, nu):  # returns the plaquette in the (mu,nu) plane;
    """Compute U(1) plaquettes in the (mu,nu) plane given `links` = arg(U)

    Example: the links for L = 4, batch_size = 1 are:
        links = [[[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9,10,11,12],
                  [13,14,15,16]],

                 [[17,18,19,20],
                  [21,22,23,24],
                  [25,26,27,28],
                  [29,30,31,32]]]

    Then, for mu=0 and nu=1:
        links[:, mu] = [[1, 2, 3, 4], #takes the mu=0-th element;
                        [5, 6, 7, 8],
                        [9,10,11,12],
                        [13,14,15,16]]

        torch.roll(links[:, nu], -1, mu + 1) = [[29, 30, 31, 32], #rolls the first row;
                                                [17, 18, 19, 20],
                                                [21, 22, 23, 24],
                                                [25, 26, 27, 28]]

        torch.roll(links[:, mu], -1, nu + 1) = [[2, 3, 4, 1], #rolls the first column;
                                                [6, 7, 8, 5],
                                                [10,11,12,9],
                                                [14,15,16,13]]

        links[:, nu] = [[17,18,19,20], #takes the nu=1-th element;
                        [21,22,23,24],
                        [25,26,27,28],
                        [29,30,31,32]]
    """
    return torch_mod(
        links[:, mu]
        + torch.roll(links[:, nu], -1, mu + 1)
        - torch.roll(links[:, mu], -1, nu + 1)
        - links[:, nu]
        + 2 * torch.pi
    )


def u1_2x1_loops(links, mu, nu):  # returns 2x1 loops;
    return torch_mod(
        links[:, mu]
        + torch.roll(links[:, mu], -1, mu + 1)
        + torch.roll(links[:, nu], -2, mu + 1)
        - torch.roll(torch.roll(links[:, mu], -1, nu + 1), -1, mu + 1)
        - torch.roll(links[:, mu], -1, nu + 1)
        - links[:, nu]
        + 2 * torch.pi
    )


def compute_u1_2x1_loops(links):  # stacks all the 2x1 loops into a unique tensor;
    return torch.stack((u1_2x1_loops(links, 0, 1), u1_2x1_loops(links, 1, 0)), 1)


class U1GaugeAction:
    def __init__(self, beta):
        self.beta = beta

    def __call__(self, cfgs):
        Nd = cfgs.shape[1]
        action_density = 0
        for mu in range(Nd):
            for nu in range(mu + 1, Nd):
                action_density = action_density + torch.cos(
                    compute_u1_plaq(cfgs, mu, nu)
                )
        return -self.beta * torch.sum(action_density, dim=tuple(range(1, Nd + 1)))


def gauge_transform(links, alpha):
    transformed_links = links.clone()
    for mu in range(len(links.shape[2:])):
        transformed_links[:, mu] = torch_mod(
            alpha + links[:, mu] - torch.roll(alpha, -1, mu + 1)
        )
    return transformed_links


def random_gauge_transform(x, device):
    nconf, vol_shape = x.shape[0], x.shape[2:]
    return gauge_transform(
        x, 2 * np.pi * torch.rand((nconf,) + vol_shape, device=device)
    )


def topo_charge(x):
    """P01 returns a tensor with n-samples (matrices);
       each matrix contains the plaquettes in the (0,1) plane. 
       Then, for each sample matrix, all the components of each matrix are summed and divided by 2pi
       to compute the topological charge.
    """
    P01 = torch_wrap(compute_u1_plaq(x, mu=0, nu=1))
    axes = tuple(range(1, len(P01.shape)))
    return torch.sum(P01, dim=axes) / (2 * np.pi)
