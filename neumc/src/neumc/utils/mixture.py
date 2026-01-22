import numpy as np
import neumc.nf.flow
from neumc.utils.batch_function import batch_action
import torch
import math
from neumc.physics.u1 import torch_mod, U1GaugeAction
import neumc.nf.prior, neumc.nf.flow, neumc.nf.flow_abc
from torch.distributions import Categorical

"""
Personal note: each sample corresponds to two LxL matrices: the first one contains 
the horizontal links and the second one contains the vertical links. The links variables are 
saved without taking account of the direction, as a consqeuence to compute the plaquette, the signs
have to be considered.
The transformation must be a volume preserving (and invertible) transformation and
must act on two LxL matrices at a time.
"""
def target_density(x, action) -> torch.Tensor:
    return torch.exp(- action(x))

class LinksTransformation:
    def __init__(self, shape =(8,8), device = 'cpu'):
        self.device = device
        self.shape = shape
    def apply(self) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def __call__(self, config):
        return self.apply(config)

class RandomLinksTransformation(LinksTransformation): #! Work in progress; 
    """Given a certain configuration (i.e. two LxL matrices), a random matrix 
    is sampled and added to the first matrix; then its transpose is added to 
    the second matrix. In such a way the plaquette is preserved.
    """
    def __init__(self, shape, device = 'cpu'):
        super().__init__(shape, device)

    def transformation_matrices(self) -> torch.Tensor:
        m1 = torch.rand(self.shape, device=self.device) * 2 * math.pi
        m2 = m1.transpose(-2, -1)  # usa -2, -1 per gestire batch
        print("random:", m1, m2)
        return torch.stack((m1, m2))

    def apply(self, config):        
        m = self.transformation_matrices()
        return config + m  

class AdaptiveMixture:
    """Implements and performs sampling from an adaptive mixture."""
    def __init__(self, *, transformations: list, action):
        self.transformations = transformations
        self.mix_samples = []
        self.log_q_mix = [] #! not implemented yet;
        self.action = action

    def sample_from_mix(self, *, prior, layers: neumc.nf.flow_abc.Transformation, n_samples: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        rem_size = n_samples

        while rem_size > 0: 
            with torch.no_grad():
                p_a = []
                x_a_samples = []
                batch_length = min(rem_size, batch_size)
                x, logq = layers.sample(prior, batch_size=batch_length)
                for t_a in self.transformations:
                    x_a = t_a(x)
                    x_a_samples.append(x_a)
                    print(x_a)
                    p_a.append(target_density(x_a, self.action))

                p_a = torch.tensor(p_a / np.sum(p_a))
                if len(self.transformations) > 1:
                    p_a = p_a.squeeze()

                dist = Categorical(p_a) #create a categorical distribution based on the weights p_a;
                sampled_index = dist.sample()

                #!to do: with batch_size = 1 works, but there's a bug for greater values. 
                #!review the data structure. 

                x_a = x_a_samples[sampled_index]
                
            self.mix_samples.append(x_a.cpu()) 
            self.log_q_mix.append(logq.cpu()) #since this trans. does not change the volume, log_q remains the same;
            rem_size -= batch_length
        return torch.cat(self.mix_samples, 0), torch.cat(self.log_q_mix, -1) #like u_2x1, lq_2x1 in u1_rs.py;

    def __call__(self, *args, **kwargs):
        return self.sample_from_mix(*args, **kwargs)


class UniformMixture: #!work in progress
    """Implements and performs sampling from an adaptive mixture."""
    def __init__(self, *, transformations: list[callable], action):
        self.transformations = transformations
        self.mix_samples = []
        self.log_q_mix = [] #! not implemented yet;
        self.action = action

    def sample_from_mix(self, *, prior, layers: neumc.nf.flow_abc.Transformation, n_samples: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        rem_size = n_samples

        while rem_size > 0: 
            with torch.no_grad():
                p_a = []
                x_a_samples = []
                batch_length = min(rem_size, batch_size)
                x, logq = layers.sample(prior, batch_size=batch_length)
                for t_a in self.transformations:
                    x_a = t_a(x)
                    x_a_samples.append(x_a)
                    p_a.append(target_density(x_a, self.action))

                p_a = torch.tensor(p_a / np.sum(p_a))
                if len(self.transformations) > 1:
                    p_a = p_a.squeeze()

                dist = Categorical(p_a) #create a categorical distribution based on the weights p_a;
                sampled_index = dist.sample()

                #!to do: with batch_size = 1 works, but there's a bug for greater values. 
                #!review the data structure. 

                x_a = x_a_samples[sampled_index]
                
            self.mix_samples.append(x_a.cpu()) 
            self.log_q_mix.append(logq.cpu()) #since this trans. does not change the volume, log_q remains the same;
            rem_size -= batch_length
        return torch.cat(self.mix_samples, 0), torch.cat(self.log_q_mix, -1) #like u_2x1, lq_2x1 in u1_rs.py;

    def __call__(self, *args, **kwargs):
        return self.sample_from_mix(*args, **kwargs)
