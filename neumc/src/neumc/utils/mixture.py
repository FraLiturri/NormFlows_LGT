import neumc
import numpy as np
import torch
from neumc.nf.flow_abc import TransformationSequence
from torch.distributions import Categorical
from typing import override
from neumc.utils import grab
from neumc.nf.u1_model_asm import assemble_model_from_dict


"""
Personal note: each sample corresponds to two LxL matrices: the first one contains 
the horizontal links and the second one contains the vertical links. The links variables are 
saved without taking account of the direction, as a consqeuence to compute the plaquette, the signs
have to be considered.
The transformation must be a volume preserving (and invertible) transformation and
must act on two LxL matrices at a time.
"""

class Mixture:
    def __init__(self):
        pass
    @override
    def mixture(self, *args, **kwargs):
        pass
    
    def __call__(self, *args, **kwargs):
        pass

class DataLoader:
    def __init__(self, *, path_to_folder: str, beta_min: int, beta_max: int, step: float, samples_size: int, L: int = 8):
        self.path_to_folder = path_to_folder
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.step = step
        self.samples_size = samples_size
        self.L = L
        self.epsilon = 1e-10
        self.beta_values = np.arange(self.beta_min, self.beta_max + self.step - self.epsilon, self.step)
    
    def load_data(self):
        self.samples = torch.zeros(len(self.beta_values), self.samples_size, 2, self.L, self.L)
        for i in range(len(self.beta_values)):
            current_path = self.path_to_folder + f'/data_{self.beta_values[i]}.pt'
            data = torch.load(current_path, weights_only=False)
            self.samples[i] = data['phi']

        return self.samples

    def load_models(self, device: torch.device | str = "cpu") -> list:
        self.models = []
        for i in range(len(self.beta_values)):
            path = self.path_to_folder + f"/checkpoint_{self.beta_values[i]}.pt"
            checkpoint = torch.load(path, map_location=device)

            if "model_state_dict" not in checkpoint:
                raise KeyError(f"Checkpoint {path} does not contain 'model_state_dict'.")

            model_config = checkpoint.get("model_config", None)
            if model_config is None:
                raise ValueError(f"Checkpoint {path} does not contain 'model_config'.")

            config = {
                "lattice_shape": model_config.get("lattice_shape", (8, 8)),
                "masking": model_config.get("masking", "2x1"),
                "coupling": model_config.get("coupling", "cs"),
                "n_layers": model_config.get("n_layers", 24),
                "n_knots": model_config.get("n_knots", 9),  # Necessario se coupling="cs"
                "float_dtype": torch.float32,
                "nn": {
                    "hidden_channels": model_config.get("hidden_channels", [8, 8]),
                    "kernel_size": model_config.get("kernel_size", 3),
                    "dilation": model_config.get("dilation", 1),
                },
            }

            model_bundle = assemble_model_from_dict(config, device=device)
            
            layers = model_bundle["layers"]
            layers.load_state_dict(checkpoint["model_state_dict"])
            
            layers.eval()
            self.models.append(layers)

        return self.models

class TemperedMixture(Mixture):
    def __init__(self, *, changes: int = 100, dataloader : DataLoader, device : torch.device = 'cpu', L : int):
        super().__init__()
        self.samples = dataloader.samples
        self.beta_max = dataloader.beta_max
        self.beta_min = dataloader.beta_min
        self.betas = dataloader.beta_values
        self.device = device
        self.L = L

        if changes > len(self.samples[0]):
            raise ValueError("changes must be less than the number of samples")

        self.changes = changes
        self.new_samples = dataloader.samples[0].clone()

    def weights_setter(self, *, power: float = 1.5):
        weights = [k**power for k in range(self.beta_min, self.beta_max + 1)]
        weights = torch.tensor(weights) / np.sum(weights)
        dist = Categorical(weights) #building distribution;
        return weights, dist

    def sampler(self, * , weights : torch.Tensor | None = None, dist : Categorical | None = None, power : float = 1.5):
        self.random_indexes = torch.randint(0, len(self.samples[0]), (self.changes,), dtype=torch.long, device=self.device)

        if weights is None or dist is None:
            self.weights, self.dist = self.weights_setter(power = power)
        else:
            self.weights = weights
            self.dist = dist

        self.weights = self.weights.to(self.device) #!weights have to be on the same device as the samples; 
        counter = 0

        while counter < self.changes:
            sampled_index = int(self.dist.sample())
            idx = int(self.random_indexes[counter])
            self.new_samples[idx] = self.samples[sampled_index][idx]
            counter += 1

        return self.new_samples

    def mix_builder(self, models: list[TransformationSequence] | None = None, device: torch.device | str = "cpu"):
        if models is None: #sanity check; 
            if hasattr(self, "models"):
                models = self.models
            else:
                raise ValueError("Provide `models` or call `DataLoader.load_models()` first.")

        if not hasattr(self, "weights"): #sanity check; 
            self.weights, self.dist = self.weights_setter()
            self.weights = self.weights.to(self.device)

        self.mix = torch.zeros(len(self.new_samples), device=self.device)
        self.log_q = torch.zeros(len(self.new_samples), device = self.device)
        prior = neumc.nf.prior.MultivariateUniform(torch.zeros((2, self.L, self.L)), 2 * torch.pi * torch.ones(1), device=self.device)

        with torch.no_grad():
            self.mix = torch.zeros(len(self.new_samples), device= self.device)
            for index, model in enumerate(models):
                z, log_J = model.reverse(self.new_samples)
                weight = self.weights[index]  #!log_prob_z = prior.log_prob(z) has to be added;
                self.mix += weight * (prior.log_prob(z) - torch.exp(log_J))
                

            last_idx = len(self.betas)-1
            z, log_J = models[last_idx].reverse(self.new_samples)
            self.log_q =  log_J
            print(self.log_q)

        return self.mix, self.log_q #returns the entire mixture and the last density for MH; 
