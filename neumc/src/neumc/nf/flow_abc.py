# Status: done ✅
from abc import ABC, abstractmethod
from neumc.physics.u1 import torch_mod
from typing_extensions import override
from typing import Iterable
import torch
import numpy as np


class Transformation(
    torch.nn.Module, ABC
):  # Transformation is a subclass of torch.nn.Module and ABC (Abstract Base Class);
    def __init__(self):
        super().__init__()

    @abstractmethod
    @override
    def forward(
        self, z
    ) -> tuple[
        torch.Tensor, torch.Tensor
    ]:  # this method will transform the prior sample z into the target space x,
        # returning also the log-Jacobian determinant;
        """
        Transforms a batch of input configurations.

        Parameters
        ----------
        z: torch.Tensor
            configurations to transform

        Returns
        -------
        x: torch.Tensor
            Transformed configurations
        log_J: torch.Tensor
            the log of the Jacobian determinant of the transformation
        """
        ...  # equivalent to "pass";

    def reverse(
        self, x
    ) -> tuple[
        torch.Tensor, torch.Tensor
    ]:  # this method will transform the target sample x back to the prior space z;
        """
        Parameters
        ----------
        x

        Returns
        -------
        z: torch.Tensor
            Transformed configurations
        loog_J: torch.Tensor
            the log of the Jacobian determinant of the transformation
        """
        return NotImplemented  # if not overridden, returns NotImplemented;

    def sample(self, prior, batch_size: int):
        z = prior.sample_n(batch_size)
        log_prob_z = prior.log_prob(z)
        x, log_J = self.forward(z)
        #print("log_q:", log_prob_z - log_J)
        return x, log_prob_z - log_J


class TransformationSequence(
    Transformation
):  # this class implements a sequence of transformations of type Transformation;
    def __init__(self, layers: Iterable[Transformation]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    @override
    def forward(self, z) -> tuple[torch.Tensor, torch.Tensor]:
        log_J = torch.zeros(
            z.shape[0], device=z.device
        )  # z.device accesses the device where z is stored;

        for layer in self.layers:
            z, log_J_layer = layer.forward(z)
            log_J += log_J_layer  # log(a*b) = log(a) + log(b);

        return z, log_J

    @override
    def reverse(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        log_J = torch.zeros(x.shape[0], device=x.device)

        for layer in reversed(self.layers):
            x, log_J_layer = layer.reverse(x)
            log_J += log_J_layer

        return x, log_J
