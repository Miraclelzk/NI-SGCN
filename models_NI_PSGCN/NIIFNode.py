from typing import Callable

import torch
import torch.nn as nn
import math

from spikingjelly.activation_based import neuron, functional, surrogate

class NIIFNode(neuron.IFNode):
    def __init__(self, mu: float = 0.0, sigma: float = 0.2, 
                 decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = surrogate.ATan(),
                 detach_reset: bool = True, step_mode='s', backend='torch', store_v_seq: bool = False):
        """
         Normal(mu, sigma^2)
        """
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.mu = mu
        self.sigma = sigma

    def extra_repr(self):
        return super().extra_repr() + f', sigma={self.sigma}'

    @property
    def supported_backends(self):
        return 'torch'
    
    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x
        self.v += -torch.normal(torch.ones_like(self.v) * self.mu, self.sigma)
        # print("training")
