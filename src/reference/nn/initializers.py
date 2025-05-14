from typing import TYPE_CHECKING
import torch
import numpy as np
from engine.autodiff import Value

if TYPE_CHECKING:
    from ffnn import Layer 

class ZeroInit:
    def __init__(self):
        pass

    def initialize(self, layer):
        if self.seed is not None:
            np.random.seed(self.seed)
        layer.W = Value(torch.zeros(layer.output_shape, layer.input_shape), requires_grad=True)
        layer.b = Value(torch.zeros(1, layer.output_shape), requires_grad=True)

class XavierInit:
    def __init__(self, seed: int = None, uniform: bool = True):
        self.seed = seed
        self.uniform = uniform

    def initialize(self, layer):
        if self.seed is not None:
            np.random.seed(self.seed)
        if self.uniform:
            bound = np.sqrt(6.0 / (layer.input_shape + layer.output_shape))
            layer.W = Value(torch.tensor(np.random.uniform(-bound, bound, (layer.output_shape, layer.input_shape)), dtype=torch.float32), requires_grad=True)
            layer.b = Value(torch.tensor(np.random.uniform(-bound, bound, (1, layer.output_shape)), dtype=torch.float32), requires_grad=True)
        else:
            bound = np.sqrt(2.0 / (layer.input_shape + layer.output_shape))
            layer.W = Value(torch.tensor(np.random.uniform(-bound, bound, (layer.output_shape, layer.input_shape)), dtype=torch.float32), requires_grad=True)
            layer.b = Value(torch.tensor(np.random.uniform(-bound, bound, (1, layer.output_shape)), dtype=torch.float32), requires_grad=True)

class HeInit:
    def __init__(self, seed: int = None):
        self.seed = seed

    def initialize(self, layer):
        if self.seed is not None:
            np.random.seed(self.seed)
        std = np.sqrt(2.0 / layer.input_shape)
        layer.W = Value(torch.tensor(np.random.randn(layer.output_shape, layer.input_shape) * std, dtype=torch.float32), requires_grad=True)
        layer.b = Value(torch.zeros(1, layer.output_shape), requires_grad=True)

class RandomUniform:
    def __init__(self, seed: int = None, lower_bound: float = -0.1, upper_bound: float = 0.1):
        self.seed = seed
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def initialize(self, layer):
        if self.seed is not None:
            np.random.seed(self.seed)
        layer.W = Value(torch.tensor(np.random.uniform(self.lower_bound, self.upper_bound, (layer.output_shape, layer.input_shape)), dtype=torch.float32), requires_grad=True)
        layer.b = Value(torch.tensor(np.random.uniform(self.lower_bound, self.upper_bound, (1, layer.output_shape)), dtype=torch.float32), requires_grad=True)

class RandomNormal:
    def __init__(self, seed: int = None, mean: float = 0.0, variance: float = 1.0):
        self.seed = seed
        self.mean = mean
        self.variance = variance

    def initialize(self, layer):
        if self.seed is not None:
            np.random.seed(self.seed)
        std = np.sqrt(self.variance)
        layer.W = Value(torch.tensor(np.random.normal(self.mean, std, (layer.output_shape, layer.input_shape)), dtype=torch.float32), requires_grad=True)
        layer.b = Value(torch.tensor(np.random.normal(self.mean, std, (1, layer.output_shape)), dtype=torch.float32), requires_grad=True)