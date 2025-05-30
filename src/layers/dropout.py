import numpy as np
import torch
from layers.layer import Layer
from utils.autodiff import Value

class DropoutLayer(Layer):
    def __init__(self, rate, seed=None):
        super().__init__()
        self.rate = rate
        self.seed = seed
        self.mask = None
        self._rng = np.random.RandomState(seed) if seed is not None else np.random
        
        if not 0 <= rate <= 1:
            raise ValueError(f"Rate must be between 0 and 1. Got {rate}")

    def forward(self, inputs):
        if not isinstance(inputs, Value):
            raise TypeError("Input must be a Value object for autodiff compatibility")
            
        if self.training:
            if isinstance(inputs.data, torch.Tensor):
                shape = inputs.data.shape
            else:
                shape = inputs.shape
                
            mask_np = self._rng.binomial(1, 1 - self.rate, size=shape)
            mask_tensor = torch.tensor(mask_np, dtype=torch.float32, device=inputs.data.device)
            
            self.mask = mask_tensor
            if self.rate < 1.0:
                scaled_mask = Value(mask_tensor / (1 - self.rate))
                return inputs * scaled_mask
            else:
                scaled_mask = Value(mask_tensor)
                return inputs * scaled_mask
        else:
            return inputs

    # def backward(self, gradient):
        # Backward pass dropout layer
        # if self.mask is not None and hasattr(gradient, 'data'):
        #     if self.rate < 1.0:
        #         scaled_mask = self.mask / (1 - self.rate)
        #     else:
        #         scaled_mask = self.mask
        #     return gradient * Value(scaled_mask)
        # return gradient

    def load_weights(self, *args):
        """No weights to load for Dropout layer"""
        pass