import numpy as np
import torch
from layers.layer import Layer
from utils.autodiff import Value

class DropoutLayer(Layer):
    def __init__(self, rate, seed=None):
        # Initialize Dropout layer
        super().__init__()
        self.rate = rate
        self.seed = seed
        self.mask = None
        self._rng = np.random.RandomState(seed) if seed is not None else np.random
        
        if not 0 <= rate <= 1:
            raise ValueError(f"Rate must be between 0 and 1. Got {rate}")

    def forward(self, inputs, training=True):
        # Forward pass for dropout layer
        if not isinstance(inputs, Value):
            raise TypeError("Input must be a Value object for autodiff compatibility")
            
        if training:
            # Generate dropout mask using numpy but convert to torch tensor
            if isinstance(inputs.data, torch.Tensor):
                shape = inputs.data.shape
            else:
                shape = inputs.shape
                
            # Generate binary mask (0 or 1) based on dropout rate
            mask_np = self._rng.binomial(1, 1 - self.rate, size=shape)
            # Convert to torch tensor with same device and dtype as input
            mask_tensor = torch.tensor(mask_np, dtype=torch.float32, device=inputs.data.device)
            
            # Store mask for backward pass (as torch tensor)
            self.mask = mask_tensor
            
            # Create Value object for the scaled mask
            # Scale up by 1/(1-rate) to maintain same expected sum (inverted dropout)
            if self.rate < 1.0:
                scaled_mask = Value(mask_tensor / (1 - self.rate))
            else:
                scaled_mask = Value(mask_tensor)
            
            # Apply dropout using Value multiplication
            return inputs * scaled_mask
        else:
            # During inference, no dropout is applied
            return inputs

    # def backward(self, gradient):
        # Backward pass for dropout layer
        # Note: With Value class, backward pass is handled automatically
        # This method is kept for compatibility but may not be used
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