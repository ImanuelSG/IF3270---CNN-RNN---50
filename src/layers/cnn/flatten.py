from src.layers.layer import Layer
from src.utils.autodiff import Value
import torch


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, x):
        # Save the original shape for potential backward reshape if needed
        self.input_shape = x.data.shape  # (B, C, H, W) or any shape

        B = self.input_shape[0]
        flattened_size = 1
        for dim in self.input_shape[1:]:
            flattened_size *= dim

        flattened_data = x.data.reshape(B, flattened_size)

        out = Value(flattened_data, requires_grad=x.requires_grad, _children=(x,), _op='flatten')

        def _backward():
            if x.requires_grad:
                # Grad of flattened shape comes from out.grad (B, flattened_size)
                grad_input = out.grad.reshape(self.input_shape)
                x.grad += grad_input

        out._backward = _backward
        return out
    
    def get_parameters(self):
        return []
