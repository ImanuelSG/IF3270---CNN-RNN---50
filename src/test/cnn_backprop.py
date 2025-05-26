import torch
from copy import deepcopy
from src.utils.autodiff import Value
from src.layers.cnn.conv import Conv2D

def loss_fn(output: Value):
    # Simple sum of output elements to keep scalar
    return output.sum()

def numerical_gradient(layer, x, param_tensor, param_name, epsilon=1e-5):
    numerical_grads = torch.zeros_like(param_tensor)

    for idx in torch.ndindex(param_tensor.shape):
        original_value = param_tensor[idx].item()

        # Perturb +epsilon
        param_tensor[idx] = original_value + epsilon
        layer_cp = deepcopy(layer)  # Make a copy so we don’t reuse .grad values
        x_cp = deepcopy(x)
        out_pos = layer_cp.forward(x_cp)
        loss_pos = loss_fn(out_pos).data.item()

        # Perturb -epsilon
        param_tensor[idx] = original_value - epsilon
        layer_cp = deepcopy(layer)
        x_cp = deepcopy(x)
        out_neg = layer_cp.forward(x_cp)
        loss_neg = loss_fn(out_neg).data.item()

        # Restore original value
        param_tensor[idx] = original_value

        # Central difference
        grad = (loss_pos - loss_neg) / (2 * epsilon)
        numerical_grads[idx] = grad

    return numerical_grads