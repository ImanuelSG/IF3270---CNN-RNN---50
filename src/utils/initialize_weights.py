import torch
import torch.nn.init as init
import numpy as np

def initialize_weights(weight, initializer='normal', seed=None, wrap=False):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    is_numpy = isinstance(weight, np.ndarray)
    if is_numpy:
        weight = torch.tensor(weight, dtype=torch.float32)

    # Apply initialization
    if initializer == 'uniform':
        init.uniform_(weight, a=-0.1, b=0.1)
    elif initializer == 'normal':
        init.normal_(weight, mean=0.0, std=0.01)
    elif initializer == 'glorot_uniform':
        init.xavier_uniform_(weight)
    elif initializer == 'glorot_normal':
        init.xavier_normal_(weight)
    elif initializer == 'he_normal':
        init.kaiming_normal_(weight, nonlinearity='relu')
    elif initializer == 'he_uniform':
        init.kaiming_uniform_(weight, nonlinearity='relu')
    elif initializer == 'orthogonal':
        init.orthogonal_(weight)
    elif initializer == 'zeros':
        init.zeros_(weight)
    elif initializer == 'ones':
        init.ones_(weight)
    elif initializer == 'custom':
        pass  # Do nothing
    else:
        raise ValueError(f"Unknown initialization method: {initializer}")

    

    if wrap:
        from src.utils.autodiff import Value
        return Value(weight, requires_grad=True)
    else:
        weight_np = weight.detach().cpu().numpy()
        return weight_np
