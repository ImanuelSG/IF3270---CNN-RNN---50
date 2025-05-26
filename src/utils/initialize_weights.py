import torch
import torch.nn.init as init
from utils.autodiff import Value

def initialize_weights(weight, initializer='normal', seed=None, wrap=True):   
    if seed is not None:
        torch.manual_seed(seed)

    if initializer == 'uniform':
        init.uniform_(weight, a=-0.1, b=0.1)
    elif initializer == 'normal':
        init.normal_(weight, mean=0.0, std=0.01)
    elif initializer == 'glorot_uniform':
        init.xavier_uniform_(weight)  # Glorot/Xavier uniform
    elif initializer == 'glorot_normal':
        init.xavier_normal_(weight)   # Glorot/Xavier normal
    elif initializer == 'he_normal':
        init.kaiming_normal_(weight, nonlinearity='relu')  # He normal
    elif initializer == 'he_uniform':
        init.kaiming_uniform_(weight, nonlinearity='relu')  # He uniform
    elif initializer == 'orthogonal':
        init.orthogonal_(weight)
    elif initializer == 'zeros':
        init.zeros_(weight)
    elif initializer == 'ones':
        init.ones_(weight)
    elif initializer == 'custom':
        weigth = weight
    else:
        raise ValueError(f"Unknown initialization method: {initializer}")

    return Value(weight, requires_grad=True) if wrap else weight
