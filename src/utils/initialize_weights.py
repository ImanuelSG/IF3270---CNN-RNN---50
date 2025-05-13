import torch.nn.init as init
from utils.autodiff import Value
def initialize_weights(self, weight, initializer='random'):    
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
        else:
            raise ValueError(f"Unknown initialization method: {initializer}")

        # Wrap in your custom Value class
        return Value(weight, requires_grad=True)