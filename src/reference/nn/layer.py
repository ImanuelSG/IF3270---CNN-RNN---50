from .module import Module
from engine.autodiff import Value


class Layer(Module):
    """
    Dimensions:
    - self.W (output_size, input_size)
    - self.b (output_size)
    """

    ALLOWED_ACTIVATIONS = {"linear", "relu", "sigmoid", "tanh", "softmax", 'elu', "leaky_relu"}

    def __init__(self, output_shape: int, activation: str, init_method = None,  input_shape: int = None):
        if activation not in self.ALLOWED_ACTIVATIONS:
            raise ValueError(f"Invalid activation function '{activation}'. Choose from {self.ALLOWED_ACTIVATIONS}")
        
        self.activation = activation
        self.W: Value = None
        self.b: Value = None 

        self.input_shape = input_shape
        self.output_shape = output_shape # a.k.a. banyak neuron pada layer
        self.init_method = init_method

    def parameters(self):
        params = []
        if self.W is not None:
            params.append(self.W)
        if self.b is not None:
            params.append(self.b)
        return params
    
    def __call__(self, X: Value):
        if self.W is None or self.b is None:
            raise ValueError("Layer must be initialized before calling it")

        Z = X @ self.W.T() + self.b
        return getattr(Z, self.activation)()