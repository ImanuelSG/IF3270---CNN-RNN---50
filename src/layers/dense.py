import torch
from layers.layer import Layer
from utils.autodiff import Value
from utils.initialize_weights import initialize_weights

class DenseLayer(Layer):
    ALLOWED_ACTIVATIONS = {"linear", "relu", "sigmoid", "tanh", "softmax", 'elu', "leaky_relu"}

    def __init__(self, output_shape, input_shape, activation='linear', init_method='zeros', ):
        if activation not in self.ALLOWED_ACTIVATIONS:
            raise ValueError(f"Invalid activation function '{activation}'. Choose from {self.ALLOWED_ACTIVATIONS}")

        super().__init__()
        self.activation = activation
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.init_method = init_method

        self.weights: Value = initialize_weights(torch.empty(self.output_shape, input_shape), self.init_method)
        self.bias: Value = initialize_weights(torch.zeros(self.output_shape, 1), "zeros")


    def load_weights(self, weights, bias):
        """
        Load weights for the dense layer.
        weights: Tuple of tensors (weights, bias)
        """
        self.weights = Value(torch.tensor(weights.T, dtype=torch.float32), requires_grad=True)
        self.bias = Value(torch.tensor(bias, dtype=torch.float32), requires_grad=True)
        return self
        

    def forward(self, X: Value):
        Z = X @ self.weights.T() + self.bias 
        activated = getattr(Z, self.activation)()
        return activated
    
    def __call__(self, X: Value):
        return self.forward(X)
