import numpy as np
from src.layers.layer import Layer
from src.utils.initialize_weights import initialize_weights
from src.utils.activations import ACTIVATIONS


class DenseLayer(Layer):
    def __init__(self, output_shape, activation='linear', init_method=None, input_shape=None):
        if activation not in ACTIVATIONS:
            raise ValueError(f"Invalid activation '{activation}'. Choose from {list(ACTIVATIONS.keys())}")

        super().__init__()
        self.activation_name = activation
        self.activation_fn, self.activation_derivative_fn = ACTIVATIONS[activation]

        self.weights: np.ndarray = None
        self.bias: np.ndarray = None

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.init_method = init_method

        self.grad_weights = None
        self.grad_bias = None

        self.X = None
        self.Z = None
        self.A = None

    def _initialize_parameters(self, input_dim):
        if self.init_method is None:
            raise ValueError("Initialization method must be provided")

        weight_shape = (self.output_shape, input_dim)
        self.weights = initialize_weights(np.empty(weight_shape, dtype=np.float64), self.init_method)
        self.bias = np.zeros((self.output_shape,), dtype=np.float64)

    def forward(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            self._initialize_parameters(X.shape[1])
            self.input_shape = X.shape[1]

        self.X = X
        self.Z = X @ self.weights.T + self.bias
        self.A = self.activation_fn(self.Z)
        return self.A

    def backward(self, dA: np.ndarray, y_true: np.ndarray = None) -> np.ndarray:
        m = self.X.shape[0]

        if self.activation_name == "softmax":
            if y_true is None:
                raise ValueError("y_true must be provided for softmax with cross-entropy")
            dZ = self.A - y_true
        else:
            dZ = dA * self.activation_derivative_fn(self.Z)

        self.grad_weights = dZ.T @ self.X / m
        self.grad_bias = np.sum(dZ, axis=0) / m
        dX = dZ @ self.weights

        return dX

    def load_parameters(self, keras_weights: np.ndarray, keras_biases: np.ndarray = None):
        if keras_weights.ndim != 2:
            raise ValueError("Weights must be a 2D array")
        input_dim, output_dim = keras_weights.shape
        if output_dim != self.output_shape:
            raise ValueError(f"Expected output dim {self.output_shape}, got {output_dim}")

        self.weights = keras_weights.T.astype(np.float64)

        self.bias = keras_biases.astype(np.float64) if keras_biases is not None else np.zeros(self.output_shape, dtype=np.float64)
