from src.layers.layer import Layer
import numpy as np

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.X = None  

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.input_shape = X.shape
        self.X = X
        B = X.shape[0]

        X = np.transpose(X, (0, 2, 3, 1))
        return X.reshape(B, -1)

    def backward(self, dA: np.ndarray) -> np.ndarray:
        return dA.reshape(self.input_shape)

