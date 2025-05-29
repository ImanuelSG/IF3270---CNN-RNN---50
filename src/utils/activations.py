import numpy as np


def linear(Z): return Z
def linear_derivative(Z): return np.ones_like(Z)


def relu(Z): return np.maximum(0, Z)
def relu_derivative(Z): return (Z > 0).astype(np.float64)


def sigmoid(Z): return 1 / (1 + np.exp(-Z))
def sigmoid_derivative(Z):
    sig = sigmoid(Z)
    return sig * (1 - sig)


def tanh(Z): return np.tanh(Z)
def tanh_derivative(Z): return 1 - np.tanh(Z) ** 2


def softmax(Z):
    e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return e_Z / np.sum(e_Z, axis=1, keepdims=True)


def softmax_derivative(Z):
    raise NotImplementedError("Use softmax + cross-entropy simplification in loss, not derivative.")


def elu(Z, alpha=1.0): return np.where(Z > 0, Z, alpha * (np.exp(Z) - 1))
def elu_derivative(Z, alpha=1.0): return np.where(Z > 0, 1, alpha * np.exp(Z))


def leaky_relu(Z, alpha=0.01): return np.where(Z > 0, Z, alpha * Z)
def leaky_relu_derivative(Z, alpha=0.01): return np.where(Z > 0, 1, alpha)


ACTIVATIONS = {
    "linear": (linear, linear_derivative),
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "softmax": (softmax, softmax_derivative),
    "elu": (elu, elu_derivative),
    "leaky_relu": (leaky_relu, leaky_relu_derivative),
}
