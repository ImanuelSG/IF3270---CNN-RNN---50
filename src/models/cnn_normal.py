import random
import numpy as np
from src.models.model import Model
from src.layers.layer import Layer
from src.utils.loss import LossFunction

class CNNModel(Model):
    def __init__(
        self,
        layers: list[Layer] = None,
        batch_size: int = 32,
        epochs: int = 10,
        lr: float = 0.01,
        loss_fn: LossFunction = None,
        verbose: bool = True,
        optimizer: str = "adam",
        seed: int = 42,
    ):
        self.layers = layers if layers is not None else []
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.loss_fn = loss_fn
        self.verbose = verbose
        self.optimizer = optimizer
        self.opt_state = {} if optimizer == "adam" else None
        self.seed = seed

    def fit(self, X, Y, X_val=None, y_val=None):

        for i, layer in enumerate(self.layers):
            layer._id = f"layer_{i}"  

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        if self.loss_fn is None:
            raise ValueError("loss_fn must be provided at model initialization")


        N = len(X)
        if self.layers[-1].activation_name == "softmax":
            num_classes = self.layers[-1].output_shape
            Y = np.eye(num_classes)[Y.astype(int)]
            if y_val is not None:
                y_val = np.eye(num_classes)[y_val.astype(int)]


        if self.optimizer == "adam":
            self.opt_state = {}
            for layer in self.layers:
                for key,param, grad in layer.get_parameters():
                    if key not in self.opt_state:
                        self.opt_state[key] = {
                'm': np.zeros_like(param, dtype=np.float64),
                'v': np.zeros_like(param, dtype=np.float64),
                't': 0
            }

        for epoch in range(self.epochs):
            indices = np.arange(N)
            np.random.shuffle(indices)
            total_loss = 0.0

            

            X = X[indices]
            Y = Y[indices]

            for i in range(0, N, self.batch_size):
                x_batch = X[i:i + self.batch_size]
                y_batch = Y[i:i + self.batch_size]

                y_pred = self.forward(x_batch)

                loss = self.loss_fn.forward(y_pred, y_batch)
                total_loss += loss

                grad_output = self.loss_fn.backward(y_pred, y_batch)

                for idx, layer in reversed(list(enumerate(self.layers))):
                    if idx == len(self.layers) - 1 and layer.activation_name == "softmax":
                        grad_output = layer.backward(grad_output, y_batch)
                    else:
                        grad_output = layer.backward(grad_output)

                    for key, param, grad in layer.get_parameters():
                        if grad is None:
                            continue

                        if self.optimizer == "adam":
                            if key not in self.opt_state:
                                self.opt_state[key] = {
                                    'm': np.zeros_like(param, dtype=np.float64),
                                    'v': np.zeros_like(param, dtype=np.float64),
                                    't': 0
                                }

                            state = self.opt_state[key]
                            state['t'] += 1
                            t = state['t']
                            beta1, beta2 = 0.9, 0.999
                            eps = 1e-8

                            state['m'] = beta1 * state['m'] + (1 - beta1) * grad
                            state['v'] = beta2 * state['v'] + (1 - beta2) * (grad ** 2)

                            m_hat = state['m'] / (1 - beta1 ** t)
                            v_hat = state['v'] / (1 - beta2 ** t)

                            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)


            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {total_loss:.4f}")

    def predict(self, X):
        return self.forward(X)

    def load_weights(self, weights_list):
        weights_iter = iter(weights_list)  # create an iterator over the list

        for layer in self.layers:
            if isinstance(layer, Layer) and hasattr(layer, 'load_parameters'):
                try:
                    weights = next(weights_iter)  # get weights
                    bias = next(weights_iter)     # get bias
                    layer.load_parameters(weights, bias)  # pass both
                except StopIteration:
                    raise ValueError("Not enough items in weights_list to load all layers.")