import random
import torch
from src.utils.autodiff import Value
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
        loss_fn : LossFunction =None,
        verbose: bool = True
    ):
        self.layers = layers if layers is not None else []

        # Training configs
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.loss_fn = loss_fn
        self.verbose = verbose

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x: Value) -> Value:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def load_weights(self, weights_list):
        assert len(weights_list) == len(self.layers)
        for layer, weights in zip(self.layers, weights_list):
            if hasattr(layer, 'load_weights'):
                layer.load_weights(*weights)

    def fit(self, X, Y, X_val = None, y_val = None):
        """
        X: torch.Tensor, shape (N, C, H, W)
        Y: torch.Tensor, shape (N, ...)
        """
        if self.loss_fn is None:
            raise ValueError("loss_fn must be provided at model initialization")

        N = X.data.shape[0]

        if self.layers[-1].activation == "softmax":
            num_classes = self.layers[-1].output_shape
            Y = torch.nn.functional.one_hot(Y.data.to(torch.long), num_classes=num_classes)
            if y_val is not None:
                y_val = torch.nn.functional.one_hot(y_val.data.to(torch.long), num_classes=num_classes)

        for epoch in range(1):
            indices = list(range(N))
            random.shuffle(indices)
            total_loss = 0
            X.data = X.data[indices]
            Y.data = Y.data[indices]

            for i in range(0, N, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                x_batch = X.data[batch_indices]
                y_batch = Y.data[batch_indices]

                x_val = Value(x_batch, requires_grad=True)
                
                
                y_pred = self.forward(x_val)
                loss = self.loss_fn(y_pred, y_batch)

                loss.backward()
                total_loss += loss.data.item()

                for layer in self.layers:
                    for param in layer.get_parameters():
                        if param.requires_grad and param.grad is not None:
                            if self.optimizer == "sgd":
                                param.data -= self.lr * param.grad
                            elif self.optimizer == "adam":
                                state = self.opt_state[param]
                                state['t'] += 1
                                t = state['t']

                                # Update biased first moment estimate
                                state['m'] = 0.9 * state['m'] + 0.1 * param.grad
                                # Update biased second raw moment estimate
                                state['v'] = 0.999 * state['v'] + 0.001 * (param.grad ** 2)

                                # Compute bias-corrected first and second moment
                                m_hat = state['m'] / (1 - 0.9 ** t)
                                v_hat = state['v'] / (1 - 0.999 ** t)

                                # Update parameters
                                param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + 1e-8)

                            # Clear gradients
                            param.grad = torch.zeros_like(param.grad)
                            

            avg_loss = total_loss
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.4f}")
                
    def predict(self, X):
        """
        X: torch.Tensor, shape (N, C, H, W)
        """
        x_val = Value(X, requires_grad=False)
        y_pred = self.forward(x_val)
        return y_pred.data
    
    # def load_weights