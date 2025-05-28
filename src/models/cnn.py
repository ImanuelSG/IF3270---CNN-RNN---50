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
        loss_fn: LossFunction = None,
        verbose: bool = True,
        optimizer: str = "adam",
        seed: int = None,
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

        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        if self.loss_fn is None:
            raise ValueError("loss_fn must be provided at model initialization")

        N = X.data.shape[0]

        if self.layers[-1].activation == "softmax":
            num_classes = self.layers[-1].output_shape
            Y = torch.nn.functional.one_hot(Y.data.to(torch.long), num_classes=num_classes)
            if y_val is not None:
                y_val = torch.nn.functional.one_hot(y_val.data.to(torch.long), num_classes=num_classes)

        if self.optimizer == "adam":
            for layer in self.layers:
                for param in layer.get_parameters():
                    if param not in self.opt_state:
                        
                        self.opt_state[param] = {
                            'm': torch.zeros_like(param.data),
                            'v': torch.zeros_like(param.data),
                            't': 0
                        }

        for epoch in range(self.epochs):
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
                                beta1 = 0.9
                                beta2 = 0.999
                                eps = 1e-8

                                state['m'] = beta1 * state['m'] + (1 - beta1) * param.grad
                                state['v'] = beta2 * state['v'] + (1 - beta2) * (param.grad ** 2)

                                m_hat = state['m'] / (1 - beta1 ** t)
                                v_hat = state['v'] / (1 - beta2 ** t)

                                param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + eps)

                            # Clear gradients
                            param.grad = torch.zeros_like(param.grad)

            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {total_loss:.4f}")

    def predict(self, X):

       
        x_val = Value(X, requires_grad=False)
        y_pred = self.forward(x_val)
        return y_pred
