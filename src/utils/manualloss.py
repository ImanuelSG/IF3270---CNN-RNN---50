import numpy as np

class LossFunction:
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self, y_pred, y_true):
        raise NotImplementedError


class MSELoss(LossFunction):
    def forward(self, y_pred, y_true):
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_pred, y_true):
        return -2 * (y_true - y_pred) / y_true.shape[0]


class BinaryCrossEntropyLoss(LossFunction):
    def forward(self, y_pred, y_true):
        eps = 1e-9
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, y_pred, y_true):
        eps = 1e-9
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return (-(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))) / y_true.shape[0]


class CategoricalCrossEntropyLoss(LossFunction):
    def forward(self, y_pred, y_true):
        eps = 1e-9
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]

    def backward(self, y_pred, y_true):
        eps = 1e-9
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -y_true / y_pred / y_pred.shape[0]


class SparseCategoricalCrossEntropyLoss(LossFunction):
    def forward(self, y_pred, y_true):
        eps = 1e-9
        y_pred = np.clip(y_pred, eps, 1 - eps)
        batch_size = y_pred.shape[0]
        # Use advanced indexing for the correct class probabilities
        correct_class_probs = y_pred[np.arange(batch_size), y_true]
        return -np.mean(np.log(correct_class_probs))

    def backward(self, y_pred, y_true):
        eps = 1e-9
        y_pred = np.clip(y_pred, eps, 1 - eps)
        grad = np.zeros_like(y_pred)
        batch_size = y_pred.shape[0]
        grad[np.arange(batch_size), y_true] = -1 / y_pred[np.arange(batch_size), y_true]
        return grad / batch_size
