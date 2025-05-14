from engine.autodiff import Value

class LossFunction:
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

    def forward(self, y_pred, y_true):
        """Compute the loss value."""
        raise NotImplementedError
    
class MSELoss(LossFunction):
    def forward(self, y_pred :Value, y_true : Value):
        """
        Input might be (batch_size,), (batch_size, 1), or (batch_size, n_label)
        """
        return ((y_true - y_pred) ** 2).mean()

class BinaryCrossEntropyLoss(LossFunction):
    def forward(self, y_pred, y_true):
        """
        Input might be (batch_size,), (batch_size, 1), or (batch_size, n_label)
        """
        eps = 1e-9
        y_pred = y_pred.clamp(eps, 1 - eps)
        
        return -(y_true * y_pred.log() + (1 - y_true) * (1 - y_pred).log()).mean()

class CategoricalCrossEntropyLoss(LossFunction):
    def forward(self, y_pred, y_true):
        """
        Input must be (batch_size, n_class)
        """
        eps = 1e-9
        y_pred = y_pred.clamp(eps, 1 - eps)

        log_preds = y_pred.log()
        return -(y_true * log_preds).sum() / y_pred.data.shape[0]