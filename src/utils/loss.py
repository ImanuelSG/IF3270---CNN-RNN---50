from src.utils.autodiff import Value
import torch
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
    
class SparseCategoricalCrossEntropyLoss(LossFunction):
    def forward(self, y_pred, y_true):
        """
        y_pred: (batch_size, num_classes), probabilities
        y_true: (batch_size,), integer labels
        """
        eps = 1e-9
        y_pred = y_pred.clamp(eps, 1 - eps)
        log_preds = y_pred.log()

        batch_size = y_pred.data.shape[0]
        
        batch_indices = Value(torch.arange(batch_size))
        
        true_class_log_probs = log_preds.data[batch_indices.data.long(), y_true.data.long()]
        true_class_log_probs = Value(true_class_log_probs, requires_grad=y_pred.requires_grad, _children=(log_preds,), _op="gather")
        
        def _backward():
            if y_pred.requires_grad and true_class_log_probs.requires_grad:
                
                grad = torch.zeros_like(log_preds.data)
               
                grad[batch_indices.data.long(), y_true.data.long()] = true_class_log_probs.grad
                log_preds.grad += grad
        
        true_class_log_probs._backward = _backward

        return -(true_class_log_probs).mean()