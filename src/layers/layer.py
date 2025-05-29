from utils.autodiff import Value
class Layer:
    def forward(self):
        self._id = None 
        raise NotImplementedError("Forward pass not implemented.")

    def get_parameters(self):
        params = []

        if hasattr(self, "weights") and self.weights is not None:
            params.append((f"{self._id}_weights", self.weights, self.grad_weights))

        if hasattr(self, "bias") and self.bias is not None:
            params.append((f"{self._id}_bias", self.bias, self.grad_bias))

        return params
    
    def backward(self, x):
        raise NotImplementedError("Backward pass not implemented.")

    def __call__(self, x):
        return self.forward(x)
