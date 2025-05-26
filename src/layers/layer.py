from src.utils.autodiff import Value
class Layer:
    def forward(self, x : Value):
        raise NotImplementedError("Forward pass not implemented.")

    def get_parameters(self):
        """
        Returns the parameters of the layer.
        This method should be overridden by subclasses to return the layer's parameters.
        """
        raise NotImplementedError("get_parameters not implemented.")
    
    def backward(self, x):
        raise NotImplementedError("Backward pass not implemented.")

    def __call__(self, x):
        return self.forward(x)

    def load_weights(self, *args):
        raise NotImplementedError("Weight loading not implemented.")
