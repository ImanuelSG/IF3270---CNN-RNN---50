from model import Model
class CNN(Model):
    def __init__(self, layers=None):
        super().__init__(layers)
        self.layers = layers if layers is not None else []
        self.input_shape = None

    def add(self, layer):
        if self.input_shape is not None:
            layer.input_shape = self.input_shape
        self.layers.append(layer)
        self.input_shape = layer.output_shape

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)
    
    def load_weights(self, weights_list):
        """
        weights_list: List of tuples matching each layer's weights
        Example: [(W1, b1), (W2, b2), ...]
        """
        assert len(weights_list) == len(self.layers)
        for layer, weights in zip(self.layers, weights_list):
            if hasattr(layer, 'load_weights'):
                layer.load_weights(*weights)