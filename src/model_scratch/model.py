class Model:
    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) 
        return x

    def __call__(self, x):
        return self.forward(x)
    
    def train(self):
        """Set the model and all its layers to training mode"""
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()
        return self
    
    def eval(self):
        """Set the model and all its layers to evaluation mode"""
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()
        return self

    def load_weights(self, weights_list):
        """
        weights_list: List of tuples matching each layer's weights
        Example: [(W1, b1), (W2, b2), ...]
        """
        assert len(weights_list) == len(self.layers)
        for layer, weights in zip(self.layers, weights_list):
            if hasattr(layer, 'load_weights'):
                layer.load_weights(*weights)
