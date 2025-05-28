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

    def load_weights(self, weights_list):
        raw_iter = iter(weights_list)

        for layer in self.layers:
            if hasattr(layer, 'load_weights'):
                if hasattr(layer, 'bias'):
                    layer.load_weights(next(raw_iter), next(raw_iter))
                else:
                    layer.load_weights(next(raw_iter), None)