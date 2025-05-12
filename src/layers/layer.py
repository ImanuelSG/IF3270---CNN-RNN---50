class Layer:
    def forward(self, x):
        raise NotImplementedError("Forward pass not implemented.")

    def __call__(self, x):
        return self.forward(x)

    def load_weights(self, *args):
        raise NotImplementedError("Weight loading not implemented.")
