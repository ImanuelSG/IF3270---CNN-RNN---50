
class CNNModel:
    def __init__(self):
        self.layers = [
            Conv2D(filters=16, kernel_size=3),
            Pooling2D(pool_size=2),
            Conv2D(filters=32, kernel_size=3),
            Pooling2D(pool_size=2),
            Flatten(),
            Dense(input_dim=512, output_dim=128, activation='relu'),
            Dense(input_dim=128, output_dim=10, activation='softmax')
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
