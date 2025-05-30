import torch
from utils.autodiff import Value
from .model import Model
from layers.layer import Layer
from layers.lstm.unidirectionalLSTM import UnidirectionalLSTM
from layers.embedding import EmbeddingLayer

class LSTM(Model):
    def __init__(self, layers=None, batch_size=None):
        super().__init__(layers)
        self.layers = layers if layers is not None else []
        self.input_shape = None
        self.batch_size = batch_size

    def add(self, layer):
        if self.input_shape is not None:
            layer.input_shape = self.input_shape
        self.layers.append(layer)
        self.input_shape = layer.output_shape

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def batch_generator(self, X, batch_size):
        n = X.shape[0]
        for i in range(0, n, batch_size):
            yield X[i:i + batch_size]
            
    def predict(self, x):
        self.eval()
        outputs = []
        print(f"Predicting with batch size: {self.batch_size}, input shape: {x.shape}")
        for batch_X in self.batch_generator(x, self.batch_size):
            if len(self.layers) > 0 and isinstance(self.layers[0], EmbeddingLayer):
                # print("Using EmbeddingLayer, converting indices to long tensor")
                batch_X_value = torch.tensor(batch_X, dtype=torch.long)
            else:
                # print("Using non-EmbeddingLayer, converting to float tensor")
                batch_tensor = torch.tensor(batch_X, dtype=torch.float32)
                batch_X_value = Value(batch_tensor, requires_grad=False)
            out = self.forward(batch_X_value)
            outputs.append(out)
        return Value.cat(outputs, dim=0)
    
    def __call__(self, x):
        if self.batch_size is not None:
            return self.predict(x)
        else:
            return self.forward(x)

    def load_weights(self, weights_list):
        for layer, weights in zip(self.layers, weights_list):
            if hasattr(layer, 'load_weights'):
                layer.load_weights(weights)