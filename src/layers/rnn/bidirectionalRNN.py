import torch 
import utils.initialize_weights as init
from utils.autodiff import Value
from layers.layer import Layer
class BidirectionalRNN(Layer):
    def __init__(self, units, activation="tanh", kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', return_sequences=False):
        self.activation = activation
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.return_sequences = return_sequences

        # Forward RNN weights
        self.Whh_f = init.initialize_weights(torch.empty(units, units), recurrent_initializer)
        self.bxh_f = init.initialize_weights(torch.zeros(units, 1), bias_initializer)

        # Backward RNN weights
        self.Whh_b = init.initialize_weights(torch.empty(units, units), recurrent_initializer)
        self.bxh_b = init.initialize_weights(torch.zeros(units, 1), bias_initializer)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, feature_size = x.shape

        # Input → hidden weights (shared across time but separate for f/b)
        self.Wxh_f = init.initialize_weights(torch.empty(self.units, feature_size), self.kernel_initializer)
        self.Wxh_b = init.initialize_weights(torch.empty(self.units, feature_size), self.kernel_initializer)

        # Initial hidden states
        h_f = Value(torch.zeros(batch_size, self.units))
        h_b = Value(torch.zeros(batch_size, self.units))

        self.outputs_f = []
        self.outputs_b = []

        # Forward direction
        for t in range(seq_len):
            h_f = (self.Wxh_f @ Value(x[:, t]).T() + self.Whh_f @ h_f.T() + self.bxh_f).tanh().T()
            self.outputs_f.append(h_f)

        # Backward direction
        for t in reversed(range(seq_len)):
            h_b = (self.Wxh_b @ Value(x[:, t]).T() + self.Whh_b @ h_b.T() + self.bxh_b).tanh().T()
            self.outputs_b.insert(0, h_b)  # To align with forward time

        # Concatenate outputs
        self.outputs = [Value(torch.cat([f.data, b.data], dim=1)) for f, b in zip(self.outputs_f, self.outputs_b)]

        if self.return_sequences:
            return Value.stack(self.outputs, dim=1)  # (batch_size, seq_len, 2 * units)
        else:
            return self.outputs[-1]  # (batch_size, 2 * units)

    def load_weights(self, weights):
        """
        Load weights for the bidirectional RNN.
        weights: Tuple of tensors (Wxh_f, Whh_f, bxh_f, Wxh_b, Whh_b, bxh_b)
        """
        assert len(weights) == 6, "Weights must be a tuple of 6 tensors"
        self.Wxh_f.data, self.Whh_f.data, self.bxh_f.data, self.Wxh_b.data, self.Whh_b.data, self.bxh_b.data = weights

    def __call__(self, x):
        return self.forward(x)
