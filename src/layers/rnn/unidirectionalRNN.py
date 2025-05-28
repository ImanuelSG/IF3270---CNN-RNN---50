import torch
from utils.autodiff import Value
from utils.initialize_weights import initialize_weights
from layers.layer import Layer

class UnidirectionalRNN(Layer):
    def __init__(self, units, input_dim=None, batch_size=None, activation="tanh",
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                 bias_initializer='zeros', return_sequences=False):
        super().__init__()
        self.units = units
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.activation = activation
        self.return_sequences = return_sequences

        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer

        # Placeholders for weights
        self.Wxh = None
        self.Whh = None
        self.bxh = None

        # Initial hidden state
        self.h0 = None

        if input_dim is not None:
            self.build(input_dim)

        if batch_size is not None:
            self.init_hidden_state(batch_size)

    def build(self, input_dim):
        """Initialize weights when input_dim known"""
        self.Wxh = initialize_weights(torch.empty(self.units, input_dim), self.kernel_initializer)
        self.Whh = initialize_weights(torch.empty(self.units, self.units), self.recurrent_initializer)
        self.bxh = initialize_weights(torch.zeros(self.units, 1), self.bias_initializer)
        # print(f"UnidirectionalRNN built with input_dim={input_dim}, units={self.units}")
        # print(f"Weights shapes: Wxh={self.Wxh.data.shape}, Whh={self.Whh.data.shape}, bxh={self.bxh.data.shape}")

    def init_hidden_state(self, batch_size):
        """Initialize hidden state tensor"""
        self.batch_size = batch_size
        self.h0 = Value(torch.zeros(batch_size, self.units))
        # print(f"Initial hidden state h0 initialized with shape: {self.h0.data.shape}")

    def forward(self, x: Value):
        batch_size, seq_len, feature_size = x.data.shape
        # print(f"Forward pass with input shape: {x.data.shape}, batch_size: {batch_size}, seq_len: {seq_len}, feature_size: {feature_size}")

        if self.Wxh is None:
            self.build(feature_size)

        if self.h0 is None or self.batch_size != batch_size:
            self.init_hidden_state(batch_size)

        h = self.h0
        self.outputs = []

        for t in range(seq_len):
            x_t = Value(x.data[:, t])  # shape: (batch_size, feature_size)
            # print(f"Processing time step {t+1}/{seq_len}, input shape: {x_t.data.shape}") if t == 0 else None
            # Compute next hidden state
            h = (self.Wxh @ x_t.T() + self.Whh @ h.T() + self.bxh).tanh().T()
            # print(f"Hidden state at time step {t+1}: {h.data.shape}") if t == 0 else None
            self.outputs.append(h)

        if self.return_sequences:
            return Value.cat(self.outputs, 1)  # (batch_size, seq_len, units)
        else:
            return self.outputs[-1]  # (batch_size, units)
    
    def load_weights(self, weights):
        Wxh, Whh, bxh = weights
        self.Wxh = Value(torch.tensor(Wxh.T, dtype=torch.float32), requires_grad=True)
        self.Whh = Value(torch.tensor(Whh.T, dtype=torch.float32), requires_grad=True)
        self.bxh = Value(torch.tensor(bxh.reshape(-1, 1), dtype=torch.float32), requires_grad=True)
        return self

    def __call__(self, x):
        return self.forward(x)
