import torch
from utils.autodiff import Value
from utils.initialize_weights import initialize_weights
from layers.layer import Layer

class UnidirectionalRNN(Layer):
    def __init__(
        self,
        units,
        input_dim=None,
        batch_size=None,
        activation="tanh",
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        return_sequences=False
    ):
        super().__init__()
        self.units = units
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.return_sequences = return_sequences

        self.Wxh = None  # Input-to-hidden weights
        self.Whh = None  # Hidden-to-hidden weights
        self.bxh = None  # Bias

        self.h0 = None  # Initial hidden state

        if input_dim is not None:
            self.build(input_dim)
        if batch_size is not None:
            self.init_hidden_state(batch_size)

    def build(self, input_dim):
        """Initialize input and recurrent weights."""
        self.Wxh = initialize_weights(torch.empty(self.units, input_dim), self.kernel_initializer)
        self.Whh = initialize_weights(torch.empty(self.units, self.units), self.recurrent_initializer)
        self.bxh = initialize_weights(torch.zeros(self.units, 1), self.bias_initializer)

    def init_hidden_state(self, batch_size):
        """Initialize hidden state for the batch."""
        self.batch_size = batch_size
        self.h0 = Value(torch.zeros(batch_size, self.units))

    def forward(self, x: Value):
        """
        Forward pass of the RNN.
        Args:
            x (Value): Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Value: Output tensor of shape (batch_size, seq_len, units) if return_sequences=True,
                   else (batch_size, units)
        """
        batch_size, seq_len, feature_size = x.data.shape

        if self.Wxh is None:
            self.build(feature_size)
        if self.h0 is None or self.batch_size != batch_size:
            self.init_hidden_state(batch_size)

        h = self.h0
        self.outputs = []

        for t in range(seq_len):
            x_t = Value(x.data[:, t])  # Shape: (batch_size, input_dim)
            h = (self.Wxh @ x_t.T() + self.Whh @ h.T() + self.bxh).tanh().T()  # Shape: (batch_size, units)
            self.outputs.append(h)

        if self.return_sequences:
            return Value.cat([v.unsqueeze(1) for v in self.outputs], dim=1)  # Shape: (batch_size, seq_len, units)
        else:
            return self.outputs[-1]  # Shape: (batch_size, units)

    def load_weights(self, weights):
        """
        Load model weights.
        Args:
            weights (tuple): Tuple of (Wxh, Whh, bxh)
        """
        Wxh, Whh, bxh = weights
        self.Wxh = Value(torch.tensor(Wxh.T, dtype=torch.float32), requires_grad=True)
        self.Whh = Value(torch.tensor(Whh.T, dtype=torch.float32), requires_grad=True)
        self.bxh = Value(torch.tensor(bxh.reshape(-1, 1), dtype=torch.float32), requires_grad=True)
        return self

    def __call__(self, x):
        return self.forward(x)
