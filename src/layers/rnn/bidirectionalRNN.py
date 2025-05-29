import torch
import utils.initialize_weights as init
from utils.autodiff import Value
from layers.layer import Layer

class BidirectionalRNN(Layer):
    def __init__(
        self,
        units,
        input_dim=None,
        batch_size=32,
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

        self.h_f = self.h_b = None  # Hidden states

        # Initialize weights if input_dim is given
        if input_dim is not None:
            self.build(input_dim)

        # Recurrent weights (independent of input_dim)
        self.Whh_f = init.initialize_weights(torch.empty(units, units), recurrent_initializer)
        self.bxh_f = init.initialize_weights(torch.zeros(units, 1), bias_initializer)

        self.Whh_b = init.initialize_weights(torch.empty(units, units), recurrent_initializer)
        self.bxh_b = init.initialize_weights(torch.zeros(units, 1), bias_initializer)

    def build(self, input_dim):
        """Initialize input-to-hidden weights."""
        self.Wxh_f = init.initialize_weights(torch.empty(self.units, input_dim), self.kernel_initializer)
        self.Wxh_b = init.initialize_weights(torch.empty(self.units, input_dim), self.kernel_initializer)

    def init_hidden_state(self, batch_size):
        """Initialize hidden states."""
        self.batch_size = batch_size
        self.h_f = Value(torch.zeros(batch_size, self.units))
        self.h_b = Value(torch.zeros(batch_size, self.units))

    def forward(self, x: Value):
        """Perform forward pass for the bidirectional RNN."""
        batch_size, seq_len, feature_size = x.data.shape

        if not hasattr(self, "Wxh_f") or not hasattr(self, "Wxh_b"):
            self.build(feature_size)

        if self.h_f is None or self.h_b is None or self.batch_size != batch_size:
            self.init_hidden_state(batch_size)

        outputs_f, outputs_b = [], []
        h_f, h_b = self.h_f, self.h_b

        # Forward pass
        for t in range(seq_len):
            x_t = Value(x.data[:, t])
            h_f = (self.Wxh_f @ x_t.T() + self.Whh_f @ h_f.T() + self.bxh_f).tanh().T()
            outputs_f.append(h_f)

        # Backward pass
        for t in reversed(range(seq_len)):
            x_t = Value(x.data[:, t])
            h_b = (self.Wxh_b @ x_t.T() + self.Whh_b @ h_b.T() + self.bxh_b).tanh().T()
            outputs_b.insert(0, h_b)

        # Merge outputs
        merged = [Value.cat([f, b], dim=1) for f, b in zip(outputs_f, outputs_b)]

        if self.return_sequences:
            return Value.cat([v.unsqueeze(1) for v in merged], dim=1)  # Shape: (batch, seq_len, 2 * units)
        else:
            return merged[-1]  # Shape: (batch, 2 * units)

    def load_weights(self, weights):
        """
        Load weights from a tuple: (Wxh_f, Whh_f, bxh_f, Wxh_b, Whh_b, bxh_b)
        Each element should be a NumPy array or tensor with matching shape.
        """
        assert len(weights) == 6, "Expected 6 weight tensors"

        self.Wxh_f = Value(torch.tensor(weights[0].T, dtype=torch.float32), requires_grad=True)
        self.Whh_f = Value(torch.tensor(weights[1].T, dtype=torch.float32), requires_grad=True)
        self.bxh_f = Value(torch.tensor(weights[2].reshape(-1, 1), dtype=torch.float32), requires_grad=True)

        self.Wxh_b = Value(torch.tensor(weights[3].T, dtype=torch.float32), requires_grad=True)
        self.Whh_b = Value(torch.tensor(weights[4].T, dtype=torch.float32), requires_grad=True)
        self.bxh_b = Value(torch.tensor(weights[5].reshape(-1, 1), dtype=torch.float32), requires_grad=True)

    def __call__(self, x):
        return self.forward(x)
