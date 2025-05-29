import torch
from utils.autodiff import Value
from layers.layer import Layer
from layers.rnn.unidirectionalRNN import UnidirectionalRNN  # import your UnidirectionalRNN

class BidirectionalRNN(Layer):
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
        self.return_sequences = return_sequences

        # Create forward and backward RNNs
        self.forward_rnn = UnidirectionalRNN(
            units,
            input_dim,
            batch_size,
            activation,
            kernel_initializer,
            recurrent_initializer,
            bias_initializer,
            return_sequences=True  # Always True to get full sequence
        )
        self.backward_rnn = UnidirectionalRNN(
            units,
            input_dim,
            batch_size,
            activation,
            kernel_initializer,
            recurrent_initializer,
            bias_initializer,
            return_sequences=True
        )

    def forward(self, x: Value):
        """
        Perform forward and backward RNN passes, then concatenate their outputs.
        Args:
            x (Value): Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Value: Concatenated output of shape:
                (batch_size, seq_len, 2 * units) if return_sequences=True
                (batch_size, 2 * units) if return_sequences=False
        """
        # Forward RNN
        output_f = self.forward_rnn(x)  # Shape: (batch, seq_len, units)

        # Reverse input sequence for backward pass
        reversed_x = Value(x.data.flip(dims=[1]))
        output_b = self.backward_rnn(reversed_x)  # (batch, seq_len, units)

        # print(x)
        # print(reversed_x)

        # # Reverse backward output to match time order
        # output_b = Value(output_b.data.flip(dims=[1]))

        # # Concatenate along the last dimension (units)
        # merged = Value.cat([output_f, output_b], dim=2)  # (batch, seq_len, 2 * units)

        if self.return_sequences:
            # Reverse backward output to match time order
            output_b = Value(output_b.data.flip(dims=[1]))

            # Concatenate along the last dimension (units)
            merged = Value.cat([output_f, output_b], dim=2)
            return merged
        else:
            return Value.cat(
                [Value(output_f.data[:, -1, :]), Value(output_b.data[:, -1, :])], dim=1
            )  
        # Return last timestep (batch, 2 * units)

    def load_weights(self, weights):
        """
        Load weights into forward and backward RNNs.
        Args:
            weights (tuple): Tuple of 6 tensors:
                (Wxh_f, Whh_f, bxh_f, Wxh_b, Whh_b, bxh_b)
        """
        assert len(weights) == 6, "Expected 6 tensors: (Wxh_f, Whh_f, bxh_f, Wxh_b, Whh_b, bxh_b)"
        self.forward_rnn.load_weights((weights[0], weights[1], weights[2]))
        self.backward_rnn.load_weights((weights[3], weights[4], weights[5]))
        return self

    def __call__(self, x):
        return self.forward(x)
