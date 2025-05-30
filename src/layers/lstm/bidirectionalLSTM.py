from utils.autodiff import Value
from layers.layer import Layer
from layers.lstm.unidirectionalLSTM import UnidirectionalLSTM

class BidirectionalLSTM(Layer):
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

        # forward & backward LSTM
        self.forward_lstm = UnidirectionalLSTM(
            units,
            input_dim,
            batch_size,
            activation,
            kernel_initializer,
            recurrent_initializer,
            bias_initializer,
            return_sequences=True
        )
        self.backward_lstm = UnidirectionalLSTM(
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
        # Forward LSTM
        output_f = self.forward_lstm(x)

        # Reverse input sequence for backward pass
        reversed_x = Value(x.data.flip(dims=[1]))
        output_b = self.backward_lstm(reversed_x)

        if self.return_sequences:
            output_b = Value(output_b.data.flip(dims=[1]))

            merged = Value.cat([output_f, output_b], dim=2)
            return merged
        else:
            return Value.cat(
                [Value(output_f.data[:, -1, :]), Value(output_b.data[:, -1, :])], dim=1
            )

    def load_weights(self, weights):
        if len(weights) == 6:
            self.forward_lstm.load_weights((weights[0], weights[1], weights[2]))
            self.backward_lstm.load_weights((weights[3], weights[4], weights[5]))
        elif len(weights) == 24:
            # Custom format: 12 weights for forward, 12 weights for backward
            forward_weights = weights[:12]
            backward_weights = weights[12:]
            self.forward_lstm.load_weights(forward_weights)
            self.backward_lstm.load_weights(backward_weights)
        else:
            raise ValueError(f"Expected 6 or 24 weight tensors, got {len(weights)}")
        
        return self

    def get_parameters(self):
        parameters = []
        parameters.extend(self.forward_lstm.get_parameters())
        parameters.extend(self.backward_lstm.get_parameters())
        return parameters

    def __call__(self, x):
        return self.forward(x)