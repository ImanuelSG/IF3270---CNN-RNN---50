import torch
from utils.autodiff import Value
from utils.initialize_weights import initialize_weights
from layers.layer import Layer

class UnidirectionalLSTM(Layer):
    def __init__(
        self, 
        units, 
        input_dim=None, 
        batch_size=None, 
        activation="tanh",
        kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
        bias_initializer='zeros', return_sequences=False
    ):
        super().__init__()
        self.units = units
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.activation = activation
        self.return_sequences = return_sequences

        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer

        # input weight matrix
        self.Wxi = None  # input gate
        self.Wxf = None  # forget gate
        self.Wxo = None  # output gate
        self.Wxc = None  # candidate values

        # hidden weight matrix
        self.Whi = None  # input gate
        self.Whf = None  # forget gate
        self.Who = None  # output gate
        self.Whc = None  # candidate values

        # Bias 
        self.bi = None   # input gate bias
        self.bf = None   # forget gate bias
        self.bo = None   # futput gate bias
        self.bc = None   # candidate bias

        # Initial states h0 dan c0
        self.h0 = None   # Initial hidden state
        self.c0 = None   # Initial cell state

        if input_dim is not None:
            self.build(input_dim)

        if batch_size is not None:
            self.init_states(batch_size)

    def build(self, input_dim):
        # Input weight matrix
        self.Wxi = initialize_weights(torch.empty(self.units, input_dim), self.kernel_initializer)
        self.Wxf = initialize_weights(torch.empty(self.units, input_dim), self.kernel_initializer)
        self.Wxo = initialize_weights(torch.empty(self.units, input_dim), self.kernel_initializer)
        self.Wxc = initialize_weights(torch.empty(self.units, input_dim), self.kernel_initializer)

        # Hidden weight matrix
        self.Whi = initialize_weights(torch.empty(self.units, self.units), self.recurrent_initializer)
        self.Whf = initialize_weights(torch.empty(self.units, self.units), self.recurrent_initializer)
        self.Who = initialize_weights(torch.empty(self.units, self.units), self.recurrent_initializer)
        self.Whc = initialize_weights(torch.empty(self.units, self.units), self.recurrent_initializer)

        # Bias vector
        self.bi = initialize_weights(torch.zeros(self.units, 1), self.bias_initializer)
        self.bf = initialize_weights(torch.zeros(self.units, 1), self.bias_initializer)  # Initialize forget gate bias to 1
        self.bo = initialize_weights(torch.zeros(self.units, 1), self.bias_initializer)
        self.bc = initialize_weights(torch.zeros(self.units, 1), self.bias_initializer)

    def init_states(self, batch_size):
        self.batch_size = batch_size
        self.h0 = Value(torch.zeros(batch_size, self.units))
        self.c0 = Value(torch.zeros(batch_size, self.units))

    def forward(self, x: Value):
        batch_size, seq_len, feature_size = x.data.shape

        if self.Wxi is None:
            self.build(feature_size)

        if self.h0 is None or self.batch_size != batch_size:
            self.init_states(batch_size)

        h = self.h0
        c = self.c0
        self.outputs = []

        for t in range(seq_len):
            x_t = Value(x.data[:, t])
            
            # Input gate
            i_t = (x_t @ self.Wxi.T() + h @ self.Whi.T() + self.bi.T()).sigmoid()
            
            # Forget gate
            f_t = (x_t @ self.Wxf.T() + h @ self.Whf.T() + self.bf.T()).sigmoid()
            
            # Output gate
            o_t = (x_t @ self.Wxo.T() + h @ self.Who.T() + self.bo.T()).sigmoid()
            
            # Candidate value
            c_tilde = (x_t @ self.Wxc.T() + h @ self.Whc.T() + self.bc.T()).tanh()
            
            # Update cell state
            c = f_t * c + i_t * c_tilde
            
            # Update hidden state
            h = o_t * c.tanh()
            
            self.outputs.append(h)

        if self.return_sequences:
            return Value.cat([v.unsqueeze(1) for v in self.outputs], dim=1)  # (batch_size, seq_len, units)
        else:
            return self.outputs[-1]  # (batch_size, units)
    
    def load_weights(self, weights):
        if len(weights) == 3:
            # TensorFlow LSTM format: [kernel, recurrent_kernel, bias]
            kernel, recurrent_kernel, bias = weights
            kernel = kernel.T
            input_dim = kernel.shape[1]
            
            self.Wxi = Value(torch.tensor(kernel[:self.units, :], dtype=torch.float32), requires_grad=True)
            self.Wxf = Value(torch.tensor(kernel[self.units:2*self.units, :], dtype=torch.float32), requires_grad=True)
            self.Wxc = Value(torch.tensor(kernel[2*self.units:3*self.units, :], dtype=torch.float32), requires_grad=True)
            self.Wxo = Value(torch.tensor(kernel[3*self.units:4*self.units, :], dtype=torch.float32), requires_grad=True)
            
            recurrent_kernel = recurrent_kernel.T
            self.Whi = Value(torch.tensor(recurrent_kernel[:self.units, :], dtype=torch.float32), requires_grad=True)
            self.Whf = Value(torch.tensor(recurrent_kernel[self.units:2*self.units, :], dtype=torch.float32), requires_grad=True)
            self.Whc = Value(torch.tensor(recurrent_kernel[2*self.units:3*self.units, :], dtype=torch.float32), requires_grad=True)
            self.Who = Value(torch.tensor(recurrent_kernel[3*self.units:4*self.units, :], dtype=torch.float32), requires_grad=True)
            
            self.bi = Value(torch.tensor(bias[:self.units].reshape(-1, 1), dtype=torch.float32), requires_grad=True)
            self.bf = Value(torch.tensor(bias[self.units:2*self.units].reshape(-1, 1), dtype=torch.float32), requires_grad=True)
            self.bc = Value(torch.tensor(bias[2*self.units:3*self.units].reshape(-1, 1), dtype=torch.float32), requires_grad=True)
            self.bo = Value(torch.tensor(bias[3*self.units:4*self.units].reshape(-1, 1), dtype=torch.float32), requires_grad=True)
            
        elif len(weights) == 12:
            # Custom format: (Wxi, Wxf, Wxo, Wxc, Whi, Whf, Who, Whc, bi, bf, bo, bc)
            Wxi, Wxf, Wxo, Wxc, Whi, Whf, Who, Whc, bi, bf, bo, bc = weights
            
            self.Wxi = Value(torch.tensor(Wxi.T, dtype=torch.float32), requires_grad=True)
            self.Wxf = Value(torch.tensor(Wxf.T, dtype=torch.float32), requires_grad=True)
            self.Wxo = Value(torch.tensor(Wxo.T, dtype=torch.float32), requires_grad=True)
            self.Wxc = Value(torch.tensor(Wxc.T, dtype=torch.float32), requires_grad=True)
            
            self.Whi = Value(torch.tensor(Whi, dtype=torch.float32), requires_grad=True)
            self.Whf = Value(torch.tensor(Whf, dtype=torch.float32), requires_grad=True)
            self.Who = Value(torch.tensor(Who, dtype=torch.float32), requires_grad=True)
            self.Whc = Value(torch.tensor(Whc, dtype=torch.float32), requires_grad=True)
            
            self.bi = Value(torch.tensor(bi.reshape(-1, 1), dtype=torch.float32), requires_grad=True)
            self.bf = Value(torch.tensor(bf.reshape(-1, 1), dtype=torch.float32), requires_grad=True)
            self.bo = Value(torch.tensor(bo.reshape(-1, 1), dtype=torch.float32), requires_grad=True)
            self.bc = Value(torch.tensor(bc.reshape(-1, 1), dtype=torch.float32), requires_grad=True)
        else:
            raise ValueError(f"Expected 3 or 12 weight matrices, got {len(weights)}")
        
        return self

    def get_parameters(self):
        parameters = []
        if self.Wxi is not None:
            parameters.extend([
                self.Wxi, self.Wxf, self.Wxo, self.Wxc,
                self.Whi, self.Whf, self.Who, self.Whc,
                self.bi, self.bf, self.bo, self.bc
            ])
        return parameters

    def __call__(self, x):
        return self.forward(x)