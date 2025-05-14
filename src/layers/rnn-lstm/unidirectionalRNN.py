import torch 
import utils.initialize_weights as init
from utils.autodiff import Value

class UnidirectionalRNN:
    def __init__(self, units, activation="tanh", kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', return_sequences=False):
        # Initialize parameters
        self.activation = activation
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.return_sequences = return_sequences

        # Initialize weights
        self.Whh = init.initialize_weights(torch.empty(units, units), recurrent_initializer)
        
        # Initialize biases
        self.bxh = init.initialize_weights(torch.zeros(units, 1), bias_initializer)

    def forward(self, x: torch.Tensor): # Shape: (batch_size, seq_len, feature_size)
        """
        x: Input tensor of shape (batch_size, seq_len, feature_size)
        """
        
        batch_size, seq_len, feature_size = x.shape
        self.Wxh = init.initialize_weights(torch.empty(self.units, feature_size), self.kernel_initializer)
        h = Value(torch.zeros(batch_size, self.units))  # Initial hidden state
        self.outputs = [] # Shape: (seq_len, batch_size, units)
        self.hidden_states = [h]
        
        if self.activation == "tanh":
            for t in range(seq_len):
                h = (self.Wxh @ Value(x[:,t]).T() + self.Whh @ h.T() + self.bxh).tanh().T()  # Shape: (batch_size, units)
                self.hidden_states.append(h)
                self.outputs.append(h)

        if self.return_sequences:
            out_tensor = torch.stack(self.outputs, dim=1) # Shape: (batch_size, seq_len, units)
        else:
            out_tensor = h.data  # Shape: (batch_size, units)

        out = Value(out_tensor, requires_grad=True)
        def _backward():
            if self.return_sequences:
                for t, h in enumerate(self.outputs):
                    if h.grad is None:
                        h.grad = out.grad[:, t]
                    else:
                        h.grad += out.grad[:, t]
            else:
                if self.outputs[-1].grad is None:
                    self.outputs[-1].grad = out.grad
                else:
                    self.outputs[-1].grad += out.grad

            for h in reversed(self.hidden_states):
                h.backward()
        out._backward = _backward
        out._prev = {self.Wxh, self.Whh, self.bxh}
        out._prev.update(self.hidden_states)   
        return out
    
    def __call__(self, x):
        return self.forward(x)

    # def backward(self, grad_output=None):
    #     if grad_output is not None:
    #         if self.return_sequences:
    #             for t, h in enumerate(self.outputs):
    #                 h.grad = grad_output[:, t] # Shape: (batch_size, units)
    #         else:
    #             self.hidden_states[-1].grad = grad_output
    #     else:
    #         pass

    #     for h in reversed(self.hidden_states):
    #         h.backward()