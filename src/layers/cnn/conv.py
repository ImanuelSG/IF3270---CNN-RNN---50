from src.layers.layer import Layer
import torch
from src.utils.initialize_weights import initialize_weights
from src.utils.autodiff import Value

class Conv2D(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding=0, 
                 activation=None, use_bias=True, kernel_initializer="glorot_uniform", 
                 bias_initializer="zeros"):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        
        self.weights = None
        self.bias = None
        self.input_channels = None
    
    def _initialize_parameters(self, input_channels):  
        self.input_channels = input_channels
        
        weight_shape = (self.filters, input_channels, self.kernel_size[0], self.kernel_size[1])
        weight_tensor = torch.empty(weight_shape, dtype=torch.float32)
        self.weights = initialize_weights(weight_tensor, self.kernel_initializer)
        
        if self.use_bias:
            bias_tensor = torch.empty(self.filters, dtype=torch.float32)
            self.bias = initialize_weights(bias_tensor, self.bias_initializer)

    
    def _calculate_output_size(self, input_height, input_width):
        output_height = (input_height + 2 * self.padding[0] - self.kernel_size[0]) // self.strides[0] + 1
        output_width = (input_width + 2 * self.padding[1] - self.kernel_size[1]) // self.strides[1] + 1
        return output_height, output_width
    
    def _pad_input(self, x):
        if self.padding[0] == 0 and self.padding[1] == 0:
            return x
        
        padding = (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
        padded_data = torch.nn.functional.pad(x.data, padding, mode='constant', value=0)
        
        return Value(padded_data, requires_grad=x.requires_grad, _children=(x,), _op="pad")
    
    def forward(self, x : Value):
        batch_size, input_channels, input_height, input_width = x.data.shape
        
        if self.weights is None or self.bias is None:
            self._initialize_parameters(input_channels)
        
        x_padded = self._pad_input(x) if any(self.padding) else x
        
        padded_height, padded_width = x_padded.data.shape[2], x_padded.data.shape[3]
        output_height, output_width = self._calculate_output_size(padded_height, padded_width)
        
        output = []

        for b in range(batch_size):
            batch_out = []
            for f in range(self.filters):
                filter_out = []
                for oh in range(output_height):
                    row_out = []
                    for ow in range(output_width):
                        h_start = oh * self.strides[0]
                        h_end = h_start + self.kernel_size[0]
                        w_start = ow * self.strides[1]
                        w_end = w_start + self.kernel_size[1]

                        conv_sum = Value(0.0)
                        for c in range(input_channels):
                            input_patch = x_padded[b][c][h_start:h_end, w_start:w_end]
                            kernel_patch = self.weights[f][c]
                            conv_sum = conv_sum + (input_patch * kernel_patch).sum()
                        if self.use_bias:
                            conv_sum += self.bias[f]
                        row_out.append(conv_sum)
                    filter_out.append(row_out)
                batch_out.append(filter_out)
            output.append(batch_out)

        final_output = []
        for b in range(batch_size):
            for f in range(self.filters):
                for i in range(output_height):
                    for j in range(output_width):
                        final_output.append(output[b][f][i][j])
        
        # def print_child(v, max_depth=3):
        #     if max_depth < 0:
        #         return
        #     if (v._prev is not None):
        #         print(f"Child: {v._prev}, Op: {v._op}, Data: {v.data} at depth {max_depth}")
        #         for child in v._prev:
        #             print_child(child, max_depth - 1)
        
        # for v in final_output:
        #     print_child(v)

        out_tensor = torch.stack([v.data for v in final_output])
        shape = (batch_size, self.filters, output_height, output_width)
        out_tensor = out_tensor.view(shape)

        out = Value(out_tensor, requires_grad=True, _children=tuple(final_output), _op="conv2d_output")

        def _backward():
            if out.grad is None:
                return
            grad_tensor = out.grad
            idx = 0
            for b in range(batch_size):
                for f in range(self.filters):
                    for i in range(output_height):
                        for j in range(output_width):
                            final_output[idx].grad += grad_tensor[b, f, i, j]
                            idx += 1

        out._backward = _backward


        if self.activation:
            return getattr(out, self.activation)()

        return out
    
    def get_parameters(self):
        """Return all trainable parameters"""
        params = [self.weights]
        if self.use_bias:
            params.append(self.bias)
        return params
    def backward(self, x):
        return super().backward(x)
