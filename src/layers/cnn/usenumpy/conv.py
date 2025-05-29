import numpy as np
from src.layers.layer import Layer
from src.utils.initialize_weights import initialize_weights
from src.utils.activations import ACTIVATIONS


class Conv2D(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding=0,
                 activation=None, use_bias=True, kernel_initializer="glorot_uniform",
                 bias_initializer="zeros"):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)

        if isinstance(padding, str):
            if padding.lower() == 'same':
                self.padding_mode = 'same'
                self.padding = None
            elif padding.lower() == 'valid':
                self.padding_mode = 'valid'
                self.padding = (0, 0)
            else:
                raise ValueError(f"Unsupported padding mode: {padding}")
        else:
            self.padding_mode = 'manual'
            self.padding = padding if isinstance(padding, tuple) else (padding, padding)

        self.activation_name = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        if activation:
            if activation not in ACTIVATIONS:
                raise ValueError(f"Invalid activation '{activation}'. Choose from {list(ACTIVATIONS.keys())}")
            self.activation_fn, self.activation_derivative_fn = ACTIVATIONS[activation]
        else:
            self.activation_fn = self.activation_derivative_fn = None

        self.weights = None
        self.bias = None
        self.grad_weights = None
        self.grad_bias = None

        self.input = None
        self.input_padded = None
        self.input_channels = None
        self.Z = None
        self.A = None
        self.padding_h = None
        self.padding_w = None

    def _calculate_same_padding(self, input_height, input_width):
        pad_h = max(0, (self.kernel_size[0] - 1))
        pad_w = max(0, (self.kernel_size[1] - 1))
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return (pad_top, pad_bottom), (pad_left, pad_right)

    def _initialize_parameters(self, input_channels):
        self.input_channels = input_channels
        shape = (self.filters, input_channels, *self.kernel_size)
        self.weights = initialize_weights(np.empty(shape, dtype=np.float64), self.kernel_initializer)
        if self.use_bias:
            self.bias = initialize_weights(np.empty((self.filters,), dtype=np.float64), self.bias_initializer)

    def _pad_input(self, x, padding_h, padding_w):
        if padding_h == (0, 0) and padding_w == (0, 0):
            return x
        return np.pad(x, ((0, 0), (0, 0), padding_h, padding_w), mode='constant')

    def _calculate_output_size(self, H, W, padding_h, padding_w):
        out_H = (H + sum(padding_h) - self.kernel_size[0]) // self.strides[0] + 1
        out_W = (W + sum(padding_w) - self.kernel_size[1]) // self.strides[1] + 1
        return out_H, out_W

    def forward(self, x):
        self.input = x
        N, C, H, W = x.shape

        if self.weights is None:
            self._initialize_parameters(C)

        if self.padding_mode == 'same':
            padding_h, padding_w = self._calculate_same_padding(H, W)
        elif self.padding_mode == 'valid':
            padding_h, padding_w = (0, 0), (0, 0)
        else:
            padding_h = (self.padding[0], self.padding[0])
            padding_w = (self.padding[1], self.padding[1])

        self.padding_h = padding_h
        self.padding_w = padding_w

        x_pad = self._pad_input(x, padding_h, padding_w)
        self.input_padded = x_pad

        out_H, out_W = self._calculate_output_size(H, W, padding_h, padding_w)
        out = np.zeros((N, self.filters, out_H, out_W), dtype=np.float64)

        for n in range(N):
            if n % 100 == 0:
                print(f"Processing batch {n + 1}/{N}")
            for f in range(self.filters):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * self.strides[0]
                        w_start = j * self.strides[1]
                        h_end = h_start + self.kernel_size[0]
                        w_end = w_start + self.kernel_size[1]

                        region = x_pad[n, :, h_start:h_end, w_start:w_end]
                        out[n, f, i, j] = np.sum(region * self.weights[f])
                        if self.use_bias:
                            out[n, f, i, j] += self.bias[f]

        self.Z = out
        self.A = self.activation_fn(out) if self.activation_fn else out
        return self.A

    def backward(self, dA):
        N, _, out_H, out_W = dA.shape
        dx_padded = np.zeros_like(self.input_padded)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias) if self.use_bias else None

        if self.activation_fn:
            dZ = dA * self.activation_derivative_fn(self.Z)
        else:
            dZ = dA

        for n in range(N):
            for f in range(self.filters):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * self.strides[0]
                        w_start = j * self.strides[1]
                        h_end = h_start + self.kernel_size[0]
                        w_end = w_start + self.kernel_size[1]

                        region = self.input_padded[n, :, h_start:h_end, w_start:w_end]
                        self.grad_weights[f] += dZ[n, f, i, j] * region
                        dx_padded[n, :, h_start:h_end, w_start:w_end] += dZ[n, f, i, j] * self.weights[f]
                        if self.use_bias:
                            self.grad_bias[f] += dZ[n, f, i, j]

        ph_top, ph_bottom = self.padding_h
        pw_left, pw_right = self.padding_w
        if (ph_top, ph_bottom) == (0, 0) and (pw_left, pw_right) == (0, 0):
            return dx_padded
        return dx_padded[:, :, ph_top:-ph_bottom or None, pw_left:-pw_right or None]

    def load_parameters(self, weights, bias=None):
        weights = weights.transpose(3, 2, 0, 1)
        self.input_channels = weights.shape[1]
        if weights.shape != (self.filters, self.input_channels, self.kernel_size[0], self.kernel_size[1]):
            raise ValueError(f"Expected weights of shape {(self.filters, self.input_channels, *self.kernel_size)}, but got {weights.shape}")
        self.weights = weights.astype(np.float64)

        if self.use_bias:
            if bias is None:
                raise ValueError("Bias is required but not provided.")
            if bias.shape != (self.filters,):
                raise ValueError(f"Expected bias of shape ({self.filters},), but got {bias.shape}")
            self.bias = bias.astype(np.float64)
