import numpy as np
from src.layers.layer import Layer


class Pooling(Layer):
    def __init__(self, pool_size, pool_type='max', strides=None, padding=0):
        super().__init__()
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.pool_type = pool_type.lower()
        self.strides = strides if strides else self.pool_size
        self.strides = self.strides if isinstance(self.strides, tuple) else (self.strides, self.strides)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

        valid_types = ['max', 'avg', 'average', 'globalmax', 'globalavg', 'globalaverage']
        if self.pool_type not in valid_types:
            raise ValueError(f"Invalid pool_type: {self.pool_type}. Must be one of {valid_types}")

        # saved for backward
        self.input_shape = None
        self.output_shape = None
        self.max_indices = None  # for max pooling backward
        self.input_padded = None  # keep padded input for backward

    def _pad_input(self, x):
        if self.padding == (0, 0):
            return x
        if self.pool_type == 'max' or self.pool_type == 'globalmax':
            pad_val = -np.inf
        else:
            pad_val = 0.0
        pad_top, pad_left = self.padding
        pad_bottom, pad_right = self.padding
        # np.pad pads as ((before_1, after_1), (before_2, after_2), ...)
        padded = np.pad(
            x,
            ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=pad_val,
        )
        return padded

    def _calculate_output_size(self, input_height, input_width):
        if self.pool_type.startswith('global'):
            return 1, 1
        out_h = (input_height + 2 * self.padding[0] - self.pool_size[0]) // self.strides[0] + 1
        out_w = (input_width + 2 * self.padding[1] - self.pool_size[1]) // self.strides[1] + 1
        return out_h, out_w

    def forward(self, x: np.ndarray):
        # x shape: (B, C, H, W)
        self.input_shape = x.shape
        B, C, H, W = self.input_shape

        # Pad input if needed
        x_padded = self._pad_input(x)
        self.input_padded = x_padded
        H_padded, W_padded = x_padded.shape[2], x_padded.shape[3]

        H_out, W_out = self._calculate_output_size(H, W)
        self.output_shape = (B, C, H_out, W_out)

        out = np.zeros((B, C, H_out, W_out), dtype=x.dtype)

        if self.pool_type in ['max', 'globalmax']:
            # Max pooling
            self.max_indices = np.zeros_like(out, dtype=tuple)  # will store indices as tuples

            for b in range(B):
                for c in range(C):
                    for oh in range(H_out):
                        for ow in range(W_out):
                            h_start = oh * self.strides[0]
                            w_start = ow * self.strides[1]
                            h_end = h_start + self.pool_size[0]
                            w_end = w_start + self.pool_size[1]

                            window = x_padded[b, c, h_start:h_end, w_start:w_end]
                            max_idx = np.unravel_index(np.argmax(window, axis=None), window.shape)
                            out[b, c, oh, ow] = window[max_idx]

                            # Save max indices relative to padded input for backward
                            self.max_indices[b, c, oh, ow] = (h_start + max_idx[0], w_start + max_idx[1])

        elif self.pool_type in ['avg', 'average', 'globalavg', 'globalaverage']:
            # Average pooling
            if self.pool_type.startswith('global'):
                # global average pooling over entire H, W
                for b in range(B):
                    for c in range(C):
                        out[b, c, 0, 0] = np.mean(x_padded[b, c])
            else:
                for b in range(B):
                    for c in range(C):
                        for oh in range(H_out):
                            for ow in range(W_out):
                                h_start = oh * self.strides[0]
                                w_start = ow * self.strides[1]
                                h_end = h_start + self.pool_size[0]
                                w_end = w_start + self.pool_size[1]

                                window = x_padded[b, c, h_start:h_end, w_start:w_end]
                                out[b, c, oh, ow] = np.mean(window)
        else:
            raise NotImplementedError(f"Pooling type {self.pool_type} not implemented.")
        return out

    def backward(self, dout: np.ndarray):
        """
        dout: gradient of loss wrt output, shape = (B, C, H_out, W_out)
        Returns:
            dx: gradient of loss wrt input, shape = self.input_shape
        """

        B, C, H, W = self.input_shape
        dx_padded = np.zeros_like(self.input_padded)

        H_out, W_out = self.output_shape[2], self.output_shape[3]

        if self.pool_type in ['max', 'globalmax']:
            # Backprop only through max locations
            for b in range(B):
                for c in range(C):
                    for oh in range(H_out):
                        for ow in range(W_out):
                            h_idx, w_idx = self.max_indices[b, c, oh, ow]
                            dx_padded[b, c, h_idx, w_idx] += dout[b, c, oh, ow]

        elif self.pool_type in ['avg', 'average', 'globalavg', 'globalaverage']:
            # Backprop average gradient distributed equally
            if self.pool_type.startswith('global'):
                area = self.input_padded.shape[2] * self.input_padded.shape[3]
                for b in range(B):
                    for c in range(C):
                        dx_padded[b, c] += dout[b, c, 0, 0] / area
            else:
                pool_area = self.pool_size[0] * self.pool_size[1]
                for b in range(B):
                    for c in range(C):
                        for oh in range(H_out):
                            for ow in range(W_out):
                                h_start = oh * self.strides[0]
                                w_start = ow * self.strides[1]
                                h_end = h_start + self.pool_size[0]
                                w_end = w_start + self.pool_size[1]

                                dx_padded[b, c, h_start:h_end, w_start:w_end] += dout[b, c, oh, ow] / pool_area
        else:
            raise NotImplementedError(f"Pooling type {self.pool_type} not implemented.")

        # Remove padding if any
        pad_top, pad_left = self.padding
        pad_bottom, pad_right = self.padding

        if pad_top == 0 and pad_left == 0:
            dx = dx_padded
        else:
            dx = dx_padded[:, :, pad_top:dx_padded.shape[2] - pad_bottom, pad_left:dx_padded.shape[3] - pad_right]

        return dx

