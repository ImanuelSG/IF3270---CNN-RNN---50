from src.layers.layer import Layer
from src.utils.autodiff import Value
import torch


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

        self.max_indices = None
        self.input_shape = None
        self.output_shape = None

    def _calculate_output_size(self, input_height, input_width):
        if self.pool_type in ['globalmax', 'globalavg', 'globalaverage']:
            return 1, 1
        output_height = (input_height + 2 * self.padding[0] - self.pool_size[0]) // self.strides[0] + 1
        output_width = (input_width + 2 * self.padding[1] - self.pool_size[1]) // self.strides[1] + 1
        return output_height, output_width

    def _pad_input(self, x):
        if self.padding == (0, 0):
            return x
        pad_val = float('-inf') if self.pool_type == 'max' else 0.0
        padding = (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
        padded = torch.nn.functional.pad(x.data, padding, value=pad_val)
        return Value(padded, requires_grad=x.requires_grad, _children=(x,), _op='pad')

    def _max_pooling(self, x_padded, B, C, H_out, W_out):
        output = [[[[
            Value(float('-inf')) for _ in range(W_out)
        ] for _ in range(H_out)] for _ in range(C)] for _ in range(B)]

        self.max_indices = [[[[None for _ in range(W_out)]
                               for _ in range(H_out)] for _ in range(C)] for _ in range(B)]

        for b in range(B):
            for c in range(C):
                for oh in range(H_out):
                    for ow in range(W_out):
                        h_start = oh * self.strides[0]
                        w_start = ow * self.strides[1]
                        h_end = h_start + self.pool_size[0]
                        w_end = w_start + self.pool_size[1]

                        max_val = Value(float('-inf'))
                        max_idx = (0, 0)
                        for i in range(h_start, h_end):
                            for j in range(w_start, w_end):
                                val = Value(x_padded.data[b, c, i, j])
                                if val.data > max_val.data:
                                    max_val = val
                                    max_idx = (i, j)
                        output[b][c][oh][ow] = max_val
                        self.max_indices[b][c][oh][ow] = max_idx

        # Convert nested list of Values to tensor
        result_tensor = torch.stack([
            torch.stack([
                torch.stack([
                    torch.stack([v.data for v in row], dim=0)
                    for row in channel
                ], dim=0)
                for channel in batch
            ], dim=0)
            for batch in output
        ], dim=0)

        out = Value(result_tensor, requires_grad=x_padded.requires_grad, _children=(x_padded,), _op='max_pool')

        def _backward():
            if x_padded.requires_grad:
                grad_input = torch.zeros_like(x_padded.data)
                for b in range(B):
                    for c in range(C):
                        for oh in range(H_out):
                            for ow in range(W_out):
                                i, j = self.max_indices[b][c][oh][ow]
                                grad_input[b, c, i, j] += out.grad[b, c, oh, ow]
                x_padded.grad += grad_input

        out._backward = _backward
        return out

    def forward(self, x):
        self.input_shape = x.data.shape
        B, C, H, W = self.input_shape

        if self.pool_type.startswith('global'):
            # Global pooling over full H, W
            if 'max' in self.pool_type:
                # Global max pooling uses _max_pooling logic but on full input
                # We'll implement similarly here:
                output = [[[
                    [Value(float('-inf')) for _ in range(1)]
                ] for _ in range(C)] for _ in range(B)]

                self.max_indices = [[[
                    [None] for _ in range(C)
                ] for _ in range(B)]]

                for b in range(B):
                    for c in range(C):
                        max_val = Value(float('-inf'))
                        max_idx = (0, 0)
                        for i in range(H):
                            for j in range(W):
                                val = Value(x.data[b, c, i, j])
                                if val.data > max_val.data:
                                    max_val = val
                                    max_idx = (i, j)
                        output[b][c][0][0] = max_val
                        self.max_indices[b][c][0][0] = max_idx

                result_tensor = torch.stack([
                    torch.stack([
                        torch.stack([v[0][0].data for v in batch_channel], dim=0)
                        for batch_channel in batch
                    ], dim=0)
                    for batch in output
                ], dim=0)

                out = Value(result_tensor, requires_grad=x.requires_grad, _children=(x,), _op='global_max_pool')

                def _backward():
                    if x.requires_grad:
                        grad_input = torch.zeros_like(x.data)
                        for b in range(B):
                            for c in range(C):
                                i, j = self.max_indices[b][c][0][0]
                                grad_input[b, c, i, j] += out.grad[b, c, 0, 0]
                        x.grad += grad_input

                out._backward = _backward
                return out

            else:
                # Global average pooling using native autodiff:
                output_vals = []
                for b in range(B):
                    batch_vals = []
                    for c in range(C):
                        total = Value(0.0)
                        for i in range(H):
                            for j in range(W):
                                total += x[b, c, i, j]  # Value
                        avg = total / Value(H * W)
                        batch_vals.append([[avg]])
                    output_vals.append(batch_vals)

                result_tensor = torch.stack([
                    torch.stack([
                        torch.stack([v[0][0].data for v in batch_channel], dim=0)
                        for batch_channel in batch
                    ], dim=0)
                    for batch in output_vals
                ], dim=0)

                return Value(result_tensor, requires_grad=x.requires_grad, _children=(x,), _op='global_avg_pool')

        else:
            x_padded = self._pad_input(x)
            padded_H, padded_W = x_padded.data.shape[2], x_padded.data.shape[3]
            H_out, W_out = self._calculate_output_size(padded_H, padded_W)
            self.output_shape = (B, C, H_out, W_out)

            if self.pool_type == 'max':
                return self._max_pooling(x_padded, B, C, H_out, W_out)

            else:  # avg pooling via native autodiff
                output_vals = []
                for b in range(B):
                    batch_vals = []
                    for c in range(C):
                        channel_vals = []
                        for oh in range(H_out):
                            row_vals = []
                            for ow in range(W_out):
                                h_start = oh * self.strides[0]
                                w_start = ow * self.strides[1]
                                h_end = h_start + self.pool_size[0]
                                w_end = w_start + self.pool_size[1]

                                total = Value(0.0)
                                for i in range(h_start, h_end):
                                    for j in range(w_start, w_end):
                                        total += x_padded[b, c, i, j]  # Value addition
                                avg = total / Value(self.pool_size[0] * self.pool_size[1])
                                row_vals.append(avg)
                            channel_vals.append(row_vals)
                        batch_vals.append(channel_vals)
                    output_vals.append(batch_vals)

                result_tensor = torch.stack([
                    torch.stack([
                        torch.stack([
                            torch.stack([v.data for v in row], dim=0)
                            for row in channel
                        ], dim=0)
                        for channel in batch
                    ], dim=0)
                    for batch in output_vals
                ], dim=0)

                return Value(result_tensor, requires_grad=x.requires_grad, _children=(x,), _op='avg_pool')
    
    def get_parameters(self):
        return []
