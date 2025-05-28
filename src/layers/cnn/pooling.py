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

                        max_val = Value(float('-inf'), )
                        max_idx = (0, 0)
                        for i in range(h_start, h_end):
                            for j in range(w_start, w_end):
                                val = x_padded[b, c, i, j]
                                if val.data > max_val.data:
                                    max_val = val
                                    max_idx = (i, j)
                        output[b][c][oh][ow] = max_val
                        self.max_indices[b][c][oh][ow] = max_idx

        final_output = []
        for b in range(B):
            for c in range(C):
                for oh in range(H_out):
                    for ow in range(W_out):
                        final_output.append(output[b][c][oh][ow])
        

        out_tensor = torch.stack([v.data for v in final_output])
        shape = (B, C, H_out, W_out)
        out_tensor = out_tensor.view(shape)

        out = Value(out_tensor, requires_grad=x_padded.requires_grad, _children=tuple(final_output), _op='max_pool')

        def _backward():
            if out.grad is None:
                return
            
            grad_tensor = out.grad
            idx = 0
            for b in range(B):
                for c in range(C):
                    for oh in range(H_out):
                        for ow in range(W_out):
                           
                            final_output[idx].grad += grad_tensor[b, c, oh, ow]
                            idx += 1

        out._backward = _backward
        return out

    def forward(self, x):
      
        self.input_shape = x.data.shape
        B, C, H, W = self.input_shape

        if self.pool_type.startswith('global'):
            # Global pooling over full H, W
            if 'max' in self.pool_type:
                # Global max pooling
                output = [[[
                    [Value(float('-inf'))]
                ] for _ in range(C)] for _ in range(B)]

                self.max_indices = [[[
                    [None]
                ] for _ in range(C)] for _ in range(B)]

                for b in range(B):
                    for c in range(C):
                        max_val = Value(float('-inf'))
                        max_idx = (0, 0)
                        for i in range(H):
                            for j in range(W):
                                val = x[b, c, i, j]
                                if val.data > max_val.data:
                                    max_val = val
                                    max_idx = (i, j)
                        output[b][c][0][0] = max_val
                        self.max_indices[b][c][0][0] = max_idx

                # Flatten output for stacking
                final_output = []
                for b in range(B):
                    for c in range(C):
                        final_output.append(output[b][c][0][0])

                out_tensor = torch.stack([v.data for v in final_output])
                shape = (B, C, 1, 1)
                out_tensor = out_tensor.view(shape)

                out = Value(out_tensor, requires_grad=x.requires_grad, _children=tuple(final_output), _op='global_max_pool')

                def _backward():
                    if out.grad is None:
                        return
                    
                    grad_tensor = out.grad
                    idx = 0
                    for b in range(B):
                        for c in range(C):
                            final_output[idx].grad += grad_tensor[b, c, 0, 0]
                            idx += 1

                out._backward = _backward
                return out

            else:
                output_vals = []
                for b in range(B):
                    for c in range(C):
                        total = Value(0.0)
                        for i in range(H):
                            for j in range(W):
                                total += x[b, c, i, j]
                        avg = total / Value(H * W)
                        output_vals.append(avg)

                out_tensor = torch.stack([v.data for v in output_vals])
                shape = (B, C, 1, 1)
                out_tensor = out_tensor.view(shape)

                out = Value(out_tensor, requires_grad=x.requires_grad, _children=tuple(output_vals), _op='global_avg_pool')

                def _backward():
                    if out.grad is None:
                        return
                    
                    grad_tensor = out.grad
                    idx = 0
                    for b in range(B):
                        for c in range(C):
                            output_vals[idx].grad += grad_tensor[b, c, 0, 0]
                            idx += 1

                out._backward = _backward
                return out

        else:
            x_padded = self._pad_input(x)
            padded_H, padded_W = x_padded.data.shape[2], x_padded.data.shape[3]
            H_out, W_out = self._calculate_output_size(padded_H, padded_W)
            self.output_shape = (B, C, H_out, W_out)

            if self.pool_type == 'max':
                return self._max_pooling(x_padded, B, C, H_out, W_out)

            else:  # avg pooling
                output_vals = []
                for b in range(B):
                    for c in range(C):
                        for oh in range(H_out):
                            for ow in range(W_out):
                                h_start = oh * self.strides[0]
                                w_start = ow * self.strides[1]
                                h_end = h_start + self.pool_size[0]
                                w_end = w_start + self.pool_size[1]

                                total = Value(0.0)
                                for i in range(h_start, h_end):
                                    for j in range(w_start, w_end):
                                        total += x_padded[b, c, i, j]
                                avg = total / Value(self.pool_size[0] * self.pool_size[1])
                                output_vals.append(avg)

                out_tensor = torch.stack([v.data for v in output_vals])
                shape = (B, C, H_out, W_out)
                out_tensor = out_tensor.view(shape)

                out = Value(out_tensor, requires_grad=x.requires_grad, _children=tuple(output_vals), _op='avg_pool')

                def _backward():
                    if out.grad is None:
                        return
                    
                    grad_tensor = out.grad
                    idx = 0
                    for b in range(B):
                        for c in range(C):
                            for oh in range(H_out):
                                for ow in range(W_out):
                                    output_vals[idx].grad += grad_tensor[b, c, oh, ow]
                                    idx += 1

                out._backward = _backward
                return out
    
    def get_parameters(self):
        return []