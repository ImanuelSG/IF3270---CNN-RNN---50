import torch

class Value:
    """
    This class implements micrograd-like Value class that supports autodiff (hopefully it implements it correctly) and utilizes torch.Tensor's parallel operation
    References:
    https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
    https://github.com/mattjj/autodidact/blob/master/autograd/numpy/numpy_vjps.py
    https://douglasorr.github.io/2021-11-autodiff/article.html

    We are working to ensure correctness up to 2D only :D
    """
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        if isinstance(data, torch.Tensor):
            self.data = data.float()
        else:
            self.data = torch.tensor(data, dtype=torch.float32)

        self.   requires_grad = requires_grad
        self.grad = torch.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    def T(self):
        out = Value(torch.t(self.data), requires_grad=self.requires_grad, _children=(self,), _op="transpose")

        def _backward():
            if self.requires_grad:
                self.grad += torch.t(out.grad)

        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op="+")

        def _backward():
            if self.requires_grad:
                self.grad += self._reduce_grad(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += self._reduce_grad(out.grad, other.data.shape)

        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op="*")

        def _backward():
            if self.requires_grad:
                self.grad += self._reduce_grad(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += self._reduce_grad(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op="@")

        def _backward():
            if self.requires_grad:
                if self.data.ndim == 0:
                    self.grad += (other.data * out.grad).sum()
                elif self.data.ndim == 1 and other.data.ndim == 1:
                    self.grad += out.grad * other.data
                elif self.data.ndim == 2 and other.data.ndim == 1:
                    self.grad += out.grad[:, None] * other.data
                elif self.data.ndim == 1 and other.data.ndim == 2:
                    self.grad += other.data @ out.grad
                else:
                    self.grad += out.grad @ other.data.T

            if other.requires_grad:
                if other.data.ndim == 0:
                    other.grad += (self.data * out.grad).sum()
                elif self.data.ndim == 1 and other.data.ndim == 1:
                    other.grad += out.grad * self.data
                elif self.data.ndim == 2 and other.data.ndim == 1:
                    other.grad += out.grad @ self.data
                elif self.data.ndim == 1 and other.data.ndim == 2:
                    other.grad += self.data[:, None] * out.grad
                else:
                    other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op="/")

        def _backward():
            if self.requires_grad:
                self.grad += self._reduce_grad(out.grad / other.data, self.data.shape)
            if other.requires_grad:
                other.grad += self._reduce_grad(-out.grad * self.data / (other.data ** 2), other.data.shape)

        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Exponent must be int or float"
        out = Value(self.data ** other, requires_grad=self.requires_grad, _children=(self,), _op=f"**{other}")

        def _backward():
            if self.requires_grad:
                self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out
    
    def exp(self):
        out = Value(torch.exp(self.data), requires_grad=self.requires_grad, _children=(self,), _op="exp")

        def _backward():
            if self.requires_grad:
                self.grad += out.data * out.grad

        out._backward = _backward
        return out
    
    def log(self):
        out = Value(torch.log(self.data), requires_grad=self.requires_grad, _children=(self,), _op="log")

        def _backward():
            if self.requires_grad:
                self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out

    def abs(self):
        out = Value(torch.abs(self.data), requires_grad=self.requires_grad, _children=(self,), _op="abs")

        def _backward():
            self.grad += torch.sign(self.data) * out.grad

        out._backward = _backward
        return out

    def sum(self):
        out = Value(self.data.sum(), requires_grad=self.requires_grad, _children=(self,), _op="sum")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.expand_as(self.data)
        
        out._backward = _backward
        return out

    def mean(self):
        return self.sum() / self.data.numel()
    
    def linear(self):
        return self

    def relu(self):
        out = Value(self.data.clamp(min=0), requires_grad=self.requires_grad, _children=(self,), _op="relu")

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0).float() * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        out = Value(self.data.tanh(), requires_grad=self.requires_grad, _children=(self,), _op="tanh")

        def _backward():
            if self.requires_grad:
                self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        sig = 1 / (1 + torch.exp(-self.data))
        out = Value(sig, requires_grad=self.requires_grad, _children=(self,), _op="sigmoid")

        def _backward():
            if self.requires_grad:
                self.grad += (out.data * (1 - out.data)) * out.grad

        out._backward = _backward
        return out
    
    def leaky_relu(self, negative_slope=0.01):
        out = Value(torch.where(self.data > 0, self.data, negative_slope * self.data), 
                    requires_grad=self.requires_grad, _children=(self,), _op="leaky_relu")

        def _backward():
            if self.requires_grad:
                grad_mask = torch.where(self.data > 0, 1, negative_slope)
                self.grad += grad_mask * out.grad

        out._backward = _backward
        return out
    
    def elu(self, alpha=1.0):
        out = Value(torch.where(self.data > 0, self.data, alpha * (torch.exp(self.data) - 1)),
                    requires_grad=self.requires_grad, _children=(self,), _op="elu")

        def _backward():
            if self.requires_grad:
                grad_mask = torch.where(self.data > 0, 1, out.data + alpha)
                self.grad += grad_mask * out.grad

        out._backward = _backward
        return out

    def softmax(self):
        exp_values = torch.exp(self.data - self.data.max(dim=-1, keepdim=True).values)
        softmax_values = exp_values / exp_values.sum(dim=-1, keepdim=True)
        out = Value(softmax_values, requires_grad=self.requires_grad, _children=(self,), _op="softmax")

        def _backward():
            if self.requires_grad:
                """
                Ref: https://youtu.be/46GOMwvjayc?feature=shared
                Normally for a vector input, the formula is: self.grad += out.data * out.grad - (out.data.T @ out.grad) * out.data
                where @ of the cross term is actually an inner product resulting in scalar

                When we have out.data of dim [batch_size, output_size],
                what we want is a cross term result in the dimension [batch_size, 1].
                Algorithmically:
                    for each instance of data (row)
                        reduce that row of data by multiplying it (eq. with dot product) with the corresponding row in out.grad

                The above reduction is equivalent with doing element wise multiplication followed by a sum in the second (last) axis.
                """
                cross_term = (out.data * out.grad).sum(dim=-1, keepdim=True)

                grad_input = out.data * out.grad - cross_term * out.data
                self.grad += grad_input

        out._backward = _backward
        return out
    
    def clamp(self, min_val, max_val):
        out = Value(torch.clamp(self.data, min_val, max_val), requires_grad=self.requires_grad, _children=(self,), _op="clamp")

        def _backward():
            grad_mask = (self.data >= min_val) & (self.data <= max_val)
            self.grad += grad_mask.float() * out.grad

        out._backward = _backward
        return out
    
    @staticmethod
    def stack(values, dim=0):
        """
        Stack a list of Value instances along a specified dimension.
        This is similar to torch.stack but works with Value objects.
        """
        assert isinstance(dim, int) and dim >= 0, "dim must be a non-negative integer"
        assert len(values) > 0, "values must not be empty"
        assert all(isinstance(v, Value) for v in values), "All elements must be Value instances"

        data_stack = torch.stack([v.data for v in values], dim=dim)
        requires_grad = any(v.requires_grad for v in values)
        out = Value(data_stack, requires_grad=requires_grad, _children=tuple(values), _op=f"stack_dim{dim}")

        def _backward():
            if not out.requires_grad:
                return
            grads = torch.unbind(out.grad, dim=dim)
            for v, g in zip(values, grads):
                if v.requires_grad:
                    if v.grad is None:
                        v.grad = torch.zeros_like(v.data)
                    v.grad += g

        out._backward = _backward
        return out

    def backward(self):
        assert self.requires_grad, "Called backward on a Value that does not require grad"

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = torch.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __neg__(self):
        return self * -1
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rmul__(self, other):
        return self * other
    
    def __rtruediv__(self, other):
        return self / other

    def _reduce_grad(self, grad, target_shape):
        """
        Heavily inspired by autodidact repository
        """
        while len(grad.shape) > len(target_shape):
            grad = grad.sum(dim=0)

        for axis, size in enumerate(target_shape):
            if size == 1:
                grad = grad.sum(dim=axis, keepdim=True)

        return grad
