from layer import Layer
class Conv2D(Layer):
    def __init__(self, filters, kernel_size, strides =(1,1), padding=0):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.bias = None
        self.weights = None

    def forward(self, x):
        # Implement the forward pass for 2D convolution
        for 

        pass

    def backward(self, x):
        return super().backward(x)
