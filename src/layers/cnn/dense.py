from src.layers.layer import Layer
from src.utils.autodiff import Value
from src.utils.initialize_weights import initialize_weights

class DenseLayer(Layer):
    ALLOWED_ACTIVATIONS = {"linear", "relu", "sigmoid", "tanh", "softmax", 'elu', "leaky_relu"}

    def __init__(self, output_shape, activation='linear', init_method=None, input_shape=None):
        if activation not in self.ALLOWED_ACTIVATIONS:
            raise ValueError(f"Invalid activation function '{activation}'. Choose from {self.ALLOWED_ACTIVATIONS}")

        super().__init__()
        self.activation = activation

        self.weights: Value = None 
        self.bias: Value = None  

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.init_method = init_method
    
    def get_parameters(self):
        params = []
        if self.weights is not None:
            params.append(self.weights)
        if self.bias is not None:
            params.append(self.bias)
        return params

    def _initialize_parameters(self, input_dim):
        if self.init_method is None:
            raise ValueError("Initialization method must be provided")

        weight_shape = (self.output_shape, input_dim)
        self.weights = initialize_weights(weight_shape, self.init_method)

        bias_shape = (self.output_shape,)
        self.bias = initialize_weights(bias_shape, "zeros")

    def forward(self, X: Value):
        
        if self.weights is None or self.bias is None:
            input_dim = X.data.shape[1]  
            self._initialize_parameters(input_dim)
            self.input_shape = input_dim

       
        Z = X @ self.weights.T() + self.bias 


        activated = getattr(Z, self.activation)()
        return activated
