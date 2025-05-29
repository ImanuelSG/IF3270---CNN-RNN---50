import torch
import torch.nn.init as init
from utils.autodiff import Value
from utils.initialize_weights import initialize_weights
from layers.layer import Layer
class EmbeddingLayer(Layer):
    def __init__(self, vocab_size, embedding_dim, initializer='normal'):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # print(f"Initializing EmbeddingLayer with vocab_size={vocab_size}, embedding_dim={embedding_dim}, initializer={initializer}")
        self.weights = initialize_weights(torch.empty(vocab_size, embedding_dim), initializer)
    
    def forward(self, indices: torch.Tensor):
        # print(f"Forward pass in EmbeddingLayer with shape={indices.shape}, data type={indices.dtype}")
        out = Value(self.weights.data[indices], requires_grad=self.weights.requires_grad)

        # def _backward():
        #     if self.weights.requires_grad:
        #         self.weights.grad.index_add_(
        #             0,
        #             indices.view(-1), # shape (batch_size, vocab_size)
        #             out.grad.view(-1, self.embedding_dim) # shape (batch_size * vocab_size, embedding_dim)
        #         )

        # out._backward = _backward
        # out._prev = {self.weights}
        return out

    def load_weights(self, weights):
        """
        Load weights for the embedding layer.
        weights: torch.Tensor of shape (vocab_size, embedding_dim)
        """
        assert len(weights) == 1
        weights = weights[0] if isinstance(weights, (list, tuple)) else weights
        assert weights.shape == (self.vocab_size, self.embedding_dim), "Weights shape mismatch"
        if isinstance(self.weights, Value):
            self.weights.data = torch.tensor(weights, dtype=torch.float32)
        else:
            self.weights = Value(torch.tensor(weights, dtype=torch.float32), requires_grad=True)
        return self

    def __call__(self, indices):
        return self.forward(indices)
    
