import torch
import torch.nn.init as init
from utils.autodiff import Value
from utils.initialize_weights import initialize_weights
class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_dim, initializer='normal'):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weights = initialize_weights(torch.empty(vocab_size, embedding_dim), initializer)
    
    def forward(self, indices):
        out = Value(self.weights.data[indices], requires_grad=self.weights.requires_grad)

        def _backward():
            if self.weights.requires_grad:
                self.weights.grad.index_add_(
                    0,
                    indices.view(-1), # shape (batch_size, vocab_size)
                    out.grad.view(-1, self.embedding_dim) # shape (batch_size * vocab_size, embedding_dim)
                )

        out._backward = _backward
        out._prev = {self.weights}
        return out

    def __call__(self, indices):
        return self.forward(indices)
    
