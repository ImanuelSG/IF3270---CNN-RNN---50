class Module:
    def zero_grad(self):
        for p in self.parameters():
            if p is not None and p.grad is not None:
                p.grad.zero_()

    def parameters(self):
        return []