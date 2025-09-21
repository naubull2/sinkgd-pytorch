import torch
from torch.optim.optimizer import Optimizer


class SinkGD(Optimizer):
    def __init__(self, params, lr=1e-4, sinkhorn_iterations=5, eps=1e-6):
        """
        A simple implementation of the SinkGD optimizer.
        Args:
            params (iterable)        : Iterable of parameters to optimize.
            lr (float)               : Learning rate.
            sinkhorn_iterations (int): Number of alternating row/column normalizations.
            eps (float)              : Small value to prevent division by zero.
        """
        defaults = dict(lr=lr, sinkhorn_iterations=sinkhorn_iterations, eps=eps)
        super(SinkGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            iterations = group['sinkhorn_iterations']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                
                # We only apply Sinkhorn to 2D matrices (weights of Linear layers)
                if grad.dim() == 2:
                    normalized_grad = grad.clone()

                    # The core Sinkhorn alternating normalization loop
                    for _ in range(iterations):
                        # Row normalization
                        row_norm = torch.linalg.norm(normalized_grad, ord=2, dim=1, keepdim=True)
                        normalized_grad = normalized_grad / (row_norm + eps)
                        
                        # Column normalization
                        col_norm = torch.linalg.norm(normalized_grad, ord=2, dim=0, keepdim=True)
                        normalized_grad = normalized_grad / (col_norm + eps)
                    
                    update_tensor = normalized_grad
                else:
                    # For non-matrix params (biases, etc.), just normalize the vector.
                    update_tensor = grad / (torch.linalg.norm(grad) + eps)

                p.add_(update_tensor, alpha=-lr)
                
        return loss
