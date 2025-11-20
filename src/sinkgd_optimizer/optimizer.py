import torch
import math
from torch.optim.optimizer import Optimizer


class SinkGD(Optimizer):
    """
    Implements the SinkGD (Sinkhorn Gradient Descent) optimizer proposed in
    'Gradient Multi-Normalization for Stateless and Scalable LLM Training' (Scetbon et al., 2025).

    Key notes regarding Gradient Explosion:
    1. Default LR is 1e-3 (Standard), NOT 2e-2
       - Paper's specific setup where the authors used 512 batchsize, use common sense of LR scaling when running on smaller batches.
    2. Implements 'embedding_heuristic' to ensure Embeddings use Adam, not SinkGD.
    3. Applies strict geometric scaling (sqrt_n, sqrt_m) as per paper.

    Args:
        params (iterable)         : Iterable of parameters to optimize.
        lr (float): Global learning rate. RECOMMENDATION: 1e-3. 
                    (If you use 0.02 like the paper, Adam layers will likely diverge).
        sinkhorn_iter (int)       : Number of normalization iterations (L). Default: 5.
        eps (float)               : Small value to prevent division by zero.
        linear_lr_scale (float)   : Scaling factor (alpha) for SinkGD layers. 
            - If lr=1e-3, try linear_lr_scale=1.0.
            - If lr=2e-2, you MUST use linear_lr_scale=0.05.
        embedding_heuristic (bool): If True, auto-detects embeddings by shape if names are missing.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        sinkhorn_iter: int = 5,
        linear_lr_scale: float = 0.1, # Changed to 0.1 assuming target lr=1e-4
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,       # Default to 0 as the paper didn't use weight decay
        embedding_heuristic: bool = True # Enabled by default for safety
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = dict(
            lr=lr,
            sinkhorn_iter=sinkhorn_iter,
            linear_lr_scale=linear_lr_scale,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            embedding_heuristic=embedding_heuristic
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            
            param_names = group.get('name', None) 

            for idx, p in enumerate(group['params']):
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('SinkGD does not support sparse gradients')
                    grads.append(p.grad)
                    
                    # --- Classification Logic ---
                    is_sinkhorn_eligible = (p.ndim > 1)
                    
                    # 1. Name check (if provided at init)
                    if is_sinkhorn_eligible and param_names:
                        if 'embed' in param_names: is_sinkhorn_eligible = False

                    # 2. Heuristic check (Critical for stability if names missing)
                    if is_sinkhorn_eligible and group['embedding_heuristic']:
                        rows, cols = p.shape
                        # Embeddings are usually [Vocab, Hidden]. Vocab (rows) >> Hidden (cols)
                        if rows > cols * 5: 
                             is_sinkhorn_eligible = False

                    if not is_sinkhorn_eligible:
                        # Prepare Adam State
                        state = self.state[p]
                        if len(state) == 0:
                            state['step'] = 0
                            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        exp_avgs.append(state['exp_avg'])
                        exp_avg_sqs.append(state['exp_avg_sq'])
                        state_steps.append(state['step'])
                    else:
                        # Marker for SinkGD
                        exp_avgs.append(None)
                        exp_avg_sqs.append(None)
                        state_steps.append(None)
            
            self._sinkgd_step(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                group
            )

        return loss

    def _sinkgd_step(self, params, grads, exp_avgs, exp_avg_sqs, state_steps, group):
        lr = group['lr']
        weight_decay = group['weight_decay']
        sinkhorn_iter = group['sinkhorn_iter']
        linear_scale = group['linear_lr_scale']
        beta1, beta2 = group['betas']
        eps = group['eps']

        for i, param in enumerate(params):
            grad = grads[i]
            is_adam_param = exp_avgs[i] is not None

            # Apply weight decay globally
            if weight_decay != 0:
                param.mul_(1 - lr * weight_decay)

            if not is_adam_param:
                # --- SINKGD Update (Stateless) ---
                X = grad.clone()
                m, n = X.shape
                
                sqrt_n = math.sqrt(n)
                sqrt_m = math.sqrt(m)

                for _ in range(sinkhorn_iter):
                    # Row Norm
                    row_norms = torch.linalg.vector_norm(X, dim=1, keepdim=True)
                    # Safety: ensure we don't divide by near-zero
                    row_norms.clamp_min_(eps)
                    X.div_(row_norms).mul_(sqrt_n)

                    # Col Norm
                    col_norms = torch.linalg.vector_norm(X, dim=0, keepdim=True)
                    col_norms.clamp_min_(eps)
                    X.div_(col_norms).mul_(sqrt_m)
                
                # Scale down if needed (alpha)
                effective_lr = lr * linear_scale
                param.add_(X, alpha=-effective_lr)

            else:
                # --- AdamW Update (Stateful) ---
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step = state_steps[i] + 1
                
                self.state[param]['step'] = step

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** step)).add_(eps)
                step_size = lr / (1 - beta1 ** step)
                
                param.addcdiv_(exp_avg, denom, value=-step_size)
