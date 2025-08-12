from src.client.fedavg import FedAvgClient
import random
import math
import torch

class FedSignClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        # Get parameter p from config, default to 0.5 if not specified.
        # If p is a string, handle special cases like "float('inf')".
        p_val = getattr(self.args.fedsign, "p", 1000)
        if isinstance(p_val, str):
            p_str = p_val.strip().lower()
            if p_str in ["float('inf')", "inf", "infty", "infinity"]:
                self.p = float('inf')
            else:
                self.p = float(p_val)
        else:
            self.p = float(p_val)


    def oracle(self, mode='l_inf_norm'):
        """
        Adjusts gradients based on the value of p:
        - If self.p equals 2, the gradients are left unchanged.
        - Otherwise, there are two modes:
            1. 'l1norm': L1 norm of the gradient * elementwise sign of the gradient.
            2. 'threshold': 0 if |grad| < epsilon, otherwise sign(grad).
        """
        if self.p == 2:
            return  # Do not modify the gradients if p is exactly 2

        # Define epsilon for threshold mode
        epsilon = 1e-5

        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_data = param.grad.data

                if mode == 'l_inf_norm':
                    # Mode 1: Replace gradient with (L1 norm) * (sign of grad)
                    l1_norm = grad_data.abs().sum()
                    param.grad.data = l1_norm * grad_data.sign()

                elif mode == 'l_inf_norm_threshold':
                    # Compute the L1 norm of the gradient
                    l1_norm = grad_data.abs().sum()
                    # Apply thresholding and multiply by the L1 norm
                    param.grad.data = l1_norm * torch.where(
                        grad_data.abs() < epsilon,
                        torch.zeros_like(grad_data),
                        grad_data.sign()
                    )

                else:
                    raise ValueError(f"Unknown mode '{mode}'. Use 'l1norm' or 'threshold'.")





    def fit(self):
        self.model.train()
        self.dataset.train()
        for epoch in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()

                # Process gradients using the oracle method
                self.oracle()

                # Update parameters using optimizer.step() which will use the modified gradients.
                self.optimizer.step()

                # Optionally, include the learning rate scheduler step.
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
