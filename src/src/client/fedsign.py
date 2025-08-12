from src.client.fedavg import FedAvgClient
import random
import math
import torch

class FedSignClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        # Keep the parameter p as a float, default to 0.5
        self.p = float(getattr(self.args.fedsign, "p", 0.5))


    def oracle(self):
        """
        Adjusts gradients based on the value of p:
        - Replace gradient with (L1 norm) * (sign of grad).
        """
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_data = param.grad.data
                # Compute the L1 norm of the gradient
                l1_norm = grad_data.abs().sum()
                # Update the gradient using the L1 norm and the sign of the gradient
                param.grad.data = l1_norm * grad_data.sign()


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
