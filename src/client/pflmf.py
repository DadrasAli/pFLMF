from src.client.fedper import FedPerClient


import sys
import torch

class pFLMFClient(FedPerClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.v_steps = float(getattr(self.args.pflmf, "num_steps_v", 1))
        self.u_steps = float(getattr(self.args.pflmf, "num_steps_u", 1))
        self.lr_v = float(getattr(self.args.pflmf, "lr_v", 0.0001))
        self.lr_u = float(getattr(self.args.pflmf, "lr_u", 0.001))

def fit(self):
    self.model.train()

    # --- v-steps ---
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = self.lr_v

    for _ in range(int(self.v_steps)):
        for x, y in self.trainloader:
            if len(x) <= 1:
                continue

            x, y = x.to(self.device), y.to(self.device)
            logit = self.model(x)
            loss = self.criterion(logit, y)

            self.optimizer.zero_grad()
            loss.backward()

            # Handle classifier gradients
            if hasattr(self.model, 'classifier') and isinstance(self.model.classifier, torch.nn.Module):
                self.model.classifier.zero_grad()
            else:
                for name, param in self.model.named_parameters():
                    if name in self.personal_params_name and param.grad is not None:
                        param.grad.zero_()

            self.optimizer.step()

    # --- u-steps ---
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = self.lr_u

    for _ in range(int(self.u_steps)):
        for x, y in self.trainloader:
            if len(x) <= 1:
                continue

            x, y = x.to(self.device), y.to(self.device)
            logit = self.model(x)
            loss = self.criterion(logit, y)

            self.optimizer.zero_grad()
            loss.backward()

            # Handle classifier gradients
            if hasattr(self.model, 'classifier') and isinstance(self.model.classifier, torch.nn.Module):
                self.model.classifier.zero_grad()
            else:
                for name, param in self.model.named_parameters():
                    if name in self.personal_params_name and param.grad is not None:
                        param.grad.zero_()

            self.optimizer.step()
