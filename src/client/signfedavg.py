from src.client.fedavg import FedAvgClient
import random
import math
from copy import deepcopy
import torch

class SignFedAvgClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.lr_server = float(getattr(self.args.signfedavg, "lr_server", 1))


    def package(self):
        """Package data that client needs to transmit to the server."""
        model_params = self.model.state_dict()
        
        lr_scheduler_state = {} if self.lr_scheduler is None else deepcopy(self.lr_scheduler.state_dict())
        
        client_package = dict(
            weight=len(self.trainset),
            eval_results=self.eval_results,
            regular_model_params={
                key: model_params[key].clone().cpu() for key in self.regular_params_name
            },
            personal_model_params={
                key: model_params[key].clone().cpu()
                for key in self.personal_params_name
            },
            optimizer_state=deepcopy(self.optimizer.state_dict()),
            lr_scheduler_state=lr_scheduler_state,
        )
        
        # Print the client's current learning rate in green
        #print(f"\033[92mClient {self.client_id} current learning rate: {self.optimizer.param_groups[0]['lr']:.6f}\033[0m")
        
    # If returning the difference, compute the l1 norm times the elementwise sign of the difference.
        if self.return_diff:
            model_params_diff = {}
            for key, param_new in client_package["regular_model_params"].items():
                param_old = self.regular_model_params[key]
                param_diff = param_old - param_new
                l1_norm = param_diff.abs().sum()
                dimension = param_diff.numel()  # Total number of elements in the tensor
                model_params_diff[key] = self.lr_server * (l1_norm * param_diff.sign()) / dimension  #(dimension ** 0.5)
            client_package["model_params_diff"] = model_params_diff
            client_package.pop("regular_model_params")
        return client_package

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

                # Normal gradient update
                self.optimizer.step()

