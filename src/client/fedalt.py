from src.client.fedper import FedPerClient


class FedAltClient(FedPerClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.num_steps_base = int(getattr(self.args.fedalt, "num_steps_base", 1))
        self.num_steps_classifier = int(getattr(self.args.fedalt, "num_steps_classifier", 1))

    def fit(self):
        self.model.train()
        self.dataset.train()
        for _ in range(self.num_steps_base):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.model.classifier.zero_grad()
                self.optimizer.step()

        for _ in range(self.num_steps_classifier):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.model.base.zero_grad()
                self.optimizer.step()


