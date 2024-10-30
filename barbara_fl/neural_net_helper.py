import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class NeuralNet(nn.Module):
    def __init__(self) -> None:
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output


class NetHelper:
    def __init__(
        self, model, data, device_to_use="gpu", learning_rating=0.7, epochs=8
    ) -> None:
        self.learning_rate: float = learning_rating
        self.epochs: int = epochs

        train_kwargs = {}
        test_kwargs = {}

        match device_to_use:
            case "gpu":
                cuda_kwargs = {
                    "num_workers": 1,
                    "pin_memory": True,
                    "shuffle": True,
                }
                self.device = torch.device("cuda")
                train_kwargs.update(cuda_kwargs)
                test_kwargs.update(cuda_kwargs)
            case _:
                self.device = torch.device("cpu")

        self.model: NeuralNet = copy.deepcopy(model).to(self.device)
        self.data_size = data["train_samples"]

        X_train, y_train, train_samples = (
            data["x_train"],
            data["y_train"],
            data["train_samples"],
        )
        X_test, y_test, test_samples = (
            data["x_test"],
            data["y_test"],
            data["test_samples"],
        )

        self.train_data = TensorDataset(X_train, y_train)
        self.test_data = TensorDataset(X_test, y_test)

        self.train_loader = DataLoader(
            self.train_data, train_samples, **train_kwargs
        )
        self.test_loader = DataLoader(
            self.test_data, test_samples, **test_kwargs
        )

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=learning_rating
        )
        self.loss_function = nn.NLLLoss()

    def train(self):
        self.model.train()

        loss: torch.Tensor = torch.Tensor()
        for epoch in range(self.epochs):
            self.model.train()
            for batch_idx, (X, y) in enumerate(self.train_loader):
                y = y.type(torch.LongTensor)

                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss_function(output, y)
                loss.backward()

                self.optimizer.step()

        return loss.data

    def train_loss(self):
        self.model.train()

        total_loss = 0.0
        loss: torch.Tensor = torch.Tensor()
        for batch_idx, (X, y) in enumerate(self.train_loader):
            y = y.type(torch.LongTensor)
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss_function(output, y)
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def test(self):
        self.model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in self.test_loader:
                y = y.type(torch.LongTensor)
                X, y = X.to(self.device), y.to(self.device)

                output = self.model(X)
                _, predicted = torch.max(output.data, 1)
                total += y.size(0)
                correct += predicted.eq(y.view_as(predicted)).sum().item()

        return 100 * (correct / total)
