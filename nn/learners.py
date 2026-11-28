import os
import numpy as np
import torch

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader


"""
learners
--------
Learner base class copied from Lab 2 (learners.py). ValueLearner follows ClassificationLearner:
same train loop with validation every epoch, 'leader' checkpoint at the lowest validation loss.
New compared to Lab 2: the accuracy is 'result predicted on the right side of 0.5' on decisive
positions, and export_weights() for numpy inference in the engine.
"""


class Learner():

    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def save(self, label):
        if not os.path.exists(os.path.join("checkpoints")):
            os.makedirs(os.path.join("checkpoints"))
        torch.save(self.model.state_dict(), os.path.join("checkpoints", self.model.__class__.__name__ + "_" + label))

    def load(self, label):
        self.model.load_state_dict(torch.load(os.path.join("checkpoints", self.model.__class__.__name__ + "_" + label), weights_only=True))

    def export_weights(self, path: str):
        # Save the weights as numpy arrays, so nn_utility.py can run the network without PyTorch
        arrays = {name: tensor.detach().cpu().numpy() for name, tensor in self.model.state_dict().items()}
        np.savez(path, **arrays)


class ValueLearner(Learner):

    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
        super().__init__(model, criterion, optimizer)

    def train(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, epochs: int, device: torch.device):

        self.model.to(device)

        train_losses = []
        valid_losses = []
        valid_accuracies = []

        for epoch in range(epochs):

            self.model.train()  # set model in training mode

            train_loss = 0

            for data, targets in train_dataloader:
                data = data.to(device)
                targets = targets.to(device)

                # Zero the gradients of the model parameters
                self.optimizer.zero_grad()
                # Calculate the output of the model
                outputs = self.model(data)
                # Calculate the loss
                loss = self.criterion(outputs, targets)
                # Calculate the gradients wrt. the loss
                loss.backward()
                # Update the model parameters based on the gradients
                self.optimizer.step()

                # Weigh by the batch size: the last minibatch is smaller, so a mean over batches would be skewed
                train_loss += loss.item() * len(data)

            train_loss = train_loss / len(train_dataloader.dataset)

            self.model.eval()  # set model in evaluation mode

            # evaluate the model on the validation dataset
            valid_loss, valid_accuracy = self._evaluate(valid_dataloader, device)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)

            print(f"train epoch [{epoch + 1}/{epochs}]:\ttrain loss = {train_loss:.4f}\tvalid loss = {valid_loss:.4f}\tvalid accuracy = {100 * valid_accuracy:.2f}%")

            if valid_loss <= min(valid_losses):
                self.save("leader")

        print()

        self.save("final")

        return (train_losses, valid_losses, valid_accuracies)

    def test(self, test_dataloader: DataLoader, device: torch.device):

        self.model.to(device)

        self.model.eval()  # set model in evaluation mode

        # evaluate the model on the test dataset
        test_loss, test_accuracy = self._evaluate(test_dataloader, device)

        return (test_loss, test_accuracy)

    @torch.no_grad()
    def _evaluate(self, dataloader, device):
        eval_loss = 0
        correct = 0
        decisive = 0

        for data, targets in dataloader:
            data = data.to(device)
            targets = targets.to(device)

            # Calculate the output of the model
            outputs = self.model(data)
            # Calculate the loss (mean over the minibatch) and weigh it by the minibatch size
            loss = self.criterion(outputs, targets)
            eval_loss += loss.item() * len(data)

            # The output is a logit: the sigmoid gives the predicted score for the side to move
            predictions = torch.sigmoid(outputs)
            # Only positions where one side is clearly better (more than ~80 cp), a balanced position has no right side of 0.5
            mask = (targets - 0.5).abs() > 0.05
            correct += ((predictions > 0.5) == (targets > 0.5))[mask].sum().item()
            decisive += mask.sum().item()

        # Aggregated over samples, not batches
        eval_loss = eval_loss / len(dataloader.dataset)
        eval_accuracy = correct / decisive

        return (eval_loss, eval_accuracy)
