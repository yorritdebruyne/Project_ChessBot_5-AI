import torch

from fiveaichess.utilities.encoding import NUM_FEATURES


"""
model
-----
The networks, same structure as models.py in Lab 2.
ValueNetwork: 768 -> 128 -> 32 -> 1, the logit of the score for the side to move
(between 0 and 1 after the sigmoid, the engine flips it to white's view).
"""


class ValueNetwork(torch.nn.Module):
    def __init__(self, hidden1: int = 128, hidden2: int = 32):
        super().__init__()
        # Hidden sizes as arguments: model selection (L4) only changes the constructor call
        # Small on purpose: the search evaluates every leaf with this network

        self.fc1 = torch.nn.Linear(NUM_FEATURES, hidden1)
        self.fc2 = torch.nn.Linear(hidden1, hidden2)
        self.fc3 = torch.nn.Linear(hidden2, 1)

    def forward(self, x: torch.Tensor):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(1) # Shape (batch, 1) -> (batch,), same shape as the targets