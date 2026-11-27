import torch

from fiveaichess.utilities.encoding import NUM_FEATURES


"""
model
-----
The networks, same structure as models.py in Lab 2.
ValueNetwork: 768 -> 128 -> 32 -> 1, the logit of the probability that white wins.
"""


class ValueNetwork(torch.nn.Module):
    def __init__(self, hidden1: int = 128, hidden2: int = 32):
        super().__init__()
        # Hidden sizes as arguments: model selection (L4) only changes the constructor call
        # Small on purpose: the search evaluates every leaf with this network

        # TODO: Define the three layers (like NeuralNetworkClassificationModel):
        #       layer1 = Linear(NUM_FEATURES, hidden1) + ReLU
        #       layer2 = Linear(hidden1, hidden2) + ReLU
        #       layer3 = Linear(hidden2, 1), no activation: BCEWithLogitsLoss applies the sigmoid itself

    def forward(self, x: torch.Tensor):
        # TODO: Forward tensor x through all layers

        # Shape (batch, 1) -> (batch,), same shape as the targets
        # TODO: return x.squeeze(1)
        raise NotImplementedError
