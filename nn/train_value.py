import math
import torch

from nn.dataset import PositionsDataset
from nn.model import ValueNetwork
from nn.learners import ValueLearner
from nn.utilities import plot_training_process


"""
train_value
-----------
Trains the value network, same structure as run_classification.py in Lab 2.
Data: the Lichess sample (Stockfish evaluations), shards written by build_dataset_lichess.py.
Labels = score for the side to move between 0 and 1 (see dataset.py).

Run from the project root: python -m nn.train_value
"""

## PARAMETERS

random_seed = 1

shards_dir = "data/shards_test"
# More data than Kasparov -> fewer epochs needed and a bigger batch size is fine (e.g. epochs = 5, batch_size_train = 1024)
weights_path = "nn/value_weights.npz"

batch_size_train = 512
batch_size_test = 4096

epochs = 3

learning_rate = 0.001
weight_decay = 0  # model selection: compare 0 vs 1e-4

## INITIALISATION

# Manually set seeds for random number generators -> ensures reproducibility of results

torch.manual_seed(random_seed)

## HARDWARE ACCELERATION

# The network is small: CPU is fast enough, and the engine also runs it on the CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("device:", device)
print()

## DATA

train_dataset = PositionsDataset(shards_dir, "train")
valid_dataset = PositionsDataset(shards_dir, "val")
test_dataset = PositionsDataset(shards_dir, "test")

print("train positions:", len(train_dataset))
print("valid positions:", len(valid_dataset))
print("test positions:", len(test_dataset))
print()

# Already split in build_dataset_lichess.py (fixed split via a hash of the FEN), so no random_split here
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size_test, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False)

## MODEL DEFINITION

# Create an instance of the model
model = ValueNetwork()

# Define the loss function: one output (logit) with a target between 0 and 1 -> binary cross-entropy
criterion = torch.nn.BCEWithLogitsLoss()

# Define our optimizer: Adam needs a much lower learning rate than SGD
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
print("neural network architecture:")
print()
print(model)
print()

learner = ValueLearner(model, criterion, optimizer)

## BASELINE

# Loss of always predicting the mean label of the training set: the network has to beat this clearly,
# otherwise it only learned the label distribution
mean_target = train_dataset.targets.mean().item()
# Logit of the mean label: the same prediction for every position, without looking at the board
logit = math.log(mean_target / (1 - mean_target))
baseline_loss = criterion(torch.full_like(test_dataset.targets, logit), test_dataset.targets).item()

print(f"baseline (predict mean {mean_target:.3f}):\ttest loss = {baseline_loss:.4f}")
print()

## TRAIN MODEL

train_losses, valid_losses, valid_accuracies = learner.train(train_dataloader, valid_dataloader, epochs, device)

min_valid_loss = min(valid_losses)
min_valid_loss_index = valid_losses.index(min_valid_loss)
min_valid_accuracy = valid_accuracies[min_valid_loss_index]

print(f"train results:\tvalid loss = {min_valid_loss:.4f}\tvalid accuracy = {100 * min_valid_accuracy:.2f}% (epoch {min_valid_loss_index + 1})")
print()

# plot the training loss versus the validation loss (saved for the report)
plot_training_process(train_losses, valid_losses, "results/value_training_process.png")

## TESTING

# load the best version of the model parameters
learner.load("leader")

test_loss, test_accuracy = learner.test(test_dataloader, device)

print(f"test results:\ttest loss = {test_loss:.4f}\ttest accuracy = {100 * test_accuracy:.2f}%")
print()

## EXPORT

# numpy weights for nn_utility.py (Phase 3c)
learner.export_weights(weights_path)
print("saved", weights_path)
