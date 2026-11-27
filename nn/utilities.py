import matplotlib.pyplot as plt


"""
utilities
---------
Plot function copied from Lab 2 (utilities.py), with saving added for the report.
"""


def plot_training_process(train_losses, valid_losses, path=None):
    figure = plt.figure()
    ax = figure.add_subplot()
    ax.plot(train_losses, label="Training loss")
    ax.plot(valid_losses, label="Validation loss")
    ax.set_title("Training process")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    figure.tight_layout()
    if path is not None:
        figure.savefig(path)
    plt.show()
