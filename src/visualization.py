import matplotlib.pyplot as plt


def plot_loss(train_losses, val_losses):

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        epochs,
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")

    plt.legend()
    plt.grid(True)

    plt.show()


def plot_accuracy(val_accuracies):

    epochs = range(1, len(val_accuracies) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        val_accuracies,
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")

    plt.legend()
    plt.grid(True)

    plt.show()

    