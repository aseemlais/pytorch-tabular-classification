import torch


def evaluate(model, data_loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for X_batch, y_batch in data_loader:

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch.float().view(-1, 1)
            )

            total_loss += loss.item()

            predicted_classes = (predictions >= 0.5).float()

            correct += (
                predicted_classes == y_batch.float().view(-1, 1)
            ).sum().item()

            total += y_batch.size(0)

    average_loss = total_loss / len(data_loader)
    accuracy = correct / total

    return average_loss, accuracy

