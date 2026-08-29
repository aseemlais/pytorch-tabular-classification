import torch


def evaluate(model, data_loader, criterion, return_predictions=False):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_targets = []

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

            if return_predictions:
                all_predictions.extend(
                    predicted_classes.squeeze(1).cpu().numpy()
                )

                all_targets.extend(
                    y_batch.cpu().numpy()
                )

    average_loss = total_loss / len(data_loader)
    accuracy = correct / total

    if return_predictions:
        return (
            average_loss,
            accuracy,
            all_targets,
            all_predictions
        )

    return average_loss, accuracy
