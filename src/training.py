import torch
import torch.nn as nn


def create_loss_and_optimizer(model, learning_rate=0.001):
    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    return criterion, optimizer


def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()

    total_loss = 0.0

    for X_batch, y_batch in train_loader:

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(X_batch)

        # Calculate loss
        loss = criterion(
            predictions,
            y_batch.float().view(-1, 1)
        )

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    return average_loss

