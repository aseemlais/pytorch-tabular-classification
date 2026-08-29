import joblib
from src.data_loader import (
    load_data,
    split_features_target,
    split_data,
    scale_data,
    convert_to_tensors
)


data_df = load_data()

X, y = split_features_target(data_df)

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_data(
    X_train,
    X_val,
    X_test
)
joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Scaler saved successfully!")

(
    X_train_tensor,
    X_val_tensor,
    X_test_tensor,
    y_train_tensor,
    y_val_tensor,
    y_test_tensor
) = convert_to_tensors(
    X_train_scaled,
    X_val_scaled,
    X_test_scaled,
    y_train,
    y_val,
    y_test
)


print("X train:")
print("Shape:", X_train_tensor.shape)
print("Dtype:", X_train_tensor.dtype)

print("\ny train:")
print("Shape:", y_train_tensor.shape)
print("Dtype:", y_train_tensor.dtype)


from src.dataset import RiceDataset
train_dataset = RiceDataset(
    X_train_tensor,
    y_train_tensor
)

val_dataset = RiceDataset(
    X_val_tensor,
    y_val_tensor
)

test_dataset = RiceDataset(
    X_test_tensor,
    y_test_tensor
)

print("Training samples:", len(train_dataset))

sample_X, sample_y = train_dataset[0]

print("Sample X shape:", sample_X.shape)
print("Sample y:", sample_y)


from torch.utils.data import DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

X_batch, y_batch = next(iter(train_loader))

print("\nBatch X shape:", X_batch.shape)
print("Batch y shape:", y_batch.shape)

from src.model import RiceClassifier
model = RiceClassifier()

# print(model)
X_batch, y_batch = next(iter(train_loader))

predictions = model(X_batch)

print("\nBatch shape:", X_batch.shape)
print("Prediction shape:", predictions.shape)
print("Predictions:")
print(predictions[:5])

from src.training import create_loss_and_optimizer
model = RiceClassifier()
criterion, optimizer = create_loss_and_optimizer(model)

loss = criterion(predictions, y_batch.float().view(-1, 1))

print("\nLoss:", loss.item())

# Save the initial loss
initial_loss = loss.item()

# Clear previous gradients
optimizer.zero_grad()

# Backpropagation
loss.backward()

# Update model parameters
optimizer.step()

# Forward pass again after updating weights
new_predictions = model(X_batch)

new_loss = criterion(
    new_predictions,
    y_batch.float().view(-1, 1)
)

print("\nInitial loss:", initial_loss)
print("New loss:", new_loss.item())

from src.training import create_loss_and_optimizer, train_one_epoch
criterion, optimizer = create_loss_and_optimizer(model)

 
from src.evaluation import evaluate
EPOCHS = 10

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(EPOCHS):

    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    val_loss, val_accuracy = evaluate(
        model,
        val_loader,
        criterion
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

from src.visualization import plot_loss, plot_accuracy
plot_loss(train_losses, val_losses)

plot_accuracy(val_accuracies)

test_loss, test_accuracy, test_targets, test_predictions = evaluate(
    model,
    test_loader,
    criterion,
    return_predictions=True
)

print("\nFinal Test Results")
print("------------------")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

cm = confusion_matrix(
    test_targets,
    test_predictions
)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        test_targets,
        test_predictions
    )
)

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks([0, 1], ["Class 0", "Class 1"])
plt.yticks([0, 1], ["Class 0", "Class 1"])

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.show()

import torch
torch.save(
    model.state_dict(),
    "models/rice_classifier.pth"
)

print("\nModel saved successfully!")
