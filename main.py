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

# print("X train:")
# print("Shape:", X_train_tensor.shape)
# print("Dtype:", X_train_tensor.dtype)

# print("\ny train:")
# print("Shape:", y_train_tensor.shape)
# print("Dtype:", y_train_tensor.dtype)


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

# print("Training samples:", len(train_dataset))

sample_X, sample_y = train_dataset[0]

# print("Sample X shape:", sample_X.shape)
# print("Sample y:", sample_y)


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

