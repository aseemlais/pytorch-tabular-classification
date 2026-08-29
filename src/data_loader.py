import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/raw/rice-type-classification/riceClassification.csv"


def load_data():
    data_df = pd.read_csv(DATA_PATH)

    # Remove rows containing missing values
    data_df.dropna(inplace=True)

    # Remove ID because it is not a useful feature
    data_df.drop(["id"], axis=1, inplace=True)

    return data_df


def split_features_target(data_df):
    X = data_df.drop("Class", axis=1)
    y = data_df["Class"]

    return X, y


def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def convert_to_tensors(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.to_numpy())
    y_val_tensor = torch.tensor(y_val.to_numpy())
    y_test_tensor = torch.tensor(y_test.to_numpy())

    return (
        X_train_tensor,
        X_val_tensor,
        X_test_tensor,
        y_train_tensor,
        y_val_tensor,
        y_test_tensor
    )
