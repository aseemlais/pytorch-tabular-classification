import torch
import joblib

from src.model import RiceClassifier


# Load scaler
scaler = joblib.load(
    "models/scaler.pkl"
)

# Create model architecture
model = RiceClassifier()

# Load trained parameters
model.load_state_dict(
    torch.load(
        "models/rice_classifier.pth",
        map_location="cpu"
    )
)

# Evaluation mode
model.eval()

print("Model and scaler loaded successfully!")
import pandas as pd
DATA_PATH = "data/raw/rice-type-classification/riceClassification.csv"

data_df = pd.read_csv(DATA_PATH)

data_df.dropna(inplace=True)

data_df.drop(["id"], axis=1, inplace=True)

X = data_df.drop("Class", axis=1)
y = data_df["Class"]
from sklearn.model_selection import train_test_split
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
sample = X_test.iloc[0]

actual_class = y_test.iloc[0]

print("\nRaw sample:")
print(sample)

print("\nActual class:", actual_class)
sample_array = sample.to_numpy().reshape(1, -1)
sample_scaled = scaler.transform(sample_array)
sample_tensor = torch.tensor(
    sample_scaled,
    dtype=torch.float32
)
with torch.no_grad():

    probability = model(sample_tensor)

predicted_class = (
    probability >= 0.5
).int().item()
print("\nPrediction probability:", probability.item())
print("Predicted class:", predicted_class)
print("Actual class:", actual_class)
