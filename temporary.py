import pandas as pd
import joblib

# Load dataset
data = pd.read_csv("dataset.csv")

# Load selected features
features = joblib.load("features.pkl")

# Select only required columns
test_data = data[features]

# Save new CSV
test_data.to_csv("test_data.csv", index=False)

print("test_data.csv created!")