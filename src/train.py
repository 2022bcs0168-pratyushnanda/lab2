import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from data_loader import load_data
from preprocessing import preprocess
from model import get_model

# Paths
DATA_PATH = "dataset/winequality-red.csv"
MODEL_DIR = "outputs/models"
RESULTS_DIR = "outputs/results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data
X, y = load_data(DATA_PATH)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing (CHANGE FLAG IF NEEDED)
X_train, X_test = preprocess(X_train, X_test, scale=False)

# Model
model = get_model()
model.fit(X_train, y_train)

# Evaluation
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

# Print metrics (REQUIRED)
print(f"MSE: {mse}")
print(f"R2_SCORE: {r2}")

# Save model
model_path = f"{MODEL_DIR}/model.joblib"
joblib.dump(model, model_path)

# Save metrics
results = {
    "mse": mse,
    "r2_score": r2
}

results_path = f"{RESULTS_DIR}/metrics.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

print("Training completed successfully.")

