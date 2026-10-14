from pathlib import Path
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").values.ravel()

print(f"X_train shape : {X_train.shape}")
print(f"y_train shape : {y_train.shape}")

print("\nCreating Logistic Regression model...")

model = LogisticRegression(
    max_iter=5000,
    solver="saga",
    multi_class="multinomial",
    n_jobs=-1,
    random_state=42
)

print("Training started...\n")

model.fit(X_train, y_train)

print("Training completed")

model_path = CHECKPOINT_DIR / "log_reg_weights.pkl"

joblib.dump(model, model_path)

print(f"\nModel saved to:")
print(model_path)

print("\nDone!")
