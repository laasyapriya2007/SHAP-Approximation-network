from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

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

print("\nCreating Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)

print("\nTraining started...")

model.fit(X_train, y_train)

print("Training completed")

model_path = CHECKPOINT_DIR / "random_forest_weights.pkl"

joblib.dump(model, model_path)

print(f"\nModel saved to:\n{model_path}")

print("\nDone!")
